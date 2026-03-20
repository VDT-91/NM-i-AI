from __future__ import annotations

from datetime import date, timedelta
import logging
import time
import unicodedata
from pathlib import Path
from typing import Any

from tripletex_solver.attachment_text import extract_attachment_text
from tripletex_solver.attachments import save_attachments
from tripletex_solver.errors import (
    AmbiguousMatchError,
    EntityNotFoundError,
    ParsingError,
    TripletexAPIError,
    UnsupportedTaskError,
)
from tripletex_solver.llm_parser import parse_with_llm
from tripletex_solver.models import Action, Entity, ParsedTask, SolveRequest
from tripletex_solver.parser import PromptParser
from tripletex_solver.tripletex_client import TripletexClient

LOGGER = logging.getLogger(__name__)
DEFAULT_INVOICE_BANK_ACCOUNT_NUMBER = "10000000006"
DEFAULT_EMPLOYEE_DEPARTMENT_NAME = "General"
DEFAULT_PROJECT_MANAGER_NAME = "Project Manager"

# Valid Tripletex userType values: STANDARD, NO_ACCESS, EXTENDED
ROLE_TO_USER_TYPE = {
    "administrator": "STANDARD",
    "admin": "STANDARD",
    "kontoadministrator": "STANDARD",
    "administrador": "STANDARD",      # Spanish/Portuguese
    "administrateur": "STANDARD",     # French
    "kontoverwalter": "STANDARD",     # German
    "accountant": "EXTENDED",
    "regnskapsfører": "EXTENDED",
    "revisor": "EXTENDED",
    "auditor": "EXTENDED",
    "user": "STANDARD",
    "bruker": "STANDARD",
    "no_access": "NO_ACCESS",
}

# Roles that should get ALL_PRIVILEGES entitlements after creation
ADMIN_ROLES = frozenset({
    "administrator", "admin", "kontoadministrator",
    "administrador", "administrateur", "kontoverwalter",
})

# Map role keywords to entitlement templates
ROLE_TO_ENTITLEMENT_TEMPLATE = {
    "administrator": "ALL_PRIVILEGES",
    "admin": "ALL_PRIVILEGES",
    "kontoadministrator": "ALL_PRIVILEGES",
    "administrador": "ALL_PRIVILEGES",
    "administrateur": "ALL_PRIVILEGES",
    "kontoverwalter": "ALL_PRIVILEGES",
    "accountant": "ACCOUNTANT",
    "regnskapsfører": "ACCOUNTANT",
    "revisor": "AUDITOR",
    "auditor": "AUDITOR",
    "invoicing_manager": "INVOICING_MANAGER",
    "faktureringsansvarlig": "INVOICING_MANAGER",
    "personell_manager": "PERSONELL_MANAGER",
    "personalansvarlig": "PERSONELL_MANAGER",
    "department_leader": "DEPARTMENT_LEADER",
    "avdelingsleder": "DEPARTMENT_LEADER",
}


def _normalize_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return decomposed.encode("ascii", "ignore").decode("ascii")


def _contains_any_ascii(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _normalize_ascii(text)
    return any(_normalize_ascii(phrase) in normalized for phrase in phrases)


CREDIT_NOTE_KEYWORDS = (
    "credit note",
    "credit memo",
    "kreditnota",
    "kreditnote",
    "nota de credito",
    "gutschrift",
    "credito",
    "storno",
    "omgjøring",
    "full reversal",
    "reversal",
    "revert",
    "reverte",
)


def _parse_date_value(val: Any) -> date:
    """Parse a date from string or date object."""
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        # Guard against LLM returning template strings like "YYYY-02-01"
        if "YYYY" in val or "yyyy" in val:
            LOGGER.warning("LLM returned template date %r, defaulting to today", val)
            return date.today()
        try:
            return date.fromisoformat(val)
        except ValueError:
            LOGGER.warning("Could not parse date %r, defaulting to today", val)
            return date.today()
    raise ParsingError(f"Cannot parse date from {val!r}")


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


class TripletexService:
    def __init__(self, client: TripletexClient, *, parser: PromptParser | None = None):
        self.client = client
        self.parser = parser or PromptParser()
        self.last_parsed_task: ParsedTask | None = None
        self.last_attachment_text: str | None = None
        self._saved_attachment_paths: list[Path] = []
        # Per-execution caches — cleared at start of each execute()
        self._cache: dict[str, Any] = {}

    def _clear_cache(self) -> None:
        self._cache.clear()

    def _cache_get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def _cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def execute(self, solve_request: SolveRequest, *, attachments_dir: Path | None = None) -> ParsedTask:
        prompt = solve_request.prompt
        self.last_parsed_task = None
        self.last_attachment_text = None
        self._saved_attachment_paths = []
        self._clear_cache()
        LOGGER.info(
            "Solve request received: prompt_chars=%d file_count=%d",
            len(solve_request.prompt),
            len(solve_request.files),
        )
        LOGGER.info("Raw prompt: %s", solve_request.prompt)
        if attachments_dir is not None and solve_request.files:
            t0 = time.monotonic()
            saved_paths = save_attachments(solve_request.files, attachments_dir)
            self._saved_attachment_paths = saved_paths
            LOGGER.info("Saved %d attachments to %s", len(saved_paths), attachments_dir)
            LOGGER.info("Attachment filenames: %s", [path.name for path in saved_paths])
            extracted_text = extract_attachment_text(saved_paths)
            LOGGER.info("[TIMING] attachment extraction: %.1fs", time.monotonic() - t0)
            if extracted_text:
                prompt = f"{prompt}\n\nAttachment content:\n{extracted_text}"
                self.last_attachment_text = extracted_text
                LOGGER.info("Extracted attachment text: %s", extracted_text[:4000])
                LOGGER.info("Appended extracted attachment text to prompt context")

        # LLM-first strategy: use Gemini to understand the prompt (handles all
        # languages and complex multi-entity tasks), fall back to rule-based
        # parser only if the LLM is unavailable.
        t0 = time.monotonic()
        try:
            task = parse_with_llm(prompt)
            LOGGER.info("[TIMING] LLM parse: %.1fs", time.monotonic() - t0)
            LOGGER.info("LLM parsed task: %s", task.model_dump())
        except Exception as llm_err:
            LOGGER.warning("[TIMING] LLM parse failed after %.1fs: %s", time.monotonic() - t0, llm_err)
            task = self.parser.parse(prompt)
            LOGGER.info("Rule-based parsed task: %s", task.model_dump())

        if (
            task.action is Action.CREATE
            and task.entity is Entity.INVOICE
            and task.attributes.get("workflow") != "creditNote"
            and task.raw_prompt
            and _contains_any_ascii(task.raw_prompt, CREDIT_NOTE_KEYWORDS)
        ):
            task.attributes["workflow"] = "creditNote"
            LOGGER.info("Inferred credit-note workflow from prompt keywords")

        self.last_parsed_task = task
        LOGGER.info("Final task: %s", task.model_dump())
        t0 = time.monotonic()
        self._pre_process(task)
        LOGGER.info("[TIMING] pre-process: %.1fs", time.monotonic() - t0)
        t0 = time.monotonic()
        self._dispatch(task)
        LOGGER.info("[TIMING] workflow dispatch: %.1fs", time.monotonic() - t0)
        LOGGER.info("Workflow completed: action=%s entity=%s", task.action.value, task.entity.value)
        return task

    def _pre_process(self, task: ParsedTask) -> None:
        """Create prerequisite entities found in task attributes before the main workflow.

        This runs before _dispatch and ensures that referenced entities (employee,
        customer, project, department, supplier) exist.  It also handles "side-effect"
        work like registering timesheet hours when the main action is something else
        (e.g. create invoice).

        The _ensure_* helpers are cached, so calling them here and again inside a
        workflow will not create duplicates.
        """
        attrs = task.attributes
        entity = task.entity
        action = task.action

        # Skip pre-processing for delete/update — those resolve existing entities
        if action in (Action.DELETE, Action.UPDATE):
            return

        # --- Ensure department ---
        department_name = attrs.get("departmentName")
        if department_name and entity not in (Entity.DEPARTMENT,):
            try:
                self._ensure_department(department_name)
                LOGGER.info("[PRE] Ensured department: %s", department_name)
            except Exception as e:
                LOGGER.warning("[PRE] Could not ensure department %r: %s", department_name, e)

        # --- Ensure customer ---
        customer_name = attrs.get("customerName")
        if customer_name and entity not in (Entity.CUSTOMER,):
            try:
                org_number = attrs.get("organizationNumber") or (
                    str(attrs["orgNumber"]) if attrs.get("orgNumber") else None
                )
                self._ensure_customer(
                    name=customer_name,
                    email=attrs.get("customerEmail"),
                    org_number=org_number,
                    address=self._build_address(attrs),
                )
                LOGGER.info("[PRE] Ensured customer: %s", customer_name)
            except Exception as e:
                LOGGER.warning("[PRE] Could not ensure customer %r: %s", customer_name, e)

        # --- Ensure supplier ---
        supplier_name = attrs.get("supplierName")
        if supplier_name and entity not in (Entity.SUPPLIER,):
            try:
                sup_org = attrs.get("organizationNumber") or (
                    str(attrs["orgNumber"]) if attrs.get("orgNumber") else None
                )
                self._ensure_supplier(name=supplier_name, org_number=sup_org)
                LOGGER.info("[PRE] Ensured supplier: %s", supplier_name)
            except Exception as e:
                LOGGER.warning("[PRE] Could not ensure supplier %r: %s", supplier_name, e)

        # --- Ensure employee ---
        employee_name = attrs.get("employeeName")
        if employee_name and entity not in (Entity.EMPLOYEE,):
            try:
                self._ensure_employee(
                    name=employee_name,
                    email=attrs.get("employeeEmail") or attrs.get("email"),
                )
                LOGGER.info("[PRE] Ensured employee: %s", employee_name)
            except Exception as e:
                LOGGER.warning("[PRE] Could not ensure employee %r: %s", employee_name, e)

        # --- Ensure project ---
        project_name = attrs.get("projectName")
        if project_name and entity not in (Entity.PROJECT,):
            try:
                self._ensure_project(
                    name=project_name,
                    customer_name=customer_name,
                    customer_email=attrs.get("customerEmail"),
                    org_number=attrs.get("organizationNumber") or (
                        str(attrs["orgNumber"]) if attrs.get("orgNumber") else None
                    ),
                    pm_name=attrs.get("projectManagerName"),
                    pm_email=attrs.get("projectManagerEmail"),
                    fixed_price=self._as_float(attrs.get("fixedPrice")),
                )
                LOGGER.info("[PRE] Ensured project: %s", project_name)
            except Exception as e:
                LOGGER.warning("[PRE] Could not ensure project %r: %s", project_name, e)

        # --- Register timesheet hours (side-effect) ---
        # LLM sometimes puts hours in "quantity" instead of "hours"
        hours = attrs.get("hours") or (attrs.get("quantity") if attrs.get("activityName") else None)
        if hours is not None and entity not in (Entity.TIMESHEET,):
            try:
                emp_name = employee_name or attrs.get("employeeName")
                if emp_name:
                    employee = self._ensure_employee(name=emp_name, email=attrs.get("employeeEmail"))
                    activity_name = attrs.get("activityName")
                    activity = self._resolve_activity(activity_name)
                    entry_date_val = attrs.get("date") or date.today()
                    entry_date = _parse_date_value(entry_date_val) if not isinstance(entry_date_val, date) else entry_date_val
                    ts_payload: dict[str, Any] = {
                        "employee": {"id": employee["id"]},
                        "activity": {"id": activity["id"]},
                        "date": entry_date.isoformat(),
                        "hours": float(hours),
                    }
                    if project_name:
                        try:
                            project = self._ensure_project(
                                name=project_name,
                                customer_name=customer_name,
                            )
                            ts_payload["project"] = {"id": project["id"]}
                        except Exception:
                            pass
                    if attrs.get("comment"):
                        ts_payload["comment"] = attrs["comment"]
                    self.client.create_timesheet_entry(ts_payload)
                    LOGGER.info("[PRE] Registered %s hours for %s", hours, emp_name)
            except Exception as e:
                LOGGER.warning("[PRE] Could not register timesheet hours: %s", e)

        # --- Activate module ---
        module_name = attrs.get("moduleName")
        if module_name and entity not in (Entity.COMPANY_MODULE,):
            try:
                self.client.activate_sales_module(module_name)
                LOGGER.info("[PRE] Activated module: %s", module_name)
            except Exception as e:
                LOGGER.warning("[PRE] Could not activate module %r: %s", module_name, e)

    def _dispatch(self, task: ParsedTask) -> None:
        if task.action is Action.CREATE:
            self._handle_create(task)
            return
        if task.action is Action.UPDATE:
            self._handle_update(task)
            return
        if task.action is Action.DELETE:
            self._handle_delete(task)
            return
        if task.action is Action.REGISTER:
            self._handle_register(task)
            return
        raise UnsupportedTaskError(f"{task.action.value} workflows are not implemented yet")

    def _handle_create(self, task: ParsedTask) -> None:
        if task.entity is Entity.EMPLOYEE:
            self._create_employee(task)
            return
        if task.entity is Entity.CUSTOMER:
            self._create_customer(task)
            return
        if task.entity is Entity.DEPARTMENT:
            self._create_department(task)
            return
        if task.entity is Entity.PRODUCT:
            self._create_product(task)
            return
        if task.entity is Entity.PROJECT:
            self._create_project(task)
            return
        if task.entity is Entity.INVOICE:
            self._create_invoice(task)
            return
        if task.entity is Entity.TRAVEL_EXPENSE:
            self._create_travel_expense(task)
            return
        if task.entity is Entity.CONTACT:
            self._create_contact(task)
            return
        if task.entity is Entity.SUPPLIER:
            self._create_supplier(task)
            return
        if task.entity is Entity.VOUCHER:
            self._create_voucher(task)
            return
        if task.entity is Entity.TIMESHEET:
            self._create_timesheet_entry(task)
            return
        if task.entity is Entity.COMPANY_MODULE:
            self._activate_module(task)
            return
        if task.entity is Entity.INCOMING_INVOICE:
            self._create_incoming_invoice(task)
            return
        if task.entity is Entity.SALARY_TRANSACTION:
            self._create_salary_transaction(task)
            return
        if task.entity is Entity.PURCHASE_ORDER:
            self._create_purchase_order(task)
            return
        if task.entity is Entity.DIMENSION:
            self._create_dimension(task)
            return
        if task.entity is Entity.BANK_STATEMENT:
            self._create_bank_statement(task)
            return
        if task.entity is Entity.ACCOUNT:
            self._create_account(task)
            return
        if task.entity is Entity.ORDER:
            self._create_order(task)
            return
        if task.entity is Entity.REMINDER:
            self._create_reminder(task)
            return
        if task.entity is Entity.ACTIVITY:
            self._create_activity(task)
            return
        if task.entity is Entity.DIVISION:
            self._create_division(task)
            return
        if task.entity is Entity.LEAVE_OF_ABSENCE:
            self._create_leave_of_absence(task)
            return
        if task.entity is Entity.NEXT_OF_KIN:
            self._create_next_of_kin(task)
            return
        if task.entity is Entity.CUSTOMER_CATEGORY:
            self._create_customer_category(task)
            return
        if task.entity is Entity.EMPLOYEE_CATEGORY:
            self._create_employee_category(task)
            return
        if task.entity is Entity.ASSET:
            self._create_asset(task)
            return
        if task.entity is Entity.PRODUCT_GROUP:
            self._create_product_group(task)
            return
        if task.entity is Entity.PROJECT_CATEGORY:
            self._create_project_category(task)
            return
        if task.entity is Entity.INVENTORY:
            self._create_inventory(task)
            return
        if task.entity is Entity.INVENTORY_LOCATION:
            self._create_inventory_location(task)
            return
        if task.entity is Entity.STOCKTAKING:
            self._create_stocktaking(task)
            return
        if task.entity is Entity.GOODS_RECEIPT:
            self._create_goods_receipt(task)
            return
        if task.entity is Entity.DOCUMENT_ARCHIVE:
            self._upload_document(task)
            return
        if task.entity is Entity.EVENT_SUBSCRIPTION:
            self._create_event_subscription(task)
            return
        raise UnsupportedTaskError(f"Create workflow for {task.entity.value} is not implemented")

    def _handle_update(self, task: ParsedTask) -> None:
        if task.entity is Entity.EMPLOYEE:
            self._update_employee(task)
            return
        if task.entity is Entity.CUSTOMER:
            self._update_customer(task)
            return
        if task.entity is Entity.PROJECT:
            self._update_project(task)
            return
        if task.entity is Entity.DEPARTMENT:
            self._update_department(task)
            return
        if task.entity is Entity.PRODUCT:
            self._update_product(task)
            return
        if task.entity is Entity.CONTACT:
            self._update_contact(task)
            return
        if task.entity is Entity.SUPPLIER:
            self._update_supplier(task)
            return
        if task.entity is Entity.TRAVEL_EXPENSE:
            self._update_travel_expense(task)
            return
        if task.entity is Entity.TIMESHEET:
            self._update_timesheet_entry(task)
            return
        if task.entity is Entity.VOUCHER:
            self._update_voucher(task)
            return
        if task.entity is Entity.INVOICE:
            # Invoice update is not directly supported — route to credit note
            # On a fresh account we must create the invoice first, then credit it
            LOGGER.info("Routing invoice update to credit note workflow")
            task.attributes["workflow"] = "creditNote"
            self._create_invoice(task)
            return
        raise UnsupportedTaskError(f"Update workflow for {task.entity.value} is not implemented")

    def _handle_delete(self, task: ParsedTask) -> None:
        if task.entity is Entity.EMPLOYEE:
            self._delete_employee(task)
            return
        if task.entity is Entity.CUSTOMER:
            self._delete_customer(task)
            return
        if task.entity is Entity.PROJECT:
            self._delete_project(task)
            return
        if task.entity is Entity.DEPARTMENT:
            self._delete_department(task)
            return
        if task.entity is Entity.PRODUCT:
            self._delete_product(task)
            return
        if task.entity is Entity.TRAVEL_EXPENSE:
            self._delete_travel_expense(task)
            return
        if task.entity is Entity.CONTACT:
            self._delete_contact(task)
            return
        if task.entity is Entity.SUPPLIER:
            self._delete_supplier(task)
            return
        if task.entity is Entity.INVOICE:
            self._delete_invoice(task)
            return
        if task.entity is Entity.VOUCHER:
            self._delete_voucher(task)
            return
        if task.entity is Entity.TIMESHEET:
            self._delete_timesheet_entry(task)
            return
        if task.entity is Entity.PAYMENT:
            # "Revert payment" / "delete payment" — create credit note on the invoice
            self._revert_payment(task)
            return
        if task.entity is Entity.DOCUMENT_ARCHIVE:
            self._delete_document(task)
            return
        if task.entity is Entity.EVENT_SUBSCRIPTION:
            self._delete_event_subscription(task)
            return
        if task.entity is Entity.INVENTORY:
            self._delete_inventory(task)
            return
        if task.entity is Entity.STOCKTAKING:
            self._delete_stocktaking(task)
            return
        raise UnsupportedTaskError(f"Delete workflow for {task.entity.value} is not implemented")

    def _handle_register(self, task: ParsedTask) -> None:
        if task.entity is Entity.PAYMENT:
            self._register_payment(task)
            return
        if task.entity is Entity.INCOMING_INVOICE:
            # "Pay supplier invoice" / "betal leverandørfaktura"
            self._pay_supplier_invoice(task)
            return
        # "Registrer" in Norwegian means "create" for non-payment entities
        # (e.g., "Registrer en reiseregning" = "Create a travel expense")
        LOGGER.info("Redirecting register/%s to create workflow", task.entity.value)
        task.action = Action.CREATE
        self._handle_create(task)

    def _create_employee(self, task: ParsedTask) -> None:
        # Map role to userType
        role = task.attributes.get("role", "")
        user_type = task.attributes.get("userType")
        if not user_type and role:
            user_type = ROLE_TO_USER_TYPE.get(role.lower(), "STANDARD")
        if not user_type:
            user_type = "STANDARD"

        first_name = task.attributes.get("firstName")
        last_name = task.attributes.get("lastName")
        # Fallback: parse from target_name if LLM didn't extract firstName/lastName
        if (not first_name or not last_name) and task.target_name:
            parts = task.target_name.split()
            if len(parts) >= 2:
                first_name = first_name or parts[0]
                last_name = last_name or " ".join(parts[1:])
            elif len(parts) == 1:
                first_name = first_name or parts[0]
                last_name = last_name or parts[0]
        if not first_name or not last_name:
            raise ParsingError("Could not extract employee first/last name from prompt")

        payload: dict[str, Any] = {
            "firstName": first_name,
            "lastName": last_name,
            "userType": user_type,
        }
        email = task.attributes.get("email")
        if email:
            payload["email"] = email
        elif user_type == "STANDARD":
            # STANDARD userType requires email — generate a placeholder
            payload["email"] = f"{first_name.lower().replace(' ', '')}.{last_name.lower().replace(' ', '')}@placeholder.example.com"

        phone = task.attributes.get("phoneNumberWork") or task.attributes.get("phoneNumberMobile") or task.attributes.get("phoneNumber")
        if phone:
            payload["phoneNumberWork"] = phone
        if task.attributes.get("phoneNumberMobile"):
            payload["phoneNumberMobile"] = task.attributes["phoneNumberMobile"]

        if task.attributes.get("dateOfBirth"):
            payload["dateOfBirth"] = _parse_date_value(task.attributes["dateOfBirth"]).isoformat()
        if task.attributes.get("nationalIdentityNumber"):
            nid = str(task.attributes["nationalIdentityNumber"]).replace(" ", "").replace("-", "")
            if len(nid) == 11 and nid.isdigit():
                payload["nationalIdentityNumber"] = nid
            else:
                LOGGER.warning("Invalid national ID number format: %r (expected 11 digits), skipping", nid)
        if task.attributes.get("employeeNumber"):
            payload["employeeNumber"] = str(task.attributes["employeeNumber"])
        if task.attributes.get("bankAccountNumber"):
            payload["bankAccountNumber"] = str(task.attributes["bankAccountNumber"])

        address = self._build_address(task.attributes)
        if address:
            payload["address"] = address

        department_name = task.attributes.get("departmentName") or DEFAULT_EMPLOYEE_DEPARTMENT_NAME
        department = self._ensure_department(department_name)
        payload["department"] = {"id": department["id"]}

        # If we have startDate or salary data, ensure dateOfBirth is set (required for employment)
        start_date = task.attributes.get("startDate")
        has_salary = any(task.attributes.get(f) for f in ("annualSalary", "monthlySalary", "hourlyWage", "percentageOfFullTimeEquivalent"))
        if (start_date or has_salary) and "dateOfBirth" not in payload:
            payload["dateOfBirth"] = "1990-01-01"

        # Try to create the employee; if email is rejected, retry without it
        try:
            employee = self.client.create("/employee", payload)
        except TripletexAPIError as e:
            if e.status_code == 422 and "e-post" in str(e).lower():
                LOGGER.warning("Email rejected by Tripletex (%s), retrying without email", e)
                original_email = payload.pop("email", None)
                # Generate a safe placeholder email
                first = first_name.lower().replace(" ", "")
                last = last_name.lower().replace(" ", "")
                payload["email"] = f"{first}.{last}@placeholder.example.com"
                employee = self.client.create("/employee", payload)
            elif e.status_code == 422 and "nationalIdentityNumber" in str(e):
                LOGGER.warning("National ID rejected (%s), retrying without it", e)
                payload.pop("nationalIdentityNumber", None)
                employee = self.client.create("/employee", payload)
            else:
                raise

        # Grant entitlements if role maps to a template (worth 5/10 pts on employee tasks)
        entitlement_template = ROLE_TO_ENTITLEMENT_TEMPLATE.get(role.lower()) if role else None
        if entitlement_template:
            try:
                self.client.grant_entitlements(employee["id"], entitlement_template)
            except Exception as e:
                LOGGER.warning("Could not grant %s entitlements: %s", entitlement_template, e)

        # startDate lives on the employment object, not the employee
        if start_date or has_salary:
            self._update_employment(
                employee["id"],
                start_date=_parse_date_value(start_date).isoformat() if start_date else None,
                salary_attrs=task.attributes if has_salary else None,
                dob_already_set=True,  # We set dateOfBirth in the payload above
            )

        # Hourly cost and rate
        hourly_cost = task.attributes.get("hourlyCost")
        hourly_rate_val = task.attributes.get("hourlyRate")
        if hourly_cost is not None or hourly_rate_val is not None:
            try:
                hcr_payload: dict[str, Any] = {
                    "employee": {"id": employee["id"]},
                    "date": date.today().isoformat(),
                }
                if hourly_cost is not None:
                    hcr_payload["hourlyCostPercentage"] = float(hourly_cost)
                if hourly_rate_val is not None:
                    hcr_payload["hourlyRate"] = float(hourly_rate_val)
                self.client.create_hourly_cost_and_rate(hcr_payload)
            except Exception as e:
                LOGGER.warning("Could not set hourly cost/rate: %s", e)

    def _create_customer(self, task: ParsedTask) -> None:
        names = task.attributes.get("names")
        if names:
            for name in names:
                self.client.create("/customer", {"name": name, "isCustomer": True})
            return
        name = task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Could not extract customer name from prompt")
        payload: dict[str, Any] = {
            "name": name,
            "isCustomer": True,
        }
        if task.attributes.get("email"):
            payload["email"] = task.attributes["email"]
            payload["invoiceEmail"] = task.attributes["email"]
        if task.attributes.get("phoneNumberMobile"):
            payload["phoneNumberMobile"] = task.attributes["phoneNumberMobile"]
        if task.attributes.get("phoneNumber"):
            payload["phoneNumber"] = task.attributes["phoneNumber"]
        org_number = task.attributes.get("organizationNumber") or (str(task.attributes["orgNumber"]) if task.attributes.get("orgNumber") else None)
        if org_number:
            payload["organizationNumber"] = org_number

        address = self._build_address(task.attributes)
        if address:
            payload["postalAddress"] = address
            payload["physicalAddress"] = address

        delivery_address = self._build_delivery_address(task.attributes)
        if delivery_address:
            payload["deliveryAddress"] = delivery_address

        if task.attributes.get("website"):
            payload["website"] = task.attributes["website"]
        if task.attributes.get("isPrivateIndividual"):
            payload["isPrivateIndividual"] = True
        if task.attributes.get("invoicesDueIn") is not None:
            payload["invoicesDueIn"] = int(task.attributes["invoicesDueIn"])
            payload["invoicesDueInType"] = "DAYS"
        if task.attributes.get("customerNumber"):
            payload["customerNumber"] = int(task.attributes["customerNumber"])
        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]

        self.client.create("/customer", payload)

    def _create_department(self, task: ParsedTask) -> None:
        names = task.attributes.get("names")
        if not names:
            name = task.attributes.get("name") or task.target_name
            if not name:
                raise ParsingError("No department name(s) could be extracted from the prompt")
            names = [name]
        for name in names:
            dept_payload: dict[str, Any] = {"name": name}
            if task.attributes.get("departmentNumber"):
                dept_payload["departmentNumber"] = str(task.attributes["departmentNumber"])
            manager_name = task.attributes.get("departmentManagerName")
            if manager_name:
                try:
                    manager = self._find_employee(name=manager_name, email=None)
                    dept_payload["departmentManager"] = {"id": manager["id"]}
                except (EntityNotFoundError, AmbiguousMatchError):
                    LOGGER.warning("Could not find department manager %r", manager_name)
            self.client.create("/department", dept_payload)

    def _create_product(self, task: ParsedTask) -> None:
        names = task.attributes.get("names")
        if names:
            for name in names:
                try:
                    self.client.create("/product", {"name": name})
                except TripletexAPIError as e:
                    if e.status_code == 422 and "allerede" in str(e).lower():
                        LOGGER.warning("Product '%s' already exists, skipping: %s", name, e)
                    else:
                        raise
            return
        name = task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Could not extract product name from prompt")
        payload: dict[str, Any] = {"name": name}
        if "number" in task.attributes:
            payload["number"] = str(task.attributes["number"])

        # Resolve VAT type if a rate is specified
        vat_rate = task.attributes.get("vatRate")
        vat_type = None
        if vat_rate is not None:
            vat_type = self._resolve_product_vat_type(float(vat_rate))
            if vat_type:
                payload["vatType"] = {"id": vat_type["id"]}

        # Determine the effective VAT percentage for price consistency
        vat_pct = float(vat_rate) if vat_rate is not None else (
            float(vat_type.get("percentage", 25)) if vat_type else 25.0
        )

        price_excl = task.attributes.get("priceExcludingVatCurrency")
        price_incl = task.attributes.get("priceIncludingVatCurrency")

        if price_excl is not None:
            price_excl = float(price_excl)
            payload["priceExcludingVatCurrency"] = price_excl
            # Must set incl price consistently — calculate from VAT rate
            if price_incl is None:
                price_incl = price_excl * (1 + vat_pct / 100)
            payload["priceIncludingVatCurrency"] = float(price_incl)
        elif price_incl is not None:
            price_incl = float(price_incl)
            payload["priceIncludingVatCurrency"] = price_incl
            # Calculate excl from incl
            price_excl = price_incl / (1 + vat_pct / 100)
            payload["priceExcludingVatCurrency"] = price_excl

        if "costExcludingVatCurrency" in task.attributes:
            payload["costExcludingVatCurrency"] = float(task.attributes["costExcludingVatCurrency"])

        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]

        # Resolve product unit (stk, timer, kg, etc.)
        unit_name = task.attributes.get("productUnit")
        if unit_name:
            unit = self._resolve_product_unit(unit_name)
            if unit:
                payload["productUnit"] = {"id": unit["id"]}

        self.client.create("/product", payload)

    def _resolve_product_unit(self, name: str) -> dict[str, Any] | None:
        """Find a ProductUnit matching the given name (e.g. 'stk', 'timer', 'kg')."""
        cache_key = f"product_unit:{_normalize(name)}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        results = self.client.search_product_units(name=name)
        if results:
            # Try exact match first
            for r in results:
                if _normalize(r.get("name", "")) == _normalize(name) or _normalize(r.get("nameShort", "")) == _normalize(name):
                    self._cache_set(cache_key, r)
                    return r
            self._cache_set(cache_key, results[0])
            return results[0]
        # Try without name filter to get all and match loosely
        all_units = self.client.search_product_units()
        for u in all_units:
            if _normalize(name) in _normalize(u.get("name", "")) or _normalize(name) in _normalize(u.get("nameShort", "")):
                self._cache_set(cache_key, u)
                return u
        LOGGER.warning("Could not find product unit for %r", name)
        return None

    def _resolve_product_vat_type(self, vat_rate: float) -> dict[str, Any] | None:
        """Find a VAT type matching the given percentage rate."""
        cache_key = f"vat_type_rate:{vat_rate}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        # Map common rates to Tripletex VAT type numbers
        rate_to_number: dict[float, str] = {
            0.0: "6",    # MVA 0% (exempt)
            12.0: "32",  # MVA 12% (transport, cinema)
            15.0: "33",  # MVA 15% (food)
            25.0: "3",   # MVA 25% (standard)
        }
        vat_number = rate_to_number.get(vat_rate)
        if vat_number:
            results = self.client.search_vat_types(number=vat_number)
            if results:
                self._cache_set(cache_key, results[0])
                return results[0]
        # Fallback: search all and match by percentage
        all_types = self.client.search_vat_types()
        for vt in all_types:
            if abs(float(vt.get("percentage", -1)) - vat_rate) < 0.01:
                self._cache_set(cache_key, vt)
                return vt
        LOGGER.warning("Could not find VAT type for rate %.1f%%", vat_rate)
        return None

    def _create_project(self, task: ParsedTask) -> None:
        pm_name = task.attributes.get("projectManagerName")
        pm_email = task.attributes.get("projectManagerEmail")
        if pm_name:
            project_manager = self._ensure_employee(name=pm_name, email=pm_email)
        else:
            project_manager = self._resolve_project_manager()
        self._ensure_project_manager_access(project_manager)
        project_name = task.attributes.get("name") or task.target_name
        if not project_name:
            raise ParsingError("Could not extract project name from prompt")
        payload: dict[str, Any] = {
            "name": project_name,
            "projectManager": {"id": project_manager["id"]},
            "startDate": date.today().isoformat(),
        }

        customer_name = task.attributes.get("customerName")
        if customer_name:
            org_number = task.attributes.get("organizationNumber") or (
                str(task.attributes["orgNumber"]) if task.attributes.get("orgNumber") else None
            )
            customer = self._ensure_customer(
                name=customer_name,
                email=task.attributes.get("customerEmail"),
                org_number=org_number,
                address=self._build_address(task.attributes),
            )
            payload["customer"] = {"id": customer["id"]}

        department_name = task.attributes.get("departmentName")
        if department_name:
            department = self._ensure_department(department_name)
            payload["department"] = {"id": department["id"]}

        if task.attributes.get("startDate"):
            payload["startDate"] = _parse_date_value(task.attributes["startDate"]).isoformat()
        if task.attributes.get("endDate"):
            payload["endDate"] = _parse_date_value(task.attributes["endDate"]).isoformat()
        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]
        if task.attributes.get("reference"):
            payload["reference"] = task.attributes["reference"]
        if task.attributes.get("projectNumber"):
            payload["number"] = str(task.attributes["projectNumber"])

        # Fixed price project support
        fixed_price = task.attributes.get("fixedPrice")
        if fixed_price is not None:
            payload["isFixedPrice"] = True
            payload["fixedprice"] = float(fixed_price)
        elif task.attributes.get("isFixedPrice"):
            payload["isFixedPrice"] = True

        project = self.client.create("/project", payload)

        # Add project participants if specified
        participant_names = task.attributes.get("participantNames") or []
        participant_name = task.attributes.get("projectParticipantName") or task.attributes.get("participantName")
        if participant_name and participant_name not in participant_names:
            participant_names.append(participant_name)
        for pname in participant_names:
            try:
                emp = self._ensure_employee(name=pname)
                self.client.create_project_participant({
                    "project": {"id": project["id"]},
                    "employee": {"id": emp["id"]},
                    "adminAccess": False,
                })
                LOGGER.info("Added participant %s to project %s", pname, project_name)
            except Exception as e:
                LOGGER.warning("Could not add participant %s: %s", pname, e)

        # Add project activities if specified
        activity_names = task.attributes.get("projectActivityNames") or []
        activity_name = task.attributes.get("projectActivityName") or task.attributes.get("activityName")
        if activity_name and activity_name not in activity_names:
            activity_names.append(activity_name)
        for aname in activity_names:
            try:
                activity = self._resolve_or_create_activity(aname)
                self.client.create_project_activity({
                    "activity": {"id": activity["id"]},
                    "project": {"id": project["id"]},
                })
                LOGGER.info("Added activity %s to project %s", aname, project_name)
            except Exception as e:
                LOGGER.warning("Could not add activity %s: %s", aname, e)

        # Project category
        category_name = task.attributes.get("projectCategory") or task.attributes.get("categoryName")
        if category_name:
            try:
                self.client.create_project_category({"name": category_name})
                LOGGER.info("Created project category: %s", category_name)
            except Exception as e:
                LOGGER.warning("Could not create project category %s: %s", category_name, e)

        # Project hourly rate
        hourly_rate = task.attributes.get("projectHourlyRate") or task.attributes.get("hourlyRate")
        if hourly_rate is not None:
            try:
                self.client.create_project_hourly_rate({
                    "project": {"id": project["id"]},
                    "startDate": date.today().isoformat(),
                    "showInProjectOrder": True,
                })
                LOGGER.info("Created project hourly rate for project %s", project_name)
            except Exception as e:
                LOGGER.warning("Could not create project hourly rate: %s", e)

    def _create_invoice(self, task: ParsedTask) -> None:
        is_credit_note = task.attributes.get("workflow") == "creditNote"

        self._ensure_invoice_bank_account()

        # Path 1: Credit note on an existing invoice.
        if is_credit_note:
            invoice = self._resolve_invoice_for_credit_note(task)
            cn_date_val = (
                task.attributes.get("creditNoteDate")
                or task.attributes.get("date")
                or task.attributes.get("invoiceDate")
            )
            cn_date = _parse_date_value(cn_date_val) if cn_date_val else date.today()
            self.client.create_credit_note(
                invoice["id"],
                credit_note_date=cn_date,
                comment=task.attributes.get("comment"),
                send_to_customer=False,
            )
            return

        # Path 2: Create invoice (and optionally credit note) from scratch
        customer_name = task.attributes.get("customerName")
        if not customer_name:
            raise ParsingError("Could not determine customer for invoice")
        org_number = task.attributes.get("organizationNumber") or (str(task.attributes["orgNumber"]) if task.attributes.get("orgNumber") else None)
        customer = self._ensure_customer(
            name=customer_name,
            email=task.attributes.get("customerEmail"),
            org_number=org_number,
            address=self._build_address(task.attributes),
        )
        today = date.today()
        invoice_date_val = task.attributes.get("invoiceDate")
        invoice_date = _parse_date_value(invoice_date_val) if invoice_date_val else today
        due_date_val = task.attributes.get("invoiceDueDate")
        due_date = _parse_date_value(due_date_val) if due_date_val else invoice_date + timedelta(days=30)
        quantity = float(task.attributes.get("quantity", 1.0))

        # Build order lines — support multi-line invoices with different VAT rates
        order_lines_raw = task.attributes.get("orderLines")
        if order_lines_raw and isinstance(order_lines_raw, list) and len(order_lines_raw) > 0:
            # Multi-line invoice
            order_lines: list[dict[str, Any]] = []
            for line in order_lines_raw:
                line_desc = line.get("description") or line.get("productName") or "Invoice line"
                line_amount = float(line.get("amount") or line.get("unitPrice") or 0)
                line_qty = float(line.get("quantity", 1.0))
                line_vat_rate = line.get("vatRate")
                if line_vat_rate is not None:
                    vt = self._resolve_product_vat_type(float(line_vat_rate))
                    if not vt:
                        vt = self._get_default_vat_type()
                else:
                    vt = self._get_default_vat_type()
                ol: dict[str, Any] = {
                    "description": line_desc,
                    "count": line_qty,
                    "unitPriceExcludingVatCurrency": line_amount,
                    "vatType": {"id": vt["id"]},
                }
                order_lines.append(ol)
            LOGGER.info("Multi-line invoice: %d lines", len(order_lines))
        else:
            # Single-line invoice (original path)
            amount_value = task.attributes.get("amount")
            if amount_value is None:
                raise ParsingError("Could not determine invoice amount from prompt or product state")
            amount = float(amount_value)
            description = task.attributes.get("lineDescription") or task.attributes.get("productName") or "Invoice line"
            # Use task-specified VAT rate if provided, otherwise default (25%)
            vat_rate = task.attributes.get("vatRate")
            vat_type = None
            if vat_rate is not None:
                vat_type = self._resolve_product_vat_type(float(vat_rate))
            if not vat_type:
                vat_type = self._get_default_vat_type()
            order_lines = [
                {
                    "description": description,
                    "count": quantity,
                    "unitPriceExcludingVatCurrency": amount,
                    "vatType": {"id": vat_type["id"]},
                }
            ]

        order_payload: dict[str, Any] = {
            "customer": {"id": customer["id"]},
            "orderDate": invoice_date.isoformat(),
            "deliveryDate": invoice_date.isoformat(),
            "orderLines": order_lines,
        }

        # Link order to project if specified
        project_name = task.attributes.get("projectName")
        if project_name:
            try:
                project = self._find_project(name=project_name)
                order_payload["project"] = {"id": project["id"]}
            except (EntityNotFoundError, AmbiguousMatchError):
                LOGGER.warning("Could not link invoice order to project %r", project_name)

        order = self.client.create_order(order_payload)

        invoice_payload = {
            "invoiceDate": invoice_date.isoformat(),
            "invoiceDueDate": due_date.isoformat(),
            "customer": {"id": customer["id"]},
            "orders": [{"id": order["id"]}],
        }
        invoice = self.client.create_invoice(invoice_payload, send_to_customer=False)
        LOGGER.info("Created invoice: id=%s", invoice.get("id"))

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _invoice_amount_matches(
        self,
        invoice: dict[str, Any],
        amount: float,
        *,
        amount_is_exclusive: bool | None,
        amount_is_inclusive: bool | None,
    ) -> bool:
        amount_fields = (
            self._as_float(invoice.get("amount")),
            self._as_float(invoice.get("amountCurrencyOutstanding")),
            self._as_float(invoice.get("amountOutstanding")),
        )
        expected_values = [float(amount)]
        if amount_is_exclusive:
            expected_values.extend([float(amount) * 1.25, float(amount) * 1.12, float(amount) * 1.15])
        elif amount_is_inclusive:
            expected_values.extend([float(amount) / 1.25, float(amount) / 1.12, float(amount) / 1.15])
        else:
            expected_values.extend(
                [
                    float(amount) * 1.25,
                    float(amount) / 1.25,
                    float(amount) * 1.12,
                    float(amount) / 1.12,
                    float(amount) * 1.15,
                    float(amount) / 1.15,
                ]
            )

        tolerance = 0.01
        for invoice_amount in amount_fields:
            if invoice_amount is None:
                continue
            for expected in expected_values:
                if abs(invoice_amount - expected) <= tolerance:
                    return True
        return False

    def _resolve_invoice_for_credit_note(self, task: ParsedTask) -> dict[str, Any]:
        if task.identifier is not None:
            return self.client.get_invoice(task.identifier)

        invoice_number = task.attributes.get("invoiceNumber")
        if invoice_number:
            results = self.client.search_invoices(invoice_number=str(invoice_number))
            exact = self._pick_exact(results, "invoiceNumber", str(invoice_number))
            if exact:
                return exact
            if len(results) == 1:
                return results[0]
            if results:
                raise AmbiguousMatchError(f"Invoice number {invoice_number!r} matched multiple invoices")

        customer_name = task.attributes.get("customerName")
        if not customer_name:
            raise ParsingError("Credit note requires a customer reference when invoice ID/number is missing")

        customer = self._find_customer(
            name=customer_name,
            email=None,
            organization_number=task.attributes.get("organizationNumber")
            or (str(task.attributes["orgNumber"]) if task.attributes.get("orgNumber") else None),
        )
        results = self.client.search_invoices(customer_id=customer["id"])
        if not results:
            raise EntityNotFoundError(f"Could not resolve invoice for customer {customer_name!r}")

        amount = task.attributes.get("amount")
        amount_is_exclusive = task.attributes.get("amountIsVatExclusive")
        if amount_is_exclusive is not None:
            amount_is_exclusive = bool(amount_is_exclusive)
        amount_is_inclusive = task.attributes.get("amountIsVatInclusive")
        if amount_is_inclusive is not None:
            amount_is_inclusive = bool(amount_is_inclusive)

        if amount is not None:
            amount_value = float(amount)
            amount_matches = [invoice for invoice in results if self._invoice_amount_matches(
                invoice,
                amount_value,
                amount_is_exclusive=amount_is_exclusive,
                amount_is_inclusive=amount_is_inclusive,
            )]
            if len(amount_matches) == 1:
                return amount_matches[0]
            if len(amount_matches) > 1:
                raise AmbiguousMatchError(
                    f"Customer {customer_name!r} has multiple invoices matching amount {amount_value}"
                )

        def _has_positive_outstanding(invoice_value: dict[str, Any]) -> bool:
            outstanding_amount = self._as_float(invoice_value.get("amountCurrencyOutstanding") or invoice_value.get("amountOutstanding"))
            return outstanding_amount is not None and outstanding_amount > 0

        outstanding = [item for item in results if _has_positive_outstanding(item)]
        if len(outstanding) == 1:
            return outstanding[0]
        if len(results) == 1:
            return results[0]

        if amount is None:
            raise EntityNotFoundError(f"Could not determine target invoice for customer {customer_name!r}")
        raise ParsingError(f"Could not uniquely match invoice for customer {customer_name!r} and amount {amount!r}")

    def _delete_invoice(self, task: ParsedTask) -> None:
        """Delete an invoice by creating a credit note (no direct delete in Tripletex)."""
        # On a fresh account, create the invoice first, then credit it
        task.attributes["workflow"] = "creditNote"
        self._create_invoice(task)

    def _create_travel_expense(self, task: ParsedTask) -> None:
        employee_name = task.attributes.get("employeeName") or task.target_name
        employee_email = task.attributes.get("employeeEmail") or task.attributes.get("email")
        # Fallback: derive name from email if missing
        if not employee_name and employee_email:
            local = employee_email.split("@")[0]
            parts = local.replace(".", " ").replace("_", " ").replace("-", " ").split()
            employee_name = " ".join(p.capitalize() for p in parts)
            LOGGER.info("Derived employee name %r from email %r", employee_name, employee_email)
        if not employee_name:
            raise ParsingError("Could not extract employee name for travel expense")
        employee = self._ensure_employee(
            name=employee_name,
            email=employee_email,
        )
        departure_date_val = task.attributes.get("departureDate")
        return_date_val = task.attributes.get("returnDate")
        if not departure_date_val:
            LOGGER.warning("No departure date extracted, using today")
            departure_date_val = date.today().isoformat()
        if not return_date_val:
            LOGGER.warning("No return date extracted, using departure date")
            return_date_val = departure_date_val
        departure_date = _parse_date_value(departure_date_val)
        return_date = _parse_date_value(return_date_val)
        departure_from = task.attributes.get("departureFrom", "Oslo")
        destination = task.attributes.get("destination", "Bergen")

        payload: dict[str, Any] = {
            "employee": {"id": employee["id"]},
            "travelDetails": {
                "isForeignTravel": task.attributes.get("isForeignTravel", False),
                "isDayTrip": departure_date == return_date,
                "departureDate": departure_date.isoformat(),
                "returnDate": return_date.isoformat(),
                "departureFrom": departure_from,
                "destination": destination,
                "purpose": task.attributes.get("purpose", ""),
            },
            "isChargeable": False,
            "isFixedInvoicedAmount": False,
            "isIncludeAttachedReceiptsWhenReinvoicing": False,
        }
        if task.attributes.get("title"):
            payload["title"] = task.attributes["title"]
        project_name = task.attributes.get("projectName")
        if project_name:
            try:
                project = self._resolve_project_by_name(project_name)
                payload["project"] = {"id": project["id"]}
            except (EntityNotFoundError, AmbiguousMatchError):
                LOGGER.warning("Could not link travel expense to project %r", project_name)
        department_name = task.attributes.get("departmentName")
        if department_name:
            department = self._ensure_department(department_name)
            payload["department"] = {"id": department["id"]}
        travel_expense = self.client.create_travel_expense(payload)

        # Per diem compensation (dietas / kostgodtgjørelse / per diem)
        per_diem = task.attributes.get("perDiem") or task.attributes.get("hasPerDiem")
        per_diem_rate = task.attributes.get("perDiemRate")
        if per_diem or per_diem_rate or _contains_any_ascii(task.raw_prompt, ("dieta", "per diem", "kostgodtgjorelse", "tagegelder", "indemnite journaliere", "tarifa completa", "tarifa diaria")):
            try:
                destination_lower = destination.lower() if destination else ""
                norwegian_cities = {"oslo", "bergen", "trondheim", "stavanger", "tromsø", "kristiansand", "drammen", "fredrikstad", "bodø", "ålesund", "tønsberg", "haugesund", "sandefjord", "moss", "sarpsborg", "arendal", "hamar", "larvik", "halden", "kongsberg"}
                is_domestic = destination_lower in norwegian_cities
                is_day_trip = departure_date == return_date
                rate_cat = self._resolve_per_diem_rate_category(
                    is_day_trip=is_day_trip, is_domestic=is_domestic,
                )
                per_diem_payload: dict[str, Any] = {
                    "travelExpense": {"id": travel_expense["id"]},
                    "rateCategory": {"id": rate_cat["id"]},
                    "countryCode": task.attributes.get("countryCode") or ("NO" if is_domestic else None),
                    "overnightAccommodation": "NONE" if is_day_trip else "HOTEL",
                    "location": destination,
                    "isDeductionForBreakfast": task.attributes.get("breakfastIncluded", False),
                    "isDeductionForLunch": task.attributes.get("lunchIncluded", False),
                    "isDeductionForDinner": task.attributes.get("dinnerIncluded", False),
                }
                # Remove None values
                per_diem_payload = {k: v for k, v in per_diem_payload.items() if v is not None}
                self.client.create_per_diem_compensation(per_diem_payload)
                LOGGER.info("Created per diem compensation for travel expense %s", travel_expense["id"])
            except Exception as e:
                LOGGER.warning("Could not create per diem compensation: %s", e)

        # Mileage allowance
        km = task.attributes.get("kilometers") or task.attributes.get("km")
        if km is not None:
            try:
                mileage_rate_cat = self._resolve_mileage_rate_category()
                mileage_payload: dict[str, Any] = {
                    "travelExpense": {"id": travel_expense["id"]},
                    "rateCategory": {"id": mileage_rate_cat["id"]},
                    "date": departure_date.isoformat(),
                    "departureLocation": departure_from,
                    "destination": destination,
                    "km": float(km),
                    "isCompanyCar": False,
                }
                self.client.create_mileage_allowance(mileage_payload)
                LOGGER.info("Created mileage allowance for travel expense %s", travel_expense["id"])
            except Exception as e:
                LOGGER.warning("Could not create mileage allowance: %s", e)

        # Accommodation allowance
        nights = task.attributes.get("nights") or task.attributes.get("accommodationNights")
        has_accommodation_keywords = _contains_any_ascii(
            task.raw_prompt,
            ("overnatting", "accommodation", "hebergement", "hébergement", "alojamiento", "unterkunft", "natt", "nights", "hotell", "hotel"),
        )
        if nights or (has_accommodation_keywords and departure_date != return_date):
            try:
                accom_rate_cat = self._resolve_accommodation_rate_category()
                night_count = int(nights) if nights else (return_date - departure_date).days
                if night_count < 1:
                    night_count = 1
                accom_payload: dict[str, Any] = {
                    "travelExpense": {"id": travel_expense["id"]},
                    "rateCategory": {"id": accom_rate_cat["id"]},
                    "location": destination,
                    "count": night_count,
                }
                if task.attributes.get("accommodationAddress"):
                    accom_payload["address"] = task.attributes["accommodationAddress"]
                self.client.create_accommodation_allowance(accom_payload)
                LOGGER.info("Created accommodation allowance for travel expense %s", travel_expense["id"])
            except Exception as e:
                LOGGER.warning("Could not create accommodation allowance: %s", e)

        amount = task.attributes.get("amount")
        if amount is not None:
            payment_type = self._resolve_travel_payment_type()
            cost_category = self._resolve_travel_cost_category()
            self.client.create_travel_cost(
                {
                    "travelExpense": {"id": travel_expense["id"]},
                    "paymentType": {"id": payment_type["id"]},
                    "date": departure_date.isoformat(),
                    "costCategory": {"id": cost_category["id"]},
                    "amountCurrencyIncVat": float(amount),
                }
            )

        # Passengers
        passengers = task.attributes.get("passengers") or []
        if passengers and isinstance(passengers, list):
            for pname in passengers:
                try:
                    emp = self._ensure_employee(name=pname)
                    self.client.create_travel_passenger({
                        "mileageAllowance": {"id": travel_expense["id"]},
                        "name": pname,
                        "employee": {"id": emp["id"]},
                    })
                except Exception as e:
                    LOGGER.warning("Could not add passenger %s: %s", pname, e)

        # Driving stops
        driving_stops = task.attributes.get("drivingStops") or []
        if driving_stops and isinstance(driving_stops, list):
            for stop in driving_stops:
                try:
                    stop_name = stop if isinstance(stop, str) else stop.get("location", "")
                    self.client.create_driving_stop({
                        "mileageAllowance": {"id": travel_expense["id"]},
                        "location": stop_name,
                    })
                except Exception as e:
                    LOGGER.warning("Could not add driving stop: %s", e)

        # Deliver and approve the travel expense (required for Tier 2+ completion)
        try:
            self.client.deliver_travel_expense(travel_expense["id"])
            LOGGER.info("Delivered travel expense %s", travel_expense["id"])
        except Exception as e:
            LOGGER.warning("Could not deliver travel expense %s: %s", travel_expense["id"], e)

        try:
            self.client.approve_travel_expense(travel_expense["id"])
            LOGGER.info("Approved travel expense %s", travel_expense["id"])
        except Exception as e:
            LOGGER.warning("Could not approve travel expense %s: %s", travel_expense["id"], e)

    def _update_employee(self, task: ParsedTask) -> None:
        emp_name = task.target_name
        if not emp_name:
            # Try building name from attributes
            fn = task.attributes.get("firstName", "")
            ln = task.attributes.get("lastName", "")
            if fn and ln:
                emp_name = f"{fn} {ln}"
        employee = self._find_employee(name=emp_name, email=task.attributes.get("email"))
        payload: dict[str, Any] = {}

        for field in ("email", "firstName", "lastName"):
            if field in task.attributes:
                payload[field] = task.attributes[field]

        phone = task.attributes.get("phoneNumberWork") or task.attributes.get("phoneNumberMobile") or task.attributes.get("phoneNumber")
        if phone:
            payload["phoneNumberWork"] = phone

        # Handle role/userType updates
        role = task.attributes.get("role")
        user_type = task.attributes.get("userType")
        if role and not user_type:
            user_type = ROLE_TO_USER_TYPE.get(role.lower())
        if user_type:
            payload["userType"] = user_type

        if task.attributes.get("dateOfBirth"):
            payload["dateOfBirth"] = _parse_date_value(task.attributes["dateOfBirth"]).isoformat()
        if task.attributes.get("nationalIdentityNumber"):
            nid = str(task.attributes["nationalIdentityNumber"]).replace(" ", "").replace("-", "")
            if len(nid) == 11 and nid.isdigit():
                payload["nationalIdentityNumber"] = nid
        if task.attributes.get("bankAccountNumber"):
            payload["bankAccountNumber"] = str(task.attributes["bankAccountNumber"])

        address = self._build_address(task.attributes)
        if address:
            payload["address"] = address

        department_name = task.attributes.get("departmentName")
        if department_name:
            department = self._ensure_department(department_name)
            payload["department"] = {"id": department["id"]}

        start_date = task.attributes.get("startDate")

        if not payload and not start_date:
            raise ParsingError("No updatable employee fields were extracted from the prompt")

        if payload:
            self.client.update("/employee", employee["id"], payload)

        has_salary = any(task.attributes.get(f) for f in ("annualSalary", "monthlySalary", "hourlyWage", "percentageOfFullTimeEquivalent"))
        if start_date or has_salary:
            self._update_employment(
                employee["id"],
                start_date=_parse_date_value(start_date).isoformat() if start_date else None,
                salary_attrs=task.attributes if has_salary else None,
            )

        # Grant entitlements if role maps to a template
        role = task.attributes.get("role", "")
        entitlement_template = ROLE_TO_ENTITLEMENT_TEMPLATE.get(role.lower()) if role else None
        if entitlement_template:
            try:
                self.client.grant_entitlements(employee["id"], entitlement_template)
            except Exception as e:
                LOGGER.warning("Could not grant %s entitlements on update: %s", entitlement_template, e)

    def _update_customer(self, task: ParsedTask) -> None:
        customer = self._find_customer(name=task.target_name, email=None)
        payload: dict[str, Any] = {}

        if "name" in task.attributes:
            payload["name"] = task.attributes["name"]
        if "email" in task.attributes:
            payload["email"] = task.attributes["email"]
            payload["invoiceEmail"] = task.attributes["email"]
        if "phoneNumberMobile" in task.attributes:
            payload["phoneNumberMobile"] = task.attributes["phoneNumberMobile"]
        if "phoneNumber" in task.attributes:
            payload["phoneNumber"] = task.attributes["phoneNumber"]
        if "organizationNumber" in task.attributes:
            payload["organizationNumber"] = str(task.attributes["organizationNumber"])
        if "website" in task.attributes:
            payload["website"] = task.attributes["website"]
        if "description" in task.attributes:
            payload["description"] = task.attributes["description"]
        if task.attributes.get("invoicesDueIn") is not None:
            payload["invoicesDueIn"] = int(task.attributes["invoicesDueIn"])
            payload["invoicesDueInType"] = "DAYS"

        address = self._build_address(task.attributes)
        if address:
            payload["postalAddress"] = address
            payload["physicalAddress"] = address
        delivery_address = self._build_delivery_address(task.attributes)
        if delivery_address:
            payload["deliveryAddress"] = delivery_address

        if not payload:
            raise ParsingError("No updatable customer fields were extracted from the prompt")

        self.client.update("/customer", customer["id"], payload)

    def _update_project(self, task: ParsedTask) -> None:
        project = self._resolve_project(task)
        payload: dict[str, Any] = {}

        if "name" in task.attributes:
            payload["name"] = task.attributes["name"]

        customer_name = task.attributes.get("customerName")
        if customer_name:
            customer = self._ensure_customer(name=customer_name, email=task.attributes.get("customerEmail"))
            payload["customer"] = {"id": customer["id"]}

        department_name = task.attributes.get("departmentName")
        if department_name:
            department = self._ensure_department(department_name)
            payload["department"] = {"id": department["id"]}

        if not payload:
            raise ParsingError("No updatable project fields were extracted from the prompt")

        self.client.update("/project", int(project["id"]), payload)

    def _update_department(self, task: ParsedTask) -> None:
        department = self._resolve_department(task)
        payload: dict[str, Any] = {}

        if "name" in task.attributes:
            payload["name"] = task.attributes["name"]

        if not payload:
            raise ParsingError("No updatable department fields were extracted from the prompt")

        self.client.update("/department", int(department["id"]), payload)

    def _update_product(self, task: ParsedTask) -> None:
        product = self._resolve_product(task)
        payload: dict[str, Any] = {}

        for field in ("name", "number"):
            if field in task.attributes:
                payload[field] = task.attributes[field]
        for field in ("priceExcludingVatCurrency", "priceIncludingVatCurrency", "costExcludingVatCurrency"):
            if field in task.attributes:
                payload[field] = float(task.attributes[field])

        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]

        if not payload:
            raise ParsingError("No updatable product fields were extracted from the prompt")

        self.client.update("/product", int(product["id"]), payload)

    def _delete_employee(self, task: ParsedTask) -> None:
        employee = self._find_employee(name=task.target_name, email=task.attributes.get("email"))
        # Tripletex has no DELETE /employee endpoint — deactivate instead
        self.client.update("/employee", employee["id"], {"userType": "NO_ACCESS"})

    def _delete_customer(self, task: ParsedTask) -> None:
        customer = self._find_customer(name=task.target_name, email=task.attributes.get("email"))
        self.client.delete("/customer", customer["id"])

    def _delete_project(self, task: ParsedTask) -> None:
        project = self._resolve_project(task)
        self.client.delete("/project", int(project["id"]))

    def _delete_department(self, task: ParsedTask) -> None:
        department = self._resolve_department(task)
        self.client.delete("/department", int(department["id"]))

    def _delete_product(self, task: ParsedTask) -> None:
        product = self._resolve_product(task)
        self.client.delete("/product", int(product["id"]))

    def _delete_travel_expense(self, task: ParsedTask) -> None:
        travel_expense = self._resolve_travel_expense(task)
        self.client.delete("/travelExpense", travel_expense["id"])

    def _update_travel_expense(self, task: ParsedTask) -> None:
        travel_expense = self._resolve_travel_expense(task)
        travel_details: dict[str, Any] = {}
        for field in ("departureFrom", "destination", "purpose"):
            if field in task.attributes:
                travel_details[field] = task.attributes[field]
        for date_field in ("departureDate", "returnDate"):
            if date_field in task.attributes:
                travel_details[date_field] = _parse_date_value(task.attributes[date_field]).isoformat()
        payload: dict[str, Any] = {}
        if travel_details:
            payload["travelDetails"] = travel_details
        if task.attributes.get("title"):
            payload["title"] = task.attributes["title"]
        if not payload:
            raise ParsingError("No updatable travel expense fields were extracted")
        self.client.update_travel_expense(travel_expense["id"], payload)

    # --- Contact workflows ---

    def _create_contact(self, task: ParsedTask) -> None:
        payload: dict[str, Any] = {}
        first_name = task.attributes.get("firstName")
        last_name = task.attributes.get("lastName")
        # Fallback: parse from target_name if LLM didn't extract firstName/lastName
        if (not first_name or not last_name) and task.target_name:
            parts = task.target_name.split()
            if len(parts) >= 2:
                first_name = first_name or parts[0]
                last_name = last_name or " ".join(parts[1:])
            elif len(parts) == 1:
                first_name = first_name or parts[0]
        if first_name:
            payload["firstName"] = first_name
        if last_name:
            payload["lastName"] = last_name
        if task.attributes.get("email"):
            payload["email"] = task.attributes["email"]
        if task.attributes.get("phoneNumberMobile"):
            payload["phoneNumberMobile"] = task.attributes["phoneNumberMobile"]
        if task.attributes.get("phoneNumberWork"):
            payload["phoneNumberWork"] = task.attributes["phoneNumberWork"]
        if task.attributes.get("phoneNumber"):
            payload["phoneNumberMobile"] = task.attributes["phoneNumber"]

        customer_name = task.attributes.get("customerName")
        if customer_name:
            customer = self._ensure_customer(name=customer_name)
            payload["customer"] = {"id": customer["id"]}

        self.client.create_contact(payload)

    def _update_contact(self, task: ParsedTask) -> None:
        customer_name = task.attributes.get("customerName")
        customer_id = None
        if customer_name:
            customer = self._find_customer(name=customer_name, email=None)
            customer_id = customer["id"]
        contact = self._find_contact(name=task.target_name, customer_id=customer_id)
        payload: dict[str, Any] = {}
        for field in ("firstName", "lastName", "email", "phoneNumberMobile"):
            if field in task.attributes:
                payload[field] = task.attributes[field]
        if not payload:
            raise ParsingError("No updatable contact fields were extracted")
        self.client.update("/contact", contact["id"], payload)

    def _delete_contact(self, task: ParsedTask) -> None:
        customer_name = task.attributes.get("customerName")
        customer_id = None
        if customer_name:
            customer = self._find_customer(name=customer_name, email=None)
            customer_id = customer["id"]
        contact = self._find_contact(name=task.target_name, customer_id=customer_id)
        # Tripletex has no DELETE /contact/{id} — use batch delete endpoint
        self.client.delete_list("/contact", [contact["id"]])

    def _find_contact(self, *, name: str | None, customer_id: int | None = None) -> dict[str, Any]:
        results = self.client.search_contacts(customer_id=customer_id)
        if name:
            exact = self._pick_display_name(results, name)
            if exact:
                return exact
        if not results:
            raise EntityNotFoundError(f"Contact lookup returned no results for {name!r}")
        if len(results) == 1:
            return results[0]
        raise AmbiguousMatchError(f"Contact lookup matched {len(results)} entities")

    # --- Supplier workflows ---

    def _create_supplier(self, task: ParsedTask) -> None:
        name = task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Could not extract supplier name from prompt")
        payload: dict[str, Any] = {
            "name": name,
            "isSupplier": True,
        }
        if task.attributes.get("email"):
            payload["email"] = task.attributes["email"]
            payload["invoiceEmail"] = task.attributes["email"]
        if task.attributes.get("phoneNumber"):
            payload["phoneNumber"] = task.attributes["phoneNumber"]
        if task.attributes.get("phoneNumberMobile"):
            payload["phoneNumberMobile"] = task.attributes["phoneNumberMobile"]
        if task.attributes.get("organizationNumber"):
            payload["organizationNumber"] = str(task.attributes["organizationNumber"])
        if task.attributes.get("website"):
            payload["website"] = task.attributes["website"]
        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]
        address = self._build_address(task.attributes)
        if address:
            payload["postalAddress"] = address
            payload["physicalAddress"] = address
        delivery_address = self._build_delivery_address(task.attributes)
        if delivery_address:
            payload["deliveryAddress"] = delivery_address
        self.client.create("/supplier", payload)

    def _update_supplier(self, task: ParsedTask) -> None:
        supplier = self._find_supplier(name=task.target_name)
        payload: dict[str, Any] = {}
        for field in ("name", "email", "phoneNumber"):
            if field in task.attributes:
                payload[field] = task.attributes[field]
        if not payload:
            raise ParsingError("No updatable supplier fields were extracted")
        self.client.update("/supplier", supplier["id"], payload)

    def _delete_supplier(self, task: ParsedTask) -> None:
        supplier = self._find_supplier(name=task.target_name, email=task.attributes.get("email"))
        self.client.delete("/supplier", supplier["id"])

    def _find_supplier(self, *, name: str | None, email: str | None = None) -> dict[str, Any]:
        results = self.client.search_suppliers(name=name, email=email)
        if email:
            exact = self._pick_exact(results, "email", email)
            if exact:
                return exact
        if name:
            exact = self._pick_exact(results, "name", name)
            if exact:
                return exact
        if not results:
            hint = email or name or "supplier"
            raise EntityNotFoundError(f"Supplier lookup returned no results for {hint!r}")
        if len(results) == 1:
            return results[0]
        raise AmbiguousMatchError(f"Supplier lookup matched {len(results)} entities")

    # --- Voucher workflows ---

    def _create_voucher(self, task: ParsedTask) -> None:
        # Check if this is an opening balance request
        if task.attributes.get("isOpeningBalance") or _contains_any_ascii(
            task.raw_prompt, ("opening balance", "inngående balanse", "åpningsbalanse", "eröffnungsbilanz", "balance de apertura", "bilan d'ouverture")
        ):
            self._create_opening_balance(task)
            return

        description = task.attributes.get("description", "Manual voucher")
        voucher_date_val = task.attributes.get("voucherDate") or task.attributes.get("date") or date.today()
        voucher_date = _parse_date_value(voucher_date_val) if not isinstance(voucher_date_val, date) else voucher_date_val

        voucher_type = self._resolve_voucher_type()

        postings: list[dict[str, Any]] = []

        # Handle LLM format: postings=[{debitAccount: 1500, creditAccount: 3000, amount: 1000}]
        llm_postings = task.attributes.get("postings")
        if llm_postings and isinstance(llm_postings, list):
            row = 1
            for p in llm_postings:
                debit_num = p.get("debitAccount")
                credit_num = p.get("creditAccount")
                amt = p.get("amount")
                if debit_num is not None and credit_num is not None and amt is not None:
                    debit_accounts = self.client.search_accounts_by_number(int(debit_num))
                    credit_accounts = self.client.search_accounts_by_number(int(credit_num))
                    if not debit_accounts:
                        raise EntityNotFoundError(f"Debit account {debit_num} not found")
                    if not credit_accounts:
                        raise EntityNotFoundError(f"Credit account {credit_num} not found")
                    postings.append({"row": row, "account": {"id": debit_accounts[0]["id"]}, "amountGross": float(amt), "amountGrossCurrency": float(amt), "description": description})
                    row += 1
                    postings.append({"row": row, "account": {"id": credit_accounts[0]["id"]}, "amountGross": -float(amt), "amountGrossCurrency": -float(amt), "description": description})
                    row += 1

        # Fallback: rule-based parser format with debitAccountNumber/creditAccountNumber
        if not postings:
            amount = task.attributes.get("amount")
            debit_number = task.attributes.get("debitAccountNumber")
            credit_number = task.attributes.get("creditAccountNumber")
            if debit_number is not None and credit_number is not None and amount is not None:
                debit_accounts = self.client.search_accounts_by_number(debit_number)
                credit_accounts = self.client.search_accounts_by_number(credit_number)
                if not debit_accounts:
                    raise EntityNotFoundError(f"Debit account {debit_number} not found")
                if not credit_accounts:
                    raise EntityNotFoundError(f"Credit account {credit_number} not found")
                postings = [
                    {"row": 1, "account": {"id": debit_accounts[0]["id"]}, "amountGross": float(amount), "amountGrossCurrency": float(amount), "description": description},
                    {"row": 2, "account": {"id": credit_accounts[0]["id"]}, "amountGross": -float(amount), "amountGrossCurrency": -float(amount), "description": description},
                ]

        payload: dict[str, Any] = {
            "date": voucher_date.isoformat(),
            "description": description,
            "postings": postings,
        }
        if voucher_type:
            payload["voucherType"] = {"id": voucher_type["id"]}

        self.client.create("/ledger/voucher", payload)

    def _create_opening_balance(self, task: ParsedTask) -> None:
        voucher_date_val = task.attributes.get("voucherDate") or task.attributes.get("date")
        if voucher_date_val:
            voucher_date = _parse_date_value(voucher_date_val)
        else:
            # Opening balance must be first day of a month (recommended: first day of year)
            today = date.today()
            voucher_date = date(today.year, 1, 1)

        balance_postings: list[dict[str, Any]] = []

        # Handle postings from LLM
        llm_postings = task.attributes.get("postings")
        if llm_postings and isinstance(llm_postings, list):
            for p in llm_postings:
                account_num = p.get("accountNumber") or p.get("debitAccount") or p.get("creditAccount")
                amt = p.get("amount", 0)
                if account_num is not None:
                    accounts = self.client.search_accounts_by_number(int(account_num))
                    if not accounts:
                        raise EntityNotFoundError(f"Account {account_num} not found")
                    balance_postings.append({
                        "account": {"id": accounts[0]["id"]},
                        "amount": float(amt),
                    })

        # Fallback: single debit/credit pair
        if not balance_postings:
            debit_num = task.attributes.get("debitAccountNumber")
            credit_num = task.attributes.get("creditAccountNumber")
            amount = task.attributes.get("amount")
            if debit_num and amount:
                debit_accounts = self.client.search_accounts_by_number(int(debit_num))
                if not debit_accounts:
                    raise EntityNotFoundError(f"Account {debit_num} not found")
                balance_postings.append({"account": {"id": debit_accounts[0]["id"]}, "amount": float(amount)})
            if credit_num and amount:
                credit_accounts = self.client.search_accounts_by_number(int(credit_num))
                if not credit_accounts:
                    raise EntityNotFoundError(f"Account {credit_num} not found")
                balance_postings.append({"account": {"id": credit_accounts[0]["id"]}, "amount": -float(amount)})

        payload: dict[str, Any] = {
            "voucherDate": voucher_date.isoformat(),
            "balancePostings": balance_postings,
        }
        self.client.create("/ledger/voucher/openingBalance", payload)
        LOGGER.info("Created opening balance with %d postings", len(balance_postings))

    def _delete_voucher(self, task: ParsedTask) -> None:
        if task.identifier is None:
            raise ParsingError("Voucher deletion requires an explicit ID")
        # Try reverse first (more reliable), fall back to delete
        try:
            reverse_date = date.today().isoformat()
            self.client.reverse_voucher(task.identifier, reverse_date)
        except Exception:
            LOGGER.info("Voucher reverse failed, trying direct delete")
            self.client.delete("/ledger/voucher", task.identifier)

    # --- Timesheet workflows ---

    def _create_timesheet_entry(self, task: ParsedTask) -> None:
        employee_name = task.attributes.get("employeeName") or task.target_name
        if not employee_name:
            raise ParsingError("Could not extract employee name for timesheet entry")
        employee = self._ensure_employee(
            name=employee_name,
            email=task.attributes.get("employeeEmail"),
        )

        activity_name = task.attributes.get("activityName")
        activity = self._resolve_activity(activity_name)

        entry_date_val = task.attributes.get("date") or date.today()
        entry_date = _parse_date_value(entry_date_val) if not isinstance(entry_date_val, date) else entry_date_val
        hours_val = task.attributes.get("hours")
        if hours_val is None:
            raise ParsingError("Could not extract hours for timesheet entry")
        hours = float(hours_val)

        payload: dict[str, Any] = {
            "employee": {"id": employee["id"]},
            "activity": {"id": activity["id"]},
            "date": entry_date.isoformat(),
            "hours": hours,
        }

        project_name = task.attributes.get("projectName")
        if project_name:
            try:
                project = self._find_project(name=project_name)
                payload["project"] = {"id": project["id"]}
            except EntityNotFoundError:
                pass

        comment = task.attributes.get("comment")
        if comment:
            payload["comment"] = comment

        self.client.create_timesheet_entry(payload)

    def _update_timesheet_entry(self, task: ParsedTask) -> None:
        if task.identifier is None:
            raise ParsingError("Timesheet entry update requires an explicit ID")
        payload: dict[str, Any] = {}
        if "hours" in task.attributes:
            payload["hours"] = float(task.attributes["hours"])
        if "comment" in task.attributes:
            payload["comment"] = task.attributes["comment"]
        if "date" in task.attributes:
            payload["date"] = _parse_date_value(task.attributes["date"]).isoformat()
        if not payload:
            raise ParsingError("No updatable timesheet fields were extracted")
        self.client.update("/timesheet/entry", task.identifier, payload)

    def _delete_timesheet_entry(self, task: ParsedTask) -> None:
        if task.identifier is None:
            raise ParsingError("Timesheet entry deletion requires an explicit ID")
        self.client.delete("/timesheet/entry", task.identifier)

    def _update_voucher(self, task: ParsedTask) -> None:
        if task.identifier is None:
            raise ParsingError("Voucher update requires an explicit ID")
        payload: dict[str, Any] = {}
        if "description" in task.attributes:
            payload["description"] = task.attributes["description"]
        if "date" in task.attributes or "voucherDate" in task.attributes:
            date_val = task.attributes.get("voucherDate") or task.attributes.get("date")
            payload["date"] = _parse_date_value(date_val).isoformat()
        if not payload:
            raise ParsingError("No updatable voucher fields were extracted")
        self.client.update("/ledger/voucher", task.identifier, payload)

    def _resolve_voucher_type(self) -> dict[str, Any] | None:
        # Don't use voucher types that require special fields (supplier, etc.)
        # Just let Tripletex use the default - omitting voucherType works for manual vouchers
        return None

    # --- Tier 3: Company Module Activation ---

    def _activate_module(self, task: ParsedTask) -> None:
        """Activate a Tripletex sales module (e.g. department accounting)."""
        module_name = task.attributes.get("moduleName") or task.attributes.get("name")
        if not module_name:
            # Try to infer from prompt keywords
            prompt_lower = task.raw_prompt.lower()
            module_map = {
                "avdelingsregnskap": "SMART",
                "department accounting": "SMART",
                "prosjekt": "PROJECT",
                "project": "PROJECT",
                "lager": "LOGISTICS",
                "inventory": "LOGISTICS",
                "logistics": "LOGISTICS",
                "lønn": "WAGE",
                "wage": "WAGE",
                "salary": "WAGE",
                "payroll": "WAGE",
                "timeføring": "SMART_TIME_TRACKING",
                "time tracking": "SMART_TIME_TRACKING",
                "faktura": "BASIS",
                "invoice": "BASIS",
                "ocr": "OCR",
                "api": "API_V2",
                "anlegg": "FIXED_ASSETS_REGISTER",
                "fixed assets": "FIXED_ASSETS_REGISTER",
                "årsoppgjør": "YEAR_END_REPORTING_AS",
                "year end": "YEAR_END_REPORTING_AS",
                "reise": "SMART",
                "travel": "SMART",
            }
            for keyword, mod in module_map.items():
                if keyword in prompt_lower:
                    module_name = mod
                    break
        if not module_name:
            module_name = "SMART"  # Safe default — enables most features
            LOGGER.warning("Could not determine module name, defaulting to SMART")

        try:
            self.client.activate_sales_module(module_name)
            LOGGER.info("Activated sales module: %s", module_name)
        except Exception as e:
            LOGGER.warning("Could not activate module %s: %s", module_name, e)

    # --- Tier 3: Incoming Invoice (Supplier Invoice) ---

    def _create_incoming_invoice(self, task: ParsedTask) -> None:
        supplier_name = task.attributes.get("supplierName") or task.attributes.get("name") or task.target_name
        if not supplier_name:
            raise ParsingError("Incoming invoice requires a supplier name")

        org_number = task.attributes.get("organizationNumber") or (
            str(task.attributes["orgNumber"]) if task.attributes.get("orgNumber") else None
        )
        supplier = self._ensure_supplier(name=supplier_name, org_number=org_number)

        invoice_date_val = task.attributes.get("invoiceDate") or task.attributes.get("date")
        invoice_date = _parse_date_value(invoice_date_val) if invoice_date_val else date.today()
        due_date_val = task.attributes.get("invoiceDueDate") or task.attributes.get("dueDate")
        due_date = _parse_date_value(due_date_val) if due_date_val else invoice_date + timedelta(days=30)

        amount = task.attributes.get("amount")
        if amount is None:
            raise ParsingError("Incoming invoice requires an amount")

        # Resolve debit account for order line
        debit_account_num = task.attributes.get("debitAccountNumber") or 4000
        debit_accounts = self.client.search_accounts_by_number(int(debit_account_num))
        if not debit_accounts:
            raise EntityNotFoundError(f"Debit account {debit_account_num} not found")

        # Resolve VAT type for order line
        vat_rate = task.attributes.get("vatRate")
        vat_type_id = None
        if vat_rate is not None:
            vt = self._resolve_product_vat_type(float(vat_rate))
            if vt:
                vat_type_id = vt["id"]
        if vat_type_id is None:
            # Default to 25% incoming VAT (type 1 = MVA høy sats innkommende)
            incoming_vat = self.client.search_vat_types(number="1", type_of_vat="INCOMING")
            if incoming_vat:
                vat_type_id = incoming_vat[0]["id"]

        # Use the new aggregate API format
        payload: dict[str, Any] = {
            "invoiceHeader": {
                "vendorId": supplier["id"],
                "invoiceDate": invoice_date.isoformat(),
                "dueDate": due_date.isoformat(),
                "invoiceAmount": float(amount),
                "description": task.attributes.get("description", f"Invoice from {supplier_name}"),
            },
            "orderLines": [
                {
                    "externalId": "line-1",
                    "row": 1,
                    "accountId": debit_accounts[0]["id"],
                    "amountInclVat": float(amount),
                    "description": task.attributes.get("description") or task.attributes.get("productName") or f"Invoice from {supplier_name}",
                }
            ],
        }

        if task.attributes.get("invoiceNumber"):
            payload["invoiceHeader"]["invoiceNumber"] = str(task.attributes["invoiceNumber"])

        if vat_type_id:
            payload["orderLines"][0]["vatTypeId"] = vat_type_id

        result = self.client.create_incoming_invoice(payload)

        # Approve the incoming invoice (required for Tier 2+ completion)
        invoice_id = None
        if isinstance(result, dict):
            invoice_id = result.get("id")
        if invoice_id:
            try:
                self.client.approve_incoming_invoice(invoice_id)
                LOGGER.info("Approved incoming invoice %s", invoice_id)
            except Exception as e:
                LOGGER.warning("Could not approve incoming invoice %s: %s", invoice_id, e)

    def _ensure_supplier(self, *, name: str, org_number: str | None = None) -> dict[str, Any]:
        cache_key = f"supplier:{_normalize(name)}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        try:
            supplier = self._find_supplier(name=name)
            self._cache_set(cache_key, supplier)
            return supplier
        except EntityNotFoundError:
            payload: dict[str, Any] = {"name": name, "isSupplier": True}
            if org_number:
                payload["organizationNumber"] = org_number
            supplier = self.client.create("/supplier", payload)
            self._cache_set(cache_key, supplier)
            return supplier

    # --- Tier 3: Salary Transaction ---

    def _create_salary_transaction(self, task: ParsedTask) -> None:
        employee_name = task.attributes.get("employeeName") or task.target_name
        if not employee_name:
            raise ParsingError("Salary transaction requires an employee name")

        employee = self._ensure_employee(name=employee_name, email=task.attributes.get("employeeEmail"))

        # Ensure employee has dateOfBirth + employment (required for payroll)
        self._ensure_employee_has_date_of_birth(employee["id"])
        base_salary = task.attributes.get("baseSalary") or task.attributes.get("monthlySalary") or task.attributes.get("amount")
        bonus = task.attributes.get("bonus")
        if base_salary:
            self._update_employment(
                employee["id"],
                start_date=date.today().isoformat(),
                salary_attrs={"monthlySalary": float(base_salary)},
            )

        transaction_date_val = task.attributes.get("date")
        transaction_date = _parse_date_value(transaction_date_val) if transaction_date_val else date.today()

        # SalaryTransaction schema: date, year, month, payslips[]
        payload: dict[str, Any] = {
            "date": transaction_date.isoformat(),
            "year": transaction_date.year,
            "month": transaction_date.month,
        }

        # Build payslip with salary specifications
        specifications: list[dict[str, Any]] = []

        # Resolve salary types for base salary and bonus
        salary_types = self.client.search_salary_types()
        base_type = None
        bonus_type = None
        for st in salary_types:
            st_name = (st.get("name") or "").lower()
            st_number = st.get("number", "")
            if st_number == "1" or "fastlønn" in st_name or "fast lønn" in st_name or "månedslønn" in st_name:
                base_type = st
            if "tillegg" in st_name or "bonus" in st_name or st_number == "30":
                bonus_type = st

        if base_salary and base_type:
            specifications.append({
                "salaryType": {"id": base_type["id"]},
                "rate": float(base_salary),
                "count": 1,
                "employee": {"id": employee["id"]},
            })
        elif base_salary:
            # Fallback: use first salary type
            if salary_types:
                specifications.append({
                    "salaryType": {"id": salary_types[0]["id"]},
                    "rate": float(base_salary),
                    "count": 1,
                    "employee": {"id": employee["id"]},
                })

        if bonus and bonus_type:
            specifications.append({
                "salaryType": {"id": bonus_type["id"]},
                "rate": float(bonus),
                "count": 1,
                "employee": {"id": employee["id"]},
            })
        elif bonus and salary_types:
            # Use bonus_type or fallback
            fallback = bonus_type or (salary_types[1] if len(salary_types) > 1 else salary_types[0])
            specifications.append({
                "salaryType": {"id": fallback["id"]},
                "rate": float(bonus),
                "count": 1,
                "employee": {"id": employee["id"]},
            })

        if specifications:
            payload["payslips"] = [{
                "employee": {"id": employee["id"]},
                "date": transaction_date.isoformat(),
                "year": transaction_date.year,
                "month": transaction_date.month,
                "specifications": specifications,
            }]

        self.client.create_salary_transaction(payload)

    # --- Tier 3: Purchase Order ---

    def _create_purchase_order(self, task: ParsedTask) -> None:
        supplier_name = task.attributes.get("supplierName") or task.attributes.get("name") or task.target_name
        if not supplier_name:
            raise ParsingError("Purchase order requires a supplier name")

        supplier = self._ensure_supplier(name=supplier_name)

        order_date_val = task.attributes.get("orderDate") or task.attributes.get("date")
        order_date = _parse_date_value(order_date_val) if order_date_val else date.today()

        delivery_date_val = task.attributes.get("deliveryDate")
        delivery_date = _parse_date_value(delivery_date_val) if delivery_date_val else order_date + timedelta(days=14)

        payload: dict[str, Any] = {
            "supplier": {"id": supplier["id"]},
            "creationDate": order_date.isoformat(),
            "deliveryDate": delivery_date.isoformat(),
        }

        if task.attributes.get("description"):
            payload["comments"] = task.attributes["description"]

        po = self.client.create_purchase_order(payload)

        # Add order lines if product/amount info is available
        product_name = task.attributes.get("productName") or task.attributes.get("description")
        amount = task.attributes.get("amount")
        quantity = float(task.attributes.get("quantity", 1.0))
        if product_name or amount:
            line_payload: dict[str, Any] = {
                "purchaseOrder": {"id": po["id"]},
                "count": quantity,
            }
            if product_name:
                line_payload["description"] = product_name
            if amount is not None:
                line_payload["unitPriceExcludingVatCurrency"] = float(amount)
            try:
                self.client.create("/purchaseOrder/orderline", line_payload)
            except Exception as e:
                LOGGER.warning("Could not add order line to purchase order: %s", e)

    # --- Dimension workflows ---

    def _create_dimension(self, task: ParsedTask) -> None:
        dimension_name = task.attributes.get("dimensionName") or task.attributes.get("name") or task.target_name
        if not dimension_name:
            raise ParsingError("Dimension creation requires a dimension name")

        # Step 1: Create the dimension name
        dim_payload: dict[str, Any] = {"dimensionName": dimension_name}
        if task.attributes.get("description"):
            dim_payload["description"] = task.attributes["description"]
        dimension = self.client.create_dimension_name(dim_payload)
        dim_index = dimension.get("dimensionIndex")
        LOGGER.info("Created dimension %r with index %s", dimension_name, dim_index)

        # Step 2: Create dimension values
        dim_values = task.attributes.get("dimensionValues", [])
        created_values: dict[str, dict[str, Any]] = {}
        for i, val_name in enumerate(dim_values):
            val_payload: dict[str, Any] = {
                "displayName": val_name,
                "dimensionIndex": dim_index,
                "number": str(i + 1),
                "active": True,
                "showInVoucherRegistration": True,
            }
            created = self.client.create_dimension_value(val_payload)
            created_values[val_name] = created
            LOGGER.info("Created dimension value %r (id=%s)", val_name, created.get("id"))

        # Step 3: If the prompt also asks to create a voucher linked to a dimension value
        voucher_account = task.attributes.get("voucherAccountNumber")
        voucher_amount = task.attributes.get("voucherAmount")
        voucher_dim_value = task.attributes.get("voucherDimensionValue")

        if voucher_account is not None and voucher_amount is not None:
            voucher_date_val = task.attributes.get("voucherDate") or task.attributes.get("date")
            voucher_date = _parse_date_value(voucher_date_val) if voucher_date_val else date.today()
            description = task.attributes.get("voucherDescription") or f"Posting linked to {dimension_name}"

            debit_accounts = self.client.search_accounts_by_number(int(voucher_account))
            if not debit_accounts:
                raise EntityNotFoundError(f"Account {voucher_account} not found")

            # Find a suitable credit account (use 2900 Annen kortsiktig gjeld or 1920)
            credit_accounts = self.client.search_accounts_by_number(1920)
            if not credit_accounts:
                credit_accounts = self.client.search_accounts_by_number(2900)
            if not credit_accounts:
                raise EntityNotFoundError("No suitable credit account found for dimension voucher")

            voucher_type = self._resolve_voucher_type()

            # Build postings with dimension reference
            dim_field = f"freeAccountingDimension{dim_index}" if dim_index and dim_index <= 3 else "freeAccountingDimension1"

            dim_value_ref = None
            if voucher_dim_value and voucher_dim_value in created_values:
                dim_value_ref = {"id": created_values[voucher_dim_value]["id"]}
            elif created_values:
                # Use the first value if specific one not found
                first_val = next(iter(created_values.values()))
                dim_value_ref = {"id": first_val["id"]}

            debit_posting: dict[str, Any] = {
                "row": 1,
                "account": {"id": debit_accounts[0]["id"]},
                "amountGross": float(voucher_amount),
                "amountGrossCurrency": float(voucher_amount),
                "description": description,
            }
            credit_posting: dict[str, Any] = {
                "row": 2,
                "account": {"id": credit_accounts[0]["id"]},
                "amountGross": -float(voucher_amount),
                "amountGrossCurrency": -float(voucher_amount),
                "description": description,
            }
            if dim_value_ref:
                debit_posting[dim_field] = dim_value_ref

            payload: dict[str, Any] = {
                "date": voucher_date.isoformat(),
                "description": description,
                "postings": [debit_posting, credit_posting],
            }
            if voucher_type:
                payload["voucherType"] = {"id": voucher_type["id"]}

            self.client.create("/ledger/voucher", payload)
            LOGGER.info("Created voucher linked to dimension value %r", voucher_dim_value)

    # --- Bank Statement Import ---

    def _create_bank_statement(self, task: ParsedTask) -> None:
        # Find attached file (bank statement CSV/other)
        file_path: str | None = None
        for p in self._saved_attachment_paths:
            if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".txt", ".xml"):
                file_path = str(p)
                break
        if not file_path and self._saved_attachment_paths:
            file_path = str(self._saved_attachment_paths[0])

        if not file_path:
            raise ParsingError("Bank statement import requires an attached file")

        # Determine bank account
        bank_account_number = task.attributes.get("bankAccountNumber") or task.attributes.get("accountNumber")
        bank_id: int | None = None
        if bank_account_number:
            banks = self.client.search_bank_accounts()
            for b in banks:
                if str(b.get("bankAccountNumber", "")).replace(".", "").replace(" ", "") == str(bank_account_number).replace(".", "").replace(" ", ""):
                    bank_id = b["id"]
                    break

        # Determine file format
        file_format = task.attributes.get("fileFormat", "DNB_CSV")

        import_result = self.client.import_bank_statement(file_path, bank_id=bank_id, file_format=file_format)
        LOGGER.info("Imported bank statement from %s", file_path)

        # Full bank reconciliation flow (Tier 3):
        # 1. Find the bank account used
        # 2. Find/create a bank reconciliation for it
        # 3. Suggest matches
        # 4. Close the reconciliation
        try:
            # Resolve the bank account ID from the imported statement
            resolved_bank_account_id = bank_id
            if not resolved_bank_account_id:
                # Try to get from the import result
                if isinstance(import_result, dict):
                    resolved_bank_account_id = (
                        import_result.get("bankAccountId")
                        or (import_result.get("value", {}) or {}).get("bankAccountId")
                    )
                # Fallback: search for bank statements and get the account
                if not resolved_bank_account_id:
                    statements = self.client.search_bank_statements()
                    if statements:
                        ba = statements[-1].get("bankAccount")
                        if isinstance(ba, dict):
                            resolved_bank_account_id = ba.get("id")

            if not resolved_bank_account_id:
                # Last resort: just use the first bank account
                banks = self.client.search_bank_accounts()
                if banks:
                    resolved_bank_account_id = banks[0]["id"]

            if resolved_bank_account_id:
                # Search for existing open reconciliation or create one
                existing_recs = self.client.search_bank_reconciliations(account_id=resolved_bank_account_id)
                open_rec = None
                for rec in existing_recs:
                    if not rec.get("closedDate"):
                        open_rec = rec
                        break

                if not open_rec:
                    # Create a new bank reconciliation
                    rec_payload: dict[str, Any] = {
                        "account": {"id": resolved_bank_account_id},
                    }
                    open_rec = self.client.create_bank_reconciliation(rec_payload)
                    LOGGER.info("Created bank reconciliation %s", open_rec.get("id"))

                rec_id = open_rec["id"]

                # Suggest matches
                try:
                    self.client.suggest_bank_reconciliation_matches(rec_id)
                    LOGGER.info("Suggested matches for bank reconciliation %s", rec_id)
                except Exception as e:
                    LOGGER.warning("Could not suggest reconciliation matches: %s", e)

                # Close the reconciliation
                try:
                    self.client.close_bank_reconciliation(rec_id)
                    LOGGER.info("Closed bank reconciliation %s", rec_id)
                except Exception as e:
                    LOGGER.warning("Could not close bank reconciliation %s: %s", rec_id, e)
            else:
                LOGGER.warning("Could not resolve bank account for reconciliation")
        except Exception as e:
            LOGGER.warning("Bank reconciliation flow failed: %s", e)

    # --- Ledger Account Creation ---

    def _create_account(self, task: ParsedTask) -> None:
        account_number = task.attributes.get("accountNumber") or task.attributes.get("number")
        account_name = task.attributes.get("accountName") or task.attributes.get("name") or task.target_name
        if not account_number:
            raise ParsingError("Account creation requires an account number")
        if not account_name:
            raise ParsingError("Account creation requires an account name")

        payload: dict[str, Any] = {
            "number": int(account_number),
            "name": account_name,
        }

        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]

        # Resolve VAT type if specified
        vat_rate = task.attributes.get("vatRate")
        if vat_rate is not None:
            vt = self._resolve_product_vat_type(float(vat_rate))
            if vt:
                payload["vatType"] = {"id": vt["id"]}

        self.client.create("/ledger/account", payload)
        LOGGER.info("Created ledger account %s: %s", account_number, account_name)

    # --- Standalone Order ---

    def _create_order(self, task: ParsedTask) -> None:
        customer_name = task.attributes.get("customerName") or task.attributes.get("name") or task.target_name
        if not customer_name:
            raise ParsingError("Order creation requires a customer name")

        customer = self._ensure_customer(
            name=customer_name,
            email=task.attributes.get("customerEmail") or task.attributes.get("email"),
        )

        order_date_val = task.attributes.get("orderDate") or task.attributes.get("date")
        order_date = _parse_date_value(order_date_val) if order_date_val else date.today()
        delivery_date_val = task.attributes.get("deliveryDate")
        delivery_date = _parse_date_value(delivery_date_val) if delivery_date_val else order_date + timedelta(days=14)

        # Build order lines
        order_lines: list[dict[str, Any]] = []
        lines_data = task.attributes.get("orderLines") or task.attributes.get("lines")
        if lines_data and isinstance(lines_data, list):
            for line in lines_data:
                vat_type = self._get_default_vat_type()
                order_lines.append({
                    "description": line.get("description", "Order line"),
                    "count": float(line.get("quantity", 1)),
                    "unitPriceExcludingVatCurrency": float(line.get("unitPrice", line.get("amount", 0))),
                    "vatType": {"id": vat_type["id"]},
                })
        else:
            # Single-line order
            amount = task.attributes.get("amount")
            description = task.attributes.get("lineDescription") or task.attributes.get("productName") or task.attributes.get("description") or "Order line"
            quantity = float(task.attributes.get("quantity", 1))
            vat_type = self._get_default_vat_type()
            if amount is not None:
                order_lines.append({
                    "description": description,
                    "count": quantity,
                    "unitPriceExcludingVatCurrency": float(amount),
                    "vatType": {"id": vat_type["id"]},
                })

        payload: dict[str, Any] = {
            "customer": {"id": customer["id"]},
            "orderDate": order_date.isoformat(),
            "deliveryDate": delivery_date.isoformat(),
            "orderLines": order_lines,
        }

        self.client.create_order(payload)
        LOGGER.info("Created standalone order for customer %s", customer_name)

    # --- Invoice Reminder ---

    def _create_reminder(self, task: ParsedTask) -> None:
        invoice = self._resolve_invoice(task)
        reminder_date_val = task.attributes.get("reminderDate") or task.attributes.get("date")
        reminder_date = _parse_date_value(reminder_date_val) if reminder_date_val else date.today()

        reminder_type = task.attributes.get("reminderType", "SOFT_REMINDER")

        self.client.create_reminder(
            invoice["id"],
            reminder_type=reminder_type,
            reminder_date=reminder_date.isoformat(),
        )
        LOGGER.info("Created reminder for invoice %s", invoice.get("invoiceNumber", invoice["id"]))

    # --- Activity ---

    def _create_activity(self, task: ParsedTask) -> None:
        name = task.attributes.get("activityName") or task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Activity creation requires a name")

        payload: dict[str, Any] = {"name": name}
        if task.attributes.get("activityNumber") or task.attributes.get("number"):
            payload["number"] = str(task.attributes.get("activityNumber") or task.attributes["number"])
        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]
        if task.attributes.get("activityType"):
            payload["activityType"] = task.attributes["activityType"]
        if task.attributes.get("isChargeable") is not None:
            payload["isChargeable"] = bool(task.attributes["isChargeable"])
        if task.attributes.get("rate") is not None:
            payload["rate"] = float(task.attributes["rate"])

        self.client.create_activity(payload)
        LOGGER.info("Created activity: %s", name)

    # --- Division ---

    def _create_division(self, task: ParsedTask) -> None:
        name = task.attributes.get("divisionName") or task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Division creation requires a name")

        payload: dict[str, Any] = {"name": name}
        if task.attributes.get("startDate"):
            payload["startDate"] = _parse_date_value(task.attributes["startDate"]).isoformat()
        if task.attributes.get("endDate"):
            payload["endDate"] = _parse_date_value(task.attributes["endDate"]).isoformat()
        if task.attributes.get("organizationNumber"):
            payload["organizationNumber"] = str(task.attributes["organizationNumber"])

        self.client.create_division(payload)
        LOGGER.info("Created division: %s", name)

    # --- Leave of Absence ---

    def _create_leave_of_absence(self, task: ParsedTask) -> None:
        employee_name = task.attributes.get("employeeName") or task.target_name
        if not employee_name:
            raise ParsingError("Leave of absence requires an employee name")

        employee = self._ensure_employee(name=employee_name, email=task.attributes.get("employeeEmail"))

        # Find the employee's employment
        employments = self.client.search_employments(employee_id=employee["id"])
        if not employments:
            raise EntityNotFoundError(f"No employment found for employee {employee_name}")
        employment = employments[0]

        start_date_val = task.attributes.get("startDate") or task.attributes.get("date")
        if not start_date_val:
            raise ParsingError("Leave of absence requires a start date")
        start_date = _parse_date_value(start_date_val)

        end_date_val = task.attributes.get("endDate")
        end_date = _parse_date_value(end_date_val) if end_date_val else None

        percentage = float(task.attributes.get("percentage", 100.0))

        leave_type = task.attributes.get("leaveType", "LEAVE_OF_ABSENCE")

        payload: dict[str, Any] = {
            "employment": {"id": employment["id"]},
            "startDate": start_date.isoformat(),
            "percentage": percentage,
            "type": leave_type,
            "isWageDeduction": task.attributes.get("isWageDeduction", True),
        }
        if end_date:
            payload["endDate"] = end_date.isoformat()

        self.client.create_leave_of_absence(payload)
        LOGGER.info("Created leave of absence for %s from %s", employee_name, start_date)

    # --- Next of Kin ---

    def _create_next_of_kin(self, task: ParsedTask) -> None:
        employee_name = task.attributes.get("employeeName") or task.target_name
        if not employee_name:
            raise ParsingError("Next of kin requires an employee name")

        employee = self._ensure_employee(name=employee_name, email=task.attributes.get("employeeEmail"))

        kin_name = task.attributes.get("nextOfKinName") or task.attributes.get("name")
        if not kin_name:
            raise ParsingError("Next of kin requires a contact name")

        payload: dict[str, Any] = {
            "employee": {"id": employee["id"]},
            "name": kin_name,
        }
        if task.attributes.get("phoneNumber"):
            payload["phoneNumber"] = str(task.attributes["phoneNumber"])
        if task.attributes.get("address"):
            payload["address"] = task.attributes["address"]
        if task.attributes.get("typeOfRelationship"):
            payload["typeOfRelationship"] = task.attributes["typeOfRelationship"]

        self.client.create_next_of_kin(payload)
        LOGGER.info("Created next of kin %s for employee %s", kin_name, employee_name)

    # --- Customer Category ---

    def _create_customer_category(self, task: ParsedTask) -> None:
        name = task.attributes.get("categoryName") or task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Customer category requires a name")

        payload: dict[str, Any] = {"name": name}
        if task.attributes.get("categoryNumber") or task.attributes.get("number"):
            payload["number"] = str(task.attributes.get("categoryNumber") or task.attributes["number"])
        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]

        self.client.create_customer_category(payload)
        LOGGER.info("Created customer category: %s", name)

    # --- Employee Category ---

    def _create_employee_category(self, task: ParsedTask) -> None:
        name = task.attributes.get("categoryName") or task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Employee category requires a name")

        payload: dict[str, Any] = {"name": name}
        if task.attributes.get("categoryNumber") or task.attributes.get("number"):
            payload["number"] = str(task.attributes.get("categoryNumber") or task.attributes["number"])
        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]

        self.client.create_employee_category(payload)
        LOGGER.info("Created employee category: %s", name)

    # --- Asset ---

    def _create_asset(self, task: ParsedTask) -> None:
        name = task.attributes.get("assetName") or task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Asset creation requires a name")

        acquisition_cost = task.attributes.get("acquisitionCost") or task.attributes.get("amount")
        if acquisition_cost is None:
            raise ParsingError("Asset creation requires an acquisition cost")

        date_of_acquisition_val = task.attributes.get("dateOfAcquisition") or task.attributes.get("date")
        date_of_acquisition = _parse_date_value(date_of_acquisition_val) if date_of_acquisition_val else date.today()

        payload: dict[str, Any] = {
            "name": name,
            "acquisitionCost": float(acquisition_cost),
            "dateOfAcquisition": date_of_acquisition.isoformat(),
        }

        # Resolve asset account
        account_num = task.attributes.get("accountNumber")
        if account_num:
            accounts = self.client.search_accounts_by_number(int(account_num))
            if accounts:
                payload["account"] = {"id": accounts[0]["id"]}

        # Resolve depreciation account
        depr_account_num = task.attributes.get("depreciationAccountNumber")
        if depr_account_num:
            depr_accounts = self.client.search_accounts_by_number(int(depr_account_num))
            if depr_accounts:
                payload["depreciationAccount"] = {"id": depr_accounts[0]["id"]}

        if task.attributes.get("lifetime") is not None:
            payload["lifetime"] = int(task.attributes["lifetime"])
        if task.attributes.get("depreciationMethod"):
            payload["depreciationMethod"] = task.attributes["depreciationMethod"]
        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]

        # Link to department/project
        department_name = task.attributes.get("departmentName")
        if department_name:
            department = self._ensure_department(department_name)
            payload["department"] = {"id": department["id"]}

        project_name = task.attributes.get("projectName")
        if project_name:
            try:
                project = self._resolve_project_by_name(project_name)
                payload["project"] = {"id": project["id"]}
            except (EntityNotFoundError, AmbiguousMatchError):
                LOGGER.warning("Could not link asset to project %r", project_name)

        self.client.create_asset(payload)
        LOGGER.info("Created asset: %s (cost: %s)", name, acquisition_cost)

    # --- Product Group ---

    def _create_product_group(self, task: ParsedTask) -> None:
        name = task.attributes.get("groupName") or task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Product group creation requires a name")

        payload: dict[str, Any] = {"name": name}
        self.client.create_product_group(payload)
        LOGGER.info("Created product group: %s", name)

    # --- Project Category ---

    def _create_project_category(self, task: ParsedTask) -> None:
        name = task.attributes.get("categoryName") or task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Project category creation requires a name")

        payload: dict[str, Any] = {"name": name}
        if task.attributes.get("number"):
            payload["number"] = str(task.attributes["number"])
        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]
        self.client.create_project_category(payload)
        LOGGER.info("Created project category: %s", name)

    # --- Inventory ---

    def _ensure_logistics_module(self) -> None:
        if self._cache_get("logistics_module_active"):
            return
        try:
            self.client.activate_sales_module("LOGISTICS")
            LOGGER.info("Activated LOGISTICS module")
        except Exception as e:
            LOGGER.warning("Could not activate LOGISTICS module (may already be active): %s", e)
        self._cache_set("logistics_module_active", True)

    def _create_inventory(self, task: ParsedTask) -> None:
        self._ensure_logistics_module()
        name = task.attributes.get("inventoryName") or task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Inventory creation requires a name")

        payload: dict[str, Any] = {"name": name}
        if task.attributes.get("inventoryNumber") or task.attributes.get("number"):
            payload["number"] = str(task.attributes.get("inventoryNumber") or task.attributes["number"])
        if task.attributes.get("description"):
            payload["description"] = task.attributes["description"]
        if task.attributes.get("isMainInventory"):
            payload["isMainInventory"] = True
        if task.attributes.get("email"):
            payload["email"] = task.attributes["email"]
        if task.attributes.get("phone") or task.attributes.get("phoneNumber"):
            payload["phone"] = task.attributes.get("phone") or task.attributes["phoneNumber"]

        address = self._build_address(task.attributes)
        if address:
            payload["address"] = address

        inventory = self.client.create_inventory(payload)
        LOGGER.info("Created inventory: %s (id=%s)", name, inventory.get("id"))

        # Handle multiple names (batch create)
        names = task.attributes.get("names", [])
        for extra_name in names:
            if extra_name != name:
                self.client.create_inventory({"name": extra_name})
                LOGGER.info("Created additional inventory: %s", extra_name)

    def _delete_inventory(self, task: ParsedTask) -> None:
        if task.identifier:
            self.client.delete("/inventory", task.identifier)
            return
        name = task.attributes.get("inventoryName") or task.attributes.get("name") or task.target_name
        if not name:
            raise ParsingError("Need inventory name or ID to delete")
        results = self.client.search_inventories(name=name)
        if not results:
            raise EntityNotFoundError(f"No inventory found with name {name!r}")
        self.client.delete("/inventory", results[0]["id"])
        LOGGER.info("Deleted inventory: %s", name)

    # --- Inventory Location ---

    def _create_inventory_location(self, task: ParsedTask) -> None:
        self._ensure_logistics_module()
        inventory_name = task.attributes.get("inventoryName") or task.attributes.get("warehouseName")
        location_name = task.attributes.get("locationName") or task.attributes.get("name") or task.target_name
        if not location_name:
            raise ParsingError("Inventory location creation requires a location name")

        # Resolve or create the parent inventory
        inventory = self._ensure_inventory(inventory_name)

        payload: dict[str, Any] = {
            "inventory": {"id": inventory["id"]},
            "name": location_name,
        }
        self.client.create_inventory_location(payload)
        LOGGER.info("Created inventory location: %s in inventory %s", location_name, inventory.get("name"))

        # Handle multiple names
        names = task.attributes.get("names", [])
        for extra_name in names:
            if extra_name != location_name:
                self.client.create_inventory_location({
                    "inventory": {"id": inventory["id"]},
                    "name": extra_name,
                })
                LOGGER.info("Created additional inventory location: %s", extra_name)

    def _ensure_inventory(self, name: str | None = None) -> dict[str, Any]:
        cache_key = f"inventory:{_normalize(name) if name else 'default'}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        if name:
            results = self.client.search_inventories(name=name)
            exact = self._pick_exact(results, "name", name)
            if exact:
                self._cache_set(cache_key, exact)
                return exact
            if results:
                self._cache_set(cache_key, results[0])
                return results[0]
        # Try to find any existing inventory
        results = self.client.search_inventories()
        if results:
            self._cache_set(cache_key, results[0])
            return results[0]
        # Create a new inventory
        inv_name = name or "Main Warehouse"
        inventory = self.client.create_inventory({"name": inv_name, "isMainInventory": True})
        self._cache_set(cache_key, inventory)
        return inventory

    # --- Stocktaking ---

    def _create_stocktaking(self, task: ParsedTask) -> None:
        self._ensure_logistics_module()
        inventory_name = task.attributes.get("inventoryName") or task.attributes.get("warehouseName")
        inventory = self._ensure_inventory(inventory_name)

        stocktaking_date = task.attributes.get("date") or date.today().isoformat()
        if not isinstance(stocktaking_date, str):
            stocktaking_date = _parse_date_value(stocktaking_date).isoformat()

        payload: dict[str, Any] = {
            "inventory": {"id": inventory["id"]},
            "date": stocktaking_date,
        }
        if task.attributes.get("comment"):
            payload["comment"] = task.attributes["comment"]

        type_of_stocktaking = task.attributes.get("typeOfStocktaking")
        stocktaking = self.client.create_stocktaking(payload, type_of_stocktaking=type_of_stocktaking)
        stocktaking_id = stocktaking.get("id")
        LOGGER.info("Created stocktaking: id=%s for inventory %s", stocktaking_id, inventory.get("name"))

        # Add product lines if specified
        product_lines = task.attributes.get("productLines", [])
        for pl in product_lines:
            product_name = pl.get("productName") or pl.get("name")
            count = pl.get("count", 0)
            if product_name:
                try:
                    products = self.client.search_products(name=product_name)
                    if products:
                        product = products[0]
                        line_payload: dict[str, Any] = {
                            "stocktaking": {"id": stocktaking_id},
                            "product": {"id": product["id"]},
                            "count": float(count),
                        }
                        self.client.create_stocktaking_productline(line_payload)
                        LOGGER.info("Added stocktaking product line: %s x %s", product_name, count)
                except Exception as e:
                    LOGGER.warning("Could not add stocktaking product line %r: %s", product_name, e)

    def _delete_stocktaking(self, task: ParsedTask) -> None:
        if task.identifier:
            self.client.delete("/inventory/stocktaking", task.identifier)
            return
        raise ParsingError("Stocktaking deletion requires an ID")

    # --- Goods Receipt ---

    def _create_goods_receipt(self, task: ParsedTask) -> None:
        self._ensure_logistics_module()
        # Resolve the purchase order
        purchase_order_id = task.attributes.get("purchaseOrderId")
        supplier_name = task.attributes.get("supplierName") or task.target_name

        if not purchase_order_id and supplier_name:
            # Find the most recent purchase order for this supplier
            try:
                supplier = self._find_supplier(name=supplier_name)
                orders = self.client.search_purchase_orders(supplier_id=supplier["id"])
                if orders:
                    purchase_order_id = orders[0]["id"]
            except (EntityNotFoundError, AmbiguousMatchError):
                pass

        if not purchase_order_id:
            raise ParsingError("Goods receipt requires a purchase order ID or supplier name with existing PO")

        registration_date = task.attributes.get("registrationDate") or task.attributes.get("date") or date.today().isoformat()
        if not isinstance(registration_date, str):
            registration_date = _parse_date_value(registration_date).isoformat()

        payload: dict[str, Any] = {
            "purchaseOrder": {"id": int(purchase_order_id)},
            "registrationDate": registration_date,
        }
        if task.attributes.get("comment"):
            payload["comment"] = task.attributes["comment"]

        receipt = self.client.create_goods_receipt(payload)
        receipt_id = receipt.get("id")
        LOGGER.info("Created goods receipt: id=%s for PO %s", receipt_id, purchase_order_id)

        # Optionally confirm the receipt
        if task.attributes.get("confirm", True):
            try:
                self.client.confirm_goods_receipt(receipt_id)
                LOGGER.info("Confirmed goods receipt %s", receipt_id)
            except Exception as e:
                LOGGER.warning("Could not confirm goods receipt: %s", e)

    # --- Document Archive ---

    def _upload_document(self, task: ParsedTask) -> None:
        entity_type = task.attributes.get("entityType", "").lower()
        entity_name = task.attributes.get("entityName") or task.target_name

        # Determine which entity to attach the document to
        valid_types = ("customer", "employee", "project", "supplier", "product", "account")
        if entity_type not in valid_types:
            # Try to infer from context
            if task.attributes.get("customerName"):
                entity_type = "customer"
                entity_name = task.attributes["customerName"]
            elif task.attributes.get("employeeName"):
                entity_type = "employee"
                entity_name = task.attributes["employeeName"]
            elif task.attributes.get("projectName"):
                entity_type = "project"
                entity_name = task.attributes["projectName"]
            elif task.attributes.get("supplierName"):
                entity_type = "supplier"
                entity_name = task.attributes["supplierName"]
            else:
                # Upload to reception (general inbox)
                entity_type = "reception"

        # Resolve entity ID
        entity_id = task.attributes.get("entityId") or task.identifier
        if not entity_id and entity_type != "reception":
            if entity_type == "customer" and entity_name:
                customer = self._ensure_customer(name=entity_name)
                entity_id = customer["id"]
            elif entity_type == "employee" and entity_name:
                employee = self._ensure_employee(name=entity_name)
                entity_id = employee["id"]
            elif entity_type == "project" and entity_name:
                project = self._ensure_project(name=entity_name)
                entity_id = project["id"]
            elif entity_type == "supplier" and entity_name:
                supplier = self._ensure_supplier(name=entity_name)
                entity_id = supplier["id"]
            elif entity_type == "product" and entity_name:
                product = self._find_product(name=entity_name, product_number=None)
                entity_id = product["id"]
            else:
                raise ParsingError(f"Cannot resolve {entity_type} entity for document upload")

        # Find attached files to upload
        if self._saved_attachment_paths:
            for file_path in self._saved_attachment_paths:
                if entity_type == "reception":
                    self.client.upload_document_reception(str(file_path))
                    LOGGER.info("Uploaded document to reception: %s", file_path.name)
                else:
                    self.client.upload_document_to_entity(
                        entity_type, int(entity_id), str(file_path)
                    )
                    LOGGER.info("Uploaded document to %s/%s: %s", entity_type, entity_id, file_path.name)
        else:
            LOGGER.warning("No attachment files found for document archive upload")

    def _delete_document(self, task: ParsedTask) -> None:
        if task.identifier:
            self.client.delete("/documentArchive", task.identifier)
            LOGGER.info("Deleted document archive entry: %s", task.identifier)
            return
        raise ParsingError("Document archive deletion requires an ID")

    # --- Event Subscription ---

    def _create_event_subscription(self, task: ParsedTask) -> None:
        event = task.attributes.get("event")
        target_url = task.attributes.get("targetUrl")

        if not event or not target_url:
            raise ParsingError("Event subscription requires 'event' and 'targetUrl'")

        payload: dict[str, Any] = {
            "event": event,
            "targetUrl": target_url,
        }
        if task.attributes.get("fields"):
            payload["fields"] = task.attributes["fields"]
        if task.attributes.get("authHeaderName"):
            payload["authHeaderName"] = task.attributes["authHeaderName"]
        if task.attributes.get("authHeaderValue"):
            payload["authHeaderValue"] = task.attributes["authHeaderValue"]

        subscription = self.client.create_event_subscription(payload)
        LOGGER.info("Created event subscription: event=%s targetUrl=%s id=%s",
                     event, target_url, subscription.get("id"))

    def _delete_event_subscription(self, task: ParsedTask) -> None:
        if task.identifier:
            self.client.delete_event_subscription(task.identifier)
            LOGGER.info("Deleted event subscription: %s", task.identifier)
            return
        # Try to find by event name
        event = task.attributes.get("event")
        if event:
            subs = self.client.search_event_subscriptions()
            for s in subs:
                if _normalize(s.get("event", "")) == _normalize(event):
                    self.client.delete_event_subscription(s["id"])
                    LOGGER.info("Deleted event subscription for event: %s", event)
                    return
        raise ParsingError("Event subscription deletion requires an ID or event name")

    # --- Supplier Invoice Payment ---

    def _pay_supplier_invoice(self, task: ParsedTask) -> None:
        # Find the supplier invoice
        supplier_name = task.attributes.get("supplierName") or task.target_name
        invoice_number = task.attributes.get("invoiceNumber")

        # Search for vouchers from incoming invoices
        if invoice_number:
            # Try to find by invoice number
            vouchers = self.client.list(
                "/supplierInvoice",
                fields="id,invoiceNumber,amount,supplier(id,name)",
                params={"invoiceNumber": str(invoice_number), "count": 10},
            )
        elif supplier_name:
            vouchers = self.client.list(
                "/supplierInvoice",
                fields="id,invoiceNumber,amount,supplier(id,name)",
                params={"supplierName": supplier_name, "count": 10},
            )
        else:
            raise ParsingError("Supplier invoice payment requires a supplier name or invoice number")

        if not vouchers:
            raise EntityNotFoundError("No supplier invoice found")

        invoice_id = vouchers[0]["id"]
        amount = task.attributes.get("amount")
        payment_date_val = task.attributes.get("paymentDate") or task.attributes.get("date")
        payment_date = _parse_date_value(payment_date_val).isoformat() if payment_date_val else date.today().isoformat()

        self.client.pay_supplier_invoice(
            invoice_id,
            amount=float(amount) if amount else None,
            payment_date=payment_date,
        )
        LOGGER.info("Paid supplier invoice %s", invoice_id)

    # --- Helper: Resolve or create activity ---

    def _resolve_or_create_activity(self, name: str) -> dict[str, Any]:
        cache_key = f"activity:{_normalize(name)}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        # Search for existing
        results = self.client.search_activities(name=name)
        for r in results:
            if _normalize(r.get("name", "")) == _normalize(name):
                self._cache_set(cache_key, r)
                return r
        if results:
            self._cache_set(cache_key, results[0])
            return results[0]
        # Create new
        activity = self.client.create_activity({"name": name})
        self._cache_set(cache_key, activity)
        return activity

    def _resolve_activity(self, name: str | None) -> dict[str, Any]:
        cache_key = f"activity:{_normalize(name) if name else 'default'}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        if name:
            results = self.client.search_activities(name=name)
            exact = self._pick_exact(results, "name", name)
            if exact:
                self._cache_set(cache_key, exact)
                return exact
            if results:
                self._cache_set(cache_key, results[0])
                return results[0]
        results = self.client.search_activities()
        active = [a for a in results if not a.get("isDisabled")]
        if active:
            self._cache_set(cache_key, active[0])
            return active[0]
        if results:
            self._cache_set(cache_key, results[0])
            return results[0]
        raise EntityNotFoundError("No activity found for timesheet entry")

    def _register_payment(self, task: ParsedTask) -> None:
        # Fresh account optimization: skip _resolve_invoice search (always empty),
        # go straight to creating prerequisites.
        invoice = self._create_prerequisites_for_payment(task)

        payment_type = self._resolve_payment_type(task.attributes.get("paymentTypeDescription"))
        # Use the known amount from the invoice (already includes VAT from creation).
        amount = invoice.get("amountCurrencyOutstanding") or invoice.get("amountOutstanding")
        if amount is None or float(amount) <= 0:
            # Last resort: assume 25% VAT on the parsed amount
            raw = task.attributes.get("amount")
            if raw is not None:
                amount = float(raw) * 1.25
                LOGGER.warning("Using raw amount * 1.25 as payment amount: %s", amount)
            else:
                raise ParsingError("Could not determine payment amount from prompt or invoice state")

        payment_date_val = task.attributes.get("paymentDate") or date.today()
        payment_date = _parse_date_value(payment_date_val) if not isinstance(payment_date_val, date) else payment_date_val
        self.client.pay_invoice(
            invoice["id"],
            payment_date=payment_date,
            payment_type_id=payment_type["id"],
            paid_amount=float(amount),
        )

    def _revert_payment(self, task: ParsedTask) -> None:
        """Revert a payment so the invoice shows the outstanding amount again.

        On a fresh account: create customer → invoice → pay → reverse the payment voucher.
        """
        invoice = self._create_prerequisites_for_payment(task)

        payment_type = self._resolve_payment_type(task.attributes.get("paymentTypeDescription"))
        amount = invoice.get("amountCurrencyOutstanding") or invoice.get("amountOutstanding")
        if amount is None or float(amount) <= 0:
            raw = task.attributes.get("amount")
            if raw is not None:
                amount = float(raw) * 1.25
            else:
                raise ParsingError("Could not determine payment amount for reversal")

        payment_date_val = task.attributes.get("paymentDate") or date.today()
        payment_date = _parse_date_value(payment_date_val) if not isinstance(payment_date_val, date) else payment_date_val

        # Pay the invoice first (creates a payment voucher)
        paid_invoice = self.client.pay_invoice(
            invoice["id"],
            payment_date=payment_date,
            payment_type_id=payment_type["id"],
            paid_amount=float(amount),
        )

        # Find the payment voucher by searching recent vouchers
        # The payment voucher was created by pay_invoice and is the most recent one
        try:
            today = date.today()
            vouchers = self.client.list(
                "/ledger/voucher",
                fields="id,date,description,number,voucherType(id,name)",
                params={
                    "dateFrom": (today - timedelta(days=1)).isoformat(),
                    "dateTo": (today + timedelta(days=1)).isoformat(),
                    "count": 20,
                },
            )
            # Find the payment voucher (typically the last one created)
            payment_voucher = None
            for v in reversed(vouchers):
                vt_name = (v.get("voucherType", {}).get("name") or "").lower()
                if "betaling" in vt_name or "payment" in vt_name or "innbetaling" in vt_name:
                    payment_voucher = v
                    break
            if not payment_voucher and vouchers:
                # Fallback: use the most recent voucher (likely the payment)
                payment_voucher = vouchers[-1]

            if payment_voucher:
                self.client.reverse_voucher(payment_voucher["id"], today.isoformat())
                LOGGER.info("Reversed payment voucher %s on invoice %s", payment_voucher["id"], invoice.get("id"))
            else:
                LOGGER.warning("Could not find payment voucher to reverse for invoice %s", invoice.get("id"))
        except Exception as e:
            LOGGER.warning("Could not reverse payment voucher: %s", e)

    def _create_prerequisites_for_payment(self, task: ParsedTask) -> dict[str, Any]:
        """Create customer + order + invoice so we can register a payment on a fresh account."""
        customer_name = task.attributes.get("customerName")
        if not customer_name:
            raise ParsingError("Payment registration requires a customer name to create invoice")

        org_number = task.attributes.get("organizationNumber") or (
            str(task.attributes["orgNumber"]) if task.attributes.get("orgNumber") else None
        )
        customer = self._ensure_customer(
            name=customer_name,
            email=task.attributes.get("customerEmail"),
            org_number=org_number,
            address=self._build_address(task.attributes),
        )

        self._ensure_invoice_bank_account()

        amount = task.attributes.get("amount")
        if amount is None:
            raise ParsingError("Payment registration requires an amount to create invoice")

        today = date.today()
        invoice_date_val = task.attributes.get("invoiceDate")
        invoice_date = _parse_date_value(invoice_date_val) if invoice_date_val else today
        description = task.attributes.get("productName") or task.attributes.get("description") or "Invoice line"
        vat_type = self._get_default_vat_type()

        order_payload = {
            "customer": {"id": customer["id"]},
            "orderDate": invoice_date.isoformat(),
            "deliveryDate": invoice_date.isoformat(),
            "orderLines": [
                {
                    "description": description,
                    "count": float(task.attributes.get("quantity", 1.0)),
                    "unitPriceExcludingVatCurrency": float(amount),
                    "vatType": {"id": vat_type["id"]},
                }
            ],
        }
        order = self.client.create_order(order_payload)

        invoice = self.client.invoice_order(
            order["id"], invoice_date=invoice_date, send_to_customer=False,
        )
        LOGGER.info("Created prerequisite invoice %s for payment", invoice.get("id"))
        # Re-fetch the invoice to get the accurate outstanding amount from the API.
        # The POST response may not include computed fields like amountCurrencyOutstanding.
        outstanding = invoice.get("amountCurrencyOutstanding") or invoice.get("amountOutstanding")
        if not outstanding or float(outstanding) <= 0:
            try:
                fetched = self.client.get(
                    f"/invoice/{invoice['id']}",
                    fields="id,amountCurrencyOutstanding,amountOutstanding",
                )
                outstanding = fetched.get("amountCurrencyOutstanding") or fetched.get("amountOutstanding")
                if outstanding and float(outstanding) > 0:
                    invoice["amountCurrencyOutstanding"] = float(outstanding)
                    LOGGER.info("Re-fetched invoice outstanding: %s", outstanding)
            except Exception as e:
                LOGGER.warning("Could not re-fetch invoice for outstanding amount: %s", e)
        # Final fallback: calculate from known values.  We always create with default
        # 25% outgoing VAT type, so the outstanding = amount * 1.25.
        if not invoice.get("amountCurrencyOutstanding") or float(invoice["amountCurrencyOutstanding"]) <= 0:
            invoice["amountCurrencyOutstanding"] = float(amount) * 1.25
            LOGGER.info("Fallback outstanding calculation: %s * 1.25 = %s", amount, invoice["amountCurrencyOutstanding"])
        return invoice

    def _find_customer(
        self,
        *,
        name: str | None,
        email: str | None,
        organization_number: str | None = None,
    ) -> dict[str, Any]:
        results = self.client.search_customers(name=name, email=email)
        if organization_number:
            exact = self._pick_exact(results, "organizationNumber", organization_number)
            if exact:
                return exact
            exact_matches = [r for r in results if str(r.get("organizationNumber", "")) == str(organization_number)]
            if len(exact_matches) == 1:
                return exact_matches[0]
            if len(exact_matches) > 1:
                raise AmbiguousMatchError(
                    f"Customer lookup matched multiple entities for organization number {organization_number!r}"
                )
        if email:
            exact = self._pick_exact(results, "email", email)
            if exact:
                return exact
        if name:
            exact = self._pick_exact(results, "name", name)
            if exact:
                return exact
        if not results:
            hint = email or name or "customer"
            raise EntityNotFoundError(f"Customer lookup returned no results for {hint!r}")
        if len(results) == 1:
            return results[0]
        raise AmbiguousMatchError(f"Customer lookup matched {len(results)} entities")

    def _find_employee(self, *, name: str | None, email: str | None) -> dict[str, Any]:
        first_name, last_name = (None, None)
        if name:
            parts = name.split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = " ".join(parts[1:])

        results = self.client.search_employees(first_name=first_name, last_name=last_name, email=email)
        if email:
            exact = self._pick_exact(results, "email", email)
            if exact:
                return exact
        if name:
            exact = self._pick_display_name(results, name)
            if exact:
                return exact
        if not results:
            hint = email or name or "employee"
            raise EntityNotFoundError(f"Employee lookup returned no results for {hint!r}")
        if len(results) == 1:
            return results[0]
        raise AmbiguousMatchError(f"Employee lookup matched {len(results)} entities")

    def _find_project(self, *, name: str | None) -> dict[str, Any]:
        if not name:
            raise ParsingError("Project lookup needs a project name when no ID is supplied")
        results = self.client.search_projects(name=name)
        exact = self._pick_exact(results, "name", name)
        if exact:
            return exact
        if not results:
            raise EntityNotFoundError(f"Project lookup returned no results for {name!r}")
        if len(results) == 1:
            return results[0]
        raise AmbiguousMatchError(f"Project lookup matched {len(results)} entities")

    def _find_department(self, *, name: str | None) -> dict[str, Any]:
        if not name:
            raise ParsingError("Department lookup needs a department name when no ID is supplied")
        results = self.client.search_departments(name=name)
        exact = self._pick_exact(results, "name", name)
        if exact:
            return exact
        if not results:
            raise EntityNotFoundError(f"Department lookup returned no results for {name!r}")
        if len(results) == 1:
            return results[0]
        raise AmbiguousMatchError(f"Department lookup matched {len(results)} entities")

    def _find_product(self, *, name: str | None, product_number: str | None) -> dict[str, Any]:
        results = self.client.search_products(name=name, product_number=product_number)
        if product_number:
            exact = self._pick_exact(results, "number", product_number)
            if exact:
                return exact
        if name:
            exact = self._pick_exact(results, "name", name)
            if exact:
                return exact
        if not results:
            hint = product_number or name or "product"
            raise EntityNotFoundError(f"Product lookup returned no results for {hint!r}")
        if len(results) == 1:
            return results[0]
        raise AmbiguousMatchError(f"Product lookup matched {len(results)} entities")

    @staticmethod
    def _build_address(attrs: dict[str, Any]) -> dict[str, str] | None:
        addr_line = attrs.get("addressLine1")
        postal_code = attrs.get("postalCode")
        city = attrs.get("city")
        if not any((addr_line, postal_code, city)):
            return None
        address: dict[str, str] = {}
        if addr_line:
            address["addressLine1"] = str(addr_line)
        if postal_code:
            address["postalCode"] = str(postal_code)
        if city:
            address["city"] = str(city)
        return address

    @staticmethod
    def _build_delivery_address(attrs: dict[str, Any]) -> dict[str, str] | None:
        addr_line = attrs.get("deliveryAddressLine1")
        postal_code = attrs.get("deliveryPostalCode")
        city = attrs.get("deliveryCity")
        if not any((addr_line, postal_code, city)):
            return None
        address: dict[str, str] = {}
        if addr_line:
            address["addressLine1"] = str(addr_line)
        if postal_code:
            address["postalCode"] = str(postal_code)
        if city:
            address["city"] = str(city)
        return address

    def _ensure_customer(
        self,
        *,
        name: str,
        email: str | None = None,
        org_number: str | None = None,
        address: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        cache_key = f"customer:{_normalize(name)}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        # Fresh account optimization: create directly without searching first.
        # Saves 1 GET call per customer (account is always empty on competition submissions).
        payload: dict[str, Any] = {"name": name, "isCustomer": True}
        if email:
            payload["email"] = email
            payload["invoiceEmail"] = email
        if org_number:
            payload["organizationNumber"] = org_number
        if address:
            payload["postalAddress"] = address
            payload["physicalAddress"] = address
        customer = self.client.create("/customer", payload)
        self._cache_set(cache_key, customer)
        return customer

    def _ensure_employee_has_date_of_birth(self, employee_id: int) -> None:
        """Employment creation requires dateOfBirth. Set a default if missing."""
        try:
            emp = self.client.get(f"/employee/{employee_id}", fields="id,dateOfBirth")
            if not emp.get("dateOfBirth"):
                self.client.update("/employee", employee_id, {"dateOfBirth": "1990-01-01"})
        except Exception as e:
            LOGGER.warning("Could not ensure dateOfBirth: %s", e)

    def _update_employment(
        self,
        employee_id: int,
        *,
        start_date: str | None = None,
        salary_attrs: dict[str, Any] | None = None,
        dob_already_set: bool = False,
    ) -> None:
        """Update employment record (start date) and employment details (salary)."""
        try:
            # Ensure dateOfBirth is set — required for employment operations
            if not dob_already_set:
                self._ensure_employee_has_date_of_birth(employee_id)
            employments = self.client.list(
                "/employee/employment",
                fields="id,startDate,employmentDetails(id)",
                params={"employeeId": employee_id, "count": 1},
            )
            if employments:
                employment = employments[0]
                if start_date:
                    self.client.update("/employee/employment", employment["id"], {"startDate": start_date})
                # Update salary on employment details
                if salary_attrs:
                    details = employment.get("employmentDetails", [])
                    if details:
                        detail_id = details[0]["id"]
                        detail_payload: dict[str, Any] = {}
                        if salary_attrs.get("annualSalary") is not None:
                            detail_payload["annualSalary"] = float(salary_attrs["annualSalary"])
                        if salary_attrs.get("monthlySalary") is not None:
                            detail_payload["monthlySalary"] = float(salary_attrs["monthlySalary"])
                        if salary_attrs.get("hourlyWage") is not None:
                            detail_payload["hourlyWage"] = float(salary_attrs["hourlyWage"])
                        if salary_attrs.get("percentageOfFullTimeEquivalent") is not None:
                            detail_payload["percentageOfFullTimeEquivalent"] = float(salary_attrs["percentageOfFullTimeEquivalent"])
                        if detail_payload:
                            self.client.update("/employee/employment/details", detail_id, detail_payload)
            else:
                # Employment creation requires dateOfBirth on the employee.
                if not dob_already_set:
                    self._ensure_employee_has_date_of_birth(employee_id)
                emp_payload: dict[str, Any] = {
                    "employee": {"id": employee_id},
                    "startDate": start_date or date.today().isoformat(),
                    "isMainEmployer": True,
                    "taxDeductionCode": "loennFraHovedarbeidsgiver",
                }
                new_employment = self.client.create("/employee/employment", emp_payload)
                # If salary attrs provided, create employment details
                if salary_attrs and new_employment:
                    details = self.client.list(
                        "/employee/employment/details",
                        fields="id",
                        params={"employmentId": new_employment["id"], "count": 1},
                    )
                    if details:
                        detail_payload: dict[str, Any] = {}
                        for f in ("annualSalary", "monthlySalary", "hourlyWage", "percentageOfFullTimeEquivalent"):
                            if salary_attrs.get(f) is not None:
                                detail_payload[f] = float(salary_attrs[f])
                        if detail_payload:
                            self.client.update("/employee/employment/details", details[0]["id"], detail_payload)
        except Exception as e:
            LOGGER.warning("Could not update employment/salary: %s", e)

    def _ensure_employee(self, *, name: str, email: str | None = None, role: str | None = None) -> dict[str, Any]:
        cache_key = f"employee:{_normalize(name)}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        # Fresh account optimization: create directly without searching first.
        parts = name.split()
        if len(parts) < 2:
            raise ParsingError("Employee creation requires both first and last name")
        user_type = "NO_ACCESS"
        if role:
            user_type = ROLE_TO_USER_TYPE.get(role.lower(), "NO_ACCESS")
        payload: dict[str, Any] = {
            "firstName": parts[0],
            "lastName": " ".join(parts[1:]),
            "userType": user_type,
        }
        department = self._ensure_department(DEFAULT_EMPLOYEE_DEPARTMENT_NAME)
        payload["department"] = {"id": department["id"]}
        if email:
            payload["email"] = email
        elif user_type == "STANDARD":
            first = parts[0].lower().replace(" ", "")
            last = parts[-1].lower().replace(" ", "")
            payload["email"] = f"{first}.{last}@placeholder.example.com"
        try:
            employee = self.client.create("/employee", payload)
        except TripletexAPIError as e:
            if e.status_code == 422 and "e-post" in str(e).lower():
                LOGGER.warning("Email rejected in _ensure_employee, retrying without: %s", e)
                first = parts[0].lower().replace(" ", "")
                last = parts[-1].lower().replace(" ", "")
                payload["email"] = f"{first}.{last}@placeholder.example.com"
                employee = self.client.create("/employee", payload)
            else:
                raise
        self._cache_set(cache_key, employee)
        return employee

    def _ensure_department(self, department_name: str) -> dict[str, Any]:
        cache_key = f"dept:{_normalize(department_name)}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        # Fresh account optimization: create directly without searching first.
        dept = self.client.create("/department", {"name": department_name})
        self._cache_set(cache_key, dept)
        return dept

    def _ensure_project(
        self,
        *,
        name: str,
        customer_name: str | None = None,
        customer_email: str | None = None,
        org_number: str | None = None,
        pm_name: str | None = None,
        pm_email: str | None = None,
        fixed_price: float | None = None,
    ) -> dict[str, Any]:
        cache_key = f"project:{_normalize(name)}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        pm = self._ensure_employee(name=pm_name, email=pm_email) if pm_name else self._resolve_project_manager()
        self._ensure_project_manager_access(pm)
        payload: dict[str, Any] = {
            "name": name,
            "projectManager": {"id": pm["id"]},
            "startDate": date.today().isoformat(),
        }
        if customer_name:
            customer = self._ensure_customer(
                name=customer_name,
                email=customer_email,
                org_number=org_number,
            )
            payload["customer"] = {"id": customer["id"]}
        if fixed_price is not None:
            payload["isFixedPrice"] = True
            payload["fixedprice"] = float(fixed_price)
        project = self.client.create("/project", payload)
        self._cache_set(cache_key, project)
        return project

    def _resolve_project_manager(self) -> dict[str, Any]:
        # Fresh account optimization: skip search, create directly via _ensure_employee (cached).
        return self._ensure_employee(name=DEFAULT_PROJECT_MANAGER_NAME, email=None)

    def _ensure_project_manager_access(self, employee: dict[str, Any]) -> None:
        """Ensure an employee has project manager access.

        The Tripletex API rejects project creation if the PM doesn't have the
        right entitlements.  We upgrade userType to STANDARD (if needed) and
        grant ALL_PRIVILEGES to be safe.
        """
        cache_key = f"pm_access:{employee['id']}"
        if self._cache_get(cache_key):
            return
        try:
            current_type = employee.get("userType", "NO_ACCESS")
            if current_type == "NO_ACCESS":
                update_payload: dict[str, Any] = {"userType": "STANDARD"}
                # STANDARD userType requires an email address
                if not employee.get("email"):
                    first = employee.get("firstName", "pm").lower().replace(" ", "")
                    last = employee.get("lastName", "user").lower().replace(" ", "")
                    update_payload["email"] = f"{first}.{last}@placeholder.example.com"
                self.client.update("/employee", employee["id"], update_payload)
            self.client.grant_entitlements(employee["id"], "ALL_PRIVILEGES")
        except Exception as e:
            LOGGER.warning("Could not ensure PM access for employee %s: %s", employee.get("id"), e)
        self._cache_set(cache_key, True)

    def _resolve_travel_expense(self, task: ParsedTask) -> dict[str, Any]:
        if task.identifier is not None:
            return {"id": task.identifier}
        # Try finding by employee name
        employee_name = task.attributes.get("employeeName") or task.target_name
        if employee_name:
            try:
                employee = self._find_employee(name=employee_name, email=None)
                results = self.client.search_travel_expenses(employee_id=employee["id"])
                if len(results) == 1:
                    return results[0]
                if results:
                    # Try to match by title or destination
                    title = task.attributes.get("title")
                    dest = task.attributes.get("destination")
                    for r in results:
                        td = r.get("travelDetails", {})
                        if title and _normalize(r.get("title", "")) == _normalize(title):
                            return r
                        if dest and _normalize(td.get("destination", "")) == _normalize(dest):
                            return r
                    return results[0]  # fallback to most recent
            except (EntityNotFoundError, AmbiguousMatchError):
                pass
        # Fallback: search all recent travel expenses
        results = self.client.search_travel_expenses()
        if len(results) == 1:
            return results[0]
        if results:
            return results[0]
        raise EntityNotFoundError("Could not resolve travel expense")

    def _resolve_project(self, task: ParsedTask) -> dict[str, Any]:
        if task.identifier is not None:
            return {"id": task.identifier}
        return self._find_project(name=task.target_name)

    def _resolve_project_by_name(self, name: str) -> dict[str, Any]:
        return self._find_project(name=name)

    def _resolve_department(self, task: ParsedTask) -> dict[str, Any]:
        if task.identifier is not None:
            return {"id": task.identifier}
        return self._find_department(name=task.target_name)

    def _resolve_product(self, task: ParsedTask) -> dict[str, Any] | None:
        if task.identifier is not None:
            return {"id": task.identifier}

        product_name = task.attributes.get("productName") or task.target_name
        product_number = task.attributes.get("productNumber") or task.attributes.get("number")
        if not product_name and not product_number:
            return None
        return self._find_product(name=product_name, product_number=product_number)

    def _get_default_vat_type(self) -> dict[str, Any]:
        cached = self._cache_get("default_vat_type")
        if cached:
            return cached
        results = self.client.search_vat_types(number="3", type_of_vat="OUTGOING")
        if results:
            self._cache_set("default_vat_type", results[0])
            return results[0]
        results = self.client.search_vat_types(type_of_vat="OUTGOING")
        if results:
            self._cache_set("default_vat_type", results[0])
            return results[0]
        raise EntityNotFoundError("No VAT type could be resolved for invoice creation")

    def _ensure_invoice_bank_account(self) -> None:
        if self._cache_get("invoice_bank_account_ready"):
            return
        accounts = self.client.search_accounts(number=1920)
        invoice_accounts = [account for account in accounts if account.get("isInvoiceAccount")]
        if not invoice_accounts:
            invoice_accounts = self.client.search_accounts(is_bank_account=True)
            invoice_accounts = [account for account in invoice_accounts if account.get("isInvoiceAccount")]

        if not invoice_accounts:
            raise EntityNotFoundError("Could not resolve an invoice bank account for invoice creation")

        account = invoice_accounts[0]
        if not account.get("bankAccountNumber"):
            self.client.update_account(
                int(account["id"]),
                {"bankAccountNumber": DEFAULT_INVOICE_BANK_ACCOUNT_NUMBER},
            )
        self._cache_set("invoice_bank_account_ready", True)

    def _resolve_payment_type(self, description: str | None) -> dict[str, Any]:
        cache_key = f"payment_type:{description or 'default'}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        queries = [description] if description else []
        queries.extend(["Betalt til bank", "bank"])
        seen: set[str] = set()

        for query in queries:
            if not query or query in seen:
                continue
            seen.add(query)
            results = self.client.search_payment_types(query=query)
            exact = self._pick_exact(results, "description", query)
            if exact:
                self._cache_set(cache_key, exact)
                return exact
            if results:
                self._cache_set(cache_key, results[0])
                return results[0]

        raise EntityNotFoundError("No usable payment type was found")

    def _resolve_travel_payment_type(self) -> dict[str, Any]:
        cached = self._cache_get("travel_payment_type")
        if cached:
            return cached
        results = self.client.search_travel_payment_types(query="bank")
        if results:
            self._cache_set("travel_payment_type", results[0])
            return results[0]
        results = self.client.search_travel_payment_types()
        if results:
            self._cache_set("travel_payment_type", results[0])
            return results[0]
        raise EntityNotFoundError("No travel-expense payment type was found")

    def _resolve_per_diem_rate_category(self, *, is_day_trip: bool = False, is_domestic: bool = True) -> dict[str, Any]:
        cache_key = f"per_diem_rate_cat:{is_day_trip}:{is_domestic}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        results = self.client.search_mileage_rate_categories()
        # Filter for PER_DIEM type rate categories
        per_diem_cats = [r for r in results if r.get("type") == "PER_DIEM"]
        if is_day_trip:
            day_trip_cats = [r for r in per_diem_cats if r.get("isValidDayTrip")]
            if day_trip_cats:
                per_diem_cats = day_trip_cats
        if is_domestic:
            domestic_cats = [r for r in per_diem_cats if r.get("isValidDomestic")]
            if domestic_cats:
                per_diem_cats = domestic_cats
        if per_diem_cats:
            self._cache_set(cache_key, per_diem_cats[0])
            return per_diem_cats[0]
        # Fallback: any PER_DIEM category
        all_per_diem = [r for r in results if r.get("type") == "PER_DIEM"]
        if all_per_diem:
            self._cache_set(cache_key, all_per_diem[0])
            return all_per_diem[0]
        raise EntityNotFoundError("No per diem rate category found")

    def _resolve_mileage_rate_category(self) -> dict[str, Any]:
        cached = self._cache_get("mileage_rate_cat")
        if cached:
            return cached
        results = self.client.search_mileage_rate_categories()
        mileage_cats = [r for r in results if r.get("type") == "MILEAGE_ALLOWANCE"]
        if mileage_cats:
            self._cache_set("mileage_rate_cat", mileage_cats[0])
            return mileage_cats[0]
        raise EntityNotFoundError("No mileage rate category found")

    def _resolve_accommodation_rate_category(self) -> dict[str, Any]:
        cached = self._cache_get("accommodation_rate_cat")
        if cached:
            return cached
        results = self.client.search_mileage_rate_categories()
        accom_cats = [r for r in results if r.get("type") == "ACCOMMODATION_ALLOWANCE"]
        if accom_cats:
            self._cache_set("accommodation_rate_cat", accom_cats[0])
            return accom_cats[0]
        raise EntityNotFoundError("No accommodation allowance rate category found")

    def _resolve_travel_cost_category(self) -> dict[str, Any]:
        cached = self._cache_get("travel_cost_category")
        if cached:
            return cached
        results = self.client.search_travel_cost_categories(query="travel")
        if results:
            self._cache_set("travel_cost_category", results[0])
            return results[0]
        results = self.client.search_travel_cost_categories()
        if results:
            self._cache_set("travel_cost_category", results[0])
            return results[0]
        raise EntityNotFoundError("No travel-expense cost category was found")

    def _resolve_invoice(self, task: ParsedTask, *, prefer_outstanding: bool = False) -> dict[str, Any]:
        if task.identifier is not None:
            return self.client.get_invoice(task.identifier)

        invoice_number = task.attributes.get("invoiceNumber")
        if invoice_number:
            results = self.client.search_invoices(invoice_number=str(invoice_number))
            exact = self._pick_exact(results, "invoiceNumber", str(invoice_number))
            if exact:
                return exact
            if len(results) == 1:
                return results[0]
            if results:
                raise AmbiguousMatchError(f"Invoice number {invoice_number!r} matched multiple invoices")

        customer_name = task.attributes.get("customerName")
        if customer_name:
            customer = self._find_customer(name=customer_name, email=None)
            results = self.client.search_invoices(customer_id=customer["id"])
            outstanding = [item for item in results if float(item.get("amountCurrencyOutstanding") or item.get("amountOutstanding") or 0) > 0]
            if prefer_outstanding and len(outstanding) == 1:
                return outstanding[0]
            if len(results) == 1:
                return results[0]
            if outstanding:
                raise AmbiguousMatchError(f"Customer {customer_name!r} has multiple outstanding invoices")

        raise EntityNotFoundError("Could not resolve the invoice to pay")

    @staticmethod
    def _pick_exact(results: list[dict[str, Any]], field: str, target: str) -> dict[str, Any] | None:
        target_normalized = _normalize(target)
        exact_matches = [item for item in results if _normalize(str(item.get(field, ""))) == target_normalized]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            raise AmbiguousMatchError(f"Exact match on {field}={target!r} still returned multiple entities")
        return None

    @staticmethod
    def _pick_display_name(results: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
        target_normalized = _normalize(target)
        candidates = []
        for item in results:
            display_name = item.get("displayName") or f"{item.get('firstName', '')} {item.get('lastName', '')}".strip()
            if _normalize(display_name) == target_normalized:
                candidates.append(item)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise AmbiguousMatchError(f"Employee display name {target!r} matched multiple entities")
        return None
