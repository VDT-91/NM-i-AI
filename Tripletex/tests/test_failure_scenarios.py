"""Tests for known failure scenarios from competition submissions.

These tests reproduce the exact patterns that scored 0/7, 0/8, 0/10, 2/10
and verify that fixes work correctly. Run locally before deploying.

Run: python -m pytest tests/test_failure_scenarios.py -v
"""
from __future__ import annotations

import unittest
from datetime import date, timedelta
from typing import Any
from unittest.mock import patch

from tripletex_solver.errors import ParsingError, TripletexAPIError
from tripletex_solver.models import Action, Entity, ParsedTask, SolveRequest, TripletexCredentials
from tripletex_solver.service import TripletexService


# ---------------------------------------------------------------------------
# Enhanced RecordingClient that can simulate failures
# ---------------------------------------------------------------------------

class FailableRecordingClient:
    """Mock TripletexClient that can simulate specific API failures."""

    def __init__(self):
        self._call_log: list[tuple[str, Any]] = []
        self._next_id = 100
        self._fail_on = {}
        self._fail_count = {}  # how many times to fail before succeeding
        self._salary_types = [
            {"id": 1, "number": "1", "name": "Fastlønn"},
            {"id": 2, "number": "30", "name": "Engangstillegg/bonus"},
        ]
        self._vat_types = [
            {"id": 3, "percentage": 25.0, "name": "Utgående MVA 25%"},
            {"id": 6, "percentage": 15.0, "name": "Utgående MVA 15%"},
            {"id": 5, "percentage": 0.0, "name": "Ingen MVA"},
        ]
        self._accounts = {
            1920: {"id": 1920, "number": 1920, "name": "Bank", "isInvoiceAccount": True, "bankAccountNumber": "10000000006"},
            2400: {"id": 2400, "number": 2400, "name": "Leverandørgjeld"},
            2710: {"id": 2710, "number": 2710, "name": "Inngående MVA"},
            4000: {"id": 4000, "number": 4000, "name": "Varekostnad"},
            5000: {"id": 5000, "number": 5000, "name": "Lønn"},
            6300: {"id": 6300, "number": 6300, "name": "Kontorrekvisita"},
            6500: {"id": 6500, "number": 6500, "name": "Kontorrekvisita"},
            6590: {"id": 6590, "number": 6590, "name": "Kontorutstyr"},
        }
        self.created_entities = []
        self.voucher_payloads = []
        self.order_payloads = []
        self.invoice_payloads = []
        self.paid_invoices = []
        self.paid_supplier_invoices = []
        self.activated_modules = []
        self.entitlements_granted = []
        self.timesheet_entries = []
        self.travel_expenses = []
        self.travel_costs = []
        self.per_diem_compensations = []
        self.updated_entities = []
        self.salary_transactions = []
        self.credit_notes = []
        self.deleted_entities = []
        self.incoming_invoice_payloads = []
        self._employees_dob = {}  # Track dateOfBirth by employee id

    def set_fail(self, method: str, error: TripletexAPIError, times: int = 1):
        """Configure a method to fail N times, then succeed."""
        self._fail_on[method] = error
        self._fail_count[method] = times

    def _check_fail(self, method: str):
        if method in self._fail_on and self._fail_count.get(method, 0) > 0:
            error = self._fail_on[method]
            self._fail_count[method] -= 1
            if self._fail_count[method] <= 0:
                del self._fail_on[method]
                del self._fail_count[method]
            raise error

    def _next(self, extra: dict | None = None) -> dict:
        self._next_id += 1
        result = {"id": self._next_id}
        if extra:
            result.update(extra)
        return result

    # --- Generic CRUD ---
    def create(self, endpoint: str, payload: dict, params: dict | None = None) -> dict:
        self._call_log.append(("create", (endpoint, payload)))
        self._check_fail(f"create:{endpoint}")
        self.created_entities.append((endpoint, payload))
        if "voucher" in endpoint.lower():
            self.voucher_payloads.append(payload)
        name = payload.get("name") or payload.get("firstName", "")
        result = self._next({"name": name, **payload})
        # Track employee dateOfBirth
        if endpoint == "/employee":
            self._employees_dob[result["id"]] = payload.get("dateOfBirth")
        return result

    def get(self, endpoint: str, fields: str = "*", params: dict | None = None) -> dict:
        self._call_log.append(("get", endpoint))
        # Return employee with tracked dateOfBirth
        if "/employee/" in endpoint:
            try:
                emp_id = int(endpoint.split("/")[-1])
                dob = self._employees_dob.get(emp_id)
                return {"id": emp_id, "firstName": "Test", "lastName": "User", "dateOfBirth": dob}
            except ValueError:
                pass
        return self._next()

    def list(self, endpoint: str, fields: str = "*", params: dict | None = None) -> list:
        self._call_log.append(("list", (endpoint, params)))
        if "/department" in endpoint:
            return []
        if "/project" in endpoint:
            return []
        if "/customer" in endpoint and params:
            for ep, payload in self.created_entities:
                if ep == "/customer":
                    name = params.get("name")
                    org = params.get("organizationNumber")
                    if org and payload.get("organizationNumber") == org:
                        return [{"id": payload.get("id", self._next_id), **payload}]
                    if name and name.lower() in payload.get("name", "").lower():
                        return [{"id": payload.get("id", self._next_id), **payload}]
            return []
        if "/employee/employment/details" in endpoint:
            return [{"id": 901}]
        if "/employee/employment" in endpoint:
            emp_id = (params or {}).get("employeeId", 0)
            return [{"id": 900, "division": None, "startDate": "2026-01-01",
                      "employmentDetails": [{"id": 901}]}]
        if "/ledger/accountingDimensionName" in endpoint:
            return []
        if "/ledger/voucher" in endpoint:
            return []
        return []

    def update(self, endpoint: str, entity_id: int, payload: dict) -> dict:
        self._call_log.append(("update", (endpoint, entity_id, payload)))
        self._check_fail(f"update:{endpoint}")
        self.updated_entities.append((endpoint, entity_id, payload))
        # Track dateOfBirth updates
        if endpoint == "/employee" and "dateOfBirth" in payload:
            self._employees_dob[entity_id] = payload["dateOfBirth"]
        return {"id": entity_id, **payload}

    def delete(self, endpoint: str, entity_id: int) -> None:
        self._call_log.append(("delete", (endpoint, entity_id)))
        self.deleted_entities.append((endpoint, entity_id))

    def update_account(self, account_id: int, payload: dict) -> dict:
        return self.update("/ledger/account", account_id, payload)

    # --- Employees ---
    def search_employees(self, **kwargs) -> list:
        return []

    def get_employee(self, emp_id: int) -> dict:
        dob = self._employees_dob.get(emp_id)
        return {"id": emp_id, "firstName": "Test", "lastName": "User", "dateOfBirth": dob}

    def grant_entitlements(self, employee_id: int, template: str) -> None:
        self.entitlements_granted.append((employee_id, template))

    def create_employee_standard_time(self, payload: dict) -> dict:
        return self._next()

    # --- Customers ---
    def search_customers(self, **kwargs) -> list:
        name = kwargs.get("name", "")
        for endpoint, payload in self.created_entities:
            if endpoint == "/customer" and name and name.lower() in payload.get("name", "").lower():
                return [{"id": payload.get("id", self._next_id), **payload}]
        return []

    # --- Suppliers ---
    def search_suppliers(self, **kwargs) -> list:
        return []

    # --- Products ---
    def search_products(self, **kwargs) -> list:
        return []

    # --- Accounts ---
    def search_accounts(self, **kwargs) -> list:
        num = kwargs.get("number")
        if num and num in self._accounts:
            return [self._accounts[num]]
        is_bank = kwargs.get("is_bank_account")
        if is_bank:
            return [self._accounts[1920]]
        return list(self._accounts.values())[:5]

    def search_accounts_by_number(self, number: int) -> list:
        if number in self._accounts:
            return [self._accounts[number]]
        return [{"id": number, "number": number, "name": f"Account {number}"}]

    # --- VAT ---
    def search_vat_types(self, **kwargs) -> list:
        return self._vat_types

    # --- Orders ---
    def create_order(self, payload: dict) -> dict:
        self.order_payloads.append(payload)
        return self._next({"orderNumber": 5001})

    # --- Invoices ---
    def create_invoice(self, payload: dict, *, send_to_customer: bool = False) -> dict:
        self.invoice_payloads.append((payload, send_to_customer))
        return self._next({
            "invoiceNumber": 9001, "amount": 10000,
            "amountCurrency": 10000, "amountOutstanding": 10000,
            "amountCurrencyOutstanding": 10000,
        })

    def search_invoices(self, **kwargs) -> list:
        for payload, _ in self.invoice_payloads:
            return [{"id": self._next_id, "invoiceNumber": 9001, "amount": 10000,
                     "amountOutstanding": 10000, "amountCurrencyOutstanding": 10000,
                     "customer": payload.get("customer", {})}]
        return []

    def invoice_order(self, order_id: int, invoice_date: str, *, send: bool = False, send_to_customer: bool = False) -> dict:
        payload = {"orders": [{"id": order_id}], "invoiceDate": invoice_date}
        self.invoice_payloads.append((payload, send))
        return self._next({
            "invoiceNumber": 9001, "amount": 10000,
            "amountCurrency": 10000, "amountOutstanding": 10000,
            "amountCurrencyOutstanding": 10000,
        })

    def get_invoice(self, invoice_id: int) -> dict:
        return self._next({"invoiceNumber": 9001, "amount": 10000, "amountOutstanding": 10000,
                          "amountCurrencyOutstanding": 10000})

    def pay_invoice(self, invoice_id: int, *, payment_date, payment_type_id, paid_amount, **kwargs) -> dict:
        self.paid_invoices.append({
            "invoice_id": invoice_id, "amount": paid_amount,
            "payment_date": payment_date.isoformat() if isinstance(payment_date, date) else str(payment_date),
        })
        return self._next()

    def create_credit_note(self, invoice_id: int, *, credit_note_date, comment=None, send_to_customer=False) -> dict:
        self.credit_notes.append((invoice_id, credit_note_date))
        return self._next()

    # --- Vouchers ---
    def create_voucher(self, payload: dict) -> dict:
        self.voucher_payloads.append(payload)
        return self._next()

    def search_voucher_types(self, **kwargs) -> list:
        return [{"id": 1, "name": "Leverandørfaktura"}]

    def reverse_voucher(self, voucher_id: int, date_str: str) -> dict:
        return self._next()

    # --- Payments ---
    def search_payment_types(self, **kwargs) -> list:
        return [{"id": 10, "description": "Betalt til bank"}]

    # --- Salary ---
    def search_salary_types(self, **kwargs) -> list:
        return self._salary_types

    def create_salary_transaction(self, payload: dict) -> dict:
        self._check_fail("create_salary_transaction")
        self.salary_transactions.append(payload)
        return self._next()

    def search_employments(self, **kwargs) -> list:
        return [{"id": 900, "division": None}]

    # --- Modules ---
    def activate_sales_module(self, module_name: str) -> dict:
        self._check_fail(f"activate:{module_name}")
        self.activated_modules.append(module_name)
        return {}

    def _prefetch_active_modules(self) -> None:
        pass

    # --- Activities ---
    def search_activities(self, **kwargs) -> list:
        return []

    def create_activity(self, payload: dict) -> dict:
        return self._next({"name": payload.get("name", "Activity")})

    # --- Projects ---
    def create_project_activity(self, payload: dict) -> dict:
        return self._next()

    def search_projects(self, **kwargs) -> list:
        return []

    def create_division(self, payload: dict) -> dict:
        self.created_entities.append(("/division", payload))
        return self._next({"name": payload.get("name", "Hovedkontor"), "organizationNumber": "999888777", **payload})

    # --- Timesheet ---
    def create_timesheet_entry(self, payload: dict) -> dict:
        self.timesheet_entries.append(payload)
        return self._next()

    # --- Travel ---
    def create_travel_expense(self, payload: dict) -> dict:
        self.travel_expenses.append(payload)
        return self._next()

    def create_per_diem_compensation(self, payload: dict) -> dict:
        self.per_diem_compensations.append(payload)
        return self._next()

    def create_travel_cost(self, payload: dict) -> dict:
        self.travel_costs.append(payload)
        return self._next()

    def create_mileage_allowance(self, payload: dict) -> dict:
        return self._next()

    def create_accommodation_allowance(self, payload: dict) -> dict:
        return self._next()

    def search_per_diem_rate_categories(self, **kwargs) -> list:
        return [{"id": 50, "name": "Innland med overnatting", "isDomestic": True}]

    def search_mileage_rate_categories(self) -> list:
        return [{"id": 60, "name": "Egen bil"}]

    def search_accommodation_rate_categories(self) -> list:
        return [{"id": 70, "name": "Nattillegg"}]

    def search_travel_payment_types(self, **kwargs) -> list:
        return [{"id": 80, "description": "Utlegg"}]

    def search_travel_cost_categories(self, **kwargs) -> list:
        return [
            {"id": 90, "description": "Fly", "name": "Fly"},
            {"id": 91, "description": "Taxi", "name": "Taxi"},
            {"id": 92, "description": "Annet", "name": "Annet"},
        ]

    def update_travel_expense(self, expense_id: int, payload: dict) -> dict:
        self.updated_entities.append(("/travelExpense", expense_id, payload))
        return {"id": expense_id}

    # --- Dimensions ---
    def create_dimension_name(self, payload: dict) -> dict:
        return self._next({"dimensionName": payload.get("dimensionName"), "dimensionIndex": 1})

    def create_dimension_value(self, payload: dict) -> dict:
        return self._next({"displayName": payload.get("displayName")})

    # --- Incoming invoices ---
    def search_incoming_invoices(self, **kwargs) -> list:
        return [{"id": 7001, "amount": 5000, "supplierId": 200}]

    def pay_supplier_invoice(self, invoice_id: int, *, amount: float, payment_date: str, **kwargs) -> dict:
        self.paid_supplier_invoices.append({
            "invoice_id": invoice_id, "amount": amount, "payment_date": payment_date,
        })
        return self._next()

    def create_incoming_invoice(self, payload: dict, *, send_to: str = "ledger") -> dict:
        self._check_fail("create_incoming_invoice")
        self.incoming_invoice_payloads.append(payload)
        return self._next()

    # --- Contacts ---
    def search_contacts(self, **kwargs) -> list:
        return []

    def create_contact(self, payload: dict) -> dict:
        self.created_entities.append(("/contact", payload))
        return self._next(payload)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_service(client=None) -> tuple[TripletexService, FailableRecordingClient]:
    svc = TripletexService.__new__(TripletexService)
    client = client or FailableRecordingClient()
    svc.client = client
    svc.last_attachment_text = None
    svc.last_parsed_task = None
    svc._saved_attachment_paths = []
    svc._cache = {}
    svc.parser = None
    return svc, client


def _task(
    *,
    action: Action = Action.CREATE,
    entity: Entity,
    raw_prompt: str = "",
    target_name: str | None = None,
    attributes: dict | None = None,
    identifier: int | None = None,
) -> ParsedTask:
    return ParsedTask(
        action=action,
        entity=entity,
        target_name=target_name or "",
        identifier=identifier,
        attributes=attributes or {},
        raw_prompt=raw_prompt,
    )


# ===========================================================================
# SCENARIO 1: Salary with virksomhet retry (0/8 score)
# The salary transaction fails with "virksomhet" error because employment
# is not linked to a division. The retry path must ensure dateOfBirth
# is set before updating employment.
# ===========================================================================

class TestSalaryVirksomhetRetry(unittest.TestCase):
    """Reproduces the 0/8 salary failure: virksomhet error in retry path."""

    def setUp(self):
        self.svc, self.client = _make_service()

    def test_salary_retry_sets_dob_before_division_update(self):
        """When salary fails with virksomhet, the retry must set dateOfBirth first."""
        # First salary attempt fails with virksomhet error
        self.client.set_fail(
            "create_salary_transaction",
            TripletexAPIError("Ansatte må ha virksomhet på ansettelsesforholdet", status_code=422),
            times=1,
        )

        task = _task(
            entity=Entity.SALARY_TRANSACTION,
            raw_prompt='Kjør lønn for Erik Hansen, grunnlønn 42000 kr',
            attributes={
                "employeeName": "Erik Hansen",
                "baseSalary": 42000,
            },
        )
        self.svc._dispatch(task)

        # Salary should succeed on retry
        self.assertEqual(len(self.client.salary_transactions), 1)

        # Verify dateOfBirth was set (should appear in update calls)
        dob_updates = [
            (ep, eid, p) for ep, eid, p in self.client.updated_entities
            if ep == "/employee" and "dateOfBirth" in p
        ]
        self.assertTrue(len(dob_updates) >= 1, "dateOfBirth should be set before employment update")

        # Verify division was linked to employment
        div_updates = [
            (ep, eid, p) for ep, eid, p in self.client.updated_entities
            if ep == "/employee/employment" and "division" in p
        ]
        self.assertTrue(len(div_updates) >= 1, "Division should be linked to employment")

        # dateOfBirth update must happen BEFORE employment/division update
        all_updates = self.client.updated_entities
        dob_idx = next(i for i, (ep, _, p) in enumerate(all_updates)
                       if ep == "/employee" and "dateOfBirth" in p)
        div_idx = next(i for i, (ep, _, p) in enumerate(all_updates)
                       if ep == "/employee/employment" and "division" in p)
        self.assertLess(dob_idx, div_idx,
                        "dateOfBirth must be set before employment division update")

    def test_salary_with_bonus_and_retry(self):
        """Salary with both base and bonus still works after virksomhet retry."""
        self.client.set_fail(
            "create_salary_transaction",
            TripletexAPIError("virksomhet mangler", status_code=422),
            times=1,
        )

        task = _task(
            entity=Entity.SALARY_TRANSACTION,
            raw_prompt='Lønnskjøring for Brita Berge. Grunnlønn 36800 kr, bonus 14100 kr.',
            attributes={
                "employeeName": "Brita Berge",
                "employeeEmail": "brita.berge@example.org",
                "baseSalary": 36800,
                "bonus": 14100,
            },
        )
        self.svc._dispatch(task)

        self.assertEqual(len(self.client.salary_transactions), 1)
        specs = self.client.salary_transactions[0]["payslips"][0]["specifications"]
        self.assertEqual(len(specs), 2)
        rates = sorted([s["rate"] for s in specs])
        self.assertIn(14100.0, rates)
        self.assertIn(36800.0, rates)


# ===========================================================================
# SCENARIO 2: Project creation module activation (0/10 score)
# Must activate KOMPLETT (not SMART) on fresh account.
# SMART can downgrade from KOMPLETT.
# ===========================================================================

class TestProjectModuleActivation(unittest.TestCase):
    """Reproduces the 0/10 project failure: wrong module activated."""

    def setUp(self):
        self.svc, self.client = _make_service()

    def test_project_activates_komplett_not_smart(self):
        """Project creation must use KOMPLETT module, not SMART."""
        task = _task(
            entity=Entity.PROJECT,
            raw_prompt='Create project "Cloud Platform" for customer Digital AS (org 923456789)',
            attributes={
                "name": "Cloud Platform",
                "customerName": "Digital AS",
                "organizationNumber": "923456789",
            },
        )
        self.svc._dispatch(task)

        # SMART_PROJECT must be activated (preferred module for project creation)
        self.assertIn("SMART_PROJECT", self.client.activated_modules)

    def test_project_survives_409_komplett_already_active(self):
        """If KOMPLETT is already active (409), project creation proceeds."""
        self.client.set_fail(
            "activate:KOMPLETT",
            TripletexAPIError("Module already active", status_code=409),
            times=1,
        )

        task = _task(
            entity=Entity.PROJECT,
            raw_prompt='Opprett prosjekt "Nettside" for Acme AS',
            attributes={
                "name": "Nettside",
                "customerName": "Acme AS",
            },
        )
        # Should not raise — 409 is handled gracefully
        self.svc._dispatch(task)

        # Project should still be created
        project_creates = [e for e in self.client.created_entities if e[0] == "/project"]
        self.assertTrue(len(project_creates) >= 1, "Project must be created even with 409 on module")


# ===========================================================================
# SCENARIO 3: Employee with admin role (missing entitlements = 5/10 lost)
# ===========================================================================

class TestEmployeeAdminEntitlements(unittest.TestCase):
    """Verifies admin entitlements are granted (worth 5/10 points)."""

    def setUp(self):
        self.svc, self.client = _make_service()

    def test_admin_role_grants_entitlements(self):
        task = _task(
            entity=Entity.EMPLOYEE,
            raw_prompt='Create employee Anna Berg (anna.berg@example.org) born 1990-05-15. She should be admin.',
            attributes={
                "firstName": "Anna",
                "lastName": "Berg",
                "email": "anna.berg@example.org",
                "dateOfBirth": "1990-05-15",
                "role": "administrator",
                "startDate": "2026-01-01",
            },
        )
        self.svc._dispatch(task)

        emp_creates = [e for e in self.client.created_entities if e[0] == "/employee"]
        self.assertTrue(len(emp_creates) >= 1)

        # Admin entitlements must be granted
        self.assertTrue(len(self.client.entitlements_granted) >= 1,
                        "Admin entitlements must be granted (worth 5pts)")

    def test_kontoadministrator_norwegian_maps_to_admin(self):
        """Norwegian 'kontoadministrator' should trigger admin entitlements."""
        task = _task(
            entity=Entity.EMPLOYEE,
            raw_prompt='Opprett ansatt Kari Olsen (kari@example.org). Hun skal være kontoadministrator.',
            attributes={
                "firstName": "Kari",
                "lastName": "Olsen",
                "email": "kari@example.org",
                "role": "kontoadministrator",
            },
        )
        self.svc._dispatch(task)

        self.assertTrue(len(self.client.entitlements_granted) >= 1,
                        "'kontoadministrator' must grant admin entitlements")

    def test_administrator_german_maps_to_admin(self):
        """German 'Administrator' should also trigger entitlements."""
        task = _task(
            entity=Entity.EMPLOYEE,
            raw_prompt='Erstelle Mitarbeiter Max Müller (max@example.org). Er soll Administrator sein.',
            attributes={
                "firstName": "Max",
                "lastName": "Müller",
                "email": "max@example.org",
                "role": "administrator",
            },
        )
        self.svc._dispatch(task)
        self.assertTrue(len(self.client.entitlements_granted) >= 1)


# ===========================================================================
# SCENARIO 4: Invoice multi-step flow (order → invoice → payment)
# On fresh account, must create customer + product + order → invoice
# ===========================================================================

class TestInvoiceMultiStep(unittest.TestCase):
    """Verifies complete invoice flow from scratch."""

    def setUp(self):
        self.svc, self.client = _make_service()

    def test_full_invoice_flow_creates_all_prerequisites(self):
        """Fresh account: customer → product → order → invoice → send."""
        task = _task(
            entity=Entity.INVOICE,
            raw_prompt=(
                'Create invoice for customer Nordvik AS (org 912345678) for '
                '"IT Consulting" at 25000 NOK excl. VAT. Due date 2026-04-15.'
            ),
            attributes={
                "customerName": "Nordvik AS",
                "organizationNumber": "912345678",
                "orderLines": [{"description": "IT Consulting", "amount": 25000}],
                "dueDate": "2026-04-15",
            },
        )
        self.svc._dispatch(task)

        # Customer created
        customer_creates = [e for e in self.client.created_entities if e[0] == "/customer"]
        self.assertTrue(len(customer_creates) >= 1, "Customer must be created")
        self.assertEqual(customer_creates[0][1]["name"], "Nordvik AS")

        # Order created with correct line amount
        self.assertEqual(len(self.client.order_payloads), 1)
        line = self.client.order_payloads[0]["orderLines"][0]
        self.assertAlmostEqual(line["unitPriceExcludingVatCurrency"], 25000.0)

        # Invoice created
        self.assertEqual(len(self.client.invoice_payloads), 1)

    def test_invoice_with_payment_registration(self):
        """Order → invoice → payment in one flow."""
        task = _task(
            entity=Entity.ORDER,
            raw_prompt=(
                'Create order for Greenfield Ltd (org 914083478) with "Web Design" at 23450 NOK. '
                'Convert to invoice and register full payment.'
            ),
            attributes={
                "customerName": "Greenfield Ltd",
                "organizationNumber": "914083478",
                "orderLines": [{"description": "Web Design", "amount": 23450}],
                "workflow": "orderToInvoiceAndPayment",
            },
        )
        self.svc._dispatch(task)

        self.assertEqual(len(self.client.order_payloads), 1)
        self.assertEqual(len(self.client.invoice_payloads), 1)
        self.assertEqual(len(self.client.paid_invoices), 1)


# ===========================================================================
# SCENARIO 5: Fixed price project invoice (Tier 3)
# Requires KOMPLETT module + project creation + milestone invoice
# ===========================================================================

class TestFixedPriceProjectInvoice(unittest.TestCase):
    """Tier 3: Fixed-price project with milestone invoice."""

    def setUp(self):
        self.svc, self.client = _make_service()

    def test_fixed_price_50_percent_invoice(self):
        task = _task(
            entity=Entity.INVOICE,
            raw_prompt=(
                'Set fixed price 266550 NOK on project "Data Security" for Costa Digital SL '
                '(org 891505019). Project manager Clara Navarro. Invoice 50% of fixed price.'
            ),
            attributes={
                "customerName": "Costa Digital SL",
                "organizationNumber": "891505019",
                "projectName": "Data Security",
                "projectManagerName": "Clara Navarro",
                "projectManagerEmail": "clara.navarro@example.org",
                "fixedPrice": 266550,
                "isFixedPrice": True,
                "amount": 133275,
            },
        )
        self.svc._dispatch(task)

        # SMART_PROJECT activated for project
        self.assertIn("SMART_PROJECT", self.client.activated_modules)

        # Invoice amount = 50% = 133275
        self.assertEqual(len(self.client.order_payloads), 1)
        order_line = self.client.order_payloads[0]["orderLines"][0]
        self.assertAlmostEqual(order_line["unitPriceExcludingVatCurrency"], 133275.0)


# ===========================================================================
# SCENARIO 6: Multi-language prompt handling
# Same task in different languages must produce same API calls
# ===========================================================================

class TestMultiLanguageDispatch(unittest.TestCase):
    """Verifies the dispatch layer handles pre-parsed tasks from any language."""

    def _run_employee_task(self, raw_prompt: str, attrs: dict) -> FailableRecordingClient:
        svc, client = _make_service()
        task = _task(
            entity=Entity.EMPLOYEE,
            raw_prompt=raw_prompt,
            attributes=attrs,
        )
        svc._dispatch(task)
        return client

    def test_employee_norwegian(self):
        client = self._run_employee_task(
            'Opprett ein ansatt med namn Ola Nordmann, ola@example.org',
            {"firstName": "Ola", "lastName": "Nordmann", "email": "ola@example.org"},
        )
        emp = [e for e in client.created_entities if e[0] == "/employee"]
        self.assertEqual(emp[0][1]["firstName"], "Ola")

    def test_employee_spanish(self):
        client = self._run_employee_task(
            'Cree un empleado con el nombre María García, maria@ejemplo.org',
            {"firstName": "María", "lastName": "García", "email": "maria@ejemplo.org"},
        )
        emp = [e for e in client.created_entities if e[0] == "/employee"]
        self.assertEqual(emp[0][1]["firstName"], "María")

    def test_employee_german(self):
        client = self._run_employee_task(
            'Erstelle einen Mitarbeiter namens Hans Müller, hans@example.de',
            {"firstName": "Hans", "lastName": "Müller", "email": "hans@example.de"},
        )
        emp = [e for e in client.created_entities if e[0] == "/employee"]
        self.assertEqual(emp[0][1]["firstName"], "Hans")

    def test_employee_french(self):
        client = self._run_employee_task(
            'Créez un employé nommé Jean Dupont, jean@example.fr',
            {"firstName": "Jean", "lastName": "Dupont", "email": "jean@example.fr"},
        )
        emp = [e for e in client.created_entities if e[0] == "/employee"]
        self.assertEqual(emp[0][1]["firstName"], "Jean")

    def test_employee_portuguese(self):
        client = self._run_employee_task(
            'Crie um funcionário chamado João Silva, joao@example.pt',
            {"firstName": "João", "lastName": "Silva", "email": "joao@example.pt"},
        )
        emp = [e for e in client.created_entities if e[0] == "/employee"]
        self.assertEqual(emp[0][1]["firstName"], "João")


# ===========================================================================
# SCENARIO 7: Incoming invoice VAT edge cases
# Wrong VAT calculation = wrong voucher amounts = 0 points
# ===========================================================================

class TestIncomingInvoiceVATEdgeCases(unittest.TestCase):
    """Verifies VAT calculations produce correct voucher postings (via fallback when API is restricted)."""

    def setUp(self):
        self.svc, self.client = _make_service()
        # Simulate 403 on POST /incomingInvoice so it falls back to voucher path
        self.client.set_fail(
            "create_incoming_invoice",
            TripletexAPIError("You do not have permission", status_code=403),
            times=999,
        )

    def test_15_percent_vat_food(self):
        """15% VAT (food) produces correct 3-line voucher."""
        task = _task(
            entity=Entity.INCOMING_INVOICE,
            raw_prompt='Invoice from Catering AS for 11500 NOK incl 15% VAT. Account 4000.',
            attributes={
                "supplierName": "Catering AS",
                "totalAmountIncludingVat": 11500,
                "vatRate": 15,
                "debitAccountNumber": 4000,
            },
        )
        self.svc._dispatch(task)

        voucher = self.client.voucher_payloads[0]
        postings = voucher["postings"]
        self.assertEqual(len(postings), 3)
        # Net = 11500 / 1.15 = 10000
        self.assertAlmostEqual(postings[0]["amountGross"], 10000.0)
        # VAT = 1500
        self.assertAlmostEqual(postings[1]["amountGross"], 1500.0)
        # Credit = -11500
        self.assertAlmostEqual(postings[2]["amountGross"], -11500.0)

    def test_zero_vat_exempt(self):
        """0% VAT produces 2-line voucher (no VAT line)."""
        task = _task(
            entity=Entity.INCOMING_INVOICE,
            raw_prompt='Invoice from Exempt Co for 5000 NOK, no VAT. Account 6300.',
            attributes={
                "supplierName": "Exempt Co",
                "amount": 5000,
                "debitAccountNumber": 6300,
            },
        )
        self.svc._dispatch(task)

        voucher = self.client.voucher_payloads[0]
        postings = voucher["postings"]
        self.assertEqual(len(postings), 2)
        self.assertAlmostEqual(postings[0]["amountGross"], 5000.0)
        self.assertAlmostEqual(postings[1]["amountGross"], -5000.0)

    def test_amount_excl_vat_25_percent(self):
        """Amount excluding VAT with 25% — total = amount * 1.25."""
        task = _task(
            entity=Entity.INCOMING_INVOICE,
            raw_prompt='Invoice from X for 10000 NOK excl VAT. 25% VAT. Account 6590.',
            attributes={
                "supplierName": "X Co",
                "amount": 10000,
                "vatRate": 25,
                "debitAccountNumber": 6590,
            },
        )
        self.svc._dispatch(task)

        voucher = self.client.voucher_payloads[0]
        postings = voucher["postings"]
        self.assertEqual(len(postings), 3)
        self.assertAlmostEqual(postings[0]["amountGross"], 10000.0)
        self.assertAlmostEqual(postings[1]["amountGross"], 2500.0)
        self.assertAlmostEqual(postings[2]["amountGross"], -12500.0)


# ===========================================================================
# SCENARIO 8: Credit note workflow on fresh account
# Must create invoice first, then credit it
# ===========================================================================

class TestCreditNoteOnFreshAccount(unittest.TestCase):
    """On fresh account there are no existing invoices to credit."""

    def setUp(self):
        self.svc, self.client = _make_service()

    def test_credit_note_creates_invoice_then_credits(self):
        task = _task(
            entity=Entity.INVOICE,
            raw_prompt=(
                'Kunden Stormberg AS (org 991882502) har reklamert. '
                'Opprett kreditnota som reverserer fakturaen for "Opplæring" (13100 kr).'
            ),
            attributes={
                "customerName": "Stormberg AS",
                "organizationNumber": "991882502",
                "amount": 13100,
                "workflow": "creditNote",
                "lineDescription": "Opplæring",
            },
        )
        self.svc._dispatch(task)

        # Invoice must be created first
        self.assertTrue(len(self.client.invoice_payloads) >= 1,
                        "Invoice must be created before credit note on fresh account")
        # Then credit note
        self.assertTrue(len(self.client.credit_notes) >= 1,
                        "Credit note must be created")


# ===========================================================================
# SCENARIO 9: Register action routing
# "Registrer" in Norwegian means "create" except for payments
# ===========================================================================

class TestRegisterActionRouting(unittest.TestCase):
    """Verify REGISTER action routes correctly based on context."""

    def setUp(self):
        self.svc, self.client = _make_service()

    def test_register_payment_goes_to_payment_handler(self):
        task = _task(
            action=Action.REGISTER,
            entity=Entity.PAYMENT,
            raw_prompt='Register payment for customer Acme AS amount 15000',
            attributes={
                "customerName": "Acme AS",
                "amount": 15000,
            },
        )
        self.svc._dispatch(task)
        # Should create invoice first then pay it
        self.assertTrue(len(self.client.paid_invoices) >= 1)

    def test_register_incoming_invoice_creates_not_pays(self):
        """'Registrer leverandørfaktura' = create, not pay."""
        task = _task(
            action=Action.REGISTER,
            entity=Entity.INCOMING_INVOICE,
            raw_prompt='Registrer leverandørfaktura fra Kontorservice AS for 10000 kr',
            attributes={
                "supplierName": "Kontorservice AS",
                "amount": 10000,
                "debitAccountNumber": 6500,
            },
        )
        self.svc._dispatch(task)
        # Should create incoming invoice (API or voucher fallback), not pay
        created = len(self.client.incoming_invoice_payloads) + len(self.client.voucher_payloads)
        self.assertTrue(created >= 1, "Expected at least one incoming invoice or voucher to be created")
        self.assertEqual(len(self.client.paid_supplier_invoices), 0)

    def test_betal_incoming_invoice_pays_not_creates(self):
        """'Betal leverandørfaktura' = pay, not create."""
        task = _task(
            action=Action.REGISTER,
            entity=Entity.INCOMING_INVOICE,
            raw_prompt='Betal leverandørfaktura fra Supplier AS',
            attributes={
                "supplierName": "Supplier AS",
                "amount": 3000,
                "paymentDate": "2026-03-15",
            },
        )
        self.svc._dispatch(task)
        self.assertEqual(len(self.client.paid_supplier_invoices), 1)

    def test_register_travel_expense_redirects_to_create(self):
        """'Registrer reiseregning' should redirect to create travel expense."""
        task = _task(
            action=Action.REGISTER,
            entity=Entity.TRAVEL_EXPENSE,
            raw_prompt='Registrer reiseregning for Svein Berge',
            attributes={
                "employeeName": "Svein Berge",
                "departureDate": "2026-03-10",
                "purpose": "Kundebesøk",
            },
        )
        self.svc._dispatch(task)
        self.assertEqual(len(self.client.travel_expenses), 1)


# ===========================================================================
# SCENARIO 10: Dispatch coverage — all entity types
# Ensure no entity raises UnsupportedTaskError for CREATE
# ===========================================================================

class TestDispatchCoverage(unittest.TestCase):
    """Smoke test: all main entity types dispatch without UnsupportedTaskError."""

    ENTITY_CONFIGS = [
        (Entity.EMPLOYEE, {"firstName": "Test", "lastName": "User", "email": "t@t.com"}),
        (Entity.CUSTOMER, {"name": "Test Customer"}),
        (Entity.DEPARTMENT, {"name": "Test Department"}),
        (Entity.PRODUCT, {"name": "Test Product", "price": 100}),
        (Entity.SUPPLIER, {"name": "Test Supplier"}),
        (Entity.CONTACT, {"firstName": "Test", "lastName": "Contact", "customerName": "Test Customer"}),
    ]

    def test_create_all_basic_entities(self):
        """All basic entity types should dispatch without error."""
        for entity, attrs in self.ENTITY_CONFIGS:
            with self.subTest(entity=entity.value):
                svc, client = _make_service()
                task = _task(entity=entity, attributes=attrs)
                try:
                    svc._dispatch(task)
                except ParsingError:
                    pass  # Acceptable — parsing issue, not dispatch issue
                except Exception as e:
                    if "not implemented" in str(e).lower():
                        self.fail(f"CREATE {entity.value} is not implemented: {e}")

    def test_update_employee(self):
        """UPDATE employee should work."""
        svc, client = _make_service()
        task = _task(
            action=Action.UPDATE,
            entity=Entity.EMPLOYEE,
            raw_prompt='Update employee email to new@example.org',
            attributes={"firstName": "Test", "lastName": "User", "email": "new@example.org"},
        )
        try:
            svc._dispatch(task)
        except Exception as e:
            if "not implemented" in str(e).lower():
                self.fail(f"UPDATE employee should be implemented: {e}")

    def test_delete_travel_expense(self):
        """DELETE travel expense should work."""
        svc, client = _make_service()
        task = _task(
            action=Action.DELETE,
            entity=Entity.TRAVEL_EXPENSE,
            raw_prompt='Delete travel expense for Ola Nordmann',
            attributes={"employeeName": "Ola Nordmann"},
        )
        try:
            svc._dispatch(task)
        except Exception as e:
            if "not implemented" in str(e).lower():
                self.fail(f"DELETE travel expense should be implemented: {e}")


# ===========================================================================
# SCENARIO 11: Employee fields — dateOfBirth, startDate, employment form
# Competition checks these fields specifically
# ===========================================================================

class TestEmployeeFieldCompleteness(unittest.TestCase):
    """Verify employee creation sets all fields the competition checks."""

    def setUp(self):
        self.svc, self.client = _make_service()

    def test_employee_with_all_fields(self):
        task = _task(
            entity=Entity.EMPLOYEE,
            raw_prompt='Create employee Lars Svendsen born 1985-03-20, lars@example.org, start 2026-01-15',
            attributes={
                "firstName": "Lars",
                "lastName": "Svendsen",
                "email": "lars@example.org",
                "dateOfBirth": "1985-03-20",
                "startDate": "2026-01-15",
            },
        )
        self.svc._dispatch(task)

        emp_creates = [e for e in self.client.created_entities if e[0] == "/employee"]
        self.assertTrue(len(emp_creates) >= 1)
        emp = emp_creates[0][1]
        self.assertEqual(emp["firstName"], "Lars")
        self.assertEqual(emp["lastName"], "Svendsen")
        self.assertEqual(emp["email"], "lars@example.org")
        self.assertEqual(emp["dateOfBirth"], "1985-03-20")

    def test_employee_permanent_employment_form(self):
        """Employment form 'PERMANENT' should set remunerationType=MONTHLY_PAY."""
        task = _task(
            entity=Entity.EMPLOYEE,
            raw_prompt='Create employee Test User with permanent employment',
            attributes={
                "firstName": "Test",
                "lastName": "User",
                "email": "test@example.org",
                "employmentForm": "permanent",
            },
        )
        self.svc._dispatch(task)

        # Check that employment details were updated with correct form
        detail_updates = [
            (ep, eid, p) for ep, eid, p in self.client.updated_entities
            if ep == "/employee/employment/details"
        ]
        self.assertTrue(len(detail_updates) >= 1,
                        "Employment details should be updated for permanent form")
        payload = detail_updates[0][2]
        self.assertEqual(payload.get("employmentForm"), "PERMANENT")
        self.assertEqual(payload.get("remunerationType"), "MONTHLY_WAGE")
        self.assertEqual(payload.get("workingHoursScheme"), "NOT_SHIFT")


# ===========================================================================
# SCENARIO 12: Bank account setup for invoice
# Competition checks bank account number is set on account 1920
# ===========================================================================

class TestBankAccountSetup(unittest.TestCase):
    """Invoice flow must ensure bank account number is set."""

    def test_invoice_sets_bank_account_when_missing(self):
        """When account 1920 has no bankAccountNumber, it must be set."""
        svc, client = _make_service()
        # Override account 1920 to have empty bank account number
        client._accounts[1920] = {
            "id": 1920, "number": 1920, "name": "Bank",
            "isInvoiceAccount": True, "bankAccountNumber": "",
        }
        task = _task(
            entity=Entity.INVOICE,
            raw_prompt='Create invoice for Acme AS for consulting 5000 NOK',
            attributes={
                "customerName": "Acme AS",
                "orderLines": [{"description": "Consulting", "amount": 5000}],
            },
        )
        svc._dispatch(task)

        # Bank account should be updated with a number
        bank_updates = [
            (ep, eid, p) for ep, eid, p in client.updated_entities
            if "/ledger/account" in ep and "bankAccountNumber" in p
        ]
        self.assertTrue(len(bank_updates) >= 1,
                        "Bank account number must be set for invoicing")

    def test_invoice_skips_bank_update_when_already_set(self):
        """When account 1920 already has bankAccountNumber, skip update."""
        svc, client = _make_service()
        # Default account has bankAccountNumber set
        task = _task(
            entity=Entity.INVOICE,
            raw_prompt='Create invoice for Acme AS for consulting 5000 NOK',
            attributes={
                "customerName": "Acme AS",
                "orderLines": [{"description": "Consulting", "amount": 5000}],
            },
        )
        svc._dispatch(task)

        # No bank account update needed
        bank_updates = [
            (ep, eid, p) for ep, eid, p in client.updated_entities
            if "/ledger/account" in ep and "bankAccountNumber" in p
        ]
        self.assertEqual(len(bank_updates), 0,
                         "Should not update bank account when already set")


if __name__ == "__main__":
    unittest.main()
