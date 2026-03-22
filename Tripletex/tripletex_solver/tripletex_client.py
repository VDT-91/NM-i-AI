from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
import logging
import time
from typing import Any

import requests

from tripletex_solver.errors import TripletexAPIError

LOGGER = logging.getLogger(__name__)


class TripletexClient:
    def __init__(self, base_url: str, session_token: str, *, timeout_seconds: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.auth = ("0", session_token)
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        self._call_log: list[dict[str, Any]] = []
        self._module_activation_cache: dict[str, bool] = {}  # module_name -> success

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | list[dict[str, Any]] | None = None,
        expected_statuses: tuple[int, ...] = (200, 201, 204),
    ) -> Any:
        def _strip_internal_keys(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: _strip_internal_keys(val)
                    for key, val in value.items()
                    if not str(key).startswith("_")
                }
            if isinstance(value, list):
                return [_strip_internal_keys(item) for item in value]
            return value

        query = dict(params or {})
        sanitized_json_body = (
            _strip_internal_keys(json_body)
            if path == "/ledger/voucher" and json_body is not None
            else json_body
        )
        request_url = self._url(path)
        started_at = time.perf_counter()
        LOGGER.info(
            "Tripletex request: method=%s path=%s params=%s body_keys=%s",
            method,
            path,
            query,
            sorted(sanitized_json_body.keys()) if isinstance(sanitized_json_body, dict) else None,
        )
        response = self.session.request(
            method=method,
            url=request_url,
            params=query,
            json=sanitized_json_body,
            timeout=self.timeout_seconds,
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        LOGGER.info(
            "Tripletex response: method=%s path=%s status=%s elapsed_ms=%s",
            method,
            path,
            response.status_code,
            elapsed_ms,
        )

        # Record API call for debugging
        call_record: dict[str, Any] = {
            "method": method,
            "path": path,
            "params": query or None,
            "request_body": json.loads(json.dumps(sanitized_json_body, default=str)) if sanitized_json_body else None,
            "status": response.status_code,
            "elapsed_ms": elapsed_ms,
        }
        try:
            resp_json = response.json() if response.content else None
            # For list responses, summarize to avoid huge logs
            if isinstance(resp_json, dict) and isinstance(resp_json.get("values"), list):
                vals = resp_json["values"]
                call_record["response_summary"] = f"{len(vals)} items"
                call_record["response_body"] = vals[:3]  # first 3 items
            elif isinstance(resp_json, dict) and isinstance(resp_json.get("value"), dict):
                call_record["response_body"] = resp_json["value"]
            else:
                call_record["response_body"] = resp_json
        except (ValueError, Exception):
            call_record["response_body"] = response.text[:500] if response.text else None
        self._call_log.append(call_record)

        if response.status_code not in expected_statuses:
            raise TripletexAPIError(
                f"{method} {path} failed with {response.status_code}: {self._extract_error(response)}",
                status_code=response.status_code,
                response_text=response.text,
            )

        if response.status_code == 204 or not response.content:
            return None

        try:
            return response.json()
        except ValueError as exc:
            raise TripletexAPIError(
                f"{method} {path} returned non-JSON content",
                status_code=response.status_code,
                response_text=response.text,
            ) from exc

    @staticmethod
    def _extract_error(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip()

        if isinstance(payload, dict):
            # Include validation details for 422 errors
            msgs = payload.get("validationMessages")
            base = payload.get("message", payload.get("status", ""))
            if msgs and isinstance(msgs, list):
                details = "; ".join(
                    f"{m.get('field', '?')}: {m.get('message', '?')}" for m in msgs
                )
                return f"{base} [{details}]"
            if base:
                return str(base)
        return str(payload)

    @staticmethod
    def _unwrap_value(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict) and isinstance(payload.get("value"), dict):
            return payload["value"]
        if isinstance(payload, dict):
            return payload
        raise TripletexAPIError("Expected a JSON object in API response")

    @staticmethod
    def _unwrap_values(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict) and isinstance(payload.get("values"), list):
            return payload["values"]
        raise TripletexAPIError("Expected a list wrapper with a 'values' field")

    def get(self, path: str, *, fields: str = "", params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = dict(params or {})
        if fields:
            query["fields"] = fields
        return self._unwrap_value(self._request("GET", path, params=query))

    def list(self, path: str, *, fields: str = "", params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query = dict(params or {})
        if fields:
            query["fields"] = fields
        return self._unwrap_values(self._request("GET", path, params=query))

    def create(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._unwrap_value(self._request("POST", path, params=params, json_body=payload))

    def update(self, path: str, entity_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._unwrap_value(self._request("PUT", f"{path}/{entity_id}", json_body=payload))

    def delete(self, path: str, entity_id: int) -> None:
        self._request("DELETE", f"{path}/{entity_id}")

    def delete_list(self, path: str, ids: list[int]) -> None:
        ids_str = ",".join(str(i) for i in ids)
        self._request("DELETE", f"{path}/list", params={"ids": ids_str})

    def search_customers(
        self,
        *,
        name: str | None = None,
        email: str | None = None,
        phone: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if name:
            params["customerName"] = name
        if email:
            params["email"] = email
        if phone:
            params["phoneNumberMobile"] = phone
        return self.list(
            "/customer",
            fields="id,name,email,invoiceEmail,phoneNumber,phoneNumberMobile,displayName,organizationNumber",
            params=params,
        )

    def search_employees(
        self,
        *,
        first_name: str | None = None,
        last_name: str | None = None,
        email: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if first_name:
            params["firstName"] = first_name
        if last_name:
            params["lastName"] = last_name
        if email:
            params["email"] = email
        return self.list(
            "/employee",
            fields="id,firstName,lastName,displayName,email,phoneNumberMobile,phoneNumberWork,userType",
            params=params,
        )

    def search_departments(self, *, name: str) -> list[dict[str, Any]]:
        return self.list(
            "/department",
            fields="id,name,isInactive",
            params={"name": name, "count": 100},
        )

    def search_products(self, *, name: str | None = None, product_number: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if name:
            params["name"] = name
        if product_number:
            params["productNumber"] = product_number
        return self.list(
            "/product",
            fields="id,name,number,priceExcludingVatCurrency,priceIncludingVatCurrency",
            params=params,
        )

    def search_projects(
        self,
        *,
        name: str | None = None,
        customer_id: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if name:
            params["name"] = name
        if customer_id is not None:
            params["customerId"] = customer_id
        return self.list(
            "/project",
            fields="id,name,number,displayName,customer(id,name)",
            params=params,
        )

    def search_travel_expenses(self, *, employee_id: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if employee_id is not None:
            params["employeeId"] = employee_id
        return self.list(
            "/travelExpense",
            fields="id,travelDetails(*),employee(id,displayName)",
            params=params,
        )

    def search_travel_payment_types(self, *, query: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 20, "showOnEmployeeExpenses": True}
        if query:
            params["query"] = query
        return self.list(
            "/travelExpense/paymentType",
            fields="id,description",
            params=params,
        )

    def search_travel_cost_categories(self, *, query: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 20, "showOnEmployeeExpenses": True}
        if query:
            params["query"] = query
        return self.list(
            "/travelExpense/costCategory",
            fields="id,description",
            params=params,
        )

    def search_payment_types(self, *, query: str) -> list[dict[str, Any]]:
        return self.list(
            "/invoice/paymentType",
            fields="id,description",
            params={"query": query, "count": 100},
        )

    def search_vat_types(
        self,
        *,
        number: str | None = None,
        type_of_vat: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if number:
            params["number"] = number
        if type_of_vat:
            params["typeOfVat"] = type_of_vat
        return self.list(
            "/ledger/vatType",
            fields="id,name,number,percentage",
            params=params,
        )

    # --- Currency & Country ---

    def search_currencies(self, *, code: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if code:
            params["code"] = code
        return self.list("/currency", fields="id,code,description,factor", params=params)

    def search_countries(self, *, code: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if code:
            params["code"] = code
        return self.list("/country", fields="id,name,isoAlpha2Code,isoAlpha3Code", params=params)

    # --- Delivery Address ---

    def search_delivery_addresses(self, *, query: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if query:
            params["addressLine1"] = query
        return self.list("/deliveryAddress", fields="id,addressLine1,addressLine2,postalCode,city,country(id,name)", params=params)

    def search_accounts(
        self,
        *,
        number: int | None = None,
        is_bank_account: bool | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if number is not None:
            params["number"] = str(number)
        if is_bank_account is not None:
            params["isBankAccount"] = is_bank_account
        return self.list(
            "/ledger/account",
            fields="id,number,name,isBankAccount,isInvoiceAccount,bankAccountNumber,bankName,displayName",
            params=params,
        )

    def search_invoices(self, *, customer_id: int | None = None, invoice_number: str | None = None) -> list[dict[str, Any]]:
        today = datetime.now(UTC).date()
        params: dict[str, Any] = {
            "invoiceDateFrom": (today - timedelta(days=3650)).isoformat(),
            "invoiceDateTo": (today + timedelta(days=1)).isoformat(),
            "count": 20,
        }
        if customer_id is not None:
            params["customerId"] = str(customer_id)
        if invoice_number:
            params["invoiceNumber"] = invoice_number
        return self.list(
            "/invoice",
            fields="id,invoiceNumber,amount,amountOutstanding,amountCurrencyOutstanding,customer(id,name)",
            params=params,
        )

    def get_invoice(self, invoice_id: int) -> dict[str, Any]:
        return self.get(
            f"/invoice/{invoice_id}",
            fields="id,invoiceNumber,amount,amountOutstanding,amountCurrencyOutstanding,customer(id,name)",
        )

    def create_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/order", payload)

    def create_invoice(self, payload: dict[str, Any], *, send_to_customer: bool = False) -> dict[str, Any]:
        return self.create("/invoice", payload, params={"sendToCustomer": str(send_to_customer).lower()})

    def create_credit_note(
        self,
        invoice_id: int,
        *,
        credit_note_date: date,
        comment: str | None = None,
        send_to_customer: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "date": credit_note_date.isoformat(),
            "sendToCustomer": str(send_to_customer).lower(),
        }
        if comment:
            params["comment"] = comment
        return self._unwrap_value(self._request("PUT", f"/invoice/{invoice_id}/:createCreditNote", params=params))

    def invoice_order(self, order_id: int, *, invoice_date: date, send_to_customer: bool = False) -> dict[str, Any]:
        return self._unwrap_value(
            self._request(
                "PUT",
                f"/order/{order_id}/:invoice",
                params={
                    "invoiceDate": invoice_date.isoformat(),
                    "sendToCustomer": str(send_to_customer).lower(),
                },
            )
        )

    def pay_invoice(
        self,
        invoice_id: int,
        *,
        payment_date: date,
        payment_type_id: int,
        paid_amount: float,
        paid_amount_currency: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "paymentDate": payment_date.isoformat(),
            "paymentTypeId": payment_type_id,
            "paidAmount": paid_amount,
        }
        if paid_amount_currency is not None:
            params["paidAmountCurrency"] = paid_amount_currency
        return self._unwrap_value(self._request("PUT", f"/invoice/{invoice_id}/:payment", params=params))

    def create_travel_expense(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/travelExpense", payload)

    def create_travel_cost(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/travelExpense/cost", payload)

    def update_account(self, account_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self.update("/ledger/account", account_id, payload)

    # --- Contact ---

    def search_contacts(self, *, customer_id: int | None = None, email: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if customer_id is not None:
            params["customerId"] = customer_id
        if email:
            params["email"] = email
        return self.list(
            "/contact",
            fields="id,firstName,lastName,displayName,email,phoneNumberMobile,customer(id,name)",
            params=params,
        )

    def create_contact(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/contact", payload)

    # --- Supplier ---

    def search_suppliers(self, *, name: str | None = None, email: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if name:
            params["supplierName"] = name
        if email:
            params["email"] = email
        return self.list(
            "/supplier",
            fields="id,name,email,phoneNumber,supplierNumber",
            params=params,
        )

    # --- Voucher ---

    def search_voucher_types(self, *, name: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if name:
            params["name"] = name
        return self.list("/ledger/voucherType", fields="id,name", params=params)

    def search_accounts_by_number(self, number: int) -> list[dict[str, Any]]:
        return self.list(
            "/ledger/account",
            fields="id,number,name",
            params={"number": str(number), "count": 5},
        )

    # --- Activity / Timesheet ---

    def search_activities(self, *, name: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if name:
            params["name"] = name
        return self.list(
            "/activity",
            fields="id,name,activityType,isProjectActivity,isDisabled",
            params=params,
        )

    def create_timesheet_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/timesheet/entry", payload)

    # --- Travel Expense (update / actions) ---

    def update_travel_expense(self, expense_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self.update("/travelExpense", expense_id, payload)

    def deliver_travel_expense(self, expense_id: int) -> None:
        self._request("PUT", "/travelExpense/:deliver", params={"id": expense_id})

    def approve_travel_expense(self, expense_id: int) -> None:
        self._request("PUT", "/travelExpense/:approve", params={"id": expense_id})

    def create_mileage_allowance(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/travelExpense/mileageAllowance", payload)

    def create_per_diem_compensation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/travelExpense/perDiemCompensation", payload)

    def create_accommodation_allowance(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/travelExpense/accommodationAllowance", payload)

    def create_travel_passenger(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/travelExpense/passenger", payload)

    def create_driving_stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/travelExpense/drivingStop", payload)

    def create_cost_participant(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/travelExpense/costParticipant", payload)

    def search_mileage_rate_categories(self) -> list[dict[str, Any]]:
        return self.list(
            "/travelExpense/rateCategory",
            fields="id,name,type,isValidDayTrip,isValidDomestic",
            params={"count": 100},
        )

    # --- Employee Entitlements ---

    def grant_entitlements(self, employee_id: int, template: str) -> None:
        self._request(
            "PUT",
            "/employee/entitlement/:grantEntitlementsByTemplate",
            params={"employeeId": employee_id, "template": template},
        )

    # --- Voucher (reverse) ---

    def reverse_voucher(self, voucher_id: int, reverse_date: str) -> dict[str, Any]:
        return self._unwrap_value(
            self._request("PUT", f"/ledger/voucher/{voucher_id}/:reverse", params={"date": reverse_date})
        )

    def create_voucher(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/ledger/voucher", payload)

    # --- Invoice actions ---

    def create_reminder(
        self,
        invoice_id: int,
        *,
        reminder_type: str = "SOFT_REMINDER",
        reminder_date: str,
        dispatch_type: str = "EMAIL",
    ) -> dict[str, Any]:
        return self._unwrap_value(
            self._request(
                "PUT",
                f"/invoice/{invoice_id}/:createReminder",
                params={"type": reminder_type, "date": reminder_date, "dispatchType": dispatch_type},
            )
        )

    def send_invoice(self, invoice_id: int, *, send_type: str = "EMAIL") -> dict[str, Any]:
        return self._unwrap_value(
            self._request(
                "PUT",
                f"/invoice/{invoice_id}/:send",
                params={"sendType": send_type},
            )
        )

    # --- Company / Sales Modules ---

    def get_sales_modules(self) -> list[dict[str, Any]]:
        return self._unwrap_values(self._request("GET", "/company/salesmodules", params={"count": 100}))

    def _prefetch_active_modules(self) -> None:
        """Fetch already-active modules so we never POST a 409 for them."""
        if self._module_activation_cache:
            return  # Already fetched
        try:
            modules = self.list("/company/salesmodules", fields="name", params={"count": 100})
            for m in modules:
                name = m.get("name")
                if name:
                    self._module_activation_cache[name] = True
        except Exception:
            pass  # Non-critical — we'll just POST and handle 409 as before

    def activate_sales_module(self, module_name: str) -> dict[str, Any] | None:
        self._prefetch_active_modules()
        if module_name in self._module_activation_cache:
            if self._module_activation_cache[module_name]:
                return None  # Previously succeeded or already active
            raise TripletexAPIError(
                f"Module {module_name} activation previously failed",
                status_code=403,
                response_text="",
            )
        try:
            result = self.create("/company/salesmodules", {"name": module_name})
            self._module_activation_cache[module_name] = True
            return result
        except TripletexAPIError as e:
            # 409 = module already active — treat as success
            self._module_activation_cache[module_name] = (e.status_code == 409)
            raise

    # --- Bank Statement / Reconciliation ---

    def import_bank_statement(self, file_path: str, *, bank_id: int | None = None, file_format: str = "DNB_CSV") -> Any:
        """Import a bank statement file. Uses multipart form upload."""
        import io
        with open(file_path, "rb") as f:
            file_data = f.read()
        params: dict[str, Any] = {"fileFormat": file_format}
        if bank_id is not None:
            params["bankId"] = bank_id
        # Use requests directly for multipart upload
        response = self.session.post(
            self._url("/bank/statement/import"),
            params=params,
            files={"file": ("statement.csv", io.BytesIO(file_data), "text/csv")},
            timeout=self.timeout_seconds,
        )
        if response.status_code not in (200, 201):
            raise TripletexAPIError(
                f"Bank statement import failed: {self._extract_error(response)}",
                status_code=response.status_code,
                response_text=response.text,
            )
        return response.json() if response.content else None

    def search_bank_accounts(self) -> list[dict[str, Any]]:
        return self.list("/bank", fields="id,number,name,bankAccountNumber", params={"count": 100})

    def search_bank_reconciliations(self, *, account_id: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if account_id is not None:
            params["accountId"] = account_id
        return self.list("/bank/reconciliation", fields="id,account(id,number),closedDate", params=params)

    def create_bank_reconciliation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/bank/reconciliation", payload)

    def search_bank_statement_transactions(self, *, bank_statement_id: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if bank_statement_id is not None:
            params["bankStatementId"] = bank_statement_id
        return self.list("/bank/statement/transaction", fields="*", params=params)

    def suggest_bank_reconciliation_matches(self, reconciliation_id: int) -> Any:
        return self._request(
            "PUT", "/bank/reconciliation/match/:suggest",
            params={"bankReconciliationId": reconciliation_id},
        )

    def close_bank_reconciliation(self, reconciliation_id: int) -> dict[str, Any]:
        return self._unwrap_value(
            self._request("PUT", f"/bank/reconciliation/{reconciliation_id}", json_body={"isClosed": True})
        )

    def search_accounting_periods(
        self,
        *,
        start_from: str | None = None,
        start_to: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if start_from:
            params["startFrom"] = start_from
        if start_to:
            params["startTo"] = start_to
        return self.list(
            "/ledger/accountingPeriod",
            fields="id,number,start,end",
            params=params,
        )

    def search_ledger_postings(
        self,
        *,
        date_from: str,
        date_to: str,
        account_id: int | None = None,
        open_postings: str | None = None,
        supplier_id: int | None = None,
        customer_id: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "dateFrom": date_from,
            "dateTo": date_to,
            "count": 1000,
        }
        if account_id is not None:
            params["accountId"] = account_id
        if open_postings:
            params["openPostings"] = open_postings
        if supplier_id is not None:
            params["supplierId"] = supplier_id
        if customer_id is not None:
            params["customerId"] = customer_id
        return self.list(
            "/ledger/posting",
            fields="id,date,description,amount,amountCurrency,account(id,number,name),voucher(id,number)",
            params=params,
        )

    def search_bank_statements(self) -> list[dict[str, Any]]:
        return self.list(
            "/bank/statement",
            fields="id,bankAccount(id),fileName",
            params={"count": 100},
        )

    # --- Incoming Invoice (Supplier Invoice) ---

    def create_incoming_invoice(self, payload: dict[str, Any], *, send_to: str = "ledger") -> dict[str, Any]:
        return self.create("/incomingInvoice", payload, params={"sendTo": send_to})

    def search_incoming_invoices(self, *, supplier_id: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if supplier_id is not None:
            params["supplierId"] = supplier_id
        return self._unwrap_values(self._request("GET", "/incomingInvoice/search", params=params))

    def search_supplier_invoices(
        self,
        *,
        supplier_id: int | None = None,
        invoice_date_from: str | None = None,
        invoice_date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "count": 100,
            "invoiceDateFrom": invoice_date_from or (date.today() - timedelta(days=3650)).isoformat(),
            "invoiceDateTo": invoice_date_to or (date.today() + timedelta(days=1)).isoformat(),
        }
        if supplier_id is not None:
            params["supplierId"] = supplier_id
        return self.list(
            "/supplierInvoice",
            fields="id,invoiceNumber,invoiceDate,invoiceDueDate,amount,amountCurrency,outstandingAmount,supplier(id,name),voucher(id)",
            params=params,
        )

    def approve_incoming_invoice(self, invoice_id: int) -> None:
        self._request("PUT", f"/supplierInvoice/{invoice_id}/:approve")

    # --- Salary ---

    def create_salary_transaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/salary/transaction", payload)

    def search_salary_types(self) -> list[dict[str, Any]]:
        result = self.list("/salary/type", fields="id,name,number,description", params={"count": 100})
        if result:
            return result
        try:
            return self.list(
                "/employee/employment/employmentType/salaryType",
                fields="id,name,number,description",
                params={"count": 100},
            )
        except Exception:
            return result

    # --- Purchase Order ---

    def create_purchase_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/purchaseOrder", payload)

    def search_purchase_orders(self, *, supplier_id: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if supplier_id is not None:
            params["supplierId"] = supplier_id
        return self.list("/purchaseOrder", fields="id,number,supplier(id,name)", params=params)

    # --- Accounting Dimensions ---

    def create_dimension_name(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/ledger/accountingDimensionName", payload)

    def create_dimension_value(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/ledger/accountingDimensionValue", payload)

    def search_dimension_names(self, *, name: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if name:
            params["dimensionName"] = name
        return self.list("/ledger/accountingDimensionName", fields="id,dimensionName,dimensionIndex,active", params=params)

    def search_dimension_values(self, *, dimension_index: int | None = None, display_name: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if dimension_index is not None:
            params["dimensionIndex"] = dimension_index
        if display_name:
            params["displayName"] = display_name
        return self.list("/ledger/accountingDimensionValue/search", fields="id,displayName,dimensionIndex,number,active", params=params)

    def search_product_units(self, *, name: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if name:
            params["name"] = name
        return self.list("/product/unit", fields="id,name,nameShort,isCommon", params=params)

    # --- Activity ---

    def search_activities(self, *, name: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if name:
            params["name"] = name
        return self.list("/activity", fields="id,name,number,activityType,isChargeable,rate", params=params)

    def create_activity(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/activity", payload)

    # --- Division ---

    def create_division(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/division", payload)

    # --- Employee sub-entities ---

    def create_leave_of_absence(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/employee/employment/leaveOfAbsence", payload)

    def create_next_of_kin(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/employee/nextOfKin", payload)

    def search_employments(self, *, employee_id: int) -> list[dict[str, Any]]:
        return self.list(
            "/employee/employment",
            fields="id,employee(id),startDate,endDate,division(id,name)",
            params={"employeeId": employee_id, "count": 100},
        )

    # --- Categories ---

    def create_customer_category(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/customer/category", payload)

    def create_employee_category(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Uses batch endpoint (no single POST in spec)
        result = self._unwrap_values(
            self._request("POST", "/employee/category/list", json_body=[payload])
        )
        return result[0] if result else {}

    # --- Asset ---

    def create_asset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/asset", payload)

    # --- Project sub-entities ---

    def create_project_participant(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/project/participant", payload)

    def create_project_activity(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/project/projectActivity", payload)

    # --- Supplier Invoice Payment ---

    def pay_supplier_invoice(
        self, invoice_id: int, *, payment_type: int = 0, amount: float | None = None,
        payment_date: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"paymentType": payment_type}
        if amount is not None:
            params["amount"] = amount
        if payment_date:
            params["paymentDate"] = payment_date
        params["useDefaultPaymentType"] = True
        return self._unwrap_value(
            self._request("POST", f"/supplierInvoice/{invoice_id}/:addPayment", params=params)
        )

    # --- Employee sub-entities ---

    def create_hourly_cost_and_rate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/employee/hourlyCostAndRate", payload)

    def create_employee_standard_time(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/employee/standardTime", payload)

    # --- Salary Settings ---

    def create_company_holiday_setting(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/salary/settings/holiday", payload)

    def create_pension_scheme(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/salary/settings/pensionScheme", payload)

    def create_salary_standard_time(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/salary/settings/standardTime", payload)

    # --- Timesheet ---

    def create_timesheet_company_holiday(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/timesheet/companyHoliday", payload)

    def create_timesheet_allocated(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/timesheet/allocated", payload)

    # --- Project sub-entities ---

    def create_project_category(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/project/category", payload)

    def create_project_hourly_rate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/project/hourlyRates", payload)

    def create_project_specific_rate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/project/hourlyRates/projectSpecificRates", payload)

    def create_project_order_line(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/project/orderline", payload)

    def create_project_subcontract(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/project/subcontract", payload)

    # --- Product ---

    def create_product_group(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/product/group", payload)

    def create_supplier_product(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/product/supplierProduct", payload)

    # --- Order ---

    def create_order_group(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/order/orderGroup", payload)

    # --- Ledger ---

    def create_payment_type_out(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/ledger/paymentTypeOut", payload)

    # --- Bank ---

    def create_bank_reconciliation_match(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/bank/reconciliation/match", payload)

    # --- Inventory ---

    def create_inventory(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/inventory", payload)

    def search_inventories(self, *, name: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if name:
            params["name"] = name
        return self.list("/inventory", fields="id,name,number,isMainInventory", params=params)

    def create_inventory_location(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/inventory/location", payload)

    def search_inventory_locations(self, *, inventory_id: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"count": 100}
        if inventory_id is not None:
            params["inventoryId"] = inventory_id
        return self.list("/inventory/location", fields="id,name,number,inventory(id,name)", params=params)

    def create_stocktaking(self, payload: dict[str, Any], *, type_of_stocktaking: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if type_of_stocktaking:
            params["typeOfStocktaking"] = type_of_stocktaking
        return self.create("/inventory/stocktaking", payload, params=params or None)

    def create_stocktaking_productline(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/inventory/stocktaking/productline", payload)

    def create_product_inventory_location(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/product/inventoryLocation", payload)

    def create_goods_receipt(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/purchaseOrder/goodsReceipt", payload)

    def create_goods_receipt_line(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/purchaseOrder/goodsReceiptLine", payload)

    def confirm_goods_receipt(self, goods_receipt_id: int) -> dict[str, Any]:
        return self._unwrap_value(
            self._request("PUT", f"/purchaseOrder/goodsReceipt/{goods_receipt_id}/:receiveAndConfirm")
        )

    # --- Document Archive ---

    def upload_document_to_entity(
        self,
        entity_type: str,
        entity_id: int,
        file_path: str,
        *,
        filename: str | None = None,
    ) -> Any:
        """Upload a file to an entity's document archive.

        entity_type: one of 'customer', 'employee', 'project', 'supplier', 'account', 'product'
        """
        import io
        with open(file_path, "rb") as f:
            file_data = f.read()
        actual_filename = filename or file_path.split("/")[-1].split("\\")[-1]
        response = self.session.post(
            self._url(f"/documentArchive/{entity_type}/{entity_id}"),
            files={"file": (actual_filename, io.BytesIO(file_data), "application/octet-stream")},
            timeout=self.timeout_seconds,
        )
        if response.status_code not in (200, 201):
            raise TripletexAPIError(
                f"Document upload failed: {self._extract_error(response)}",
                status_code=response.status_code,
                response_text=response.text,
            )
        return response.json() if response.content else None

    def upload_document_reception(self, file_path: str, *, filename: str | None = None) -> Any:
        """Upload a file to the general document reception."""
        import io
        with open(file_path, "rb") as f:
            file_data = f.read()
        actual_filename = filename or file_path.split("/")[-1].split("\\")[-1]
        response = self.session.post(
            self._url("/documentArchive/reception"),
            files={"file": (actual_filename, io.BytesIO(file_data), "application/octet-stream")},
            timeout=self.timeout_seconds,
        )
        if response.status_code not in (200, 201):
            raise TripletexAPIError(
                f"Document reception upload failed: {self._extract_error(response)}",
                status_code=response.status_code,
                response_text=response.text,
            )
        return response.json() if response.content else None

    def upload_voucher_attachment(self, voucher_id: int, file_path: str) -> Any:
        """Upload a PDF/image attachment to a voucher."""
        import io
        with open(file_path, "rb") as f:
            file_data = f.read()
        filename = file_path.split("/")[-1].split("\\")[-1]
        response = self.session.post(
            self._url(f"/ledger/voucher/{voucher_id}/attachment"),
            files={"file": (filename, io.BytesIO(file_data), "application/octet-stream")},
            timeout=self.timeout_seconds,
        )
        if response.status_code not in (200, 201):
            raise TripletexAPIError(
                f"Voucher attachment upload failed: {self._extract_error(response)}",
                status_code=response.status_code,
                response_text=response.text,
            )
        return response.json() if response.content else None

    def upload_purchase_order_attachment(self, po_id: int, file_path: str) -> Any:
        """Upload an attachment to a purchase order."""
        import io
        with open(file_path, "rb") as f:
            file_data = f.read()
        filename = file_path.split("/")[-1].split("\\")[-1]
        response = self.session.post(
            self._url(f"/purchaseOrder/{po_id}/attachment"),
            files={"file": (filename, io.BytesIO(file_data), "application/octet-stream")},
            timeout=self.timeout_seconds,
        )
        if response.status_code not in (200, 201):
            raise TripletexAPIError(
                f"PO attachment upload failed: {self._extract_error(response)}",
                status_code=response.status_code,
                response_text=response.text,
            )
        return response.json() if response.content else None

    def upload_travel_expense_attachment(self, expense_id: int, file_path: str) -> Any:
        """Upload an attachment to a travel expense."""
        import io
        with open(file_path, "rb") as f:
            file_data = f.read()
        filename = file_path.split("/")[-1].split("\\")[-1]
        response = self.session.post(
            self._url(f"/travelExpense/{expense_id}/attachment"),
            files={"file": (filename, io.BytesIO(file_data), "application/octet-stream")},
            timeout=self.timeout_seconds,
        )
        if response.status_code not in (200, 201):
            raise TripletexAPIError(
                f"Travel expense attachment upload failed: {self._extract_error(response)}",
                status_code=response.status_code,
                response_text=response.text,
            )
        return response.json() if response.content else None

    # --- Event Subscription ---

    def create_event_subscription(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create("/event/subscription", payload)

    def search_event_subscriptions(self) -> list[dict[str, Any]]:
        return self.list("/event/subscription", fields="id,event,targetUrl,status", params={"count": 100})

    def delete_event_subscription(self, subscription_id: int) -> None:
        self._request("DELETE", f"/event/subscription/{subscription_id}")
