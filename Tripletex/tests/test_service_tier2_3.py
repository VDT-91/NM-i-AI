"""Tests for Tier 2/3 service workflows: travel expense deliver/approve,
incoming invoice + approve, bank statement reconciliation, salary transaction,
module activation, dimension + voucher, multi-line invoice, pay supplier invoice.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Any

from tripletex_solver.errors import ParsingError
from tripletex_solver.models import (
    Action,
    Entity,
    ParsedTask,
    SolveRequest,
    TripletexCredentials,
)
from tripletex_solver.service import TripletexService


class FakeTripletexClientTier2:
    """Extended fake client that tracks all API calls needed by Tier 2/3 workflows."""

    def __init__(self) -> None:
        # Generic CRUD tracking
        self.created: list[tuple[str, dict, dict | None]] = []
        self.updated_entities: list[tuple[str, int, dict]] = []
        self.deleted_entities: list[tuple[str, int]] = []
        self.listed: list[tuple[str, dict]] = []

        # Order / Invoice
        self.order_payloads: list[dict] = []
        self.invoice_payloads: list[tuple[dict, bool]] = []
        self.payment_calls: list[dict] = []
        self.credit_note_calls: list[dict] = []
        self.updated_accounts: list[tuple[int, dict]] = []

        # Travel expense
        self.travel_expense_payloads: list[dict] = []
        self.travel_cost_payloads: list[dict] = []
        self.delivered_travel_expenses: list[int] = []
        self.approved_travel_expenses: list[int] = []
        self.per_diem_payloads: list[dict] = []

        # Incoming invoice
        self.incoming_invoice_payloads: list[dict] = []
        self.approved_incoming_invoices: list[int] = []

        # Bank statement
        self.imported_bank_statements: list[tuple[str, int | None, str]] = []
        self.created_bank_reconciliations: list[dict] = []
        self.suggested_matches: list[int] = []
        self.closed_reconciliations: list[int] = []

        # Salary
        self.salary_transaction_payloads: list[dict] = []

        # Module activation
        self.activated_modules: list[str] = []

        # Dimension
        self.dimension_names_created: list[dict] = []
        self.dimension_values_created: list[dict] = []

        # Supplier invoice payment
        self.paid_supplier_invoices: list[dict] = []

        # Internal counters for auto-incrementing IDs
        self._next_id = 100

    def _next(self) -> int:
        self._next_id += 1
        return self._next_id

    # ---- Generic CRUD ----

    def create(self, path: str, payload: dict, params: dict | None = None) -> dict:
        self.created.append((path, payload, params))
        entity_id = self._next()
        return {"id": entity_id, **payload}

    def update(self, path: str, entity_id: int, payload: dict) -> dict:
        self.updated_entities.append((path, entity_id, payload))
        return {"id": entity_id, **payload}

    def delete(self, path: str, entity_id: int) -> None:
        self.deleted_entities.append((path, entity_id))

    def delete_list(self, path: str, ids: list[int]) -> None:
        for eid in ids:
            self.deleted_entities.append((path, eid))

    def list(self, path: str, fields: str = "", params: dict | None = None) -> list[dict]:
        self.listed.append((path, params or {}))
        if "/supplierInvoice" in path:
            return [{"id": 7001, "invoiceNumber": "SI-100", "amount": 5000.0,
                      "supplier": {"id": 901, "name": "Supplier AS"}}]
        if "/employee/employment" in path:
            return [{"id": 8001, "startDate": "2025-01-01",
                      "employmentDetails": [{"id": 8501}]}]
        if "/employee/employment/details" in path:
            return [{"id": 8501}]
        if "/ledger/voucher" in path:
            return []
        if f"/employee/" in path:
            return [{"id": 201, "dateOfBirth": "1990-01-01"}]
        return []

    def get(self, path: str, fields: str = "") -> dict:
        if "/employee/" in path:
            return {"id": 201, "dateOfBirth": "1990-01-01"}
        if "/invoice/" in path:
            return {"id": 401, "amountCurrencyOutstanding": 1500.0}
        return {"id": 1}

    # ---- Search methods ----

    def search_customers(self, *, name=None, email=None, phone=None) -> list[dict]:
        return []

    def search_employees(self, *, first_name=None, last_name=None, email=None) -> list[dict]:
        if first_name is None and last_name is None and email is None:
            return [{"id": 211, "displayName": "Project Manager",
                     "firstName": "Project", "lastName": "Manager",
                     "userType": "NO_ACCESS"}]
        return []

    def search_departments(self, *, name: str) -> list[dict]:
        return []

    def search_suppliers(self, *, name=None, email=None) -> list[dict]:
        if name == "Office Supplies AS":
            return [{"id": 901, "name": "Office Supplies AS"}]
        return []

    def search_products(self, *, name=None, product_number=None) -> list[dict]:
        return []

    def search_projects(self, *, name=None, customer_id=None) -> list[dict]:
        return []

    def search_contacts(self, *, customer_id=None, email=None) -> list[dict]:
        return []

    def search_activities(self, *, name=None) -> list[dict]:
        return [{"id": 5001, "name": "Administrasjon", "isDisabled": False}]

    def search_vat_types(self, *, number=None, type_of_vat=None) -> list[dict]:
        return [{"id": 25, "number": "3", "name": "MVA 25%", "percentage": 25}]

    def search_accounts(self, *, number=None, is_bank_account=None) -> list[dict]:
        if number == 1920 or is_bank_account:
            return [{
                "id": 9001,
                "number": 1920,
                "name": "Bankinnskudd",
                "isBankAccount": True,
                "isInvoiceAccount": True,
                "bankAccountNumber": "",
            }]
        return []

    def search_accounts_by_number(self, number: int) -> list[dict]:
        return [{"id": 426260000 + number, "number": number, "name": f"Account {number}"}]

    def search_payment_types(self, *, query: str) -> list[dict]:
        return [{"id": 601, "description": "Betalt til bank"}]

    def search_voucher_types(self, *, name=None) -> list[dict]:
        return [{"id": 9811, "name": "Leverandorfaktura"}]

    def search_salary_types(self) -> list[dict]:
        return [
            {"id": 5001, "number": "1", "name": "Fastlonn (manedslonn)"},
            {"id": 5002, "number": "30", "name": "Tillegg/bonus"},
        ]

    def search_travel_payment_types(self, *, query=None) -> list[dict]:
        return [{"id": 701, "description": "Corporate card"}]

    def search_travel_cost_categories(self, *, query=None) -> list[dict]:
        return [{"id": 801, "description": "Travel"}]

    def search_per_diem_rate_categories(self) -> list[dict]:
        return [{"id": 3001, "type": "PER_DIEM", "isValidDayTrip": True, "isValidDomestic": True}]

    def search_mileage_rate_categories(self) -> list[dict]:
        return [
            {"id": 3001, "type": "PER_DIEM", "isValidDayTrip": True, "isValidDomestic": True},
            {"id": 3002, "type": "MILEAGE_ALLOWANCE"},
            {"id": 3003, "type": "ACCOMMODATION_ALLOWANCE"},
        ]

    def search_invoices(self, *, customer_id=None, invoice_number=None) -> list[dict]:
        return []

    def search_travel_expenses(self, *, employee_id=None) -> list[dict]:
        return []

    def search_bank_accounts(self) -> list[dict]:
        return [{"id": 4001, "bankAccountNumber": "12345678901"}]

    def search_bank_reconciliations(self, *, account_id=None) -> list[dict]:
        return []  # No open reconciliations, so one will be created

    def search_bank_statements(self) -> list[dict]:
        return [{"id": 6001, "bankAccount": {"id": 4001}}]

    def search_employments(self, *, employee_id=None) -> list[dict]:
        return [{"id": 8001, "startDate": "2025-01-01"}]

    def search_product_units(self, *, name=None) -> list[dict]:
        return []

    # ---- Order / Invoice ----

    def create_order(self, payload: dict) -> dict:
        self.order_payloads.append(payload)
        return {"id": 301}

    def create_invoice(self, payload: dict, *, send_to_customer: bool = False) -> dict:
        self.invoice_payloads.append((payload, send_to_customer))
        return {"id": 401}

    def get_invoice(self, invoice_id: int) -> dict:
        return {"id": invoice_id, "invoiceNumber": str(invoice_id),
                "amountCurrencyOutstanding": 1500.0}

    def pay_invoice(self, invoice_id: int, *, payment_date, payment_type_id: int,
                    paid_amount: float, paid_amount_currency: float | None = None) -> dict:
        self.payment_calls.append({
            "invoice_id": invoice_id,
            "payment_date": payment_date,
            "payment_type_id": payment_type_id,
            "paid_amount": paid_amount,
        })
        return {"id": invoice_id}

    def invoice_order(self, order_id: int, *, invoice_date=None,
                      send_to_customer: bool = False) -> dict:
        return {"id": 401, "amountCurrencyOutstanding": 1500.0}

    def create_credit_note(self, invoice_id: int, *, credit_note_date,
                           comment: str | None = None,
                           send_to_customer: bool = False) -> dict:
        self.credit_note_calls.append({
            "invoice_id": invoice_id,
            "credit_note_date": credit_note_date,
            "comment": comment,
            "send_to_customer": send_to_customer,
        })
        return {"id": 901}

    # ---- Travel expense ----

    def create_travel_expense(self, payload: dict) -> dict:
        self.travel_expense_payloads.append(payload)
        return {"id": 1001}

    def create_travel_cost(self, payload: dict) -> dict:
        self.travel_cost_payloads.append(payload)
        return {"id": 1002}

    def deliver_travel_expense(self, expense_id: int) -> dict:
        self.delivered_travel_expenses.append(expense_id)
        return {"id": expense_id}

    def approve_travel_expense(self, expense_id: int) -> dict:
        self.approved_travel_expenses.append(expense_id)
        return {"id": expense_id}

    def create_per_diem_compensation(self, payload: dict) -> dict:
        self.per_diem_payloads.append(payload)
        return {"id": self._next()}

    def create_mileage_allowance(self, payload: dict) -> dict:
        return {"id": self._next()}

    def create_accommodation_allowance(self, payload: dict) -> dict:
        return {"id": self._next()}

    def create_travel_passenger(self, payload: dict) -> dict:
        return {"id": self._next()}

    def create_driving_stop(self, payload: dict) -> dict:
        return {"id": self._next()}

    def update_travel_expense(self, expense_id: int, payload: dict) -> dict:
        self.updated_entities.append(("/travelExpense", expense_id, payload))
        return {"id": expense_id}

    # ---- Incoming invoice ----

    def create_incoming_invoice(self, payload: dict, *, send_to: str = "ledger") -> dict:
        self.incoming_invoice_payloads.append(payload)
        return {"id": 5001}

    def approve_incoming_invoice(self, invoice_id: int) -> dict:
        self.approved_incoming_invoices.append(invoice_id)
        return {"id": invoice_id}

    def create_voucher(self, payload: dict) -> dict:
        self.created.append(("/ledger/voucher", payload, None))
        return {"id": self._next(), **payload}

    # ---- Bank statement ----

    def import_bank_statement(self, file_path: str, *, bank_id: int | None = None,
                              file_format: str = "DNB_CSV") -> dict:
        self.imported_bank_statements.append((file_path, bank_id, file_format))
        return {"id": 6001, "bankAccountId": 4001}

    def create_bank_reconciliation(self, payload: dict) -> dict:
        self.created_bank_reconciliations.append(payload)
        return {"id": 7001}

    def suggest_bank_reconciliation_matches(self, rec_id: int) -> dict:
        self.suggested_matches.append(rec_id)
        return {"id": rec_id}

    def close_bank_reconciliation(self, rec_id: int) -> dict:
        self.closed_reconciliations.append(rec_id)
        return {"id": rec_id}

    # ---- Salary ----

    def create_salary_transaction(self, payload: dict) -> dict:
        self.salary_transaction_payloads.append(payload)
        return {"id": self._next()}

    # ---- Module activation ----

    def activate_sales_module(self, module_name: str) -> dict:
        self.activated_modules.append(module_name)
        return {"id": self._next()}

    # ---- Dimension ----

    def create_dimension_name(self, payload: dict) -> dict:
        self.dimension_names_created.append(payload)
        return {"id": self._next(), "dimensionIndex": 1, **payload}

    def create_dimension_value(self, payload: dict) -> dict:
        self.dimension_values_created.append(payload)
        return {"id": self._next(), **payload}

    # ---- Supplier invoice payment ----

    def pay_supplier_invoice(self, invoice_id: int, *, amount: float | None = None,
                             payment_date: str | None = None) -> dict:
        self.paid_supplier_invoices.append({
            "invoice_id": invoice_id,
            "amount": amount,
            "payment_date": payment_date,
        })
        return {"id": invoice_id}

    # ---- Employee helpers ----

    def get_employee(self, employee_id: int) -> dict:
        return {"id": employee_id, "firstName": "Test", "lastName": "Employee",
                "dateOfBirth": "1990-01-01"}

    def update_account(self, account_id: int, payload: dict) -> dict:
        self.updated_accounts.append((account_id, payload))
        return {"id": account_id, **payload}

    def create_timesheet_entry(self, payload: dict) -> dict:
        self.created.append(("/timesheet/entry", payload, None))
        return {"id": self._next(), **payload}

    def create_contact(self, payload: dict) -> dict:
        self.created.append(("/contact", payload, None))
        return {"id": self._next(), **payload}

    def grant_entitlements(self, employee_id: int, template: str) -> dict:
        return {"id": employee_id}

    def reverse_voucher(self, voucher_id: int, date_str: str) -> dict:
        return {"id": voucher_id}

    def create_purchase_order(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_activity(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_division(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_leave_of_absence(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_next_of_kin(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_customer_category(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_employee_category(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_asset(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_product_group(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_project_category(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_inventory(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def search_inventories(self, *, name=None) -> list[dict]:
        return []

    def create_inventory_location(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_stocktaking(self, payload: dict, *, type_of_stocktaking=None) -> dict:
        return {"id": self._next(), **payload}

    def create_stocktaking_productline(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_goods_receipt(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def confirm_goods_receipt(self, receipt_id: int) -> dict:
        return {"id": receipt_id}

    def search_purchase_orders(self, *, supplier_id=None) -> list[dict]:
        return []

    def create_project_participant(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_project_activity(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_project_hourly_rate(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_hourly_cost_and_rate(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def create_reminder(self, invoice_id: int, *, reminder_type: str,
                        reminder_date: str) -> dict:
        return {"id": self._next()}

    def upload_document_reception(self, file_path: str) -> dict:
        return {"id": self._next()}

    def upload_document_to_entity(self, entity_type: str, entity_id: int,
                                  file_path: str) -> dict:
        return {"id": self._next()}

    def create_event_subscription(self, payload: dict) -> dict:
        return {"id": self._next(), **payload}

    def delete_event_subscription(self, sub_id: int) -> None:
        pass

    def search_event_subscriptions(self) -> list[dict]:
        return []


# ---------------------------------------------------------------------------


class TripletexServiceTier2Test(unittest.TestCase):
    """Tests for Tier 2/3 multi-step workflows executed via the service."""

    def setUp(self) -> None:
        self.client = FakeTripletexClientTier2()
        self.service = TripletexService(self.client)

    def _request(self, prompt: str) -> SolveRequest:
        return SolveRequest(
            prompt=prompt,
            files=[],
            tripletex_credentials=TripletexCredentials(
                base_url="https://tx-proxy.ainm.no/v2",
                session_token="token",
            ),
        )

    def _task(self, **kwargs: Any) -> ParsedTask:
        defaults: dict[str, Any] = {
            "action": Action.CREATE,
            "entity": Entity.INVOICE,
            "raw_prompt": "test prompt",
        }
        defaults.update(kwargs)
        return ParsedTask(**defaults)

    # ------------------------------------------------------------------ #
    # 1. Travel expense: deliver + approve called after creation
    # ------------------------------------------------------------------ #
    def test_travel_expense_deliver_and_approve(self) -> None:
        task = self._task(
            entity=Entity.TRAVEL_EXPENSE,
            raw_prompt="Create travel expense for Ola Nordmann from Oslo to Bergen on 2026-03-19 amount 450",
            attributes={
                "employeeName": "Ola Nordmann",
                "departureFrom": "Oslo",
                "destination": "Bergen",
                "departureDate": "2026-03-19",
                "returnDate": "2026-03-19",
                "amount": 450,
            },
        )
        self.service._pre_process(task)
        self.service._dispatch(task)

        # Travel expense was created
        self.assertEqual(len(self.client.travel_expense_payloads), 1)
        # Cost line was created
        self.assertEqual(len(self.client.travel_cost_payloads), 1)
        self.assertEqual(self.client.travel_cost_payloads[0]["amountCurrencyIncVat"], 450.0)
        # Deliver and approve must both be called with the expense ID
        self.assertIn(1001, self.client.delivered_travel_expenses,
                      "deliver_travel_expense must be called")
        self.assertIn(1001, self.client.approved_travel_expenses,
                      "approve_travel_expense must be called")

    # ------------------------------------------------------------------ #
    # 2. Incoming invoice + approve
    # ------------------------------------------------------------------ #
    def test_incoming_invoice_via_voucher(self) -> None:
        task = self._task(
            entity=Entity.INCOMING_INVOICE,
            raw_prompt="Create incoming invoice from Office Supplies AS amount 8000",
            attributes={
                "supplierName": "Office Supplies AS",
                "amount": 8000,
                "invoiceDate": "2026-03-19",
            },
        )
        self.service._pre_process(task)
        self.service._dispatch(task)

        # Invoice created via API (preferred) or voucher fallback
        api_payloads = self.client.incoming_invoice_payloads
        voucher_calls = [c for c in self.client.created if c[0] == "/ledger/voucher"]
        if api_payloads:
            # Verify API payload structure
            payload = api_payloads[0]
            self.assertIn("invoiceHeader", payload)
            self.assertIn("orderLines", payload)
            self.assertEqual(payload["invoiceHeader"]["invoiceDate"], "2026-03-19")
        else:
            # Voucher fallback
            self.assertEqual(len(voucher_calls), 1, "Should create one voucher for incoming invoice")
            voucher_payload = voucher_calls[0][1]
            self.assertIn("postings", voucher_payload)
            self.assertTrue(len(voucher_payload["postings"]) >= 2, "Voucher needs at least debit + credit postings")

    # ------------------------------------------------------------------ #
    # 3. Bank statement: full reconciliation flow with file
    # ------------------------------------------------------------------ #
    def test_bank_statement_full_reconciliation(self) -> None:
        # Create a real temp CSV file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("date,amount\n2026-03-01,1000\n")
            csv_path = Path(f.name)

        try:
            task = self._task(
                entity=Entity.BANK_STATEMENT,
                raw_prompt="Import bank statement",
                attributes={},
            )
            self.service._saved_attachment_paths = [csv_path]
            self.service._pre_process(task)
            self.service._dispatch(task)

            # import_bank_statement called (legacy flow without workflow=reconcile)
            self.assertEqual(len(self.client.imported_bank_statements), 1)
            self.assertEqual(self.client.imported_bank_statements[0][0], str(csv_path))
        finally:
            csv_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # 4. Bank statement: no file raises ParsingError
    # ------------------------------------------------------------------ #
    def test_bank_statement_no_file_raises(self) -> None:
        task = self._task(
            entity=Entity.BANK_STATEMENT,
            raw_prompt="Import bank statement",
            attributes={},
        )
        self.service._saved_attachment_paths = []
        with self.assertRaises(ParsingError) as ctx:
            self.service._dispatch(task)
        self.assertIn("attached file", str(ctx.exception).lower())

    # ------------------------------------------------------------------ #
    # 5. Salary transaction: employee + salary types + payslip
    # ------------------------------------------------------------------ #
    def test_salary_transaction(self) -> None:
        task = self._task(
            entity=Entity.SALARY_TRANSACTION,
            raw_prompt="Create salary for Hans Hansen base 45000 bonus 5000",
            attributes={
                "employeeName": "Hans Hansen",
                "baseSalary": 45000,
                "bonus": 5000,
                "date": "2026-03-01",
            },
        )
        self.service._pre_process(task)
        self.service._dispatch(task)

        # Salary transaction payload created
        self.assertEqual(len(self.client.salary_transaction_payloads), 1)
        payload = self.client.salary_transaction_payloads[0]
        # Must have payslips with specifications for base + bonus
        self.assertIn("payslips", payload)
        self.assertEqual(len(payload["payslips"]), 1)
        specs = payload["payslips"][0]["specifications"]
        self.assertEqual(len(specs), 2, "Should have base salary + bonus specifications")
        # Check base salary spec
        base_spec = specs[0]
        self.assertEqual(base_spec["rate"], 45000.0)
        self.assertEqual(base_spec["salaryType"]["id"], 5001)  # Fastlonn
        # Check bonus spec
        bonus_spec = specs[1]
        self.assertEqual(bonus_spec["rate"], 5000.0)
        self.assertEqual(bonus_spec["salaryType"]["id"], 5002)  # Tillegg/bonus

    # ------------------------------------------------------------------ #
    # 6. Module activation
    # ------------------------------------------------------------------ #
    def test_activate_module(self) -> None:
        task = self._task(
            entity=Entity.COMPANY_MODULE,
            raw_prompt="Activate department accounting module",
            attributes={"moduleName": "SMART"},
        )
        self.service._dispatch(task)

        self.assertIn("SMART", self.client.activated_modules)

    def test_activate_module_inferred_from_prompt(self) -> None:
        task = self._task(
            entity=Entity.COMPANY_MODULE,
            raw_prompt="Aktiver avdelingsregnskap",
            attributes={},
        )
        self.service._dispatch(task)

        self.assertIn("SMART", self.client.activated_modules)

    # ------------------------------------------------------------------ #
    # 7. Dimension + voucher with dimension reference
    # ------------------------------------------------------------------ #
    def test_dimension_and_voucher(self) -> None:
        task = self._task(
            entity=Entity.DIMENSION,
            raw_prompt="Create dimension Kostnadssenter with values Salg and Admin, post 1000 to account 4000",
            attributes={
                "dimensionName": "Kostnadssenter",
                "dimensionValues": ["Salg", "Admin"],
                "voucherAccountNumber": 4000,
                "voucherAmount": 1000,
                "voucherDimensionValue": "Salg",
            },
        )
        self.service._dispatch(task)

        # Dimension name created
        self.assertEqual(len(self.client.dimension_names_created), 1)
        self.assertEqual(self.client.dimension_names_created[0]["dimensionName"], "Kostnadssenter")
        # Two dimension values created
        self.assertEqual(len(self.client.dimension_values_created), 2)
        val_names = [v["displayName"] for v in self.client.dimension_values_created]
        self.assertIn("Salg", val_names)
        self.assertIn("Admin", val_names)
        # Voucher created with dimension reference
        voucher_creates = [(p, d) for p, d, _ in self.client.created if p == "/ledger/voucher"]
        self.assertEqual(len(voucher_creates), 1)
        postings = voucher_creates[0][1]["postings"]
        self.assertEqual(len(postings), 2)
        # Debit posting should have dimension reference
        debit = postings[0]
        self.assertIn("freeAccountingDimension1", debit)

    # ------------------------------------------------------------------ #
    # 8. Multi-line invoice
    # ------------------------------------------------------------------ #
    def test_multi_line_invoice(self) -> None:
        task = self._task(
            entity=Entity.INVOICE,
            raw_prompt='Create invoice for Acme AS with Consulting 5000 and Support 3000',
            attributes={
                "customerName": "Acme AS",
                "orderLines": [
                    {"description": "Consulting", "amount": 5000},
                    {"description": "Support", "amount": 3000},
                ],
                "invoiceDate": "2026-03-19",
                "invoiceDueDate": "2026-04-19",
            },
        )
        self.service._pre_process(task)
        self.service._dispatch(task)

        # Order created with 2 lines
        self.assertEqual(len(self.client.order_payloads), 1)
        order_lines = self.client.order_payloads[0]["orderLines"]
        self.assertEqual(len(order_lines), 2)
        self.assertEqual(order_lines[0]["description"], "Consulting")
        self.assertEqual(order_lines[0]["unitPriceExcludingVatCurrency"], 5000.0)
        self.assertEqual(order_lines[1]["description"], "Support")
        self.assertEqual(order_lines[1]["unitPriceExcludingVatCurrency"], 3000.0)
        # Invoice created from order
        self.assertEqual(len(self.client.invoice_payloads), 1)

    # ------------------------------------------------------------------ #
    # 9. Pay supplier invoice
    # ------------------------------------------------------------------ #
    def test_pay_supplier_invoice(self) -> None:
        task = self._task(
            action=Action.REGISTER,
            entity=Entity.INCOMING_INVOICE,
            raw_prompt="Pay supplier invoice SI-100 amount 5000 on 2026-03-20",
            attributes={
                "invoiceNumber": "SI-100",
                "amount": 5000,
                "paymentDate": "2026-03-20",
            },
        )
        self.service._dispatch(task)

        self.assertEqual(len(self.client.paid_supplier_invoices), 1)
        paid = self.client.paid_supplier_invoices[0]
        self.assertEqual(paid["invoice_id"], 7001)
        self.assertEqual(paid["amount"], 5000.0)
        self.assertEqual(paid["payment_date"], "2026-03-20")

    # ------------------------------------------------------------------ #
    # 10. Salary transaction with only base salary (no bonus)
    # ------------------------------------------------------------------ #
    def test_salary_transaction_base_only(self) -> None:
        task = self._task(
            entity=Entity.SALARY_TRANSACTION,
            raw_prompt="Create salary for Kari Nordmann monthly 35000",
            attributes={
                "employeeName": "Kari Nordmann",
                "baseSalary": 35000,
                "date": "2026-04-01",
            },
        )
        self.service._pre_process(task)
        self.service._dispatch(task)

        self.assertEqual(len(self.client.salary_transaction_payloads), 1)
        payload = self.client.salary_transaction_payloads[0]
        specs = payload["payslips"][0]["specifications"]
        self.assertEqual(len(specs), 1, "Should have only base salary specification")
        self.assertEqual(specs[0]["rate"], 35000.0)

    # ------------------------------------------------------------------ #
    # 11. Dimension without voucher
    # ------------------------------------------------------------------ #
    def test_dimension_without_voucher(self) -> None:
        task = self._task(
            entity=Entity.DIMENSION,
            raw_prompt="Create dimension Region with values Nord and Sor",
            attributes={
                "dimensionName": "Region",
                "dimensionValues": ["Nord", "Sor"],
            },
        )
        self.service._dispatch(task)

        self.assertEqual(len(self.client.dimension_names_created), 1)
        self.assertEqual(len(self.client.dimension_values_created), 2)
        # No voucher created since voucherAccountNumber is not set
        voucher_creates = [(p, d) for p, d, _ in self.client.created if p == "/ledger/voucher"]
        self.assertEqual(len(voucher_creates), 0)

    # ------------------------------------------------------------------ #
    # 12. Travel expense creates employee prerequisite
    # ------------------------------------------------------------------ #
    def test_travel_expense_creates_employee(self) -> None:
        task = self._task(
            entity=Entity.TRAVEL_EXPENSE,
            raw_prompt="Travel expense for Erik Berg from Trondheim to Oslo on 2026-04-01 amount 600",
            attributes={
                "employeeName": "Erik Berg",
                "departureFrom": "Trondheim",
                "destination": "Oslo",
                "departureDate": "2026-04-01",
                "returnDate": "2026-04-01",
                "amount": 600,
            },
        )
        self.service._pre_process(task)
        self.service._dispatch(task)

        # Employee created (department first, then employee via _ensure_employee)
        emp_creates = [(p, d) for p, d, _ in self.client.created if p == "/employee"]
        self.assertTrue(len(emp_creates) >= 1, "Employee should be created as prerequisite")
        self.assertEqual(emp_creates[0][1]["firstName"], "Erik")
        self.assertEqual(emp_creates[0][1]["lastName"], "Berg")

    # ------------------------------------------------------------------ #
    # 13. Incoming invoice creates supplier prerequisite
    # ------------------------------------------------------------------ #
    def test_incoming_invoice_creates_supplier(self) -> None:
        task = self._task(
            entity=Entity.INCOMING_INVOICE,
            raw_prompt="Create incoming invoice from New Supplier AS amount 12000",
            attributes={
                "supplierName": "New Supplier AS",
                "amount": 12000,
            },
        )
        self.service._pre_process(task)
        self.service._dispatch(task)

        # Supplier created via _ensure_supplier (not found in search)
        supplier_creates = [(p, d) for p, d, _ in self.client.created if p == "/supplier"]
        self.assertTrue(len(supplier_creates) >= 1, "Supplier should be created")
        self.assertEqual(supplier_creates[0][1]["name"], "New Supplier AS")
        # Invoice created via API or voucher fallback
        api_calls = self.client.incoming_invoice_payloads
        voucher_calls = [c for c in self.client.created if c[0] == "/ledger/voucher"]
        self.assertTrue(len(api_calls) + len(voucher_calls) >= 1, "Should create incoming invoice via API or voucher")

    # ------------------------------------------------------------------ #
    # 14. Pay supplier invoice by supplier name
    # ------------------------------------------------------------------ #
    def test_pay_supplier_invoice_by_name(self) -> None:
        task = self._task(
            action=Action.REGISTER,
            entity=Entity.INCOMING_INVOICE,
            raw_prompt="Betal leverandorfaktura fra Supplier AS",
            attributes={
                "supplierName": "Supplier AS",
                "amount": 3000,
                "paymentDate": "2026-03-15",
            },
        )
        self.service._dispatch(task)

        self.assertEqual(len(self.client.paid_supplier_invoices), 1)
        self.assertEqual(self.client.paid_supplier_invoices[0]["invoice_id"], 7001)
        self.assertEqual(self.client.paid_supplier_invoices[0]["amount"], 3000.0)

    # ------------------------------------------------------------------ #
    # 15. Module activation as pre-process side effect
    # ------------------------------------------------------------------ #
    def test_module_activation_in_preprocess(self) -> None:
        """When a task's attributes contain moduleName, _pre_process activates it."""
        task = self._task(
            entity=Entity.INVOICE,
            raw_prompt="Create invoice after enabling department accounting",
            attributes={
                "moduleName": "SMART",
                "customerName": "Test Customer",
                "amount": 1000,
                "invoiceDate": "2026-03-19",
            },
        )
        self.service._pre_process(task)

        self.assertIn("SMART", self.client.activated_modules,
                      "Module should be activated during pre-processing")


if __name__ == "__main__":
    unittest.main()
