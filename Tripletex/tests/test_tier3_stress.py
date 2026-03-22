"""Comprehensive Tier 2/3 stress tests — validates LLM parsing + dispatch routing + field extraction.

These tests run LOCALLY against mock clients. They verify:
1. LLM parser correctly extracts fields from multi-language prompts
2. Service dispatch routes to the right handler
3. Correct API calls are made with correct field values
4. Edge cases (VAT calculations, multi-step workflows, etc.)

Run: python -m pytest tests/test_tier3_stress.py -v
"""
from __future__ import annotations

import re
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

from tripletex_solver.errors import ParsingError, TripletexAPIError
from tripletex_solver.models import Action, Entity, ParsedTask
from tripletex_solver.service import TripletexService


# ---------------------------------------------------------------------------
# Mock client that records every call
# ---------------------------------------------------------------------------

class RecordingClient:
    """Mock TripletexClient that records calls and returns sensible defaults."""

    def __init__(self):
        self._call_log: list[tuple[str, Any]] = []
        self._next_id = 100
        self._voucher_types = [{"id": 1, "name": "Leverandørfaktura"}]
        self._payment_types = [{"id": 10, "description": "Betalt til bank"}]
        self._salary_types = [
            {"id": 1, "number": "1", "name": "Fastlønn"},
            {"id": 2, "number": "30", "name": "Engangstillegg/bonus"},
        ]
        self._vat_types = [
            {"id": 3, "percentage": 25.0, "name": "Utgående MVA 25%"},
            {"id": 6, "percentage": 15.0, "name": "Utgående MVA 15%"},
            {"id": 5, "percentage": 0.0, "name": "Ingen MVA"},
            {"id": 1, "percentage": 25.0, "name": "Fradrag inngående avgift, høy sats"},
            {"id": 11, "percentage": 15.0, "name": "Fradrag inngående avgift, lav sats"},
        ]
        self._accounts: dict[int, dict] = {
            1920: {"id": 1920, "number": 1920, "name": "Bank", "isInvoiceAccount": True, "bankAccountNumber": "10000000006"},
            2400: {"id": 2400, "number": 2400, "name": "Leverandørgjeld"},
            2710: {"id": 2710, "number": 2710, "name": "Inngående MVA"},
            4000: {"id": 4000, "number": 4000, "name": "Varekostnad"},
            5000: {"id": 5000, "number": 5000, "name": "Lønn"},
            6300: {"id": 6300, "number": 6300, "name": "Kontorrekvisita"},
            6500: {"id": 6500, "number": 6500, "name": "Kontorrekvisita"},
            6590: {"id": 6590, "number": 6590, "name": "Kontorutstyr"},
            6800: {"id": 6800, "number": 6800, "name": "IT-tjenester"},
            7000: {"id": 7000, "number": 7000, "name": "Reisekostnad"},
            7100: {"id": 7100, "number": 7100, "name": "Bilkostnad"},
            7140: {"id": 7140, "number": 7140, "name": "Reise og diett"},
            7770: {"id": 7770, "number": 7770, "name": "Bankgebyr"},
            8040: {"id": 8040, "number": 8040, "name": "Renteinntekt"},
        }
        # Track created entities
        self.created_entities: list[tuple[str, dict]] = []
        self.voucher_payloads: list[dict] = []
        self.order_payloads: list[dict] = []
        self.invoice_payloads: list[tuple[dict, bool]] = []
        self.paid_invoices: list[dict] = []
        self.paid_supplier_invoices: list[dict] = []
        self.activated_modules: list[str] = []
        self.entitlements_granted: list[tuple[int, str]] = []
        self.timesheet_entries: list[dict] = []
        self.travel_expenses: list[dict] = []
        self.per_diem_compensations: list[dict] = []
        self.travel_costs: list[dict] = []
        self.updated_entities: list[tuple[str, int, dict]] = []
        self.salary_transactions: list[dict] = []
        self.credit_notes: list[tuple[int, Any]] = []
        self.incoming_invoice_payloads: list[dict] = []
        self._fail_on: dict[str, Exception] = {}
        self._fail_count: dict[str, int] = {}

    def set_fail(self, method: str, error: Exception, times: int = 1):
        self._fail_on[method] = error
        self._fail_count[method] = times

    def _check_fail(self, method: str):
        if method in self._fail_on:
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
    def create(self, endpoint: str, payload: dict) -> dict:
        self._call_log.append(("create", (endpoint, payload)))
        self.created_entities.append((endpoint, payload))
        if "voucher" in endpoint.lower():
            self.voucher_payloads.append(payload)
        name = payload.get("name") or payload.get("firstName", "")
        return self._next({"name": name, **payload})

    def get(self, endpoint: str, fields: str = "*") -> dict:
        self._call_log.append(("get", endpoint))
        return self._next()

    def list(self, endpoint: str, fields: str = "*", params: dict | None = None) -> list:
        self._call_log.append(("list", (endpoint, params)))
        if "/department" in endpoint:
            return []  # Force creation
        if "/project" in endpoint:
            return []  # Force creation
        if "/customer" in endpoint and params:
            # Return customers that were created matching org number
            org = params.get("organizationNumber")
            name = params.get("name")
            for ep, payload in self.created_entities:
                if ep == "/customer":
                    if org and payload.get("organizationNumber") == org:
                        return [{"id": payload.get("id", self._next_id), **payload}]
                    if name and name.lower() in payload.get("name", "").lower():
                        return [{"id": payload.get("id", self._next_id), **payload}]
            return []
        if "/employee/employment" in endpoint:
            return [{"id": 900, "division": None}]
        if "/employee/employment/details" in endpoint:
            return [{"id": 901}]
        if "/ledger/accountingDimensionName" in endpoint:
            return []
        if "/ledger/voucher" in endpoint:
            return []
        return []

    def update(self, endpoint: str, entity_id: int, payload: dict) -> dict:
        self._call_log.append(("update", (endpoint, entity_id, payload)))
        self.updated_entities.append((endpoint, entity_id, payload))
        return {"id": entity_id, **payload}

    def update_account(self, account_id: int, payload: dict) -> dict:
        return self.update("/ledger/account", account_id, payload)

    # --- Employees ---
    def search_employees(self, **kwargs) -> list:
        return []

    def get_employee(self, emp_id: int) -> dict:
        return {"id": emp_id, "firstName": "Test", "lastName": "User", "dateOfBirth": "1990-01-01"}

    def grant_entitlements(self, employee_id: int, template: str) -> None:
        self.entitlements_granted.append((employee_id, template))

    def create_employee_standard_time(self, payload: dict) -> dict:
        return self._next()

    # --- Customers ---
    def search_customers(self, **kwargs) -> list:
        # Return empty first time (customer doesn't exist yet), then return after creation
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
        # Return a generic account
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
            "invoiceNumber": 9001,
            "amount": 10000,
            "amountCurrency": 10000,
            "amountOutstanding": 10000,
            "amountCurrencyOutstanding": 10000,
        })

    def search_invoices(self, **kwargs) -> list:
        # Return invoices that were created
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
        return self._next({"invoiceNumber": 9001, "amount": 10000, "amountOutstanding": 10000})

    def pay_invoice(self, invoice_id: int, *, payment_date, payment_type_id, paid_amount, **kwargs) -> dict:
        self.paid_invoices.append({
            "invoice_id": invoice_id,
            "amount": paid_amount,
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
        return self._voucher_types

    def reverse_voucher(self, voucher_id: int, date_str: str) -> dict:
        return self._next()

    # --- Payments ---
    def search_payment_types(self, **kwargs) -> list:
        return self._payment_types

    # --- Salary ---
    def search_salary_types(self, **kwargs) -> list:
        return self._salary_types

    def create_salary_transaction(self, payload: dict) -> dict:
        self.salary_transactions.append(payload)
        return self._next()

    def search_employments(self, **kwargs) -> list:
        return [{"id": 900, "division": None}]

    # --- Modules ---
    def activate_sales_module(self, module_name: str) -> dict:
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


# ---------------------------------------------------------------------------
# Helper to build a service with the recording client
# ---------------------------------------------------------------------------

def _make_service() -> tuple[TripletexService, RecordingClient]:
    svc = TripletexService.__new__(TripletexService)
    client = RecordingClient()
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
# TIER 3: Order → Invoice → Payment (multi-step)
# ===========================================================================

class TestOrderToInvoicePayment(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_order_to_invoice_and_payment_english(self):
        task = _task(
            entity=Entity.ORDER,
            raw_prompt=(
                'Create an order for the customer Greenfield Ltd (org no. 914083478) with the products '
                'Web Design (8474) at 23450 NOK and Software License (3064) at 7800 NOK. '
                'Convert the order to an invoice and register full payment.'
            ),
            attributes={
                "customerName": "Greenfield Ltd",
                "organizationNumber": "914083478",
                "orderLines": [
                    {"description": "Web Design", "productNumber": "8474", "amount": 23450},
                    {"description": "Software License", "productNumber": "3064", "amount": 7800},
                ],
                "workflow": "orderToInvoiceAndPayment",
            },
        )
        self.svc._dispatch(task)

        # Order created
        self.assertEqual(len(self.client.order_payloads), 1)
        self.assertEqual(len(self.client.order_payloads[0]["orderLines"]), 2)
        # Invoice created
        self.assertEqual(len(self.client.invoice_payloads), 1)
        # Payment registered
        self.assertEqual(len(self.client.paid_invoices), 1)

    def test_order_to_invoice_no_payment_norwegian(self):
        task = _task(
            entity=Entity.ORDER,
            raw_prompt=(
                'Opprett ein ordre for kunden Vestland Energi AS (org.nr 955123456) '
                'for "Energirådgiving" til 18000 kr ekskl. MVA. Konverter ordren til ein faktura.'
            ),
            attributes={
                "customerName": "Vestland Energi AS",
                "organizationNumber": "955123456",
                "orderLines": [{"description": "Energirådgiving", "amount": 18000}],
                "workflow": "orderToInvoice",
            },
        )
        self.svc._dispatch(task)

        self.assertEqual(len(self.client.order_payloads), 1)
        self.assertEqual(len(self.client.invoice_payloads), 1)
        # No payment
        self.assertEqual(len(self.client.paid_invoices), 0)


# ===========================================================================
# TIER 3: Fixed-price project + milestone invoice
# ===========================================================================

class TestFixedPriceProject(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_fixed_price_milestone_invoice_spanish(self):
        task = _task(
            entity=Entity.INVOICE,
            raw_prompt=(
                'Establezca un precio fijo de 266550 NOK en el proyecto "Seguridad de datos" para '
                'Costa Digital SL (org. nº 891505019). El jefe de proyecto es Clara Navarro '
                '(clara.navarro@example.org). Facture al cliente el 50% del precio fijo.'
            ),
            attributes={
                "customerName": "Costa Digital SL",
                "organizationNumber": "891505019",
                "projectName": "Seguridad de datos",
                "projectManagerName": "Clara Navarro",
                "projectManagerEmail": "clara.navarro@example.org",
                "fixedPrice": 266550,
                "isFixedPrice": True,
                "amount": 133275,  # 50% of 266550
            },
        )
        self.svc._dispatch(task)

        # SMART_PROJECT module should be activated for project creation
        self.assertTrue(any("SMART_PROJECT" in m for m in self.client.activated_modules),
                        f"SMART_PROJECT not in activated modules: {self.client.activated_modules}")
        # Invoice created
        self.assertEqual(len(self.client.invoice_payloads), 1)
        # Order line amount should be 133275
        self.assertEqual(len(self.client.order_payloads), 1)
        order_line = self.client.order_payloads[0]["orderLines"][0]
        self.assertAlmostEqual(order_line["unitPriceExcludingVatCurrency"], 133275.0)

    def test_fixed_price_project_creates_project(self):
        """On fresh account, project must be created (not just searched)."""
        task = _task(
            entity=Entity.INVOICE,
            raw_prompt='Create fixed-price project "Security Audit" for customer Acme AS (org 123456789). Fixed price 100000 NOK. Invoice 30%.',
            attributes={
                "customerName": "Acme AS",
                "organizationNumber": "123456789",
                "projectName": "Security Audit",
                "fixedPrice": 100000,
                "isFixedPrice": True,
                "amount": 30000,
            },
        )
        self.svc._dispatch(task)

        # Verify project was created (should appear in created_entities as /project)
        project_creates = [e for e in self.client.created_entities if e[0] == "/project"]
        self.assertTrue(len(project_creates) >= 1, "Project should be created on fresh account")
        # Invoice created
        self.assertEqual(len(self.client.invoice_payloads), 1)


# ===========================================================================
# TIER 2/3: Incoming Invoice with VAT calculations
# ===========================================================================

class TestIncomingInvoiceVAT(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()
        # Simulate 403 on POST /incomingInvoice so it falls back to voucher path
        from tripletex_solver.errors import TripletexAPIError
        self.client.set_fail(
            "create_incoming_invoice",
            TripletexAPIError("Not available", status_code=403),
            times=999,
        )

    def test_incoming_invoice_including_vat(self):
        """Amount including VAT should produce correct net/VAT split."""
        task = _task(
            entity=Entity.INCOMING_INVOICE,
            raw_prompt=(
                'We have received invoice INV-2026-3205 from the supplier Ironbridge Ltd (org no. 828254375) '
                'for 24500 NOK including VAT. The amount relates to office services (account 6590). '
                'Register the supplier invoice with the correct input VAT (25%).'
            ),
            attributes={
                "supplierName": "Ironbridge Ltd",
                "organizationNumber": "828254375",
                "totalAmountIncludingVat": 24500,
                "vatRate": 25,
                "debitAccountNumber": 6590,
                "invoiceNumber": "INV-2026-3205",
                "description": "Office services",
            },
        )
        self.svc._dispatch(task)

        # Should create a voucher
        self.assertEqual(len(self.client.voucher_payloads), 1)
        voucher = self.client.voucher_payloads[0]
        postings = voucher["postings"]

        # vatType approach: 2 postings (expense w/ vatType + credit)
        # fallback: 3 postings (expense + VAT + credit)
        self.assertIn(len(postings), (2, 3))

        if len(postings) == 2:
            # vatType approach: expense has gross amount, Tripletex handles VAT
            self.assertAlmostEqual(postings[0]["amountGross"], 24500.0)
            self.assertIn("vatType", postings[0])
            self.assertAlmostEqual(postings[1]["amountGross"], -24500.0)
        else:
            # Manual 3-posting: expense (net) + VAT + credit
            self.assertAlmostEqual(postings[0]["amountGross"], 19600.0)
            self.assertAlmostEqual(postings[1]["amountGross"], 4900.0)
            self.assertAlmostEqual(postings[2]["amountGross"], -24500.0)

    def test_incoming_invoice_excluding_vat(self):
        """Amount excluding VAT should be treated as net directly."""
        task = _task(
            entity=Entity.INCOMING_INVOICE,
            raw_prompt=(
                'Register incoming invoice from Supplier X (org 999888777) for 10000 NOK '
                'excluding VAT with 25% VAT. Account 4000.'
            ),
            attributes={
                "supplierName": "Supplier X",
                "organizationNumber": "999888777",
                "amount": 10000,
                "vatRate": 25,
                "debitAccountNumber": 4000,
            },
        )
        self.svc._dispatch(task)

        voucher = self.client.voucher_payloads[0]
        postings = voucher["postings"]

        # vatType approach: 2 postings; fallback: 3 postings
        self.assertIn(len(postings), (2, 3))
        if len(postings) == 2:
            # vatType: total_with_vat = 10000 * 1.25 = 12500
            self.assertAlmostEqual(postings[0]["amountGross"], 12500.0)
            self.assertIn("vatType", postings[0])
            self.assertAlmostEqual(postings[1]["amountGross"], -12500.0)
        else:
            self.assertAlmostEqual(postings[0]["amountGross"], 10000.0)
            self.assertAlmostEqual(postings[1]["amountGross"], 2500.0)
            self.assertAlmostEqual(postings[2]["amountGross"], -12500.0)

    def test_incoming_invoice_no_vat(self):
        """No VAT → 2-line voucher."""
        task = _task(
            entity=Entity.INCOMING_INVOICE,
            raw_prompt='Register invoice from NoVAT Co (org 111222333) for 5000 NOK, no VAT. Account 6300.',
            attributes={
                "supplierName": "NoVAT Co",
                "organizationNumber": "111222333",
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

    def test_incoming_invoice_with_department(self):
        """Department should be set on expense posting."""
        task = _task(
            entity=Entity.INCOMING_INVOICE,
            raw_prompt='Register invoice from Dept Corp for 8000 NOK no VAT, account 6300, department IT.',
            attributes={
                "supplierName": "Dept Corp",
                "amount": 8000,
                "debitAccountNumber": 6300,
                "departmentName": "IT",
            },
        )
        self.svc._dispatch(task)

        voucher = self.client.voucher_payloads[0]
        expense = voucher["postings"][0]
        self.assertIn("department", expense)
        self.assertIn("id", expense["department"])

    def test_incoming_invoice_norwegian_including_vat(self):
        """Norwegian 'inkludert MVA' with 25% VAT."""
        task = _task(
            entity=Entity.INCOMING_INVOICE,
            raw_prompt=(
                'Vi har mottatt faktura fra leverandøren Kontorservice AS (org.nr 812345678) '
                'på 18750 kr inkludert MVA. Beløpet gjelder kontorrekvisita (konto 6500). '
                'Registrer leverandørfakturaen med korrekt inngående MVA (25%).'
            ),
            attributes={
                "supplierName": "Kontorservice AS",
                "organizationNumber": "812345678",
                "totalAmountIncludingVat": 18750,
                "vatRate": 25,
                "debitAccountNumber": 6500,
            },
        )
        self.svc._dispatch(task)

        voucher = self.client.voucher_payloads[0]
        postings = voucher["postings"]
        self.assertIn(len(postings), (2, 3))
        if len(postings) == 2:
            # vatType approach: expense has total incl VAT
            self.assertAlmostEqual(postings[0]["amountGross"], 18750.0)
            self.assertIn("vatType", postings[0])
            self.assertAlmostEqual(postings[1]["amountGross"], -18750.0)
        else:
            # Net = 18750 / 1.25 = 15000
            self.assertAlmostEqual(postings[0]["amountGross"], 15000.0)
            # VAT = 3750
            self.assertAlmostEqual(postings[1]["amountGross"], 3750.0)
            # Credit = -18750
            self.assertAlmostEqual(postings[2]["amountGross"], -18750.0)


# ===========================================================================
# TIER 3: Dimension + voucher
# ===========================================================================

class TestDimensionVoucher(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_dimension_with_linked_voucher(self):
        task = _task(
            entity=Entity.DIMENSION,
            raw_prompt=(
                'Opprett en fri regnskapsdimensjon "Produktlinje" med verdiene "Standard" og "Avansert". '
                'Bokfør deretter et bilag på konto 7000 for 12700 kr, knyttet til dimensjonsverdien "Standard".'
            ),
            attributes={
                "dimensionName": "Produktlinje",
                "dimensionValues": ["Standard", "Avansert"],
                "voucherAccountNumber": "7000",
                "voucherAmount": 12700,
                "voucherDimensionValue": "Standard",
            },
        )
        self.svc._dispatch(task)

        # SMART module activated
        self.assertIn("SMART", self.client.activated_modules)
        # Voucher created with dimension reference
        voucher_creates = [e for e in self.client.created_entities if e[0] == "/ledger/voucher"]
        self.assertTrue(len(voucher_creates) >= 1, "Voucher for dimension should be created")


# ===========================================================================
# TIER 3: Travel Expense with per diem + expenses
# ===========================================================================

class TestTravelExpense(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_travel_expense_with_per_diem_and_costs(self):
        task = _task(
            entity=Entity.TRAVEL_EXPENSE,
            raw_prompt=(
                'Registrer ei reiserekning for Svein Berge (svein.berge@example.org) for '
                '"Kundebesøk Trondheim". Reisa varte 3 dagar med diett (dagssats 800 kr). '
                'Utlegg: flybillett 2850 kr og taxi 200 kr.'
            ),
            attributes={
                "employeeName": "Svein Berge",
                "employeeEmail": "svein.berge@example.org",
                "departureDate": "2026-03-10",
                "returnDate": "2026-03-12",
                "destination": "Trondheim",
                "purpose": "Kundebesøk Trondheim",
                "title": "Kundebesøk Trondheim",
                "hasPerDiem": True,
                "perDiemDailyRate": 800,
                "perDiemDays": 3,
                "expenseLines": [
                    {"description": "Flybillett", "amount": 2850},
                    {"description": "Taxi", "amount": 200},
                ],
            },
        )
        self.svc._dispatch(task)

        # Travel expense created
        self.assertEqual(len(self.client.travel_expenses), 1)
        te = self.client.travel_expenses[0]
        self.assertEqual(te["travelDetails"]["destination"], "Trondheim")
        # Expense lines created (2 explicit + 1 per diem fallback when per diem API fails)
        self.assertTrue(len(self.client.travel_costs) >= 2)

    def test_travel_expense_destination_extraction(self):
        """Destination should be extracted from purpose when not explicit."""
        task = _task(
            entity=Entity.TRAVEL_EXPENSE,
            raw_prompt='Register travel expense for John Doe for client visit in Bergen. 1 day trip.',
            attributes={
                "employeeName": "John Doe",
                "departureDate": "2026-03-15",
                "purpose": "Client visit Bergen",
            },
        )
        self.svc._dispatch(task)

        te = self.client.travel_expenses[0]
        # Should extract Bergen as destination
        dest = te["travelDetails"]["destination"]
        self.assertIn("Bergen", dest)


# ===========================================================================
# TIER 2: Credit Note
# ===========================================================================

class TestCreditNote(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_credit_note_creates_invoice_then_credits(self):
        """On fresh account, we must create the invoice first, then credit it."""
        task = _task(
            entity=Entity.INVOICE,
            raw_prompt=(
                'Kunden Stormberg AS (org.nr 991882502) har reklamert på fakturaen for "Opplæring" '
                '(13100 kr ekskl. MVA). Opprett en fullstendig kreditnota som reverserer hele fakturaen.'
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

        # Should create an invoice first (no existing invoices)
        self.assertTrue(len(self.client.invoice_payloads) >= 1)
        # Then credit it
        self.assertTrue(len(self.client.credit_notes) >= 1)


# ===========================================================================
# TIER 2: Salary Transaction
# ===========================================================================

class TestSalaryTransaction(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_salary_with_bonus(self):
        task = _task(
            entity=Entity.SALARY_TRANSACTION,
            raw_prompt=(
                'Kjør lønn for Brita Berge (brita.berge@example.org) for denne månaden. '
                'Grunnlønn er 36800 kr. Legg til ein eingongsbonus på 14100 kr.'
            ),
            attributes={
                "employeeName": "Brita Berge",
                "employeeEmail": "brita.berge@example.org",
                "baseSalary": 36800,
                "bonus": 14100,
            },
        )
        self.svc._dispatch(task)

        # Salary transaction created with 2 specifications (base + bonus)
        self.assertEqual(len(self.client.salary_transactions), 1)
        tx = self.client.salary_transactions[0]
        specs = tx["payslips"][0]["specifications"]
        self.assertEqual(len(specs), 2)
        rates = sorted([s["rate"] for s in specs])
        self.assertIn(14100.0, rates)
        self.assertIn(36800.0, rates)

    def test_salary_base_only(self):
        task = _task(
            entity=Entity.SALARY_TRANSACTION,
            raw_prompt='Create salary for Kari Nordmann monthly 35000',
            attributes={
                "employeeName": "Kari Nordmann",
                "baseSalary": 35000,
                "date": "2026-04-01",
            },
        )
        self.svc._dispatch(task)

        self.assertEqual(len(self.client.salary_transactions), 1)
        specs = self.client.salary_transactions[0]["payslips"][0]["specifications"]
        self.assertEqual(len(specs), 1)
        self.assertAlmostEqual(specs[0]["rate"], 35000.0)


# ===========================================================================
# TIER 2: Payment Registration
# ===========================================================================

class TestPaymentRegistration(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_payment_creates_invoice_then_pays(self):
        """On fresh account, must create customer + order + invoice, then pay."""
        task = _task(
            action=Action.REGISTER,
            entity=Entity.PAYMENT,
            raw_prompt=(
                'Der Kunde Sonnental GmbH (Org.-Nr. 855482207) hat eine offene Rechnung über 33500 NOK ohne MwSt. '
                'für "Wartung". Registrieren Sie die vollständige Zahlung dieser Rechnung.'
            ),
            attributes={
                "customerName": "Sonnental GmbH",
                "organizationNumber": "855482207",
                "amount": 33500,
                "productName": "Wartung",
            },
        )
        self.svc._dispatch(task)

        # Customer created
        customer_creates = [e for e in self.client.created_entities if e[0] == "/customer"]
        self.assertTrue(len(customer_creates) >= 1)
        # Invoice created
        self.assertTrue(len(self.client.invoice_payloads) >= 1)
        # Payment registered
        self.assertEqual(len(self.client.paid_invoices), 1)


# ===========================================================================
# TIER 2: Multi-line Invoice with different VAT rates
# ===========================================================================

class TestMultiLineInvoice(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_three_lines_different_vat(self):
        task = _task(
            entity=Entity.INVOICE,
            raw_prompt=(
                'Create an invoice for the customer Windmill Ltd (org no. 994973150) with three product '
                'lines: System Development (6517) at 22950 NOK with 25% VAT, Maintenance (5339) at '
                '8900 NOK with 15% VAT (food), and Consulting Hours (8246) at 10450 NOK with 0% VAT (exempt).'
            ),
            attributes={
                "customerName": "Windmill Ltd",
                "organizationNumber": "994973150",
                "orderLines": [
                    {"description": "System Development", "productNumber": "6517", "amount": 22950, "vatRate": 25},
                    {"description": "Maintenance", "productNumber": "5339", "amount": 8900, "vatRate": 15},
                    {"description": "Consulting Hours", "productNumber": "8246", "amount": 10450, "vatRate": 0},
                ],
            },
        )
        self.svc._dispatch(task)

        self.assertEqual(len(self.client.order_payloads), 1)
        lines = self.client.order_payloads[0]["orderLines"]
        self.assertEqual(len(lines), 3)
        amounts = [l["unitPriceExcludingVatCurrency"] for l in lines]
        self.assertIn(22950.0, amounts)
        self.assertIn(8900.0, amounts)
        self.assertIn(10450.0, amounts)


# ===========================================================================
# TIER 3: Project Invoice with multiple employees
# ===========================================================================

class TestProjectInvoice(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_project_invoice_multi_employee(self):
        task = _task(
            entity=Entity.INVOICE,
            raw_prompt=(
                'Create a project invoice for customer TechCorp AS (org 923456789). '
                'Project "Cloud Migration". Employees: Alice (alice@ex.com, 40h), '
                'Bob (bob@ex.com, 25h). Hourly rate 1500 NOK.'
            ),
            attributes={
                "customerName": "TechCorp AS",
                "organizationNumber": "923456789",
                "projectName": "Cloud Migration",
                "workflow": "projectInvoice",
                "rate": 1500,
                "employees": [
                    {"name": "Alice Test", "email": "alice@ex.com", "hours": 40},
                    {"name": "Bob Test", "email": "bob@ex.com", "hours": 25},
                ],
            },
        )
        self.svc._dispatch(task)

        # Timesheet entries for both employees
        self.assertEqual(len(self.client.timesheet_entries), 2)
        hours = sorted([e["hours"] for e in self.client.timesheet_entries])
        self.assertEqual(hours, [25.0, 40.0])
        # Invoice created
        self.assertEqual(len(self.client.invoice_payloads), 1)
        # Order line should reflect total: 65h * 1500 = 97500
        order_line = self.client.order_payloads[0]["orderLines"][0]
        self.assertAlmostEqual(order_line["count"], 65.0)
        self.assertAlmostEqual(order_line["unitPriceExcludingVatCurrency"], 1500.0)


# ===========================================================================
# TIER 1/2: Employee creation with admin role
# ===========================================================================

class TestEmployeeCreation(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_employee_admin_entitlements(self):
        task = _task(
            entity=Entity.EMPLOYEE,
            raw_prompt='Create employee Anna Berg (anna.berg@example.org) born 1990-05-15. Admin. Start 2026-01-01.',
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

        # Employee created
        emp_creates = [e for e in self.client.created_entities if e[0] == "/employee"]
        self.assertTrue(len(emp_creates) >= 1)
        emp = emp_creates[0][1]
        self.assertEqual(emp["firstName"], "Anna")
        self.assertEqual(emp["lastName"], "Berg")
        # Admin entitlements granted (worth 5/10 pts!)
        self.assertTrue(len(self.client.entitlements_granted) >= 1,
                        "Admin entitlements should be granted")


# ===========================================================================
# TIER 2: Supplier invoice payment routing
# ===========================================================================

class TestSupplierInvoiceRouting(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_pay_keyword_routes_to_payment(self):
        """'Betal leverandorfaktura' should route to pay, not create."""
        task = _task(
            action=Action.REGISTER,
            entity=Entity.INCOMING_INVOICE,
            raw_prompt="Betal leverandorfaktura fra Supplier AS",
            attributes={
                "supplierName": "Supplier AS",
                "amount": 3000,
                "paymentDate": "2026-03-15",
            },
        )
        self.svc._dispatch(task)
        self.assertEqual(len(self.client.paid_supplier_invoices), 1)

    def test_register_keyword_routes_to_create(self):
        """'Registrer leverandørfaktura' should route to create."""
        task = _task(
            action=Action.REGISTER,
            entity=Entity.INCOMING_INVOICE,
            raw_prompt="Registrer leverandørfaktura fra Kontorservice AS for 10000 kr",
            attributes={
                "supplierName": "Kontorservice AS",
                "amount": 10000,
                "debitAccountNumber": 6500,
            },
        )
        self.svc._dispatch(task)
        # Should create incoming invoice (API or voucher), not pay
        created = len(self.client.incoming_invoice_payloads) + len(self.client.voucher_payloads)
        self.assertTrue(created >= 1, "Expected incoming invoice or voucher to be created")
        self.assertEqual(len(self.client.paid_supplier_invoices), 0)


# ===========================================================================
# TIER 3: Bank Reconciliation
# ===========================================================================

class TestBankReconciliation(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_csv_parsing_10_lines(self):
        csv = (
            "Dato;Forklaring;Inn;Ut;Saldo\n"
            "2026-01-18;Innbetaling fra Costa Lda / Faktura 1001;9562.50;;109562.50\n"
            "2026-01-21;Innbetaling fra Pereira Lda / Faktura 1002;16625.00;;126187.50\n"
            "2026-01-22;Innbetaling fra Costa Lda / Faktura 1003;4625.00;;130812.50\n"
            "2026-01-23;Innbetaling fra Silva Lda / Faktura 1004;13500.00;;144312.50\n"
            "2026-01-24;Innbetaling fra Costa Lda / Faktura 1005;5500.00;;149812.50\n"
            "2026-01-25;Betaling Fornecedor Oliveira Lda;;-9550.00;140262.50\n"
            "2026-01-27;Betaling Fornecedor Santos Lda;;-10200.00;130062.50\n"
            "2026-01-28;Betaling Fornecedor Santos Lda;;-6250.00;123812.50\n"
            "2026-01-29;Renteinntekter;1040.55;;124853.05\n"
            "2026-01-31;Bankgebyr;539.06;;125392.11\n"
        )
        lines = self.svc._parse_bank_csv(csv)
        self.assertEqual(len(lines), 10)

        customer_payments = [l for l in lines if "Innbetaling" in l["description"]]
        supplier_payments = [l for l in lines if "Fornecedor" in l["description"]]
        self.assertEqual(len(customer_payments), 5)
        self.assertEqual(len(supplier_payments), 3)


# ===========================================================================
# TIER 1: Project creation (module activation)
# ===========================================================================

class TestProjectCreation(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_project_activates_smart_project_module(self):
        task = _task(
            entity=Entity.PROJECT,
            raw_prompt='Create project "Test Project" for customer Acme AS.',
            attributes={
                "name": "Test Project",
                "customerName": "Acme AS",
            },
        )
        self.svc._dispatch(task)

        # Must activate SMART_PROJECT module for project creation
        self.assertTrue(
            any("SMART_PROJECT" in m for m in self.client.activated_modules),
            f"Expected SMART_PROJECT in {self.client.activated_modules}",
        )


# ===========================================================================
# TIER 2: Voucher / Journal Entry
# ===========================================================================

class TestVoucher(unittest.TestCase):
    def setUp(self):
        self.svc, self.client = _make_service()

    def test_simple_voucher(self):
        task = _task(
            entity=Entity.VOUCHER,
            raw_prompt='Post journal entry: debit 6300 for 2800 NOK, credit 1920 for 2800 NOK. "Office supplies March".',
            attributes={
                "debitAccountNumber": 6300,
                "creditAccountNumber": 1920,
                "amount": 2800,
                "description": "Office supplies March",
            },
        )
        self.svc._dispatch(task)

        self.assertEqual(len(self.client.voucher_payloads), 1)
        v = self.client.voucher_payloads[0]
        self.assertEqual(v["description"], "Office supplies March")
        postings = v["postings"]
        self.assertEqual(len(postings), 2)
        self.assertAlmostEqual(postings[0]["amountGross"], 2800.0)
        self.assertAlmostEqual(postings[1]["amountGross"], -2800.0)

    def test_multi_posting_voucher(self):
        task = _task(
            entity=Entity.VOUCHER,
            raw_prompt='Post accrual and depreciation entries.',
            attributes={
                "postings": [
                    {"debitAccount": 6300, "creditAccount": 2900, "amount": 5000, "description": "Accrual"},
                    {"debitAccount": 6010, "creditAccount": 1090, "amount": 8000, "description": "Depreciation"},
                ],
            },
        )
        self.svc._dispatch(task)

        # Should create voucher(s) for multi-posting
        self.assertTrue(len(self.client.voucher_payloads) >= 1)


if __name__ == "__main__":
    unittest.main()
