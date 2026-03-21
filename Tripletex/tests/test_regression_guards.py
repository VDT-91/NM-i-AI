from __future__ import annotations

import json
import types
import unittest
from datetime import date
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from tripletex_solver import api as api_module
from tripletex_solver.errors import TripletexAPIError
from tripletex_solver.models import Action, Entity, ParsedTask
from tripletex_solver.service import TripletexService
from tripletex_solver.tripletex_client import TripletexClient


class _DummyResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 201) -> None:
        self.status_code = status_code
        self._payload = payload or {"value": {"id": 1}}
        self.text = json.dumps(self._payload)
        self.content = self.text.encode("utf-8")

    def json(self) -> dict:
        return self._payload


class _RecordingSession:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {"value": {"id": 1}}
        self.calls: list[dict] = []

    def request(self, *, method: str, url: str, params=None, json=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "json": json,
                "timeout": timeout,
            }
        )
        return _DummyResponse(self.payload)


class TripletexClientVoucherSanitizingTest(unittest.TestCase):
    def _client(self) -> tuple[TripletexClient, _RecordingSession]:
        client = TripletexClient(base_url="https://example.test", session_token="token")
        session = _RecordingSession()
        client.session = session
        return client, session

    def test_voucher_request_strips_internal_keys(self) -> None:
        client, session = self._client()

        client.create_voucher(
            {
                "date": "2026-03-31",
                "description": "Monthly closing",
                "postings": [
                    {
                        "row": 1,
                        "account": {"id": 10},
                        "amountGross": 100.0,
                        "amountGrossCurrency": 100.0,
                        "_account_number": 6300,
                    }
                ],
            }
        )

        sent = session.calls[-1]["json"]
        self.assertNotIn("_account_number", sent["postings"][0])

    def test_voucher_request_strips_internal_keys_recursively(self) -> None:
        client, session = self._client()

        client.create_voucher(
            {
                "date": "2026-03-31",
                "description": "Incoming invoice",
                "_debug": True,
                "postings": [
                    {
                        "row": 1,
                        "account": {"id": 10, "_cache": "remove-me"},
                        "amountGross": 100.0,
                        "amountGrossCurrency": 100.0,
                    }
                ],
                "voucherType": {"id": 7, "_name": "Supplier invoice"},
            }
        )

        sent = session.calls[-1]["json"]
        self.assertNotIn("_debug", sent)
        self.assertNotIn("_cache", sent["postings"][0]["account"])
        self.assertNotIn("_name", sent["voucherType"])

    def test_non_voucher_request_keeps_internal_keys(self) -> None:
        client, session = self._client()

        client.create("/employee", {"firstName": "A", "_debug": True})

        sent = session.calls[-1]["json"]
        self.assertIn("_debug", sent)

    def test_search_supplier_invoices_uses_valid_supplier_invoice_fields(self) -> None:
        client = TripletexClient(base_url="https://example.test", session_token="token")
        session = _RecordingSession({"values": []})
        client.session = session

        client.search_supplier_invoices(
            supplier_id=12,
            invoice_date_from="2026-01-01",
            invoice_date_to="2026-02-01",
        )

        sent_params = session.calls[-1]["params"]
        self.assertEqual(sent_params["supplierId"], 12)
        self.assertIn("outstandingAmount", sent_params["fields"])
        self.assertIn("invoiceDueDate", sent_params["fields"])
        self.assertNotIn("amountOutstanding", sent_params["fields"])
        self.assertNotIn("dueDate", sent_params["fields"])


class _IncomingInvoiceVoucherClient:
    def __init__(self) -> None:
        self.vouchers: list[dict] = []

    def search_accounts_by_number(self, number: int) -> list[dict]:
        if number == 2400:
            return [{"id": 2400, "number": 2400}]
        if number == 2710:
            return [{"id": 2710, "number": 2710}]
        if number == 1920:
            return [{"id": 1920, "number": 1920}]
        return []

    def list(self, path: str, *, fields: str | None = None, params: dict | None = None) -> list[dict]:
        if path != "/ledger/voucherType":
            raise AssertionError(path)
        return [
            {"id": 1, "name": "Utgående faktura"},
            {"id": 2, "name": "Leverandørfaktura"},
            {"id": 3, "name": "Purring"},
        ]

    def create_voucher(self, payload: dict) -> dict:
        self.vouchers.append(payload)
        return {"id": 991}


class IncomingInvoiceVoucherRegressionTest(unittest.TestCase):
    def test_voucher_fallback_sets_supplier_invoice_type_and_invoice_metadata(self) -> None:
        client = _IncomingInvoiceVoucherClient()
        service = TripletexService(client)

        service._create_incoming_invoice_via_voucher(
            supplier={"id": 777, "name": "Komplett"},
            amount=5000.0,
            invoice_date=date(2026, 5, 4),
            description="USB-hub",
            debit_account_id=6810,
            invoice_number="INV-2026-42",
            vat_rate=25.0,
            department={"id": 12},
        )

        voucher = client.vouchers[-1]
        self.assertEqual(voucher["voucherType"]["id"], 2)
        self.assertEqual([posting["date"] for posting in voucher["postings"]], ["2026-05-04"] * 3)
        self.assertEqual(voucher["postings"][2]["supplier"]["id"], 777)
        self.assertEqual(voucher["vendorInvoiceNumber"], "INV-2026-42")
        self.assertTrue(all(posting["invoiceNumber"] == "INV-2026-42" for posting in voucher["postings"]))


class _SupplierBankClient:
    def __init__(self) -> None:
        self.suppliers: list[dict] = []
        self.supplier_invoices_by_supplier_id: dict[int, list[dict]] = {}
        self.paid_invoices: list[dict] = []
        self.vouchers: list[dict] = []
        self.raise_on_pay = False

    def search_suppliers(self, name: str) -> list[dict]:
        return list(self.suppliers)

    def search_supplier_invoices(self, *, supplier_id: int, invoice_date_from: str, invoice_date_to: str) -> list[dict]:
        return list(self.supplier_invoices_by_supplier_id.get(supplier_id, []))

    def pay_supplier_invoice(self, invoice_id: int, *, amount: float, payment_date: str) -> None:
        self.paid_invoices.append(
            {"invoice_id": invoice_id, "amount": amount, "payment_date": payment_date}
        )
        if self.raise_on_pay:
            raise TripletexAPIError("payment failed", status_code=422, response_text="failed")

    def search_accounts_by_number(self, number: int) -> list[dict]:
        if number == 2400:
            return [{"id": 2400, "number": 2400}]
        if number == 1920:
            return [{"id": 1920, "number": 1920}]
        return []

    def create_voucher(self, payload: dict) -> dict:
        self.vouchers.append(payload)
        return {"id": 3001}


class SupplierPaymentRegressionTest(unittest.TestCase):
    def test_fresh_supplier_uses_keyword_only_ensure_supplier_and_falls_back_to_voucher(self) -> None:
        client = _SupplierBankClient()
        service = TripletexService(client)
        ensured: list[tuple[str, str | None]] = []

        def _ensure_supplier(*, name: str, org_number: str | None = None) -> dict:
            ensured.append((name, org_number))
            return {"id": 777, "name": name}

        service._ensure_supplier = _ensure_supplier  # type: ignore[method-assign]

        service._reconcile_supplier_payment("Fresh Supplier AS", 1250.0, date(2026, 3, 21))

        self.assertEqual(ensured, [("Fresh Supplier AS", None)])
        self.assertEqual(len(client.vouchers), 1)
        self.assertEqual(client.vouchers[0]["postings"][0]["supplier"]["id"], 777)

    def test_existing_supplier_pays_matching_invoice(self) -> None:
        client = _SupplierBankClient()
        client.suppliers = [{"id": 10, "name": "Leverandor Exact Supplier"}]
        client.supplier_invoices_by_supplier_id = {10: [{"id": 200, "amountCurrencyOutstanding": 500.0}]}
        service = TripletexService(client)

        service._reconcile_supplier_payment("Exact Supplier", 500.0, date(2026, 3, 21))

        self.assertEqual(client.paid_invoices[0]["invoice_id"], 200)
        self.assertEqual(client.paid_invoices[0]["amount"], 500.0)
        self.assertFalse(client.vouchers)

    def test_existing_supplier_pays_matching_invoice_using_outstanding_amount_field(self) -> None:
        client = _SupplierBankClient()
        client.suppliers = [{"id": 10, "name": "Leverandor Exact Supplier"}]
        client.supplier_invoices_by_supplier_id = {10: [{"id": 201, "outstandingAmount": 625.0}]}
        service = TripletexService(client)

        service._reconcile_supplier_payment("Exact Supplier", 625.0, date(2026, 3, 21))

        self.assertEqual(client.paid_invoices[0]["invoice_id"], 201)
        self.assertEqual(client.paid_invoices[0]["amount"], 625.0)

    def test_duplicate_prefixed_suppliers_pick_matching_invoice_by_amount(self) -> None:
        client = _SupplierBankClient()
        client.suppliers = [
            {"id": 10, "name": "Fournisseur Bernard SARL"},
            {"id": 11, "name": "Fournisseur Leroy SARL"},
            {"id": 12, "name": "Fournisseur Leroy SARL"},
        ]
        client.supplier_invoices_by_supplier_id = {
            10: [{"id": 201, "amountCurrencyOutstanding": 5300.0}],
            11: [{"id": 202, "amountCurrencyOutstanding": 12550.0}],
            12: [{"id": 203, "amountCurrencyOutstanding": 6600.0}],
        }
        service = TripletexService(client)

        service._reconcile_supplier_payment("Leroy SARL", 6600.0, date(2026, 3, 21))

        self.assertEqual(client.paid_invoices[0]["invoice_id"], 203)
        self.assertEqual(client.paid_invoices[0]["amount"], 6600.0)

    def test_payment_failure_falls_back_to_voucher(self) -> None:
        client = _SupplierBankClient()
        client.suppliers = [{"id": 10, "name": "Leverandor Exact Supplier"}]
        client.supplier_invoices_by_supplier_id = {10: [{"id": 222, "amountCurrencyOutstanding": 700.0}]}
        client.raise_on_pay = True
        service = TripletexService(client)

        service._reconcile_supplier_payment("Exact Supplier", 700.0, date(2026, 3, 21))

        self.assertEqual(client.paid_invoices[0]["invoice_id"], 222)
        self.assertEqual(len(client.vouchers), 1)


class _CustomerBankClient:
    def __init__(self) -> None:
        self.payments: list[dict] = []

    def search_customers(self, name: str) -> list[dict]:
        return [
            {"id": 10002, "name": "Nilsen AS", "displayName": "Nilsen AS (10002)"},
            {"id": 10005, "name": "Nilsen AS", "displayName": "Nilsen AS (10005)"},
        ]

    def search_invoices(self, *, customer_id: int) -> list[dict]:
        mapping = {
            10002: [{"id": 4202, "invoiceNumber": 2, "amountCurrencyOutstanding": 20562.5}],
            10005: [{"id": 4205, "invoiceNumber": 5, "amountCurrencyOutstanding": 14625.0}],
        }
        return mapping[customer_id]

    def pay_invoice(self, invoice_id: int, *, payment_date: date, payment_type_id: int, paid_amount: float) -> None:
        self.payments.append(
            {
                "invoice_id": invoice_id,
                "payment_date": payment_date.isoformat(),
                "payment_type_id": payment_type_id,
                "paid_amount": paid_amount,
            }
        )


class CustomerPaymentRegressionTest(unittest.TestCase):
    def test_duplicate_customer_names_pick_invoice_specific_customer_and_keep_partial_amount(self) -> None:
        client = _CustomerBankClient()
        service = TripletexService(client)

        service._reconcile_customer_payment(
            "Nilsen AS",
            "1005",
            14625.0,
            date(2026, 1, 26),
            {"id": 17},
        )

        self.assertEqual(client.payments, [{
            "invoice_id": 4205,
            "payment_date": "2026-01-26",
            "payment_type_id": 17,
            "paid_amount": 14625.0,
        }])

    def test_partial_customer_payment_does_not_settle_full_outstanding(self) -> None:
        client = _CustomerBankClient()
        service = TripletexService(client)

        service._reconcile_customer_payment(
            "Nilsen AS",
            "1002",
            8225.0,
            date(2026, 1, 19),
            {"id": 17},
        )

        self.assertEqual(client.payments[0]["invoice_id"], 4202)
        self.assertEqual(client.payments[0]["paid_amount"], 8225.0)


class _EmployeeDetailClient:
    def __init__(self) -> None:
        self.update_calls: list[tuple[str, int, dict]] = []
        self.create_calls: list[tuple[str, dict]] = []
        self.details: list[dict] = [{"id": 7101}]
        self.occupation_codes: list[dict] = []

    def create(self, path: str, payload: dict) -> dict:
        if path == "/employee":
            return {"id": 99}
        if path == "/employee/employment/details":
            self.create_calls.append((path, payload))
            return {"id": 7102}
        raise AssertionError(path)

    def list(self, path: str, *, fields: str | None = None, params: dict | None = None) -> list[dict]:
        if path == "/employee/employment":
            return [{"id": 7001, "employmentDetails": {"id": 7101}}]
        if path == "/employee/employment/details":
            return list(self.details)
        if path == "/employee/employment/occupationCode":
            count = int((params or {}).get("count", len(self.occupation_codes) or 0))
            return list(self.occupation_codes[:count])
        return []

    def update(self, path: str, object_id: int, payload: dict) -> dict:
        self.update_calls.append((path, object_id, payload))
        return {"id": object_id}

    def update(self, path: str, object_id: int, payload: dict) -> dict:
        self.update_calls.append((path, object_id, payload))
        return {"id": object_id}


class EmployeeEmploymentDetailRegressionTest(unittest.TestCase):
    def _service(self) -> tuple[TripletexService, _EmployeeDetailClient]:
        client = _EmployeeDetailClient()
        service = TripletexService(client)
        service._ensure_department = lambda name: {"id": 12, "name": name}  # type: ignore[method-assign]
        service._build_address = lambda attrs: None  # type: ignore[method-assign]
        service._update_employment = lambda *args, **kwargs: None  # type: ignore[method-assign]
        return service, client

    def test_monthly_employee_detail_update_uses_valid_enums(self) -> None:
        service, client = self._service()
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.EMPLOYEE,
            raw_prompt="create employee",
            attributes={
                "firstName": "Camille",
                "lastName": "Moreau",
                "email": "camille.moreau@example.org",
                "startDate": "2026-04-23",
                "employmentForm": "permanent",
                "annualSalary": 860000,
            },
        )

        service._create_employee(task)

        payload = client.update_calls[-1][2]
        self.assertEqual(payload["employmentForm"], "PERMANENT")
        self.assertEqual(payload["employmentType"], "ORDINARY")
        self.assertEqual(payload["workingHoursScheme"], "NOT_SHIFT")
        self.assertEqual(payload["remunerationType"], "MONTHLY_WAGE")

    def test_hourly_employee_detail_update_uses_hourly_enum(self) -> None:
        service, client = self._service()
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.EMPLOYEE,
            raw_prompt="create employee",
            attributes={
                "firstName": "Nora",
                "lastName": "Hour",
                "email": "nora.hour@example.org",
                "hourlyWage": 450,
            },
        )

        service._create_employee(task)

        payload = client.update_calls[-1][2]
        self.assertEqual(payload["remunerationType"], "HOURLY_WAGE")

    def test_missing_employment_details_are_created(self) -> None:
        service, client = self._service()
        client.details = []
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.EMPLOYEE,
            raw_prompt="create employee",
            attributes={
                "firstName": "Daniel",
                "lastName": "Smith",
                "startDate": "2026-04-25",
                "employmentForm": "permanent",
                "percentageOfFullTimeEquivalent": 100.0,
                "annualSalary": 740000,
            },
        )

        service._create_employee(task)

        self.assertFalse(client.update_calls)
        path, payload = client.create_calls[-1]
        self.assertEqual(path, "/employee/employment/details")
        self.assertEqual(payload["employment"]["id"], 7001)
        self.assertEqual(payload["employmentForm"], "PERMANENT")
        self.assertEqual(payload["percentageOfFullTimeEquivalent"], 100.0)
        self.assertEqual(payload["annualSalary"], 740000.0)
        self.assertEqual(payload["remunerationType"], "MONTHLY_WAGE")



    def test_occupation_code_uses_exact_match_beyond_first_five_hits(self) -> None:
        service, client = self._service()
        client.details = []
        client.occupation_codes = [
            {"id": 1, "code": "7125115", "nameNO": "Wrong 1"},
            {"id": 2, "code": "2511102", "nameNO": "Wrong 2"},
            {"id": 3, "code": "8251111", "nameNO": "Wrong 3"},
            {"id": 4, "code": "9251111", "nameNO": "Wrong 4"},
            {"id": 5, "code": "1251111", "nameNO": "Wrong 5"},
            {"id": 999, "code": "2511", "nameNO": "Exact"},
        ]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.EMPLOYEE,
            raw_prompt="create employee",
            attributes={
                "firstName": "Daniel",
                "lastName": "Smith",
                "startDate": "2026-04-25",
                "employmentForm": "permanent",
                "percentageOfFullTimeEquivalent": 100.0,
                "annualSalary": 740000,
                "occupationCode": "2511",
            },
        )

        service._create_employee(task)

        payload = client.create_calls[-1][1]
        self.assertEqual(payload["occupationCode"], {"id": 999})

    def test_occupation_code_uses_derived_prefix_match_for_styrk_family(self) -> None:
        service, client = self._service()
        client.details = []
        client.occupation_codes = [
            {"id": 129, "code": "4133132", "nameNO": "Specific 2"},
            {"id": 130, "code": "4133131", "nameNO": "Specific 1"},
            {"id": 131, "code": "4133130", "nameNO": "Generic"},
        ]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.EMPLOYEE,
            raw_prompt="create employee",
            attributes={
                "firstName": "Alejandro",
                "lastName": "Sanchez",
                "startDate": "2026-04-16",
                "employmentForm": "permanent",
                "percentageOfFullTimeEquivalent": 100.0,
                "annualSalary": 470000,
                "occupationCode": "3313",
            },
        )

        service._create_employee(task)

        payload = client.create_calls[-1][1]
        self.assertEqual(payload["occupationCode"], {"id": 131})

    def test_occupation_code_skips_ambiguous_fuzzy_matches(self) -> None:
        service, client = self._service()
        client.details = []
        client.occupation_codes = [
            {"id": 129, "code": "7125115", "nameNO": "ARBEIDSLEDER (BYGG)"},
            {"id": 688, "code": "8251111", "nameNO": "BOKTRYKKERMESTER"},
            {"id": 777, "code": "9251111", "nameNO": "ANNEN TITTEL"},
        ]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.EMPLOYEE,
            raw_prompt="create employee",
            attributes={
                "firstName": "Daniel",
                "lastName": "Smith",
                "startDate": "2026-04-25",
                "employmentForm": "permanent",
                "percentageOfFullTimeEquivalent": 100.0,
                "annualSalary": 740000,
                "occupationCode": "2511",
            },
        )

        service._create_employee(task)

        payload = client.create_calls[-1][1]
        self.assertNotIn("occupationCode", payload)


class _ExistingEmployeeRoleClient:
    def __init__(self) -> None:
        self.employee = {
            "id": 18614487,
            "firstName": "Marit",
            "lastName": "Vik",
            "displayName": "Marit Vik",
            "email": "marit.vik@example.org",
            "userType": None,
        }
        self.update_calls: list[tuple[str, int, dict]] = []

    def search_employees(self, *, first_name: str | None = None, last_name: str | None = None, email: str | None = None) -> list[dict]:
        return [dict(self.employee)]

    def update(self, path: str, object_id: int, payload: dict) -> dict:
        self.update_calls.append((path, object_id, payload))
        self.employee.update(payload)
        return dict(self.employee)


class ExistingEmployeeRoleRegressionTest(unittest.TestCase):
    def test_standard_role_upgrades_existing_employee_without_access(self) -> None:
        client = _ExistingEmployeeRoleClient()
        service = TripletexService(client)

        employee = service._ensure_employee(
            name="Marit Vik",
            email="marit.vik@example.org",
            role="standard",
        )

        self.assertEqual(client.update_calls, [("/employee", 18614487, {"userType": "STANDARD"})])
        self.assertEqual(employee["userType"], "STANDARD")


class _VoucherCorrectionClient:
    def __init__(self) -> None:
        self.vouchers: list[dict] = []

    def search_ledger_postings(self, *, date_from: str, date_to: str, account_id=None, open_postings=None, supplier_id=None, customer_id=None) -> list[dict]:
        return [
            {"voucher": {"id": 501}, "account": {"number": 6540}, "amount": 13000.0},
            {"voucher": {"id": 501}, "account": {"number": 2400}, "amount": -13000.0},
        ]

    def search_accounts_by_number(self, number: int) -> list[dict]:
        return [{"id": number, "number": number, "name": f"Account {number}"}]

    def list(self, path: str, *, fields: str | None = None, params: dict | None = None) -> list[dict]:
        if path == "/ledger/voucherType":
            return []
        raise AssertionError(path)

    def create(self, path: str, payload: dict) -> dict:
        if path != "/ledger/voucher":
            raise AssertionError(path)
        self.vouchers.append(payload)
        return {"id": 9001}


class VoucherCorrectionRegressionTest(unittest.TestCase):
    def test_missing_vat_correction_infers_counter_account_from_original_voucher(self) -> None:
        client = _VoucherCorrectionClient()
        service = TripletexService(client)
        service._resolve_voucher_type = lambda: None  # type: ignore[method-assign]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.VOUCHER,
            raw_prompt=(
                "Vi har oppdaget feil i hovedboken for januar og februar 2026. "
                "en manglende MVA-linje (konto 6540, beløp ekskl. 13000 kr mangler MVA på konto 2710)."
            ),
            attributes={
                "description": "Korreksjon av feil i hovedbok for januar og februar 2026",
                "date": "2026-02-28",
                "postings": [
                    {
                        "debitAccount": 2710,
                        "creditAccount": 1920,
                        "amount": 3250,
                        "description": "Legge til manglende MVA (konto 6540)",
                    }
                ],
            },
        )

        service._create_voucher(task)

        voucher = client.vouchers[-1]
        self.assertEqual(voucher["postings"][0]["account"]["id"], 2710)
        self.assertEqual(voucher["postings"][1]["account"]["id"], 2400)

    def test_missing_vat_correction_supports_french_account_wording(self) -> None:
        class _FrenchVoucherCorrectionClient(_VoucherCorrectionClient):
            def search_ledger_postings(self, *, date_from: str, date_to: str, account_id=None, open_postings=None, supplier_id=None, customer_id=None) -> list[dict]:
                return [
                    {"voucher": {"id": 601}, "account": {"number": 6590}, "amount": 13000.0},
                    {"voucher": {"id": 601}, "account": {"number": 2400}, "amount": -13000.0},
                ]

        client = _FrenchVoucherCorrectionClient()
        service = TripletexService(client)
        service._resolve_voucher_type = lambda: None  # type: ignore[method-assign]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.VOUCHER,
            raw_prompt=(
                "Nous avons dÃ©couvert des erreurs dans le grand livre. "
                "une ligne de TVA manquante (compte 6590, montant HT 13000 NOK, TVA manquante sur compte 2710)."
            ),
            attributes={
                "description": "Correction d'erreurs",
                "date": "2026-02-28",
                "postings": [
                    {
                        "debitAccount": 2710,
                        "creditAccount": 1920,
                        "amount": 3250,
                        "description": "Correction: ligne de TVA manquante sur compte 2710",
                    }
                ],
            },
        )

        service._create_voucher(task)

        voucher = client.vouchers[-1]
        self.assertEqual(voucher["postings"][0]["account"]["id"], 2710)
        self.assertEqual(voucher["postings"][1]["account"]["id"], 2400)


class _YearEndPrepaidClient:
    def search_accounts_by_number(self, number: int) -> list[dict]:
        if number == 1700:
            return [{"id": 1700, "number": 1700, "name": "Forskuddsbetalt kostnad"}]
        if number == 6400:
            return [{"id": 6400, "number": 6400, "name": "Forsikring"}]
        return []

    def search_ledger_postings(self, *, date_from: str, date_to: str) -> list[dict]:
        return [
            {"voucher": {"id": 10}, "account": {"number": 1700}, "amount": -1200.0},
            {"voucher": {"id": 10}, "account": {"number": 6400}, "amount": 1200.0},
        ]


class _YearEndTaxClient:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict]] = []

    def search_ledger_postings(self, *, date_from: str, date_to: str) -> list[dict]:
        return [
            {"account": {"number": 3000}, "amount": -1000.0},
            {"account": {"number": 6300}, "amount": 200.0},
            {"account": {"number": 8400}, "amount": -300.0},
        ]

    def create(self, path: str, payload: dict) -> dict:
        self.created.append((path, payload))
        return {"id": 55}


class YearEndRegressionTest(unittest.TestCase):
    def test_prepaid_reversal_prefers_history_and_normalizes_description(self) -> None:
        client = _YearEndPrepaidClient()
        service = TripletexService(client)
        service._ensure_account = lambda number: {"id": number, "number": number}  # type: ignore[method-assign]
        postings = [
            {
                "account": {"id": 6300},
                "_account_number": 6300,
                "description": "Periodisering av forskuddsbetalte kostnader",
            },
            {
                "account": {"id": 1700},
                "_account_number": 1700,
                "description": "Periodisering av forskuddsbetalte kostnader",
            },
        ]

        service._correct_prepaid_expense_in_postings(
            postings,
            "Perform year-end closing and reverse prepaid expenses",
            voucher_date=date(2025, 12, 31),
        )

        self.assertEqual(postings[0]["_account_number"], 6400)
        self.assertEqual(postings[0]["description"], "Reversering forskuddsbetalte kostnader")
        self.assertEqual(postings[1]["description"], "Reversering forskuddsbetalte kostnader")

    def test_year_end_tax_excludes_special_accounts_from_taxable_result(self) -> None:
        client = _YearEndTaxClient()
        service = TripletexService(client)
        service._ensure_account = lambda number: {"id": number, "number": number}  # type: ignore[method-assign]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.VOUCHER,
            raw_prompt="Compute tax provision of 22 % on account 8700/2920 for year-end closing.",
            attributes={},
        )

        service._post_year_end_tax_provision(task, date(2025, 12, 31), None)

        _, payload = client.created[-1]
        self.assertEqual(payload["postings"][0]["amountGross"], 176.0)
        self.assertEqual(payload["postings"][1]["amountGross"], -176.0)

    def test_norwegian_year_end_prompt_creates_separate_vouchers(self) -> None:
        client = _YearEndVoucherClient()
        service = TripletexService(client)
        service._adjust_year_end_depreciation = lambda *args, **kwargs: None  # type: ignore[method-assign]
        service._correct_prepaid_expense_in_postings = lambda *args, **kwargs: None  # type: ignore[method-assign]
        service._post_salary_provision_if_missing = lambda *args, **kwargs: None  # type: ignore[method-assign]
        service._post_year_end_prepaid_reversal = lambda *args, **kwargs: None  # type: ignore[method-assign]
        service._post_year_end_tax_provision = lambda *args, **kwargs: None  # type: ignore[method-assign]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.VOUCHER,
            raw_prompt=(
                "Utf\u00f8r forenklet \u00e5rsoppgj\u00f8r for 2025. "
                "Bokf\u00f8r hver avskrivning som et eget bilag og reverser forskuddsbetalte kostnader."
            ),
            attributes={
                "description": "Forenklet \u00e5rsoppgj\u00f8r 2025",
                "date": "2025-12-31",
                "postings": [
                    {"debitAccount": 6010, "creditAccount": 1209, "amount": 21740.0, "description": "\u00c5rlig avskrivning IT-utstyr"},
                    {"debitAccount": 6010, "creditAccount": 1209, "amount": 39691.67, "description": "\u00c5rlig avskrivning Kontormaskiner"},
                    {"debitAccount": 6010, "creditAccount": 1209, "amount": 18490.0, "description": "\u00c5rlig avskrivning Kj\u00f8ret\u00f8y"},
                    {"debitAccount": 6300, "creditAccount": 1700, "amount": 53150.0, "description": "Reversering av forskuddsbetalte kostnader"},
                ],
            },
        )

        service._create_voucher(task)

        self.assertEqual(len(client.created_vouchers), 4)
        self.assertTrue(all(len(v["postings"]) == 2 for v in client.created_vouchers))


class _YearEndVoucherClient:
    def __init__(self) -> None:
        self.created_vouchers: list[dict] = []

    def search_accounts_by_number(self, number: int) -> list[dict]:
        return [{"id": number, "number": number, "name": f"Account {number}"}]

    def create(self, path: str, payload: dict) -> dict:
        if path == "/ledger/voucher":
            self.created_vouchers.append(payload)
            return {"id": len(self.created_vouchers)}
        if path == "/ledger/account":
            return {"id": payload["number"], "number": payload["number"]}
        raise AssertionError(path)




class _DummyEntry(types.SimpleNamespace):
    pass


class _DummyChallengeLog:
    def __init__(self) -> None:
        self.entries: list[_DummyEntry] = []

    def start(self, prompt: str, file_info: list[dict]) -> _DummyEntry:
        entry = _DummyEntry(
            id=f"test-{len(self.entries) + 1}",
            prompt=prompt,
            file_info=file_info,
            parsed_action=None,
            parsed_entity=None,
            parsed_attributes=None,
            attachment_text=None,
            result=None,
            error=None,
            duration_ms=None,
            api_calls=None,
        )
        self.entries.append(entry)
        return entry

    def finish(self, entry: _DummyEntry) -> None:
        return None


class _DummyAPIClient:
    def __init__(self, *, base_url: str, session_token: str) -> None:
        self.base_url = base_url
        self.session_token = session_token
        self._call_log: list[dict] = []


class _DummySuccessService:
    def __init__(self, client: _DummyAPIClient) -> None:
        self.client = client
        self.last_parsed_task = None
        self.last_attachment_text = None

    def execute(self, request, *, attachments_dir=None):
        self.last_parsed_task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.PRODUCT,
            raw_prompt=request.prompt,
            attributes={"name": "Regression Product"},
        )
        self.client._call_log.append({"method": "GET", "path": "/health", "status": 200})
        return self.last_parsed_task


class _DummyFailingService(_DummySuccessService):
    def execute(self, request, *, attachments_dir=None):
        self.last_parsed_task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.VOUCHER,
            raw_prompt=request.prompt,
            attributes={"description": "Broken run"},
        )
        raise RuntimeError("forced failure")


class LocalSolveEndpointRegressionTest(unittest.TestCase):
    def test_solve_endpoint_success_path_runs_locally(self) -> None:
        challenge_log = _DummyChallengeLog()
        with (
            patch.object(api_module, "CHALLENGE_LOG", challenge_log),
            patch.object(api_module, "TripletexClient", _DummyAPIClient),
            patch.object(api_module, "TripletexService", _DummySuccessService),
        ):
            client = TestClient(api_module.app)
            response = client.post(
                "/solve",
                json={
                    "prompt": "Create product Regression Product",
                    "tripletex_credentials": {
                        "base_url": "https://example.test",
                        "session_token": "token",
                    },
                    "files": [],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(challenge_log.entries[-1].result, "success")
        self.assertEqual(challenge_log.entries[-1].parsed_entity, "product")

    def test_solve_endpoint_error_path_keeps_request_local_and_records_failure(self) -> None:
        challenge_log = _DummyChallengeLog()
        with (
            patch.object(api_module, "CHALLENGE_LOG", challenge_log),
            patch.object(api_module, "TripletexClient", _DummyAPIClient),
            patch.object(api_module, "TripletexService", _DummyFailingService),
        ):
            client = TestClient(api_module.app)
            response = client.post(
                "/solve",
                json={
                    "prompt": "Broken prompt",
                    "tripletex_credentials": {
                        "base_url": "https://example.test",
                        "session_token": "token",
                    },
                    "files": [],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(challenge_log.entries[-1].result, "error")
        self.assertIn("RuntimeError: forced failure", challenge_log.entries[-1].error)


class _ProjectModuleClient:
    def __init__(self) -> None:
        self.activated: list[str] = []

    def activate_sales_module(self, module_name: str):
        self.activated.append(module_name)
        return None


class _ProjectManagerAccessClient:
    def __init__(self) -> None:
        self.updated: list[tuple[str, int, dict]] = []
        self.entitlements: list[tuple[int, str]] = []

    def update(self, path: str, object_id: int, payload: dict) -> dict:
        self.updated.append((path, object_id, payload))
        return {"id": object_id}

    def grant_entitlements(self, employee_id: int, template: str) -> None:
        self.entitlements.append((employee_id, template))


class _LedgerAnalysisFailureClient:
    def __init__(self) -> None:
        self.created_projects: list[dict] = []

    def activate_sales_module(self, module_name: str):
        return None

    def search_ledger_postings(self, *, date_from: str, date_to: str) -> list[dict]:
        if date_from.startswith("2026-01"):
            return [
                {"account": {"number": 5000, "name": "Salary"}, "amount": 1000.0},
                {"account": {"number": 6300, "name": "Rent"}, "amount": 500.0},
                {"account": {"number": 7000, "name": "Fuel"}, "amount": 250.0},
            ]
        return [
            {"account": {"number": 5000, "name": "Salary"}, "amount": 2200.0},
            {"account": {"number": 6300, "name": "Rent"}, "amount": 1500.0},
            {"account": {"number": 7000, "name": "Fuel"}, "amount": 900.0},
        ]

    def create(self, path: str, payload: dict) -> dict:
        if path == "/project":
            self.created_projects.append(payload)
            raise TripletexAPIError(
                "POST /project failed with 422: Du mangler tilgang til ÃƒÂ¥ opprette nye prosjekter.",
                status_code=422,
                response_text="permission denied",
            )
        if path == "/activity":
            return {"id": 1}
        raise AssertionError(path)


class ProjectRegressionTest(unittest.TestCase):
    def test_activate_project_module_tries_all_candidates(self) -> None:
        client = _ProjectModuleClient()
        service = TripletexService(client)

        service._activate_project_module()

        self.assertEqual(
            client.activated,
            ["SMART_PROJECT", "SMART", "KOMPLETT", "PROJECT", "PROSJEKT"],
        )

    def test_project_manager_access_upgrades_null_user_type(self) -> None:
        client = _ProjectManagerAccessClient()
        service = TripletexService(client)
        employee = {
            "id": 123,
            "firstName": "Project",
            "lastName": "Manager",
            "userType": None,
            "email": "",
        }

        service._ensure_project_manager_access(employee)

        self.assertEqual(client.updated[0][0], "/employee")
        self.assertEqual(client.updated[0][1], 123)
        self.assertEqual(client.updated[0][2]["userType"], "STANDARD")
        self.assertIn("@placeholder.example.com", client.updated[0][2]["email"])
        self.assertEqual(client.entitlements, [(123, "ALL_PRIVILEGES")])

    def test_ledger_analysis_raises_when_zero_projects_created(self) -> None:
        client = _LedgerAnalysisFailureClient()
        service = TripletexService(client)
        service._resolve_project_manager = lambda: {"id": 456, "userType": "STANDARD", "email": "pm@example.org"}  # type: ignore[method-assign]
        service._ensure_project_manager_access = lambda employee: None  # type: ignore[method-assign]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.PROJECT,
            raw_prompt=(
                "Total costs increased significantly from January to February 2026. "
                "Analyze the general ledger and identify the three expense accounts with the "
                "largest increase in amount. Create an internal project for each of the three "
                "accounts using the account name. Also create an activity for each project."
            ),
            attributes={},
        )

        with self.assertRaises(TripletexAPIError):
            service._create_projects_from_ledger_analysis(task)

        self.assertEqual(len(client.created_projects), 3)


class _SalaryFallbackClient:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def activate_sales_module(self, module_name: str):
        return None

    def search_salary_types(self) -> list[dict]:
        return []

    def create_salary_transaction(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"id": 701}


class SalaryTransactionRegressionTest(unittest.TestCase):
    def test_salary_transaction_still_sends_payslip_when_salary_types_are_empty(self) -> None:
        client = _SalaryFallbackClient()
        service = TripletexService(client)
        service._ensure_employee = lambda name, email=None, role=None: {"id": 42, "name": name, "email": email}  # type: ignore[method-assign]
        service._ensure_employee_has_date_of_birth = lambda employee_id: None  # type: ignore[method-assign]
        service._ensure_employment = lambda employee_id, start_date=None: {"id": 77}  # type: ignore[method-assign]
        service._update_employment = lambda employee_id, start_date=None, salary_attrs=None: None  # type: ignore[method-assign]

        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.SALARY_TRANSACTION,
            raw_prompt="Køyr løn for Eirik Lunde denne månaden med bonus.",
            attributes={
                "employeeName": "Eirik Lunde",
                "employeeEmail": "eirik.lunde@example.org",
                "monthlySalary": 39100,
                "bonus": 8200,
            },
        )

        service._create_salary_transaction(task)

        payload = client.payloads[0]
        self.assertIn("payslips", payload)
        self.assertEqual(payload["payslips"][0]["employee"]["id"], 42)
        self.assertNotIn("specifications", payload["payslips"][0])


class _ReceiptSelectionClient:
    pass


class ReceiptSelectionRegressionTest(unittest.TestCase):
    def test_receipt_prompt_for_single_item_overrides_total_receipt_amount(self) -> None:
        client = _ReceiptSelectionClient()
        service = TripletexService(client)
        service.last_attachment_text = (
            "[Attachment: kvittering_es_02.pdf]\n"
            "Peppes Pizza\n"
            "Kundemøte lunsj 14050.00 kr\n"
            "Kontorrekvisita 330.00 kr\n"
            "Totalt: 14380.00 kr\n"
            "herav MVA 25%: 3595.00 kr\n"
        )
        captured: dict[str, Any] = {}
        service._ensure_supplier = lambda name, org_number=None: {"id": 500, "name": name}  # type: ignore[method-assign]
        service._ensure_account = lambda number: {"id": number, "number": number}  # type: ignore[method-assign]
        service._ensure_department = lambda name: {"id": 9, "name": name}  # type: ignore[method-assign]

        service._create_incoming_invoice_via_api = lambda **kwargs: captured.update(kwargs)  # type: ignore[method-assign]
        service._create_incoming_invoice_via_voucher = lambda **kwargs: (_ for _ in ()).throw(AssertionError("voucher fallback should not be used"))  # type: ignore[method-assign]

        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.INCOMING_INVOICE,
            raw_prompt=(
                "Necesitamos el gasto de Kundemøte lunsj de este recibo registrado "
                "en el departamento Drift. Usa la cuenta de gastos correcta y asegura "
                "el tratamiento correcto del IVA."
            ),
            attributes={
                "supplierName": "Peppes Pizza",
                "organizationNumber": "842689562",
                "invoiceDate": "2026-04-26",
                "totalAmountIncludingVat": 14380.0,
                "vatRate": 25,
                "debitAccountNumber": 6810,
                "departmentName": "Drift",
                "description": "Kundemøte lunsj og Kontorrekvisita",
            },
        )

        service._create_incoming_invoice(task)

        self.assertEqual(captured["description"], "Kundemøte lunsj")
        self.assertEqual(captured["amount_incl_vat"], 14050.0)


class _IncomingInvoiceApiPreferenceClient:
    pass


class IncomingInvoiceApiRegressionTest(unittest.TestCase):
    def _task(self) -> ParsedTask:
        return ParsedTask(
            action=Action.CREATE,
            entity=Entity.INCOMING_INVOICE,
            raw_prompt=(
                "Me har motteke faktura INV-2026-9559 frÃ¥ leverandÃ¸ren Elvdal AS "
                "(org.nr 935041740) pÃ¥ 36600 kr inklusiv MVA. BelÃ¸pet gjeld "
                "kontortenester (konto 7100). Registrer leverandÃ¸rfakturaen."
            ),
            attributes={
                "supplierName": "Elvdal AS",
                "organizationNumber": "935041740",
                "invoiceNumber": "INV-2026-9559",
                "invoiceDate": "2026-03-21",
                "dueDate": "2026-04-20",
                "totalAmountIncludingVat": 36600.0,
                "vatRate": 25,
                "debitAccountNumber": 7100,
                "description": "kontortenester",
            },
        )

    def test_incoming_invoice_prefers_api_before_voucher_fallback(self) -> None:
        client = _IncomingInvoiceApiPreferenceClient()
        service = TripletexService(client)
        service._ensure_supplier = lambda name, org_number=None: {"id": 10, "name": name}  # type: ignore[method-assign]
        service._ensure_account = lambda number: {"id": number, "number": number}  # type: ignore[method-assign]
        seen: list[str] = []

        service._create_incoming_invoice_via_api = lambda **kwargs: seen.append("api")  # type: ignore[method-assign]
        service._create_incoming_invoice_via_voucher = lambda **kwargs: seen.append("voucher")  # type: ignore[method-assign]

        service._create_incoming_invoice(self._task())

        self.assertEqual(seen, ["api"])

    def test_incoming_invoice_falls_back_to_voucher_on_forbidden_api(self) -> None:
        client = _IncomingInvoiceApiPreferenceClient()
        service = TripletexService(client)
        service._ensure_supplier = lambda name, org_number=None: {"id": 10, "name": name}  # type: ignore[method-assign]
        service._ensure_account = lambda number: {"id": number, "number": number}  # type: ignore[method-assign]
        seen: list[str] = []

        def _fail_api(**kwargs):
            raise TripletexAPIError("forbidden", status_code=403)

        service._create_incoming_invoice_via_api = _fail_api  # type: ignore[method-assign]
        service._create_incoming_invoice_via_voucher = lambda **kwargs: seen.append("voucher")  # type: ignore[method-assign]

        service._create_incoming_invoice(self._task())

        self.assertEqual(seen, ["voucher"])


class _SalaryProvisionClient:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict]] = []

    def create(self, path: str, payload: dict) -> dict:
        self.created.append((path, payload))
        return {"id": 1}


class SalaryProvisionRegressionTest(unittest.TestCase):
    def test_month_end_salary_provision_does_not_invent_amount_from_other_postings(self) -> None:
        client = _SalaryProvisionClient()
        service = TripletexService(client)
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.VOUCHER,
            raw_prompt=(
                "Führen Sie den Monatsabschluss für März 2026 durch. "
                "Buchen Sie die Rechnungsabgrenzung (7550 NOK pro Monat von Konto 1700 auf Aufwand). "
                "Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten "
                "228450 NOK und Nutzungsdauer 9 Jahre (lineare Abschreibung auf Konto 6010). "
                "Überprüfen Sie, ob die Saldenbilanz null ergibt. "
                "Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, "
                "Haben aufgelaufene Gehälter Konto 2900)."
            ),
            attributes={
                "postings": [
                    {"debitAccount": "6300", "creditAccount": "1700", "amount": 7550},
                    {"debitAccount": "6010", "creditAccount": "1290", "amount": 2115.28},
                ]
            },
        )

        service._post_salary_provision_if_missing(task, [], date(2026, 3, 31), None)

        self.assertEqual(client.created, [])


class _SalaryVoucherFallbackClient:
    def __init__(self) -> None:
        self.vouchers: list[dict] = []

    def search_accounts_by_number(self, number: int) -> list[dict]:
        return [{"id": number, "number": number, "name": f"Account {number}"}]

    def create_voucher(self, payload: dict) -> dict:
        self.vouchers.append(payload)
        return {"id": 999}


class SalaryVoucherFallbackRegressionTest(unittest.TestCase):
    def test_salary_like_voucher_prompt_creates_manual_salary_voucher(self) -> None:
        client = _SalaryVoucherFallbackClient()
        service = TripletexService(client)
        service._resolve_voucher_type = lambda: None  # type: ignore[method-assign]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.VOUCHER,
            raw_prompt=(
                "KjÃ¸r lÃ¸nn for Erik Nilsen denne mÃ¥neden. "
                "GrunnlÃ¸nn er 53350 kr og bonus er 11050 kr."
            ),
            attributes={
                "employeeName": "Erik Nilsen",
                "baseSalary": 53350,
                "bonus": 11050,
            },
        )

        service._create_voucher(task)

        payload = client.vouchers[0]
        # First posting: gross salary on 5000
        self.assertEqual(payload["postings"][0]["account"]["id"], 5000)
        self.assertEqual(payload["postings"][0]["amountGross"], 64400.0)
        # Verify all postings sum to zero (balanced voucher)
        total = sum(p["amountGross"] for p in payload["postings"])
        self.assertAlmostEqual(total, 0.0, places=2)


class _TravelExpenseUpgradeClient:
    def __init__(self) -> None:
        self.updated: list[tuple[str, int, dict]] = []
        self.travel_expenses: list[dict] = []

    def activate_sales_module(self, module_name: str):
        return None

    def update(self, path: str, object_id: int, payload: dict) -> dict:
        self.updated.append((path, object_id, payload))
        return {"id": object_id, **payload}

    def create_travel_expense(self, payload: dict) -> dict:
        self.travel_expenses.append(payload)
        return {"id": 901}


class TravelExpenseUpgradeRegressionTest(unittest.TestCase):
    def test_existing_employee_without_access_is_explicitly_upgraded_before_travel_expense(self) -> None:
        client = _TravelExpenseUpgradeClient()
        service = TripletexService(client)
        service._ensure_employee = lambda name, email=None, role=None: {  # type: ignore[method-assign]
            "id": 77,
            "name": name,
            "userType": None,
            "email": "",
        }

        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.TRAVEL_EXPENSE,
            raw_prompt="Registrer en reiseregning for Paul Hoffmann til Bergen.",
            attributes={
                "employeeName": "Paul Hoffmann",
                "employeeEmail": "paul.hoffmann@example.org",
                "departureDate": "2026-03-18",
                "returnDate": "2026-03-18",
                "destination": "Bergen",
                "purpose": "Kundemote Bergen",
            },
        )

        service._create_travel_expense(task)

        self.assertEqual(client.updated[0][0], "/employee")
        self.assertEqual(client.updated[0][1], 77)
        self.assertEqual(client.updated[0][2]["userType"], "STANDARD")
        self.assertEqual(client.travel_expenses[0]["employee"]["id"], 77)


class _ReceiptVoucherClient:
    def __init__(self) -> None:
        self.vouchers: list[dict] = []

    def create(self, path: str, payload: dict) -> dict:
        if path != "/ledger/voucher":
            raise AssertionError(path)
        self.vouchers.append(payload)
        return {"id": 501}


class ReceiptVoucherRegressionTest(unittest.TestCase):
    def test_receipt_split_postings_build_balanced_voucher_for_selected_line_item(self) -> None:
        client = _ReceiptVoucherClient()
        service = TripletexService(client)
        service._ensure_account = lambda number: {"id": number, "number": number}  # type: ignore[method-assign]
        service._resolve_voucher_type = lambda: None  # type: ignore[method-assign]
        service.last_attachment_text = (
            "[Attachment: kvittering_no_01.pdf]\n"
            "Storm Elektro\n"
            "Mus 10100.00 kr\n"
            "Kaffe kunderelasjon 490.00 kr\n"
            "Kaffe kunderelasjon 300.00 kr\n"
            "Totalt: 10890.00 kr\n"
            "herav MVA 25%: 2722.50 kr\n"
        )

        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.VOUCHER,
            raw_prompt="Bokfor Mus fra denne kvitteringen mot bank.",
            attributes={
                "date": "2026-03-21",
                "description": "Mus fra kvittering",
                "postings": [
                    {"debitAccount": 6540, "amount": 8080.0, "description": "Mus"},
                    {"debitAccount": 6900, "amount": 632.0, "description": "Kaffe kunderelasjon"},
                    {"debitAccount": 2710, "amount": 2178.0, "description": "MVA"},
                    {"creditAccount": 1920, "amount": 10890.0, "description": "Bank"},
                ],
            },
        )

        service._create_voucher(task)

        voucher = client.vouchers[0]
        self.assertEqual(len(voucher["postings"]), 3)
        self.assertEqual(voucher["postings"][0]["account"]["id"], 6540)
        self.assertEqual(voucher["postings"][0]["description"], "Mus")
        self.assertEqual(voucher["postings"][0]["amountGross"], 8080.0)
        self.assertEqual(voucher["postings"][1]["account"]["id"], 2710)
        self.assertEqual(voucher["postings"][1]["amountGross"], 2020.0)
        self.assertEqual(voucher["postings"][2]["account"]["id"], 1920)
        self.assertEqual(voucher["postings"][2]["amountGross"], -10100.0)


class ContainedOccupationCodeRegressionTest(unittest.TestCase):
    def test_occupation_code_prefers_unique_generic_contained_family_match(self) -> None:
        service, client = EmployeeEmploymentDetailRegressionTest()._service()
        client.details = []
        client.occupation_codes = [
            {"id": 401, "code": "3241111", "nameNO": "Specific 1"},
            {"id": 402, "code": "3241112", "nameNO": "Specific 2"},
            {"id": 403, "code": "3241110", "nameNO": "Generic"},
        ]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.EMPLOYEE,
            raw_prompt="create employee",
            attributes={
                "firstName": "Anne",
                "lastName": "Analyst",
                "startDate": "2026-05-01",
                "employmentForm": "permanent",
                "percentageOfFullTimeEquivalent": 100.0,
                "annualSalary": 810000,
                "occupationCode": "2411",
            },
        )

        service._create_employee(task)

        payload = client.create_calls[-1][1]
        self.assertEqual(payload["occupationCode"], {"id": 403})


class ReceiptPromptSelectionRegressionTest(unittest.TestCase):
    def test_hyphenated_usb_item_overrides_wrong_parser_description(self) -> None:
        client = _ReceiptSelectionClient()
        service = TripletexService(client)
        service.last_attachment_text = (
            "[Attachment: kvittering_de_01.pdf]\n"
            "Komplett\n"
            "USB-hub 14550.00 kr\n"
            "Kontorrekvisita 250.00 kr\n"
            "Totalt: 14800.00 kr\n"
            "herav MVA 25%: 3700.00 kr\n"
        )
        captured: dict[str, Any] = {}
        service._ensure_supplier = lambda name, org_number=None: {"id": 77, "name": name}  # type: ignore[method-assign]
        service._ensure_account = lambda number: {"id": number, "number": number}  # type: ignore[method-assign]
        service._ensure_department = lambda name: {"id": 5, "name": name}  # type: ignore[method-assign]
        service._create_incoming_invoice_via_api = lambda **kwargs: captured.update(kwargs)  # type: ignore[method-assign]
        service._create_incoming_invoice_via_voucher = lambda **kwargs: (_ for _ in ()).throw(AssertionError("voucher fallback should not be used"))  # type: ignore[method-assign]

        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.INCOMING_INVOICE,
            raw_prompt=(
                "Wir benotigen die USB-hub-Ausgabe aus dieser Quittung in der Abteilung Lager. "
                "Verwenden Sie das richtige Aufwandskonto und stellen Sie die korrekte MwSt.-Behandlung sicher."
            ),
            attributes={
                "supplierName": "Komplett",
                "organizationNumber": "801817378",
                "invoiceDate": "2026-02-17",
                "totalAmountIncludingVat": 14800.0,
                "vatRate": 25,
                "debitAccountNumber": 6800,
                "departmentName": "Lager",
                "description": "Kontorrekvisita",
            },
        )

        service._create_incoming_invoice(task)

        self.assertEqual(captured["description"], "USB-hub")
        self.assertEqual(captured["amount_incl_vat"], 14550.0)


class _MonthEndVoucherSanitizingClient:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict]] = []

    def search_accounts_by_number(self, number: int) -> list[dict]:
        return [{"id": number, "number": number, "name": f"Account {number}"}]

    def create(self, path: str, payload: dict) -> dict:
        self.created.append((path, payload))
        return {"id": len(self.created)}


class MonthEndVoucherSanitizingRegressionTest(unittest.TestCase):
    def test_month_end_voucher_strips_unresolved_salary_provision_from_llm_output(self) -> None:
        client = _MonthEndVoucherSanitizingClient()
        service = TripletexService(client)
        service._resolve_voucher_type = lambda: None  # type: ignore[method-assign]
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.VOUCHER,
            raw_prompt=(
                "Utfør månedsavslutning for mars 2026. Periodiser forskuddsbetalt kostnad "
                "(6450 kr per måned fra konto 1700 til kostkonto). Bokfør månedlig avskrivning "
                "for et driftsmiddel med anskaffelseskost 254100 kr og levetid 8 år "
                "(lineær avskrivning til konto 6020). Kontroller at saldobalansen går i null. "
                "Bokfør også en lønnsavsetning (debet lønnskostnad konto 5000, kredit påløpt lønn konto 2900)."
            ),
            attributes={
                "description": "Månedsavslutning mars 2026",
                "date": "2026-03-31",
                "postings": [
                    {"debitAccount": 6300, "creditAccount": 1700, "amount": 6450, "description": "Periodisering"},
                    {"debitAccount": 6020, "creditAccount": 1290, "amount": 2646.88, "description": "Avskrivning"},
                    {"debitAccount": 5000, "creditAccount": 2900, "amount": 50000, "description": "Lønnsavsetning"},
                ],
            },
        )

        service._create_voucher(task)

        self.assertEqual(len(client.created), 1)
        payload = client.created[0][1]
        accounts = [posting["account"]["id"] for posting in payload["postings"]]
        self.assertEqual(accounts, [6300, 1700, 6020, 1290])


class _SalaryTransactionForbiddenClient:
    def __init__(self) -> None:
        self.vouchers: list[dict] = []

    def activate_sales_module(self, module_name: str):
        return None

    def search_salary_types(self) -> list[dict]:
        return []

    def create_salary_transaction(self, payload: dict) -> dict:
        raise TripletexAPIError("forbidden", status_code=403)

    def search_accounts_by_number(self, number: int) -> list[dict]:
        return [{"id": number, "number": number, "name": f"Account {number}"}]

    def create_voucher(self, payload: dict) -> dict:
        self.vouchers.append(payload)
        return {"id": 77}


class SalaryTransactionForbiddenFallbackRegressionTest(unittest.TestCase):
    def test_salary_transaction_forbidden_falls_back_to_salary_voucher(self) -> None:
        client = _SalaryTransactionForbiddenClient()
        service = TripletexService(client)
        service._ensure_employee = lambda name, email=None, role=None: {"id": 42, "name": name, "email": email}  # type: ignore[method-assign]
        service._ensure_employee_has_date_of_birth = lambda employee_id: None  # type: ignore[method-assign]
        service._ensure_employment = lambda employee_id, start_date=None: {"id": 77}  # type: ignore[method-assign]
        service._update_employment = lambda employee_id, start_date=None, salary_attrs=None: None  # type: ignore[method-assign]

        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.SALARY_TRANSACTION,
            raw_prompt="Ejecute la nómina de Fernando García para este mes con bonificación.",
            attributes={
                "employeeName": "Fernando García",
                "employeeEmail": "fernando.garcia@example.org",
                "baseSalary": 55350,
                "bonus": 14600,
            },
        )

        service._create_salary_transaction(task)

        voucher = client.vouchers[0]
        # First posting: gross salary on 5000
        self.assertEqual(voucher["postings"][0]["account"]["id"], 5000)
        self.assertEqual(voucher["postings"][0]["amountGross"], 69950.0)
        # Verify all postings sum to zero (balanced voucher)
        total = sum(p["amountGross"] for p in voucher["postings"])
        self.assertAlmostEqual(total, 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
