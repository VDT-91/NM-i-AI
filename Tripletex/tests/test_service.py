from __future__ import annotations

import unittest

from tripletex_solver.models import SolveRequest, TripletexCredentials
from tripletex_solver.service import TripletexService


class FakeTripletexClient:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict, dict | None]] = []
        self.order_payloads: list[dict] = []
        self.invoice_payloads: list[tuple[dict, bool]] = []
        self.payment_calls: list[dict] = []
        self.credit_note_calls: list[dict] = []
        self.travel_expense_payloads: list[dict] = []
        self.travel_cost_payloads: list[dict] = []
        self.updated_accounts: list[tuple[int, dict]] = []
        self.updated_entities: list[tuple[str, int, dict]] = []
        self.deleted_entities: list[tuple[str, int]] = []

    def create(self, path: str, payload: dict, params: dict | None = None) -> dict:
        self.created.append((path, payload, params))
        if path == "/customer":
            return {"id": 101, **payload}
        if path == "/department":
            return {"id": 151, **payload}
        if path == "/employee":
            return {"id": 201, **payload}
        if path == "/project":
            return {"id": 401, **payload}
        if path == "/product":
            return {"id": 301, **payload}
        if path == "/supplier":
            return {"id": 901, **payload}
        if path == "/ledger/voucher":
            return {"id": 608817050, **payload}
        raise AssertionError(f"Unexpected create path: {path}")

    def search_customers(self, *, name=None, email=None, phone=None) -> list[dict]:
        if name == "Existing Customer":
            return [{"id": 111, "name": "Existing Customer", "email": "existing.customer@example.org"}]
        return []

    def search_employees(self, *, first_name=None, last_name=None, email=None) -> list[dict]:
        if first_name is None and last_name is None and email is None:
            return [{"id": 211, "displayName": "Existing Project Manager"}]
        return []

    def search_departments(self, *, name: str) -> list[dict]:
        if name in {"Sales", "Operations"}:
            return [{"id": 151 + len(name), "name": name, "isInactive": False}]
        return []

    def search_products(self, *, name=None, product_number=None) -> list[dict]:
        if name == "Support Retainer" or product_number == "SUP-001":
            return [
                {
                    "id": 321,
                    "name": "Support Retainer",
                    "number": "SUP-001",
                    "priceExcludingVatCurrency": 2500.0,
                    "priceIncludingVatCurrency": 3125.0,
                }
            ]
        if name == "Consulting":
            return [
                {
                    "id": 322,
                    "name": "Consulting",
                    "number": "CONS-1",
                    "priceExcludingVatCurrency": 1500.0,
                    "priceIncludingVatCurrency": 1875.0,
                }
            ]
        return []

    def search_projects(self, *, name=None, customer_id=None) -> list[dict]:
        if name == "Website Refresh":
            return [{"id": 411, "name": "Website Refresh"}]
        if name == "Legacy Project":
            return [{"id": 412, "name": "Legacy Project"}]
        return []

    def search_vat_types(self, *, number=None, type_of_vat=None) -> list[dict]:
        return [{"id": 25, "number": "3", "name": "High VAT"}]

    def search_accounts(self, *, number=None, is_bank_account=None) -> list[dict]:
        if number == 1920:
            return [
                {
                    "id": 901,
                    "number": 1920,
                    "name": "Bankinnskudd",
                    "isBankAccount": True,
                    "isInvoiceAccount": True,
                    "bankAccountNumber": "",
                }
            ]
        if is_bank_account:
            return [
                {
                    "id": 901,
                    "number": 1920,
                    "name": "Bankinnskudd",
                    "isBankAccount": True,
                    "isInvoiceAccount": True,
                    "bankAccountNumber": "",
                }
            ]
        return []

    def create_order(self, payload: dict) -> dict:
        self.order_payloads.append(payload)
        return {"id": 301}

    def create_invoice(self, payload: dict, *, send_to_customer: bool = False) -> dict:
        self.invoice_payloads.append((payload, send_to_customer))
        return {"id": 401}

    def get_invoice(self, invoice_id: int) -> dict:
        return {
            "id": invoice_id,
            "invoiceNumber": str(invoice_id),
            "amountCurrencyOutstanding": 1500.0,
        }

    def search_invoices(self, *, customer_id=None, invoice_number=None) -> list[dict]:
        if customer_id == 111:
            return [
                {
                    "id": 503,
                    "invoiceNumber": "3003",
                    "amount": 15000.0,
                    "amountCurrencyOutstanding": 15000.0,
                }
            ]
        if invoice_number == "1001":
            return [
                {
                    "id": 501,
                    "invoiceNumber": "1001",
                    "amountCurrencyOutstanding": 1500.0,
                }
            ]
        if invoice_number == "2002":
            return [
                {
                    "id": 502,
                    "invoiceNumber": "2002",
                    "amountCurrencyOutstanding": 0.0,
                }
            ]
        return []

    def search_payment_types(self, *, query: str) -> list[dict]:
        if query in {"Betalt til bank", "bank"}:
            return [{"id": 601, "description": "Betalt til bank"}]
        return []

    def search_travel_payment_types(self, *, query=None) -> list[dict]:
        return [{"id": 701, "description": "Corporate card"}]

    def search_travel_cost_categories(self, *, query=None) -> list[dict]:
        return [{"id": 801, "description": "Travel"}]

    def pay_invoice(
        self,
        invoice_id: int,
        *,
        payment_date,
        payment_type_id: int,
        paid_amount: float,
        paid_amount_currency: float | None = None,
    ) -> dict:
        self.payment_calls.append(
            {
                "invoice_id": invoice_id,
                "payment_date": payment_date,
                "payment_type_id": payment_type_id,
                "paid_amount": paid_amount,
                "paid_amount_currency": paid_amount_currency,
            }
        )
        return {"id": invoice_id}

    def create_credit_note(
        self,
        invoice_id: int,
        *,
        credit_note_date,
        comment: str | None = None,
        send_to_customer: bool = False,
    ) -> dict:
        self.credit_note_calls.append(
            {
                "invoice_id": invoice_id,
                "credit_note_date": credit_note_date,
                "comment": comment,
                "send_to_customer": send_to_customer,
            }
        )
        return {"id": 901}

    def create_travel_expense(self, payload: dict) -> dict:
        self.travel_expense_payloads.append(payload)
        return {"id": 1001}

    def create_travel_cost(self, payload: dict) -> dict:
        self.travel_cost_payloads.append(payload)
        return {"id": 1002}

    def update_account(self, account_id: int, payload: dict) -> dict:
        self.updated_accounts.append((account_id, payload))
        return {"id": account_id, **payload}

    def update(self, path: str, entity_id: int, payload: dict) -> dict:
        self.updated_entities.append((path, entity_id, payload))
        return {"id": entity_id, **payload}

    def delete(self, path: str, entity_id: int) -> None:
        self.deleted_entities.append((path, entity_id))

    def search_contacts(self, *, customer_id=None, email=None) -> list[dict]:
        return []

    def create_contact(self, payload: dict) -> dict:
        self.created.append(("/contact", payload, None))
        return {"id": 501, **payload}

    def search_suppliers(self, *, name=None, email=None) -> list[dict]:
        if name == "Existing Supplier":
            return [{"id": 901, "name": "Existing Supplier", "email": "info@existing.no"}]
        return []

    def search_voucher_types(self, *, name=None) -> list[dict]:
        return [{"id": 9811349, "name": "Leverandørfaktura"}]

    def search_accounts_by_number(self, number: int) -> list[dict]:
        return [{"id": 426260000 + number, "number": number, "name": f"Account {number}"}]

    def search_activities(self, *, name=None) -> list[dict]:
        if name == "Administrasjon":
            return [{"id": 5604365, "name": "Administrasjon", "isDisabled": False}]
        return [{"id": 5604365, "name": "Administrasjon", "isDisabled": False}]

    def create_timesheet_entry(self, payload: dict) -> dict:
        self.created.append(("/timesheet/entry", payload, None))
        return {"id": 175904025, **payload}

    def get(self, path: str, *, fields: str = "", params: dict | None = None) -> dict:
        return {"id": 1}

    def invoice_order(self, order_id: int, *, invoice_date, send_to_customer: bool = False) -> dict:
        return {"id": 401, "amountCurrencyOutstanding": 1875.0}

    def update_travel_expense(self, expense_id: int, payload: dict) -> dict:
        self.updated_entities.append(("/travelExpense", expense_id, payload))
        return {"id": expense_id}


class TripletexServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeTripletexClient()
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

    def test_create_invoice_flow(self) -> None:
        request = self._request(
            'Create invoice for customer "Acme AS" for product "Consulting" with amount 1500 due 2026-04-20.'
        )
        self.service.execute(request)

        self.assertEqual(self.client.updated_accounts[0], (901, {"bankAccountNumber": "10000000006"}))
        self.assertEqual(self.client.created[0][0], "/customer")
        self.assertEqual(self.client.created[0][1]["name"], "Acme AS")
        self.assertEqual(self.client.order_payloads[0]["customer"]["id"], 101)
        self.assertEqual(self.client.order_payloads[0]["orderLines"][0]["description"], "Consulting")
        self.assertEqual(self.client.order_payloads[0]["orderLines"][0]["unitPriceExcludingVatCurrency"], 1500.0)
        self.assertEqual(self.client.invoice_payloads[0][0]["orders"][0]["id"], 301)
        self.assertFalse(self.client.invoice_payloads[0][1])

    def test_create_invoice_uses_existing_product_price(self) -> None:
        request = self._request(
            'Create invoice for customer "Acme AS" for product "Support Retainer" amount 2500 due 2026-04-20.'
        )
        self.service.execute(request)

        self.assertEqual(self.client.order_payloads[0]["orderLines"][0]["description"], "Support Retainer")
        self.assertEqual(self.client.order_payloads[0]["orderLines"][0]["unitPriceExcludingVatCurrency"], 2500.0)

    def test_register_payment_flow(self) -> None:
        request = self._request(
            'Register payment for customer "Acme AS" amount 1500 on 2026-03-19.'
        )
        self.service.execute(request)

        self.assertEqual(len(self.client.payment_calls), 1)
        self.assertEqual(self.client.payment_calls[0]["payment_type_id"], 601)

    def test_create_credit_note_flow(self) -> None:
        request = self._request("Create credit note for invoice 2002 on 2026-03-20.")
        self.service.execute(request)

        self.assertEqual(len(self.client.credit_note_calls), 1)
        self.assertEqual(self.client.credit_note_calls[0]["invoice_id"], 502)
        self.assertFalse(self.client.credit_note_calls[0]["send_to_customer"])

    def test_create_credit_note_by_customer_and_amount(self) -> None:
        request = self._request("Create credit note for customer Existing Customer amount 12000 NOK sem IVA.")
        self.service.execute(request)

        self.assertEqual(len(self.client.credit_note_calls), 1)
        self.assertEqual(self.client.credit_note_calls[0]["invoice_id"], 503)
        self.assertFalse(self.client.credit_note_calls[0]["send_to_customer"])

    def test_create_travel_expense_flow(self) -> None:
        request = self._request(
            'Create travel expense for employee "Ola Nordmann" from Oslo to Bergen on 2026-03-19 amount 450.'
        )
        self.service.execute(request)

        self.assertEqual(self.client.created[0][0], "/department")
        self.assertEqual(self.client.created[1][0], "/employee")
        self.assertEqual(len(self.client.travel_expense_payloads), 1)
        self.assertEqual(self.client.travel_expense_payloads[0]["employee"]["id"], 201)
        self.assertEqual(self.client.travel_expense_payloads[0]["travelDetails"]["departureFrom"], "Oslo")
        self.assertEqual(self.client.travel_expense_payloads[0]["travelDetails"]["destination"], "Bergen")
        self.assertEqual(len(self.client.travel_cost_payloads), 1)
        self.assertEqual(self.client.travel_cost_payloads[0]["amountCurrencyIncVat"], 450.0)

    def test_create_project_creates_customer_prerequisite(self) -> None:
        request = self._request('Create project "Website Refresh" for customer "Missing Customer".')
        self.service.execute(request)

        # Fresh account optimization creates: customer, department (for PM), employee (PM), project
        customer_creates = [(p, d) for p, d, _ in self.client.created if p == "/customer"]
        project_creates = [(p, d) for p, d, _ in self.client.created if p == "/project"]
        self.assertEqual(len(customer_creates), 1)
        self.assertEqual(len(project_creates), 1)
        self.assertEqual(project_creates[0][1]["customer"]["id"], 101)
        self.assertEqual(project_creates[0][1]["projectManager"]["id"], 201)

    def test_update_project_flow(self) -> None:
        request = self._request(
            'Rename project "Website Refresh" to "Website Relaunch" for customer "Existing Customer" in department "Sales".'
        )
        self.service.execute(request)

        # Fresh account optimization: _ensure_customer creates (id=101), _ensure_department creates (id=151)
        self.assertEqual(self.client.updated_entities[-1][0], "/project")
        self.assertEqual(self.client.updated_entities[-1][1], 411)
        self.assertEqual(self.client.updated_entities[-1][2]["name"], "Website Relaunch")
        self.assertEqual(self.client.updated_entities[-1][2]["customer"]["id"], 101)
        self.assertEqual(self.client.updated_entities[-1][2]["department"]["id"], 151)

    def test_update_product_flow(self) -> None:
        request = self._request('Update product "Support Retainer" price 3000.')
        self.service.execute(request)

        self.assertEqual(
            self.client.updated_entities[0],
            ("/product", 321, {"priceExcludingVatCurrency": 3000.0}),
        )

    def test_delete_department_flow(self) -> None:
        request = self._request('Delete department "Sales".')
        self.service.execute(request)

        self.assertEqual(self.client.deleted_entities[0], ("/department", 156))

    def test_create_contact_flow(self) -> None:
        request = self._request(
            'Add contact "Kari Hansen" with email kari@acme.no to customer "Existing Customer".'
        )
        self.service.execute(request)

        contact_creates = [(p, d) for p, d, _ in self.client.created if p == "/contact"]
        self.assertEqual(len(contact_creates), 1)
        self.assertEqual(contact_creates[0][1]["firstName"], "Kari")
        self.assertEqual(contact_creates[0][1]["lastName"], "Hansen")
        self.assertEqual(contact_creates[0][1]["email"], "kari@acme.no")
        # Fresh account optimization: _ensure_customer creates directly (id=101)
        self.assertEqual(contact_creates[0][1]["customer"]["id"], 101)

    def test_create_supplier_flow(self) -> None:
        request = self._request(
            'Create supplier "Nordic Parts AS" with email orders@nordicparts.no.'
        )
        self.service.execute(request)

        supplier_creates = [(p, d) for p, d, _ in self.client.created if p == "/supplier"]
        self.assertEqual(len(supplier_creates), 1)
        self.assertEqual(supplier_creates[0][1]["name"], "Nordic Parts AS")
        self.assertEqual(supplier_creates[0][1]["email"], "orders@nordicparts.no")

    def test_create_voucher_flow(self) -> None:
        request = self._request(
            'Create voucher dated 2026-03-19 description "Office supplies" '
            'debit account 4000 credit account 1920 amount 500.'
        )
        self.service.execute(request)

        voucher_creates = [(p, d) for p, d, _ in self.client.created if p == "/ledger/voucher"]
        self.assertEqual(len(voucher_creates), 1)
        payload = voucher_creates[0][1]
        self.assertEqual(payload["description"], "Office supplies")
        self.assertEqual(len(payload["postings"]), 2)
        self.assertEqual(payload["postings"][0]["amountGross"], 500.0)
        self.assertEqual(payload["postings"][0]["amountGrossCurrency"], 500.0)
        self.assertEqual(payload["postings"][1]["amountGross"], -500.0)
        self.assertEqual(payload["postings"][1]["amountGrossCurrency"], -500.0)

    def test_delete_voucher_flow(self) -> None:
        request = self._request("Delete voucher id 608817050.")
        self.service.execute(request)

        self.assertEqual(self.client.deleted_entities[0], ("/ledger/voucher", 608817050))

    def test_create_timesheet_entry_flow(self) -> None:
        request = self._request(
            'Register 7.5 hours for employee "Ola Nordmann" on activity "Administrasjon" on 2026-03-19.'
        )
        self.service.execute(request)

        timesheet_creates = [(p, d) for p, d, _ in self.client.created if p == "/timesheet/entry"]
        self.assertEqual(len(timesheet_creates), 1)
        self.assertEqual(timesheet_creates[0][1]["hours"], 7.5)
        self.assertEqual(timesheet_creates[0][1]["activity"]["id"], 5604365)

    def test_update_travel_expense_flow(self) -> None:
        request = self._request(
            'Update travel expense id 11141899 from Oslo to "Trondheim".'
        )
        self.service.execute(request)

        self.assertEqual(self.client.updated_entities[0][0], "/travelExpense")
        self.assertEqual(self.client.updated_entities[0][1], 11141899)


if __name__ == "__main__":
    unittest.main()
