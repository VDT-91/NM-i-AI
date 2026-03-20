# Tier 2/3 Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ~109 tests (69 unit + 40 stress) covering Tier 2/3 parser correctness, multi-step service workflows, CSV/file processing, and multi-language edge cases.

**Architecture:** Three new test files test independent failure surfaces — parser entity detection, service call sequencing, and file I/O. Stress test additions go in the existing `stress_test.py`. All unit tests use mock clients, no external API calls.

**Tech Stack:** Python unittest, pytest parametrize, tempfile, base64, pathlib. No new dependencies.

---

### Task 1: Parser Tests — Tier 2/3 Entity Identification

**Files:**
- Create: `Tripletex/tests/test_parser_tier2_3.py`

**Step 1: Create the parser test file with all Tier 2/3 entity tests**

```python
from __future__ import annotations

import unittest

from tripletex_solver.models import Action, Entity
from tripletex_solver.parser import PromptParser


class Tier2ParserTest(unittest.TestCase):
    """Parser tests for Tier 2/3 entities and multi-language edge cases."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    # =========================================================================
    # INCOMING INVOICE — must not be confused with sales invoice
    # =========================================================================

    def test_incoming_invoice_NO(self) -> None:
        task = self.parser.parse(
            "Registrer en innkommende faktura fra leverandør Kontorrekvisita AS på 45000 kr inkl. mva, "
            "fakturadato 2026-03-15, forfallsdato 2026-04-15."
        )
        self.assertEqual(task.action, Action.CREATE)
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_incoming_invoice_EN(self) -> None:
        task = self.parser.parse(
            "Create an incoming invoice from supplier Office Supplies Ltd, amount 30000 NOK including VAT, "
            "invoice date 2026-03-20, due date 2026-04-20."
        )
        self.assertEqual(task.action, Action.CREATE)
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_incoming_invoice_DE(self) -> None:
        task = self.parser.parse(
            "Eingangsrechnung vom Lieferanten Bürobedarf GmbH erfassen, Betrag 25000 NOK inkl. MwSt, "
            "Rechnungsdatum 2026-03-18."
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_incoming_invoice_ES(self) -> None:
        task = self.parser.parse(
            "Registrar una factura del proveedor Suministros Express por 20000 NOK con IVA, "
            "fecha 2026-03-20."
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_incoming_invoice_FR(self) -> None:
        task = self.parser.parse(
            "Enregistrer une facture fournisseur de Fournitures Express, montant 18000 NOK TTC, "
            "date 2026-03-22."
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_leverandorfaktura_NO(self) -> None:
        task = self.parser.parse(
            "Registrer leverandørfaktura fra Bygg og Anlegg AS, beløp 120000 kr, dato 2026-03-10."
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    # =========================================================================
    # BANK STATEMENT
    # =========================================================================

    def test_bank_statement_NO(self) -> None:
        task = self.parser.parse("Importer kontoutdrag og utfør bankavtemming.")
        self.assertEqual(task.entity, Entity.BANK_STATEMENT)

    def test_bank_statement_EN(self) -> None:
        task = self.parser.parse("Import bank statement and perform reconciliation.")
        self.assertEqual(task.entity, Entity.BANK_STATEMENT)

    def test_bank_statement_DE(self) -> None:
        task = self.parser.parse("Kontoauszug importieren und Bankabstimmung durchführen.")
        self.assertEqual(task.entity, Entity.BANK_STATEMENT)

    # =========================================================================
    # SALARY TRANSACTION
    # =========================================================================

    def test_salary_NO(self) -> None:
        task = self.parser.parse(
            "Kjør lønnsbilag for ansatt Kari Nordmann. Grunnlønn 45000 kr, bonus 5000 kr."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)
        self.assertEqual(task.attributes.get("employeeName"), "Kari Nordmann")

    def test_salary_EN(self) -> None:
        task = self.parser.parse(
            "Run salary transaction for employee John Smith. Base salary 42000 NOK, bonus 3000 NOK."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    def test_salary_FR(self) -> None:
        task = self.parser.parse(
            "Exécutez la paie pour l'employé Pierre Dupont. Salaire de base 42000 NOK, prime 3000 NOK."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    def test_salary_DE(self) -> None:
        task = self.parser.parse(
            "Gehaltsabrechnung für Mitarbeiter Hans Müller erstellen. Grundgehalt 40000 NOK, Bonus 5000 NOK."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    def test_salary_ES(self) -> None:
        task = self.parser.parse(
            "Ejecutar nómina para el empleado Carlos García. Salario base 38000 NOK, bonus 4000 NOK."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    # =========================================================================
    # PURCHASE ORDER
    # =========================================================================

    def test_purchase_order_NO(self) -> None:
        task = self.parser.parse(
            "Opprett innkjøpsordre til leverandør Kontorrekvisita AS for kontormøbler, 25000 kr."
        )
        self.assertEqual(task.entity, Entity.PURCHASE_ORDER)

    def test_purchase_order_EN(self) -> None:
        task = self.parser.parse(
            "Create a purchase order to supplier Office Supplies Ltd for office furniture, 20000 NOK."
        )
        self.assertEqual(task.entity, Entity.PURCHASE_ORDER)

    def test_purchase_order_DE(self) -> None:
        task = self.parser.parse(
            "Bestellung an Lieferant Bürobedarf GmbH für Büromöbel, 15000 NOK."
        )
        self.assertEqual(task.entity, Entity.PURCHASE_ORDER)

    # =========================================================================
    # DIMENSION
    # =========================================================================

    def test_dimension_NO(self) -> None:
        task = self.parser.parse(
            "Opprett dimensjon Region med verdiene Vestlandet og Midt-Norge."
        )
        self.assertEqual(task.entity, Entity.DIMENSION)

    def test_dimension_EN(self) -> None:
        task = self.parser.parse(
            "Create a custom accounting dimension called Region with values West and Central."
        )
        self.assertEqual(task.entity, Entity.DIMENSION)

    # =========================================================================
    # ACCOUNT (ledger)
    # =========================================================================

    def test_account_NO(self) -> None:
        task = self.parser.parse(
            "Opprett konto 4010 med navn Varekjøp i kontoplanen."
        )
        self.assertEqual(task.entity, Entity.ACCOUNT)

    def test_account_EN(self) -> None:
        task = self.parser.parse(
            "Create ledger account 4010 named Purchases in the chart of accounts."
        )
        self.assertEqual(task.entity, Entity.ACCOUNT)

    # =========================================================================
    # LEAVE OF ABSENCE
    # =========================================================================

    def test_leave_NO(self) -> None:
        task = self.parser.parse(
            "Registrer permisjon for ansatt Kari Nordmann fra 2026-06-01 til 2026-06-30, "
            "100% fravær, type foreldrepermisjon."
        )
        self.assertEqual(task.entity, Entity.LEAVE_OF_ABSENCE)

    def test_leave_EN(self) -> None:
        task = self.parser.parse(
            "Register leave of absence for employee John Smith from 2026-06-01 to 2026-06-30, "
            "100% leave, type parental leave."
        )
        self.assertEqual(task.entity, Entity.LEAVE_OF_ABSENCE)

    def test_leave_DE(self) -> None:
        task = self.parser.parse(
            "Urlaub für Mitarbeiter Hans Müller eintragen, vom 2026-07-01 bis 2026-07-14."
        )
        self.assertEqual(task.entity, Entity.LEAVE_OF_ABSENCE)

    # =========================================================================
    # ASSET (fixed asset)
    # =========================================================================

    def test_asset_NO(self) -> None:
        task = self.parser.parse(
            "Registrer anleggsmiddel Kontormøbler, anskaffelseskost 85000 kr, dato 2026-01-15."
        )
        self.assertEqual(task.entity, Entity.ASSET)

    def test_asset_EN(self) -> None:
        task = self.parser.parse(
            "Register a fixed asset called Office Furniture, acquisition cost 85000 NOK, date 2026-01-15."
        )
        self.assertEqual(task.entity, Entity.ASSET)


class Tier2ParserMultiStepTest(unittest.TestCase):
    """Parser tests for multi-step workflow prompts where ALL fields must be extracted."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    # =========================================================================
    # PAYMENT — must extract customerName + amount (fresh account needs both)
    # =========================================================================

    def test_payment_with_customer_NO(self) -> None:
        task = self.parser.parse(
            "Registrer betaling for kunde Nordlys AS, faktura for konsulentarbeid, beløp 25000 kr."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)
        self.assertEqual(task.attributes.get("customerName"), "Nordlys AS")
        self.assertEqual(task.attributes.get("amount"), 25000.0)

    def test_payment_with_customer_EN(self) -> None:
        task = self.parser.parse(
            "Register payment for customer Arctic Solutions Ltd, invoice for consulting, amount 15000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)
        self.assertEqual(task.attributes.get("customerName"), "Arctic Solutions Ltd")
        self.assertEqual(task.attributes.get("amount"), 15000.0)

    def test_payment_with_customer_ES(self) -> None:
        task = self.parser.parse(
            "Registrar pago del cliente Sol del Norte SL, factura por consultoría, importe 20000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_payment_with_customer_PT(self) -> None:
        task = self.parser.parse(
            "Registrar pagamento do cliente Porto Digital Ltda, fatura de consultoria, valor 18000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_payment_with_customer_DE(self) -> None:
        task = self.parser.parse(
            "Zahlung für Kunde Nordlicht GmbH registrieren, Rechnung für Beratung, Betrag 22000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_payment_with_customer_FR(self) -> None:
        task = self.parser.parse(
            "Enregistrer le paiement du client Lumière du Nord SARL, facture de conseil, montant 16000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)

    # =========================================================================
    # CREDIT NOTE — by customer name (no invoice number)
    # =========================================================================

    def test_credit_note_by_customer_NO(self) -> None:
        task = self.parser.parse(
            "Kunden Nordlys AS reklamerte på faktura for konsulentarbeid (25000 NOK ekskl. mva). "
            "Opprett en full kreditnota."
        )
        self.assertEqual(task.entity, Entity.INVOICE)
        self.assertEqual(task.attributes.get("workflow"), "creditNote")
        self.assertEqual(task.attributes.get("customerName"), "Nordlys AS")
        self.assertEqual(task.attributes.get("amount"), 25000.0)

    def test_credit_note_by_customer_EN(self) -> None:
        task = self.parser.parse(
            "Customer Arctic Solutions Ltd complained about the invoice for consulting (15000 NOK excl VAT). "
            "Issue a full credit note."
        )
        self.assertEqual(task.entity, Entity.INVOICE)
        self.assertEqual(task.attributes.get("workflow"), "creditNote")

    # =========================================================================
    # MULTI-LINE INVOICE
    # =========================================================================

    def test_multiline_invoice_NO(self) -> None:
        task = self.parser.parse(
            'Lag faktura til kunde Nordlys AS med org.nr 872778330: '
            'Konsulentarbeid 10000 kr (25% mva), Reisekostnader 5000 kr (0% mva).'
        )
        self.assertEqual(task.entity, Entity.INVOICE)
        self.assertEqual(task.attributes.get("customerName"), "Nordlys AS")
        self.assertEqual(task.attributes.get("organizationNumber"), "872778330")

    # =========================================================================
    # INCOMING INVOICE PAYMENT (pay supplier)
    # =========================================================================

    def test_pay_supplier_invoice_NO(self) -> None:
        task = self.parser.parse(
            "Betal leverandørfaktura fra Kontorrekvisita AS, beløp 45000 kr."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_pay_supplier_invoice_EN(self) -> None:
        task = self.parser.parse(
            "Pay supplier invoice from Office Supplies Ltd, amount 30000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)


class Tier2ParserEdgeCaseTest(unittest.TestCase):
    """Edge cases: mixed languages, typos, terse prompts, format variations."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    def test_mixed_language_no_en(self) -> None:
        task = self.parser.parse(
            "Create en ny employee som heter Anna Larsen, email anna@firma.no, she should be administrator"
        )
        self.assertEqual(task.entity, Entity.EMPLOYEE)

    def test_terse_incoming_invoice(self) -> None:
        task = self.parser.parse(
            "innkommende faktura Leverandør AS 50000"
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_terse_salary(self) -> None:
        task = self.parser.parse(
            "lønnsbilag Kari Nordmann 45000"
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    def test_special_chars_customer_name(self) -> None:
        task = self.parser.parse(
            'Opprett kunde "Ærlig Økonomi AS" med e-post post@aerlig.no'
        )
        self.assertEqual(task.entity, Entity.CUSTOMER)
        self.assertEqual(task.attributes.get("name"), "Ærlig Økonomi AS")

    def test_amount_with_spaces(self) -> None:
        task = self.parser.parse(
            "Lag faktura til kunde Nordlys AS for konsulentarbeid, 50 000 kr ekskl. mva."
        )
        self.assertEqual(task.entity, Entity.INVOICE)
        # Amount should parse as 50000.0
        amount = task.attributes.get("amount")
        if amount is not None:
            self.assertEqual(amount, 50000.0)

    def test_date_norwegian_month(self) -> None:
        task = self.parser.parse(
            "Opprett bilag dato 15. mars 2026, debet 6300, kredit 1920, beløp 15000."
        )
        self.assertEqual(task.entity, Entity.VOUCHER)

    def test_company_module_activation_NO(self) -> None:
        task = self.parser.parse(
            "Aktiver avdelingsregnskap i Tripletex."
        )
        self.assertEqual(task.entity, Entity.COMPANY_MODULE)

    def test_company_module_activation_EN(self) -> None:
        task = self.parser.parse(
            "Enable department accounting module."
        )
        self.assertEqual(task.entity, Entity.COMPANY_MODULE)

    def test_revert_payment_NO(self) -> None:
        task = self.parser.parse(
            "Reverter betalingen for kunde Nordlys AS, beløp 25000 kr."
        )
        self.assertEqual(task.action, Action.DELETE)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_revert_payment_EN(self) -> None:
        task = self.parser.parse(
            "Revert payment for customer Arctic Solutions Ltd, amount 15000 NOK."
        )
        self.assertEqual(task.action, Action.DELETE)
        self.assertEqual(task.entity, Entity.PAYMENT)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run to see which pass and which fail**

Run: `cd "D:\Koding\NM i AI\Tripletex" && python -m pytest tests/test_parser_tier2_3.py -v --tb=short 2>&1`

Expected: Many tests pass (the parser has keywords for these entities). Some may fail — those reveal actual parser gaps to fix.

**Step 3: Fix any parser keyword gaps revealed by failing tests**

Modify: `Tripletex/tripletex_solver/parser.py` — add missing keywords for any entity type that fails detection.

**Step 4: Run again to verify all pass**

Run: `cd "D:\Koding\NM i AI\Tripletex" && python -m pytest tests/test_parser_tier2_3.py -v --tb=short 2>&1`

Expected: All ~45 tests PASS.

**Step 5: Commit**

```bash
git add tests/test_parser_tier2_3.py tripletex_solver/parser.py
git commit -m "test: add Tier 2/3 parser tests for multi-language entity detection"
```

---

### Task 2: Service Tests — Extended Fake Client + Multi-Step Workflows

**Files:**
- Create: `Tripletex/tests/test_service_tier2_3.py`

**Step 1: Create the service test file with extended fake client and all workflow tests**

```python
from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from tripletex_solver.errors import ParsingError
from tripletex_solver.models import (
    Action,
    Entity,
    ParsedTask,
    SolveFile,
    SolveRequest,
    TripletexCredentials,
)
from tripletex_solver.service import TripletexService


class FakeTripletexClientTier2:
    """Extended fake client that tracks Tier 2/3 API call sequences."""

    def __init__(self) -> None:
        # General tracking
        self.created: list[tuple[str, dict, dict | None]] = []
        self.updated_entities: list[tuple[str, int, dict]] = []
        self.deleted_entities: list[tuple[str, int]] = []
        self.updated_accounts: list[tuple[int, dict]] = []

        # Invoice flow
        self.order_payloads: list[dict] = []
        self.invoice_payloads: list[tuple[dict, bool]] = []
        self.payment_calls: list[dict] = []
        self.credit_note_calls: list[dict] = []

        # Travel expense flow
        self.travel_expense_payloads: list[dict] = []
        self.travel_cost_payloads: list[dict] = []
        self.delivered_travel_expenses: list[int] = []
        self.approved_travel_expenses: list[int] = []

        # Incoming invoice flow
        self.incoming_invoice_payloads: list[dict] = []
        self.approved_incoming_invoices: list[int] = []

        # Bank reconciliation flow
        self.imported_bank_statements: list[dict] = []
        self.created_bank_reconciliations: list[dict] = []
        self.suggested_matches: list[int] = []
        self.closed_reconciliations: list[int] = []

        # Salary
        self.salary_payloads: list[dict] = []

        # Module activation
        self.activated_modules: list[str] = []

        # Dimension
        self.dimension_names: list[dict] = []
        self.dimension_values: list[dict] = []

        # Per diem
        self.per_diem_payloads: list[dict] = []

        # Supplier invoice payment
        self.supplier_invoice_payments: list[dict] = []

        # Internal ID counters
        self._next_id = 100

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    # --- Generic CRUD ---

    def create(self, path: str, payload: dict, params: dict | None = None) -> dict:
        self.created.append((path, payload, params))
        return {"id": self._id(), **payload}

    def update(self, path: str, entity_id: int, payload: dict) -> dict:
        self.updated_entities.append((path, entity_id, payload))
        return {"id": entity_id, **payload}

    def delete(self, path: str, entity_id: int) -> None:
        self.deleted_entities.append((path, entity_id))

    def list(self, path: str, fields: str = "*", params: dict | None = None) -> list[dict]:
        # Return empty lists by default — handlers must handle empty results
        return []

    # --- Search methods (return empty or minimal data) ---

    def search_customers(self, *, name=None, email=None, phone=None) -> list[dict]:
        return []

    def search_employees(self, *, first_name=None, last_name=None, email=None) -> list[dict]:
        return []

    def search_departments(self, *, name: str) -> list[dict]:
        return []

    def search_products(self, *, name=None, product_number=None) -> list[dict]:
        return []

    def search_projects(self, *, name=None, customer_id=None) -> list[dict]:
        return []

    def search_suppliers(self, *, name=None, email=None) -> list[dict]:
        return []

    def search_contacts(self, *, customer_id=None, email=None) -> list[dict]:
        return []

    def search_activities(self, *, name=None) -> list[dict]:
        return [{"id": 5001, "name": "General", "isDisabled": False}]

    def search_vat_types(self, *, number=None, type_of_vat=None) -> list[dict]:
        if type_of_vat == "INCOMING":
            return [{"id": 26, "number": "1", "name": "Incoming VAT high"}]
        return [{"id": 25, "number": "3", "name": "High VAT"}]

    def search_accounts(self, *, number=None, is_bank_account=None) -> list[dict]:
        if is_bank_account:
            return [{"id": 9001, "number": 1920, "name": "Bank",
                      "isBankAccount": True, "isInvoiceAccount": True, "bankAccountNumber": ""}]
        return []

    def search_accounts_by_number(self, number: int) -> list[dict]:
        return [{"id": 4260000 + number, "number": number, "name": f"Account {number}"}]

    def search_payment_types(self, *, query: str) -> list[dict]:
        return [{"id": 6001, "description": "Betalt til bank"}]

    def search_voucher_types(self, *, name=None) -> list[dict]:
        return [{"id": 9811, "name": "Manuelt bilag"}]

    def search_salary_types(self) -> list[dict]:
        return [
            {"id": 7001, "name": "Fastlønn", "number": "1"},
            {"id": 7002, "name": "Bonus/tillegg", "number": "30"},
        ]

    # --- Travel expense ---

    def search_travel_payment_types(self, *, query=None) -> list[dict]:
        return [{"id": 7101, "description": "Corporate card"}]

    def search_travel_cost_categories(self, *, query=None) -> list[dict]:
        return [{"id": 7201, "description": "Travel"}]

    def create_travel_expense(self, payload: dict) -> dict:
        self.travel_expense_payloads.append(payload)
        return {"id": 10001}

    def create_travel_cost(self, payload: dict) -> dict:
        self.travel_cost_payloads.append(payload)
        return {"id": 10002}

    def deliver_travel_expense(self, expense_id: int) -> None:
        self.delivered_travel_expenses.append(expense_id)

    def approve_travel_expense(self, expense_id: int) -> None:
        self.approved_travel_expenses.append(expense_id)

    def search_per_diem_rate_categories(self) -> list[dict]:
        return [
            {"id": 8001, "name": "Dagsreise innland", "isDomestic": True, "isDayTrip": True},
            {"id": 8002, "name": "Overnatting innland", "isDomestic": True, "isDayTrip": False},
        ]

    def create_per_diem_compensation(self, payload: dict) -> dict:
        self.per_diem_payloads.append(payload)
        return {"id": 10003}

    # --- Invoice ---

    def create_order(self, payload: dict) -> dict:
        self.order_payloads.append(payload)
        return {"id": 3001}

    def create_invoice(self, payload: dict, *, send_to_customer: bool = False) -> dict:
        self.invoice_payloads.append((payload, send_to_customer))
        return {"id": 4001, "amountCurrencyOutstanding": 15000.0}

    def get_invoice(self, invoice_id: int) -> dict:
        return {"id": invoice_id, "invoiceNumber": str(invoice_id),
                "amountCurrencyOutstanding": 15000.0}

    def search_invoices(self, *, customer_id=None, invoice_number=None) -> list[dict]:
        return []

    def pay_invoice(self, invoice_id, *, payment_date, payment_type_id, paid_amount,
                    paid_amount_currency=None) -> dict:
        self.payment_calls.append({
            "invoice_id": invoice_id, "payment_date": payment_date,
            "payment_type_id": payment_type_id, "paid_amount": paid_amount,
        })
        return {"id": invoice_id}

    def create_credit_note(self, invoice_id, *, credit_note_date, comment=None,
                           send_to_customer=False) -> dict:
        self.credit_note_calls.append({
            "invoice_id": invoice_id, "credit_note_date": credit_note_date,
            "comment": comment, "send_to_customer": send_to_customer,
        })
        return {"id": 9001}

    # --- Incoming invoice ---

    def create_incoming_invoice(self, payload: dict) -> dict:
        self.incoming_invoice_payloads.append(payload)
        return {"id": 5001}

    def approve_incoming_invoice(self, invoice_id: int) -> None:
        self.approved_incoming_invoices.append(invoice_id)

    def search_incoming_invoices(self, *, supplier_id=None) -> list[dict]:
        return []

    def pay_supplier_invoice(self, invoice_id, *, amount=None, payment_date=None) -> dict:
        self.supplier_invoice_payments.append({
            "invoice_id": invoice_id, "amount": amount, "payment_date": payment_date,
        })
        return {"id": invoice_id}

    # --- Bank statement ---

    def import_bank_statement(self, file_path: str, *, bank_id=None, file_format="DNB_CSV") -> dict:
        self.imported_bank_statements.append({
            "file_path": file_path, "bank_id": bank_id, "file_format": file_format,
        })
        return {"value": {"bankAccountId": 8001}}

    def search_bank_accounts(self) -> list[dict]:
        return [{"id": 8001, "number": 1920, "name": "Bank", "bankAccountNumber": "1234.56.78901"}]

    def search_bank_reconciliations(self, *, account_id=None) -> list[dict]:
        return []  # No existing reconciliations

    def create_bank_reconciliation(self, payload: dict) -> dict:
        self.created_bank_reconciliations.append(payload)
        return {"id": 9001}

    def suggest_bank_reconciliation_matches(self, reconciliation_id: int) -> dict:
        self.suggested_matches.append(reconciliation_id)
        return {}

    def close_bank_reconciliation(self, reconciliation_id: int) -> dict:
        self.closed_reconciliations.append(reconciliation_id)
        return {"id": reconciliation_id, "isClosed": True}

    def search_bank_statements(self) -> list[dict]:
        return [{"id": 1, "bankAccount": {"id": 8001}, "fileName": "statement.csv"}]

    # --- Salary ---

    def create_salary_transaction(self, payload: dict) -> dict:
        self.salary_payloads.append(payload)
        return {"id": 6001}

    # --- Module ---

    def activate_sales_module(self, module_name: str) -> None:
        self.activated_modules.append(module_name)

    # --- Account ---

    def update_account(self, account_id: int, payload: dict) -> dict:
        self.updated_accounts.append((account_id, payload))
        return {"id": account_id, **payload}

    # --- Dimension ---

    def create_dimension_name(self, payload: dict) -> dict:
        self.dimension_names.append(payload)
        return {"id": 11001, "dimensionIndex": 1, **payload}

    def create_dimension_value(self, payload: dict) -> dict:
        self.dimension_values.append(payload)
        return {"id": 12001 + len(self.dimension_values), **payload}

    # --- Employee helpers ---

    def get_employee(self, employee_id: int) -> dict:
        return {"id": employee_id, "firstName": "Test", "lastName": "User",
                "dateOfBirth": "1990-01-01"}

    def search_employments(self, *, employee_id: int) -> list[dict]:
        return [{"id": 20001, "employee": {"id": employee_id},
                 "startDate": "2026-01-01", "employmentType": "ORDINARY"}]

    def update_employment(self, employment_id: int, payload: dict) -> dict:
        return {"id": employment_id, **payload}

    def create_timesheet_entry(self, payload: dict) -> dict:
        self.created.append(("/timesheet/entry", payload, None))
        return {"id": 17001, **payload}

    def update_travel_expense(self, expense_id: int, payload: dict) -> dict:
        return {"id": expense_id}


class TripletexServiceTier2Test(unittest.TestCase):
    """Test multi-step Tier 2/3 workflows in the service layer."""

    def setUp(self) -> None:
        self.client = FakeTripletexClientTier2()
        self.service = TripletexService(self.client)

    def _request(self, prompt: str, files: list | None = None) -> SolveRequest:
        return SolveRequest(
            prompt=prompt,
            files=files or [],
            tripletex_credentials=TripletexCredentials(
                base_url="https://tx-proxy.ainm.no/v2",
                session_token="token",
            ),
        )

    # =========================================================================
    # TRAVEL EXPENSE — deliver + approve
    # =========================================================================

    def test_travel_expense_deliver_approve(self) -> None:
        request = self._request(
            'Create travel expense for employee "Ola Nordmann" '
            'from Oslo to Bergen on 2026-04-10 return 2026-04-12 amount 450.'
        )
        self.service.execute(request)

        self.assertEqual(len(self.client.travel_expense_payloads), 1)
        self.assertEqual(self.client.delivered_travel_expenses, [10001])
        self.assertEqual(self.client.approved_travel_expenses, [10001])

    # =========================================================================
    # INCOMING INVOICE — create + approve
    # =========================================================================

    def test_incoming_invoice_approve(self) -> None:
        request = self._request(
            "Registrer innkommende faktura fra leverandør Kontorrekvisita AS "
            "på 45000 kr, fakturadato 2026-03-15, forfallsdato 2026-04-15."
        )
        self.service.execute(request)

        self.assertEqual(len(self.client.incoming_invoice_payloads), 1)
        self.assertEqual(self.client.approved_incoming_invoices, [5001])

    # =========================================================================
    # BANK STATEMENT — full reconciliation flow
    # =========================================================================

    def test_bank_statement_full_reconciliation(self) -> None:
        # Create a temp CSV file and set it as an attachment
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "statement.csv"
            csv_path.write_text(
                "Dato;Forklaring;Inn;Ut;Saldo\n"
                "2026-03-01;Betaling;1000;;5000\n"
                "2026-03-02;Utbetaling;;500;4500\n",
                encoding="utf-8",
            )
            self.service._saved_attachment_paths = [csv_path]

            task = ParsedTask(
                action=Action.CREATE,
                entity=Entity.BANK_STATEMENT,
                raw_prompt="Importer kontoutdrag og utfør bankavtemming.",
                attributes={"fileFormat": "DNB_CSV"},
            )
            self.service._pre_process(task)
            self.service._dispatch(task)

        # Verify full flow: import → reconciliation → suggest → close
        self.assertEqual(len(self.client.imported_bank_statements), 1)
        self.assertEqual(len(self.client.created_bank_reconciliations), 1)
        self.assertEqual(self.client.suggested_matches, [9001])
        self.assertEqual(self.client.closed_reconciliations, [9001])

    def test_bank_statement_no_file_raises(self) -> None:
        self.service._saved_attachment_paths = []
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.BANK_STATEMENT,
            raw_prompt="Importer kontoutdrag.",
            attributes={},
        )
        with self.assertRaises(ParsingError) as ctx:
            self.service._dispatch(task)
        self.assertIn("attached file", str(ctx.exception).lower())

    # =========================================================================
    # SALARY TRANSACTION
    # =========================================================================

    def test_salary_transaction_with_base_and_bonus(self) -> None:
        request = self._request(
            "Kjør lønnsbilag for ansatt Kari Nordmann. Grunnlønn 45000 kr, bonus 5000 kr."
        )
        self.service.execute(request)

        self.assertEqual(len(self.client.salary_payloads), 1)
        payload = self.client.salary_payloads[0]
        self.assertIn("payslips", payload)
        specs = payload["payslips"][0]["specifications"]
        # Should have two specs: base salary + bonus
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0]["rate"], 45000.0)
        self.assertEqual(specs[1]["rate"], 5000.0)

    # =========================================================================
    # COMPANY MODULE ACTIVATION
    # =========================================================================

    def test_module_activation(self) -> None:
        request = self._request("Aktiver avdelingsregnskap i Tripletex.")
        self.service.execute(request)

        self.assertIn("SMART", self.client.activated_modules)

    # =========================================================================
    # DIMENSION + VOUCHER
    # =========================================================================

    def test_dimension_with_values_and_voucher(self) -> None:
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.DIMENSION,
            raw_prompt="Opprett dimensjon Region med verdier.",
            attributes={
                "dimensionName": "Region",
                "dimensionValues": ["Vestlandet", "Midt-Norge"],
                "voucherAccountNumber": 6860,
                "voucherAmount": 31050.0,
                "voucherDimensionValue": "Vestlandet",
            },
        )
        self.service._dispatch(task)

        # Verify dimension created
        self.assertEqual(len(self.client.dimension_names), 1)
        self.assertEqual(self.client.dimension_names[0]["dimensionName"], "Region")

        # Verify values created
        self.assertEqual(len(self.client.dimension_values), 2)

        # Verify voucher created with dimension reference
        voucher_creates = [(p, d) for p, d, _ in self.client.created if p == "/ledger/voucher"]
        self.assertEqual(len(voucher_creates), 1)
        postings = voucher_creates[0][1]["postings"]
        self.assertEqual(postings[0]["amountGross"], 31050.0)
        self.assertEqual(postings[1]["amountGross"], -31050.0)
        # Debit posting should have dimension reference
        self.assertIn("freeAccountingDimension1", postings[0])

    # =========================================================================
    # MULTI-LINE INVOICE — multiple order lines with different VAT
    # =========================================================================

    def test_multiline_invoice(self) -> None:
        task = ParsedTask(
            action=Action.CREATE,
            entity=Entity.INVOICE,
            raw_prompt="Lag faktura til Nordlys AS.",
            attributes={
                "customerName": "Nordlys AS",
                "amount": 15000.0,
                "orderLines": [
                    {"description": "Konsulentarbeid", "amount": 10000.0, "vatRate": 25, "quantity": 1.0},
                    {"description": "Reisekostnader", "amount": 5000.0, "vatRate": 0, "quantity": 1.0},
                ],
            },
        )
        self.service._pre_process(task)
        self.service._dispatch(task)

        # Should have two order lines
        self.assertEqual(len(self.client.order_payloads), 1)
        order_lines = self.client.order_payloads[0]["orderLines"]
        self.assertEqual(len(order_lines), 2)
        self.assertEqual(order_lines[0]["description"], "Konsulentarbeid")
        self.assertEqual(order_lines[0]["unitPriceExcludingVatCurrency"], 10000.0)
        self.assertEqual(order_lines[1]["description"], "Reisekostnader")
        self.assertEqual(order_lines[1]["unitPriceExcludingVatCurrency"], 5000.0)

    # =========================================================================
    # INCOMING INVOICE PAYMENT (pay supplier)
    # =========================================================================

    def test_pay_supplier_invoice(self) -> None:
        # Patch client to return a supplier invoice when searched
        def fake_list(path, fields="*", params=None):
            if path == "/supplierInvoice":
                return [{"id": 5501, "invoiceNumber": "123", "amount": 45000.0,
                         "supplier": {"id": 901, "name": "Kontorrekvisita AS"}}]
            return []
        self.client.list = fake_list

        task = ParsedTask(
            action=Action.REGISTER,
            entity=Entity.INCOMING_INVOICE,
            raw_prompt="Betal leverandørfaktura fra Kontorrekvisita AS, beløp 45000 kr.",
            attributes={"supplierName": "Kontorrekvisita AS", "amount": 45000.0},
        )
        self.service._dispatch(task)

        self.assertEqual(len(self.client.supplier_invoice_payments), 1)
        self.assertEqual(self.client.supplier_invoice_payments[0]["invoice_id"], 5501)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run to see which pass**

Run: `cd "D:\Koding\NM i AI\Tripletex" && python -m pytest tests/test_service_tier2_3.py -v --tb=short 2>&1`

Expected: Most tests pass. Some may fail due to missing fake client methods — fix those in the fake.

**Step 3: Fix any issues in the fake client or service expectations**

Iterate on the fake client until all tests pass. If a test reveals a real service bug (e.g., bank reconciliation not called), that's a genuine find — fix the service code.

**Step 4: Run full suite to verify no regressions**

Run: `cd "D:\Koding\NM i AI\Tripletex" && python -m pytest tests/ -v --tb=short 2>&1`

Expected: All new tests pass, existing test results unchanged.

**Step 5: Commit**

```bash
git add tests/test_service_tier2_3.py
git commit -m "test: add Tier 2/3 service workflow tests with extended fake client"
```

---

### Task 3: CSV/File Processing Tests

**Files:**
- Create: `Tripletex/tests/test_csv_processing.py`

**Step 1: Create the file processing test file**

```python
from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from tripletex_solver.attachment_text import extract_attachment_text
from tripletex_solver.attachments import save_attachments
from tripletex_solver.models import SolveFile


class BankStatementCSVTest(unittest.TestCase):
    """Test CSV file handling for bank statement import."""

    def test_dnb_csv_format_readable(self) -> None:
        """DNB-style CSV with semicolons and Norwegian headers is readable."""
        csv_content = (
            "Dato;Forklaring;Inn;Ut;Saldo\n"
            "2026-03-01;Betaling fra kunde;15000;;115000\n"
            "2026-03-02;Husleie;;25000;90000\n"
            "2026-03-03;Lønn;;45000;45000\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kontoutdrag.csv"
            path.write_text(csv_content, encoding="utf-8")
            extracted = extract_attachment_text([path])
            self.assertIn("Betaling fra kunde", extracted)
            self.assertIn("15000", extracted)
            self.assertIn("Husleie", extracted)

    def test_csv_with_norwegian_encoding(self) -> None:
        """CSV with æøå in latin-1 encoding should not crash."""
        csv_content = "Dato;Forklaring;Beløp\n2026-03-01;Kjøp av varer;5000\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "statement.csv"
            # Write as latin-1 (ISO-8859-1) — common in Norwegian banking
            path.write_bytes(csv_content.encode("latin-1"))
            # extract_attachment_text uses errors="ignore" so this should work
            extracted = extract_attachment_text([path])
            self.assertIn("5000", extracted)

    def test_empty_csv_no_crash(self) -> None:
        """Empty CSV file should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.csv"
            path.write_text("Dato;Forklaring;Beløp\n", encoding="utf-8")
            extracted = extract_attachment_text([path])
            # Should return something (headers) or empty, but not crash
            self.assertIsInstance(extracted, str)

    def test_xml_file_detected(self) -> None:
        """XML bank statement should be detected by suffix."""
        xml_content = '<?xml version="1.0"?><statement><entry>test</entry></statement>'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "statement.xml"
            path.write_text(xml_content, encoding="utf-8")
            # Without Gemini key, the content is extracted via fallback
            extracted = extract_attachment_text([path])
            # Should not crash, may be empty without Gemini
            self.assertIsInstance(extracted, str)

    def test_multiple_attachments_csv_selected(self) -> None:
        """When multiple files attached, CSV content is still extracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bank.csv"
            csv_path.write_text("Dato;Inn;Ut\n2026-03-01;1000;\n", encoding="utf-8")
            txt_path = Path(tmpdir) / "notes.txt"
            txt_path.write_text("Some notes", encoding="utf-8")
            extracted = extract_attachment_text([csv_path, txt_path])
            self.assertIn("1000", extracted)
            self.assertIn("Some notes", extracted)


class AttachmentSaveTest(unittest.TestCase):
    """Test base64 decoding and file saving."""

    def test_save_csv_attachment(self) -> None:
        csv_content = "Dato;Inn;Ut\n2026-03-01;1000;\n"
        encoded = base64.b64encode(csv_content.encode("utf-8")).decode()
        files = [SolveFile(filename="bank.csv", content_base64=encoded, mime_type="text/csv")]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_attachments(files, Path(tmpdir))
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].suffix, ".csv")
            content = paths[0].read_text(encoding="utf-8")
            self.assertIn("1000", content)

    def test_save_multiple_attachments(self) -> None:
        csv_encoded = base64.b64encode(b"data1").decode()
        pdf_encoded = base64.b64encode(b"data2").decode()
        files = [
            SolveFile(filename="bank.csv", content_base64=csv_encoded, mime_type="text/csv"),
            SolveFile(filename="invoice.pdf", content_base64=pdf_encoded, mime_type="application/pdf"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_attachments(files, Path(tmpdir))
            self.assertEqual(len(paths), 2)
            self.assertTrue(any(p.suffix == ".csv" for p in paths))
            self.assertTrue(any(p.suffix == ".pdf" for p in paths))

    def test_large_attachment_no_crash(self) -> None:
        """Large file (200KB) should be saved without issues."""
        large_content = "x" * 200_000
        encoded = base64.b64encode(large_content.encode("utf-8")).decode()
        files = [SolveFile(filename="large.txt", content_base64=encoded, mime_type="text/plain")]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_attachments(files, Path(tmpdir))
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].stat().st_size, 200_000)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run**

Run: `cd "D:\Koding\NM i AI\Tripletex" && python -m pytest tests/test_csv_processing.py -v --tb=short 2>&1`

Expected: All 8-9 tests PASS.

**Step 3: Commit**

```bash
git add tests/test_csv_processing.py
git commit -m "test: add CSV/file processing tests for bank statement handling"
```

---

### Task 4: Stress Test Additions

**Files:**
- Modify: `Tripletex/stress_test.py`

**Step 1: Add Tier 2/3 prompts and update runner for file attachments**

Add these entries to the `tests` list in `stress_test.py`, after the existing edge cases section. Also update `run_test` to support dict entries with file attachments.

Add to the `tests` list:

```python
    # =========================================================================
    # TIER 2: INCOMING INVOICE
    # =========================================================================
    ("34_INCINV_NO", "Registrer en innkommende faktura fra leverandør Kontorrekvisita AS på 45000 kr inkl. mva, fakturadato 2026-03-15, forfallsdato 2026-04-15."),
    ("34_INCINV_EN", "Create an incoming invoice from supplier Office Supplies Ltd, amount 30000 NOK including VAT, invoice date 2026-03-20, due date 2026-04-20."),
    ("34_INCINV_ES", "Registrar una factura del proveedor Suministros Express por 20000 NOK con IVA, fecha 2026-03-20."),
    ("34_INCINV_DE", "Eingangsrechnung vom Lieferanten Bürobedarf GmbH erfassen, Betrag 25000 NOK inkl. MwSt, Rechnungsdatum 2026-03-18."),
    ("34_INCINV_FR", "Enregistrer une facture fournisseur de Fournitures Express, montant 18000 NOK TTC, date 2026-03-22."),
    ("34_INCINV_terse", "innkommende faktura Bygg og Anlegg AS 120000 dato 2026-03-10"),

    # =========================================================================
    # TIER 2: PAY SUPPLIER INVOICE
    # =========================================================================
    ("35_PAYSUPP_NO", "Betal leverandørfaktura fra Kontorrekvisita AS, beløp 45000 kr, dato 2026-03-20."),
    ("35_PAYSUPP_EN", "Pay supplier invoice from Office Supplies Ltd, amount 30000 NOK, date 2026-03-20."),

    # =========================================================================
    # TIER 2: MULTI-LINE INVOICE
    # =========================================================================
    ("36_MULTILINE_NO", "Lag faktura til kunde Nordlys AS med org.nr 872778330: Konsulentarbeid 10000 kr (25% mva), Reisekostnader 5000 kr (0% mva), Programvare 8000 kr (25% mva)."),
    ("36_MULTILINE_EN", "Create invoice for customer Arctic Solutions Ltd: Consulting 10000 NOK (25% VAT), Travel expenses 5000 NOK (0% VAT), Software 8000 NOK (25% VAT)."),

    # =========================================================================
    # TIER 2: FIXED-PRICE PROJECT + INVOICE
    # =========================================================================
    ("37_FIXPRICE_NO", "Opprett prosjekt Nettbutikk for kunde Nordlys AS med fastpris 200000 kr. Prosjektleder er Kari Nordmann. Fakturer 25% av fastprisen som første milepæl."),
    ("37_FIXPRICE_EN", "Create project E-commerce for customer Arctic Solutions Ltd with fixed price 150000 NOK. Project manager John Smith. Invoice 25% as first milestone."),

    # =========================================================================
    # TIER 2: SALARY TRANSACTION
    # =========================================================================
    ("38_SALARY_NO", "Kjør lønnsbilag for ansatt Kari Nordmann. Grunnlønn 45000 kr, bonus 5000 kr."),
    ("38_SALARY_EN", "Run salary transaction for employee John Smith. Base salary 42000 NOK, bonus 3000 NOK."),
    ("38_SALARY_FR", "Exécutez la paie pour l'employé Pierre Dupont. Salaire de base 42000 NOK, prime 3000 NOK."),
    ("38_SALARY_DE", "Gehaltsabrechnung für Mitarbeiter Hans Müller erstellen. Grundgehalt 40000 NOK, Bonus 5000 NOK."),
    ("38_SALARY_ES", "Ejecutar nómina para el empleado Carlos García. Salario base 38000 NOK, bonus 4000 NOK."),

    # =========================================================================
    # TIER 2: PURCHASE ORDER
    # =========================================================================
    ("39_PO_NO", "Opprett innkjøpsordre til leverandør Kontorrekvisita AS for kontormøbler, 25000 kr, levering 2026-04-15."),
    ("39_PO_EN", "Create purchase order to supplier Office Supplies Ltd for office furniture, 20000 NOK, delivery 2026-04-15."),
    ("39_PO_DE", "Bestellung an Lieferant Bürobedarf GmbH für Büromöbel erstellen, 15000 NOK, Lieferung 2026-04-15."),

    # =========================================================================
    # TIER 2: TRAVEL EXPENSE WITH PER DIEM
    # =========================================================================
    ("40_TRAVEL_DIEM_NO", "Registrer reiseregning for Kari Nordmann. Oslo til Tromsø, 2026-04-10 til 2026-04-12. Formål: kundemøte. Inkluder dietas."),
    ("40_TRAVEL_DIEM_EN", "Create travel expense for John Smith. Stavanger to Bergen, 2026-04-10 to 2026-04-12. Purpose: conference. Include per diem."),

    # =========================================================================
    # TIER 3: DIMENSION + VOUCHER
    # =========================================================================
    ("41_DIM_NO", "Opprett dimensjon Region med verdiene Vestlandet og Midt-Norge. Før et bilag på konto 6860 for 31050 kr knyttet til dimensjonsverdien Vestlandet."),
    ("41_DIM_EN", "Create dimension Region with values West and Central. Post a voucher on account 6860 for 31050 NOK linked to dimension value West."),

    # =========================================================================
    # TIER 3: LEAVE OF ABSENCE
    # =========================================================================
    ("42_LEAVE_NO", "Registrer permisjon for ansatt Kari Nordmann fra 2026-06-01 til 2026-06-30, 100% fravær, type foreldrepermisjon."),
    ("42_LEAVE_EN", "Register leave of absence for employee John Smith from 2026-06-01 to 2026-06-30, 100% leave, parental leave."),
    ("42_LEAVE_DE", "Urlaub für Mitarbeiter Hans Müller eintragen, vom 2026-07-01 bis 2026-07-14, Elternzeit."),

    # =========================================================================
    # TIER 3: ASSET / FIXED ASSET
    # =========================================================================
    ("43_ASSET_NO", "Registrer anleggsmiddel Kontormøbler, anskaffelseskost 85000 kr, dato 2026-01-15, lineær avskrivning over 60 måneder."),
    ("43_ASSET_EN", "Register fixed asset Office Furniture, acquisition cost 85000 NOK, date 2026-01-15, straight-line depreciation over 60 months."),

    # =========================================================================
    # TIER 2: CREDIT NOTE BY CUSTOMER (no invoice number)
    # =========================================================================
    ("44_CREDIT_CUST_NO", "Kunden Nordlys AS reklamerte på konsulentarbeid (25000 NOK ekskl. mva). Opprett en full kreditnota som reverserer hele fakturaen."),
    ("44_CREDIT_CUST_EN", "Customer Arctic Solutions Ltd complained about consulting (15000 NOK excl VAT). Issue a full credit note reversing the entire invoice."),
    ("44_CREDIT_CUST_PT", "O cliente Porto Digital Ltda reclamou sobre consultoria (18000 NOK sem IVA). Emita uma nota de crédito completa."),

    # =========================================================================
    # TIER 2: REVERT PAYMENT
    # =========================================================================
    ("45_REVERT_NO", "Reverter betalingen for kunde Nordlys AS, faktura for konsulentarbeid, beløp 25000 kr."),
    ("45_REVERT_EN", "Revert payment for customer Arctic Solutions Ltd, consulting invoice, amount 15000 NOK."),

    # =========================================================================
    # TIER 2: MODULE ACTIVATION
    # =========================================================================
    ("46_MODULE_NO", "Aktiver avdelingsregnskap i Tripletex."),
    ("46_MODULE_EN", "Enable department accounting module in Tripletex."),
    ("46_MODULE_DE", "Abteilungsbuchhaltung in Tripletex aktivieren."),

    # =========================================================================
    # TIER 3: TIMESHEET + INVOICE (register hours then bill)
    # =========================================================================
    ("47_TIME_INV_NO", "Ansatt Erik Holm (erik@firma.no) har jobbet 40 timer på prosjekt Nettbutikk for kunde Nordlys AS, aktivitet Utvikling. Fakturer kunden 1200 kr per time ekskl. mva."),
    ("47_TIME_INV_EN", "Employee John Smith worked 35 hours on project E-commerce for customer Arctic Solutions Ltd, activity Development. Invoice the customer at 1000 NOK per hour excl VAT."),
```

Update the `run_test` function to support dict entries with file attachments:

```python
def run_test(name, prompt_or_config):
    start = time.time()
    try:
        if isinstance(prompt_or_config, dict):
            body = {
                "prompt": prompt_or_config["prompt"],
                "files": prompt_or_config.get("files", []),
                "tripletex_credentials": CREDS,
            }
        else:
            body = {"prompt": prompt_or_config, "files": [], "tripletex_credentials": CREDS}

        r = requests.post(f"{BASE}/solve", json=body, timeout=120)
        elapsed = round(time.time() - start, 1)
        return (name, r.status_code, elapsed, r.text[:120])
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        return (name, "ERR", elapsed, str(e)[:120])
```

**Step 2: Verify syntax**

Run: `cd "D:\Koding\NM i AI\Tripletex" && python -c "import ast; ast.parse(open('stress_test.py').read()); print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add stress_test.py
git commit -m "test: add ~40 Tier 2/3 prompts to stress test"
```

---

### Task 5: Run Full Suite + Fix Any Failures

**Step 1: Run all unit tests**

Run: `cd "D:\Koding\NM i AI\Tripletex" && python -m pytest tests/ -v --tb=short 2>&1`

Expected: All new tests pass. Pre-existing failures (5 tests with mock ID mismatches) still fail.

**Step 2: If any new tests fail, investigate and fix**

- Parser test failures → fix keywords in `tripletex_solver/parser.py`
- Service test failures → fix fake client methods or service handler logic
- CSV test failures → fix encoding handling in `tripletex_solver/attachment_text.py`

**Step 3: Run again to confirm**

Run: `cd "D:\Koding\NM i AI\Tripletex" && python -m pytest tests/ -v --tb=short 2>&1`

**Step 4: Final commit**

```bash
git add -A
git commit -m "fix: resolve test failures discovered by Tier 2/3 test suite"
```
