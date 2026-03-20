"""Tests for Tier 2/3 entity identification, multi-step workflows, and edge cases."""

import unittest

from tripletex_solver.models import Action, Entity
from tripletex_solver.parser import PromptParser


class Tier2ParserTest(unittest.TestCase):
    """Entity identification across multiple languages for Tier 2/3 entities."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    # --- Incoming Invoice ---

    def test_incoming_invoice_norwegian(self) -> None:
        task = self.parser.parse(
            'Opprett innkommende faktura fra "Leverandor AS" belop 50000 NOK.'
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)
        self.assertEqual(task.action, Action.CREATE)

    def test_incoming_invoice_norwegian_leverandorfaktura(self) -> None:
        task = self.parser.parse(
            'Registrer leverandorfaktura fra "Bygg og Rør AS" belop 125000 NOK.'
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_incoming_invoice_english(self) -> None:
        task = self.parser.parse(
            'Create incoming invoice from "Parts Ltd" amount 25000 NOK.'
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)
        self.assertEqual(task.action, Action.CREATE)

    def test_incoming_invoice_english_supplier_invoice(self) -> None:
        task = self.parser.parse(
            'Create supplier invoice from "Nordic Supplies" amount 75000 NOK.'
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_incoming_invoice_german(self) -> None:
        task = self.parser.parse(
            'Erstelle Eingangsrechnung von "Lieferant GmbH" Betrag 10000.'
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_incoming_invoice_spanish(self) -> None:
        task = self.parser.parse(
            'Crea factura del proveedor de "Suministros SA" por 30000 NOK.'
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_incoming_invoice_french(self) -> None:
        task = self.parser.parse(
            'Cree facture fournisseur de "Piezas SARL" montant 15000.'
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    # --- Bank Statement ---

    def test_bank_statement_norwegian(self) -> None:
        task = self.parser.parse("Opprett kontoutdrag for bankkonto 1920.")
        self.assertEqual(task.entity, Entity.BANK_STATEMENT)
        self.assertEqual(task.action, Action.CREATE)

    def test_bank_statement_english(self) -> None:
        task = self.parser.parse("Create bank statement for account 1920.")
        self.assertEqual(task.entity, Entity.BANK_STATEMENT)

    def test_bank_statement_german(self) -> None:
        task = self.parser.parse("Erstelle Kontoauszug fur Konto 1920.")
        self.assertEqual(task.entity, Entity.BANK_STATEMENT)

    # --- Salary Transaction ---

    def test_salary_transaction_norwegian(self) -> None:
        task = self.parser.parse(
            'Opprett lonnsbilag for "Ola Nordmann" belop 35000 NOK.'
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)
        self.assertEqual(task.action, Action.CREATE)

    def test_salary_transaction_english(self) -> None:
        task = self.parser.parse(
            'Create salary transaction for "Ola Nordmann" amount 35000 NOK.'
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    def test_salary_transaction_english_payroll(self) -> None:
        task = self.parser.parse("Run payroll for March 2026.")
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    def test_salary_transaction_french(self) -> None:
        task = self.parser.parse("Executez la paie pour mars 2026.")
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    def test_salary_transaction_german(self) -> None:
        task = self.parser.parse("Erstelle Gehaltsabrechnung fur Marz 2026.")
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    def test_salary_transaction_spanish(self) -> None:
        task = self.parser.parse("Ejecutar nomina de marzo 2026.")
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    # --- Purchase Order ---

    def test_purchase_order_norwegian(self) -> None:
        task = self.parser.parse(
            'Opprett innkjopsordre til "Leverandor AS" for 10 stk produkt.'
        )
        self.assertEqual(task.entity, Entity.PURCHASE_ORDER)
        self.assertEqual(task.action, Action.CREATE)

    def test_purchase_order_english(self) -> None:
        task = self.parser.parse(
            'Create purchase order to "Supplier AS" for 10 units.'
        )
        self.assertEqual(task.entity, Entity.PURCHASE_ORDER)

    def test_purchase_order_german(self) -> None:
        task = self.parser.parse(
            'Erstelle Bestellung an "Lieferant GmbH" fur 20 Stuck.'
        )
        self.assertEqual(task.entity, Entity.PURCHASE_ORDER)

    # --- Dimension ---

    def test_dimension_norwegian(self) -> None:
        task = self.parser.parse('Opprett dimensjon "Kostnadssenter".')
        self.assertEqual(task.entity, Entity.DIMENSION)
        self.assertEqual(task.action, Action.CREATE)

    def test_dimension_english(self) -> None:
        task = self.parser.parse('Create dimension "Cost Center".')
        self.assertEqual(task.entity, Entity.DIMENSION)

    # --- Account ---

    def test_account_norwegian(self) -> None:
        task = self.parser.parse("Opprett konto 4010 Innkjop av varer.")
        self.assertEqual(task.entity, Entity.ACCOUNT)
        self.assertEqual(task.action, Action.CREATE)

    def test_account_english(self) -> None:
        task = self.parser.parse('Create ledger account 4010 "Purchase of goods".')
        self.assertEqual(task.entity, Entity.ACCOUNT)

    # --- Leave of Absence ---

    def test_leave_of_absence_norwegian(self) -> None:
        task = self.parser.parse(
            'Registrer permisjon for "Ola Nordmann" fra 2026-04-01 til 2026-04-15.'
        )
        self.assertEqual(task.entity, Entity.LEAVE_OF_ABSENCE)
        self.assertEqual(task.action, Action.CREATE)

    def test_leave_of_absence_english(self) -> None:
        task = self.parser.parse(
            'Register leave of absence for "Ola Nordmann" from 2026-04-01 to 2026-04-15.'
        )
        self.assertEqual(task.entity, Entity.LEAVE_OF_ABSENCE)

    def test_leave_of_absence_german(self) -> None:
        task = self.parser.parse(
            'Erstelle Urlaub fur "Ola Nordmann" vom 2026-04-01 bis 2026-04-15.'
        )
        self.assertEqual(task.entity, Entity.LEAVE_OF_ABSENCE)

    # --- Asset ---

    def test_asset_norwegian(self) -> None:
        task = self.parser.parse(
            'Opprett anleggsmiddel "Server" verdi 150000 NOK.'
        )
        self.assertEqual(task.entity, Entity.ASSET)
        self.assertEqual(task.action, Action.CREATE)

    def test_asset_english(self) -> None:
        task = self.parser.parse(
            'Create fixed asset "Server" value 150000 NOK.'
        )
        self.assertEqual(task.entity, Entity.ASSET)


class Tier2ParserMultiStepTest(unittest.TestCase):
    """Multi-step workflow parsing for payments, credit notes, and supplier invoices."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    # --- Payment with customer ---

    def test_payment_norwegian(self) -> None:
        task = self.parser.parse(
            "Registrer betaling for faktura 1001 belop 5000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_payment_english(self) -> None:
        task = self.parser.parse(
            "Register payment for invoice 2001 amount 7500 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_payment_spanish(self) -> None:
        task = self.parser.parse(
            "Registrar pago de factura 3001 monto 10000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_payment_portuguese(self) -> None:
        task = self.parser.parse(
            "Registrar pagamento da fatura 4001 valor 8000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_payment_german(self) -> None:
        task = self.parser.parse(
            "Zahlung registrieren fur Rechnung 5001 Betrag 6000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_payment_french(self) -> None:
        task = self.parser.parse(
            "Enregistrer le paiement de facture 6001 montant 9000 NOK."
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.PAYMENT)

    # --- Credit note (no invoice number) ---

    def test_credit_note_norwegian_no_invoice(self) -> None:
        task = self.parser.parse(
            'Opprett kreditnota for kunde "Acme AS" belop 12000 NOK uten mva.'
        )
        self.assertEqual(task.action, Action.CREATE)
        self.assertEqual(task.entity, Entity.INVOICE)
        self.assertEqual(task.attributes["workflow"], "creditNote")
        self.assertEqual(task.attributes["customerName"], "Acme AS")

    def test_credit_note_english_no_invoice(self) -> None:
        task = self.parser.parse(
            'Create credit note for customer "Acme AS" amount 12000 NOK excluding vat.'
        )
        self.assertEqual(task.action, Action.CREATE)
        self.assertEqual(task.entity, Entity.INVOICE)
        self.assertEqual(task.attributes["workflow"], "creditNote")
        self.assertEqual(task.attributes["customerName"], "Acme AS")

    # --- Pay supplier invoice ---

    def test_pay_supplier_invoice_norwegian(self) -> None:
        task = self.parser.parse(
            'Betal leverandorfaktura fra "Nordic Parts AS" belop 25000 NOK.'
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_pay_supplier_invoice_english(self) -> None:
        task = self.parser.parse(
            'Pay supplier invoice from "Nordic Parts AS" amount 25000 NOK.'
        )
        self.assertEqual(task.action, Action.REGISTER)
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    # --- Revert payment ---

    def test_revert_payment_norwegian(self) -> None:
        task = self.parser.parse("Slett betaling for faktura 1001.")
        self.assertEqual(task.action, Action.DELETE)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_revert_payment_english(self) -> None:
        task = self.parser.parse("Delete payment for invoice 2001.")
        self.assertEqual(task.action, Action.DELETE)
        self.assertEqual(task.entity, Entity.PAYMENT)


class Tier2ParserEdgeCaseTest(unittest.TestCase):
    """Edge cases: mixed language, terse prompts, special chars, etc."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    def test_mixed_language_no_en(self) -> None:
        """Norwegian entity keyword with English action word."""
        task = self.parser.parse(
            'Create innkommende faktura fra "Supplier AS" amount 50000 NOK.'
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)
        self.assertEqual(task.action, Action.CREATE)

    def test_terse_incoming_invoice(self) -> None:
        """Minimal prompt with just entity keyword and data."""
        task = self.parser.parse(
            "Opprett innkommende faktura Leverandor AS 50000."
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)
        self.assertEqual(task.action, Action.CREATE)

    def test_norwegian_special_chars_preserved(self) -> None:
        """Names with Norwegian special characters are preserved."""
        task = self.parser.parse(
            'Opprett leverandorfaktura fra "Bjorn AEroey Handverk AS" belop 10000 NOK.'
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_company_module_norwegian(self) -> None:
        task = self.parser.parse("Aktiver modul avdelingsregnskap.")
        self.assertEqual(task.entity, Entity.COMPANY_MODULE)
        self.assertEqual(task.action, Action.CREATE)

    def test_company_module_english(self) -> None:
        task = self.parser.parse("Enable module department accounting.")
        self.assertEqual(task.entity, Entity.COMPANY_MODULE)
        self.assertEqual(task.action, Action.CREATE)

    def test_revert_payment_norwegian_edge(self) -> None:
        """Revert / delete payment phrased with 'fjern'."""
        task = self.parser.parse("Fjern betaling for faktura 3001.")
        self.assertEqual(task.action, Action.DELETE)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_revert_payment_english_remove(self) -> None:
        """Remove payment phrased with 'remove'."""
        task = self.parser.parse("Remove payment for invoice 4001.")
        self.assertEqual(task.action, Action.DELETE)
        self.assertEqual(task.entity, Entity.PAYMENT)

    def test_salary_run_payroll_english(self) -> None:
        """'Run payroll' maps to salary transaction."""
        task = self.parser.parse("Run payroll for employees in March 2026.")
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)


if __name__ == "__main__":
    unittest.main()
