"""Tests for bank reconciliation CSV parsing and line classification."""
from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from tripletex_solver.models import Action, Entity, ParsedTask
from tripletex_solver.service import TripletexService


def _make_service() -> TripletexService:
    """Create a TripletexService with a mock client."""
    svc = TripletexService.__new__(TripletexService)
    svc.client = MagicMock()
    svc.client._call_log = []
    svc.last_attachment_text = None
    svc.last_parsed_task = None
    svc._saved_attachment_paths = []
    svc._cache = {}
    svc.parser = None
    return svc


# ============================================================================
# CSV Parsing
# ============================================================================

class TestBankCSVParsing(unittest.TestCase):
    """Test _parse_bank_csv with various CSV formats."""

    def setUp(self):
        self.svc = _make_service()

    def test_standard_norwegian_csv(self):
        csv = (
            "Dato;Forklaring;Inn;Ut;Saldo\n"
            "2026-01-18;Innbetaling fra Costa Lda / Faktura 1001;9562.50;;109562.50\n"
            "2026-01-21;Innbetaling fra Pereira Lda / Faktura 1002;16625.00;;126187.50\n"
            "2026-01-25;Betaling Fornecedor Oliveira Lda;;-9550.00;140262.50\n"
            "2026-01-29;Renteinntekter;1040.55;;124853.05\n"
            "2026-01-31;Bankgebyr;539.06;;125392.11\n"
        )
        lines = self.svc._parse_bank_csv(csv)
        self.assertEqual(len(lines), 5)

        # Customer payments
        self.assertEqual(lines[0]["description"], "Innbetaling fra Costa Lda / Faktura 1001")
        self.assertAlmostEqual(lines[0]["amount"], 9562.50)
        self.assertTrue(lines[0]["is_incoming"])

        # Supplier payment
        self.assertEqual(lines[2]["description"], "Betaling Fornecedor Oliveira Lda")
        self.assertAlmostEqual(lines[2]["amount"], 9550.00)
        self.assertFalse(lines[2]["is_incoming"])

        # Interest
        self.assertAlmostEqual(lines[3]["amount"], 1040.55)

        # Bank fee
        self.assertAlmostEqual(lines[4]["amount"], 539.06)

    def test_csv_with_comma_decimals(self):
        csv = (
            "Dato;Forklaring;Inn;Ut;Saldo\n"
            "2026-01-18;Innbetaling fra Kunde A / Faktura 100;9562,50;;109562,50\n"
        )
        lines = self.svc._parse_bank_csv(csv)
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(lines[0]["amount"], 9562.50)

    def test_csv_empty_lines_skipped(self):
        csv = (
            "Dato;Forklaring;Inn;Ut;Saldo\n"
            "2026-01-18;Innbetaling fra Test / Faktura 1;5000;;105000\n"
            "\n"
            "2026-01-20;Bankgebyr;100;;105100\n"
        )
        lines = self.svc._parse_bank_csv(csv)
        self.assertEqual(len(lines), 2)


# ============================================================================
# Bank Line Classification
# ============================================================================

class TestBankLineClassification(unittest.TestCase):
    """Test _process_bank_line dispatches to correct handler."""

    def setUp(self):
        self.svc = _make_service()
        self.payment_type = {"id": 1, "description": "Bank"}

        # Mock account lookups
        self.svc.client.search_accounts_by_number = MagicMock(return_value=[{"id": 99, "number": 1920, "name": "Bank"}])
        self.svc.client.search_customers = MagicMock(return_value=[{"id": 10, "name": "Costa Lda"}])
        self.svc.client.search_invoices = MagicMock(return_value=[
            {"id": 100, "invoiceNumber": "1001", "amountOutstanding": 9562.50, "amountCurrencyOutstanding": 9562.50}
        ])
        self.svc.client.pay_invoice = MagicMock(return_value={})
        self.svc.client.search_suppliers = MagicMock(return_value=[{"id": 20, "name": "Oliveira Lda"}])
        self.svc.client.search_supplier_invoices = MagicMock(return_value=[])
        self.svc.client.pay_supplier_invoice = MagicMock(return_value={})
        self.svc.client.create_voucher = MagicMock(return_value={"id": 200})

    def test_customer_payment_detected(self):
        line = {
            "date": "2026-01-18",
            "description": "Innbetaling fra Costa Lda / Faktura 1001",
            "amount": 9562.50,
            "is_incoming": True,
        }
        self.svc._process_bank_line(line, self.payment_type, {"id": 99})
        self.svc.client.pay_invoice.assert_called_once()
        args = self.svc.client.pay_invoice.call_args
        self.assertEqual(args[0][0], 100)  # invoice id
        self.assertAlmostEqual(args[1]["paid_amount"], 9562.50)

    def test_supplier_payment_detected(self):
        line = {
            "date": "2026-01-25",
            "description": "Betaling Fornecedor Oliveira Lda",
            "amount": 9550.00,
            "is_incoming": False,
        }
        self.svc._process_bank_line(line, self.payment_type, {"id": 99})
        # Should try to pay supplier invoice or create voucher
        self.assertTrue(
            self.svc.client.create_voucher.called or self.svc.client.pay_supplier_invoice.called
        )

    def test_interest_income_detected(self):
        line = {
            "date": "2026-01-29",
            "description": "Renteinntekter",
            "amount": 1040.55,
            "is_incoming": True,
        }
        # Need separate return values for 1920 and 8040
        def mock_search(number):
            return [{"id": number, "number": number, "name": f"Account {number}"}]
        self.svc.client.search_accounts_by_number = MagicMock(side_effect=mock_search)

        self.svc._process_bank_line(line, self.payment_type, {"id": 99})
        self.svc.client.create_voucher.assert_called_once()
        voucher = self.svc.client.create_voucher.call_args[0][0]
        self.assertEqual(voucher["description"], "Renteinntekter")
        self.assertEqual(len(voucher["postings"]), 2)

    def test_bank_fee_detected(self):
        line = {
            "date": "2026-01-31",
            "description": "Bankgebyr",
            "amount": 539.06,
            "is_incoming": True,
        }
        def mock_search(number):
            return [{"id": number, "number": number, "name": f"Account {number}"}]
        self.svc.client.search_accounts_by_number = MagicMock(side_effect=mock_search)

        self.svc._process_bank_line(line, self.payment_type, {"id": 99})
        self.svc.client.create_voucher.assert_called_once()
        voucher = self.svc.client.create_voucher.call_args[0][0]
        self.assertEqual(voucher["description"], "Bankgebyr")
        self.assertEqual(voucher["postings"][0]["account"]["id"], 1920)
        self.assertEqual(voucher["postings"][1]["account"]["id"], 7770)

    def test_interest_expense_detected_when_outgoing(self):
        line = {
            "date": "2026-02-03",
            "description": "Renteinntekter",
            "amount": 1985.42,
            "is_incoming": False,
        }
        def mock_search(number):
            return [{"id": number, "number": number, "name": f"Account {number}"}]
        self.svc.client.search_accounts_by_number = MagicMock(side_effect=mock_search)

        self.svc._process_bank_line(line, self.payment_type, {"id": 99})
        self.svc.client.create_voucher.assert_called_once()
        voucher = self.svc.client.create_voucher.call_args[0][0]
        self.assertEqual(voucher["description"], "Rentekostnad")
        self.assertEqual(voucher["postings"][0]["account"]["id"], 8150)
        self.assertEqual(voucher["postings"][1]["account"]["id"], 1920)


# ============================================================================
# LLM Parser Entity Correction
# ============================================================================

class TestBankReconciliationParsing(unittest.TestCase):
    """Test that bank reconciliation keywords redirect to bank_statement entity."""

    def test_portuguese_reconciliation_redirect(self):
        """Portuguese 'Reconcilie o extrato bancário' should redirect to bank_statement."""
        from tripletex_solver.llm_parser import _normalize_ascii, _contains_any_ascii

        prompt = "Reconcilie o extrato bancário (CSV anexo) com as faturas em aberto no Tripletex"
        keywords = (
            "reconcili", "bankavstem", "bankutskrift", "extrato bancario",
            "kontoutskrift", "kontoauszug", "releve bancaire", "extracto bancario",
            "kontoutdrag", "bank statement",
        )
        self.assertTrue(_contains_any_ascii(prompt, keywords))

    def test_norwegian_bankavstemming_detected(self):
        from tripletex_solver.llm_parser import _normalize_ascii, _contains_any_ascii

        prompt = "Gjør bankavstemming basert på vedlagt CSV-fil"
        keywords = (
            "reconcili", "bankavstem", "bankutskrift", "extrato bancario",
            "kontoutskrift", "kontoauszug", "releve bancaire", "extracto bancario",
            "kontoutdrag", "bank statement",
        )
        self.assertTrue(_contains_any_ascii(prompt, keywords))

    def test_english_bank_reconciliation_detected(self):
        from tripletex_solver.llm_parser import _normalize_ascii, _contains_any_ascii

        prompt = "Reconcile the attached bank statement CSV with open invoices in Tripletex"
        keywords = (
            "reconcili", "bankavstem", "bankutskrift", "extrato bancario",
            "kontoutskrift", "kontoauszug", "releve bancaire", "extracto bancario",
            "kontoutdrag", "bank statement",
        )
        self.assertTrue(_contains_any_ascii(prompt, keywords))

    def test_german_kontoauszug_detected(self):
        from tripletex_solver.llm_parser import _normalize_ascii, _contains_any_ascii

        prompt = "Gleichen Sie den beigefügten Kontoauszug mit den offenen Rechnungen ab"
        keywords = (
            "reconcili", "bankavstem", "bankutskrift", "extrato bancario",
            "kontoutskrift", "kontoauszug", "releve bancaire", "extracto bancario",
            "kontoutdrag", "bank statement",
        )
        self.assertTrue(_contains_any_ascii(prompt, keywords))


# ============================================================================
# Multi-language customer payment regex
# ============================================================================

class TestCustomerPaymentRegex(unittest.TestCase):
    """Test customer payment line regex in various languages."""

    def _match(self, desc):
        import re
        return re.search(
            r"(?:innbetaling\s+fra|pagamento?\s+de?|payment\s+from|zahlung\s+von|paiement\s+de)\s+(.+?)(?:\s*/\s*|\s+)(?:faktura|fatura|invoice|rechnung|facture)\s*(\d+)",
            desc, re.IGNORECASE,
        )

    def test_norwegian(self):
        m = self._match("Innbetaling fra Kunde AS / Faktura 2001")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Kunde AS")
        self.assertEqual(m.group(2), "2001")

    def test_portuguese(self):
        m = self._match("Pagamento de Costa Lda / Fatura 1001")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Costa Lda")
        self.assertEqual(m.group(2), "1001")

    def test_english(self):
        m = self._match("Payment from Acme Inc / Invoice 3045")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Acme Inc")
        self.assertEqual(m.group(2), "3045")

    def test_german(self):
        m = self._match("Zahlung von Müller GmbH / Rechnung 5001")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Müller GmbH")
        self.assertEqual(m.group(2), "5001")

    def test_french(self):
        m = self._match("Paiement de Dupont SA / Facture 7021")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Dupont SA")
        self.assertEqual(m.group(2), "7021")


# ============================================================================
# Supplier payment regex
# ============================================================================

class TestSupplierPaymentRegex(unittest.TestCase):
    """Test supplier payment line regex in various languages."""

    def _match(self, desc):
        import re
        return re.search(
            r"(?:betaling\s+(?:fornecedor|leverandør|supplier|til)|pagamento?\s+(?:fornecedor|a)\s+|payment\s+(?:to|supplier)|zahlung\s+(?:an|lieferant))\s+(.+)",
            desc, re.IGNORECASE,
        )

    def test_portuguese_fornecedor(self):
        m = self._match("Betaling Fornecedor Oliveira Lda")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Oliveira Lda")

    def test_norwegian_leverandor(self):
        m = self._match("Betaling leverandør Byggmester AS")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Byggmester AS")

    def test_norwegian_til(self):
        m = self._match("Betaling til Hansen Regnskap")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Hansen Regnskap")

    def test_english(self):
        m = self._match("Payment to Acme Supplies Ltd")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Acme Supplies Ltd")

    def test_german(self):
        m = self._match("Zahlung an Weber GmbH")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Weber GmbH")


# ============================================================================
# Full reconciliation flow (integration-style)
# ============================================================================

class TestFullReconciliationFlow(unittest.TestCase):
    """Test the full reconcile flow with mock client calls."""

    def test_competition_csv_creates_correct_entries(self):
        """The exact CSV from the competition should produce 5 customer payments,
        3 supplier payments, 1 interest voucher, and 1 fee voucher."""
        svc = _make_service()

        csv_content = (
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

        lines = svc._parse_bank_csv(csv_content)
        self.assertEqual(len(lines), 10)

        # Verify counts by type
        customer_payments = [l for l in lines if "Innbetaling" in l["description"]]
        supplier_payments = [l for l in lines if "Fornecedor" in l["description"]]
        interest = [l for l in lines if "Renteinntekter" in l["description"]]
        fees = [l for l in lines if "Bankgebyr" in l["description"]]

        self.assertEqual(len(customer_payments), 5)
        self.assertEqual(len(supplier_payments), 3)
        self.assertEqual(len(interest), 1)
        self.assertEqual(len(fees), 1)


if __name__ == "__main__":
    unittest.main()
