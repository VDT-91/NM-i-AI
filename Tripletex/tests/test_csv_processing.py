from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from tripletex_solver.attachment_text import extract_attachment_text
from tripletex_solver.attachments import save_attachments
from tripletex_solver.models import SolveFile


class BankStatementCSVTest(unittest.TestCase):
    """Test CSV file handling for bank statement scenarios."""

    def test_dnb_csv_format_readable(self) -> None:
        """DNB-style semicolon-separated CSV with Norwegian headers is extracted."""
        csv_content = (
            "Dato;Forklaring;Inn;Ut;Saldo\n"
            "01.03.2026;Betaling fra kunde;15000.00;;120000.00\n"
            "02.03.2026;Husleie;;8500.00;111500.00\n"
            "03.03.2026;Betaling fra kunde;23000.00;;134500.00\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "banktransaksjoner.csv"
            path.write_text(csv_content, encoding="utf-8")
            extracted = extract_attachment_text([path])
            self.assertIn("Betaling fra kunde", extracted)
            self.assertIn("15000.00", extracted)
            self.assertIn("8500.00", extracted)
            self.assertIn("Saldo", extracted)

    def test_csv_with_norwegian_encoding(self) -> None:
        """CSV encoded as latin-1 with Norwegian characters does not crash."""
        csv_content = (
            "Dato;Forklaring;Belop\n"
            "01.03.2026;Losning av lan;5000.00\n"
            "02.03.2026;Overfoering til sparekonto;3000.00\n"
            "03.03.2026;Betaling - Bjorn Aasen;7500.00\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kontoutskrift.csv"
            path.write_bytes(csv_content.encode("latin-1"))
            extracted = extract_attachment_text([path])
            self.assertIsInstance(extracted, str)
            self.assertIn("5000.00", extracted)
            self.assertIn("3000.00", extracted)
            self.assertIn("7500.00", extracted)

    def test_empty_csv_no_crash(self) -> None:
        """CSV with only a header line does not raise an exception."""
        csv_content = "Dato;Forklaring;Inn;Ut;Saldo\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.csv"
            path.write_text(csv_content, encoding="utf-8")
            extracted = extract_attachment_text([path])
            self.assertIsInstance(extracted, str)

    def test_xml_file_detected(self) -> None:
        """XML file does not crash even without Gemini API key."""
        xml_content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<banktransaksjon>\n"
            "  <dato>2026-03-01</dato>\n"
            "  <belop>15000.00</belop>\n"
            "  <beskrivelse>Betaling fra kunde</beskrivelse>\n"
            "</banktransaksjon>\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "transaksjon.xml"
            path.write_text(xml_content, encoding="utf-8")
            extracted = extract_attachment_text([path])
            self.assertIsInstance(extracted, str)

    def test_multiple_attachments(self) -> None:
        """Both CSV and TXT contents appear in extracted text."""
        csv_content = (
            "Dato;Forklaring;Belop\n"
            "01.03.2026;Innbetaling;42000.00\n"
        )
        txt_content = "Kundenavn: Ola Nordmann\nFakturanummer: 12345\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bank.csv"
            csv_path.write_text(csv_content, encoding="utf-8")
            txt_path = Path(tmpdir) / "notat.txt"
            txt_path.write_text(txt_content, encoding="utf-8")
            extracted = extract_attachment_text([csv_path, txt_path])
            self.assertIn("42000.00", extracted)
            self.assertIn("Ola Nordmann", extracted)
            self.assertIn("12345", extracted)


class AttachmentSaveTest(unittest.TestCase):
    """Test base64 decoding and file saving via save_attachments."""

    def test_save_csv_attachment(self) -> None:
        """A base64-encoded CSV SolveFile is saved with correct extension and content."""
        csv_content = "Dato;Forklaring;Belop\n01.03.2026;Betaling;9500.00\n"
        encoded = base64.b64encode(csv_content.encode("utf-8")).decode()
        solve_file = SolveFile(
            filename="kontoutskrift.csv",
            content_base64=encoded,
            mime_type="text/csv",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_attachments([solve_file], Path(tmpdir))
            self.assertEqual(len(saved), 1)
            self.assertTrue(saved[0].name.endswith(".csv"))
            written = saved[0].read_text(encoding="utf-8")
            self.assertIn("9500.00", written)
            self.assertIn("Betaling", written)

    def test_save_multiple_attachments(self) -> None:
        """Two SolveFiles (CSV + PDF) are both saved with correct extensions."""
        csv_content = "header1;header2\nval1;val2\n"
        pdf_content = b"%PDF-1.4 fake pdf content"
        csv_file = SolveFile(
            filename="data.csv",
            content_base64=base64.b64encode(csv_content.encode("utf-8")).decode(),
            mime_type="text/csv",
        )
        pdf_file = SolveFile(
            filename="invoice.pdf",
            content_base64=base64.b64encode(pdf_content).decode(),
            mime_type="application/pdf",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_attachments([csv_file, pdf_file], Path(tmpdir))
            self.assertEqual(len(saved), 2)
            extensions = {p.suffix for p in saved}
            self.assertIn(".csv", extensions)
            self.assertIn(".pdf", extensions)

    def test_large_attachment_no_crash(self) -> None:
        """A 200KB text file is saved correctly with correct size."""
        large_content = "A" * 200_000
        encoded = base64.b64encode(large_content.encode("utf-8")).decode()
        solve_file = SolveFile(
            filename="large_file.txt",
            content_base64=encoded,
            mime_type="text/plain",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_attachments([solve_file], Path(tmpdir))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].stat().st_size, 200_000)
            content = saved[0].read_text(encoding="utf-8")
            self.assertEqual(len(content), 200_000)


if __name__ == "__main__":
    unittest.main()
