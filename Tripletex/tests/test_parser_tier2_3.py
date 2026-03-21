"""Parser tests for Tier 2/3 entities and multi-language edge cases.

Tests the RULE-BASED parser only. In production, the LLM parser (Gemini)
handles most Tier 2/3 prompts. Tests marked @skip require LLM parsing.
"""
from __future__ import annotations

import unittest

from tripletex_solver.models import Action, Entity
from tripletex_solver.parser import PromptParser

# Tests that need LLM parser (rule-based parser lacks these language keywords)
LLM_ONLY = unittest.skip("LLM-only: rule-based parser lacks keywords for this language/entity")


class IncomingInvoiceParserTest(unittest.TestCase):
    """Incoming invoice must NOT be confused with sales invoice."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    def test_incoming_invoice_NO(self) -> None:
        task = self.parser.parse(
            "Registrer en innkommende faktura fra leverandør Kontorrekvisita AS på 45000 kr inkl. mva, "
            "fakturadato 2026-03-15, forfallsdato 2026-04-15."
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)
        self.assertIn("supplierName", task.attributes)

    def test_incoming_invoice_EN(self) -> None:
        task = self.parser.parse(
            "Register an incoming invoice from supplier Office Supplies Ltd for 85000 NOK excluding VAT."
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    @LLM_ONLY
    def test_incoming_invoice_DE(self) -> None:
        task = self.parser.parse(
            "Eingangsrechnung vom Lieferanten Bürobedarf GmbH über 21100 NOK einschließlich MwSt erfassen."
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    def test_incoming_invoice_FR(self) -> None:
        task = self.parser.parse(
            "Enregistrer une facture fournisseur de Fournitures Express pour 72000 NOK hors TVA."
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    @LLM_ONLY
    def test_incoming_invoice_ES(self) -> None:
        task = self.parser.parse(
            "Registrar una factura de proveedor de Suministros SL por 55000 NOK sin IVA."
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)

    @LLM_ONLY
    def test_incoming_invoice_with_account_DE(self) -> None:
        """Real game pattern: German incoming invoice with account + VAT."""
        task = self.parser.parse(
            'Wir haben die Rechnung INV-2026-7058 vom Lieferanten Bergwerk GmbH (Org.-Nr. 968598546) '
            'über 21100 NOK einschließlich MwSt. erhalten. Der Betrag betrifft Bürodienstleistungen '
            '(Konto 6300). Erfassen Sie die Lieferantenrechnung mit der korrekten Vorsteuer (25 %).'
        )
        self.assertEqual(task.entity, Entity.INCOMING_INVOICE)


class SalaryParserTest(unittest.TestCase):
    """Salary transaction parsing — exact competition patterns."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    @LLM_ONLY
    def test_salary_NO(self) -> None:
        task = self.parser.parse(
            "Kjør lønn for Jonas Hansen (jonas.hansen@example.org) for denne måneden. "
            "Grunnlønn er 40000 kr. Legg til en engangsbonus på 10600 kr i tillegg til grunnlønnen."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    @LLM_ONLY
    def test_salary_NN(self) -> None:
        task = self.parser.parse(
            "Køyr løn for Brita Stølsvik (brita.stlsvik@example.org) for denne månaden. "
            "Grunnløn er 36000 kr. Legg til ein eingongsbonus på 15400 kr i tillegg til grunnløna."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    def test_salary_EN(self) -> None:
        task = self.parser.parse(
            "Run payroll for employee John Smith (john@company.com) for this month. "
            "Base salary is 52000 NOK. Add a one-time bonus of 12000 NOK."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    def test_salary_DE(self) -> None:
        task = self.parser.parse(
            "Gehaltsabrechnung für Mitarbeiter Thomas Braun (thomas.braun@example.org) "
            "für April 2026. Grundgehalt 58000 NOK, Bonus 8000 NOK."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    @LLM_ONLY
    def test_salary_ES(self) -> None:
        task = self.parser.parse(
            "Ejecute la nómina de Fernando García (fernando.garcia@example.org) para este mes. "
            "El salario base es de 55350 NOK. Añada una bonificación única de 14600 NOK."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    @LLM_ONLY
    def test_salary_FR(self) -> None:
        task = self.parser.parse(
            "Exécutez la paie pour l'employé Marc Dupont (marc.dupont@example.org) pour ce mois. "
            "Salaire de base 45000 NOK. Ajoutez une prime unique de 8000 NOK."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    @LLM_ONLY
    def test_salary_with_voucher_hint_NO(self) -> None:
        """Real game pattern: salary with voucher fallback hint."""
        task = self.parser.parse(
            "Kjør lønn for Jonas Hansen (jonas.hansen@example.org) for denne måneden. "
            "Grunnlønn er 40000 kr. Legg til en engangsbonus på 10600 kr i tillegg til grunnlønnen. "
            "Dersom lønns-API-et ikke fungerer, kan du bruke manuelle bilag på lønnskontoer "
            "(5000-serien) for å registrere lønnskostnaden."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)

    @LLM_ONLY
    def test_salary_PT(self) -> None:
        task = self.parser.parse(
            "Execute a folha de pagamento para o funcionário Pedro Silva (pedro.silva@example.org) "
            "para este mês. Salário base 42000 NOK. Adicione um bónus único de 9500 NOK."
        )
        self.assertEqual(task.entity, Entity.SALARY_TRANSACTION)


class TravelExpenseParserTest(unittest.TestCase):
    """Travel expense parsing — per diem, multi-day, expense lines."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    @LLM_ONLY
    def test_travel_with_perdiem_NN(self) -> None:
        """Real game pattern: Nynorsk 5-day trip with per diem."""
        task = self.parser.parse(
            'Registrer ei reiserekning for Svein Berge (svein.berge@example.org) for '
            '"Kundebesøk Trondheim". Reisa varte 5 dagar med diett (dagssats 800 kr). '
            'Utlegg: flybillett 2850 kr og taxi 200 kr.'
        )
        self.assertEqual(task.entity, Entity.TRAVEL_EXPENSE)

    @LLM_ONLY
    def test_travel_with_perdiem_FR(self) -> None:
        """Real game pattern: French 3-day trip with per diem."""
        task = self.parser.parse(
            'Enregistrez une note de frais de déplacement pour Adam Martin (adam.martin@example.org) '
            'pour "Visite client Oslo". Le voyage a duré 3 jours avec indemnités journalières '
            '(taux journalier 750 NOK). Dépenses : billet d\'avion 3200 NOK et taxi 350 NOK.'
        )
        self.assertEqual(task.entity, Entity.TRAVEL_EXPENSE)

    @LLM_ONLY
    def test_travel_with_perdiem_PT(self) -> None:
        """Real game pattern: Portuguese 2-day trip."""
        task = self.parser.parse(
            'Registe uma despesa de viagem para Inês Sousa (ines.sousa@example.org) referente a '
            '"Visita cliente Bergen". A viagem durou 2 dias com ajudas de custo (taxa diária 800 NOK). '
            'Despesas: bilhete de avião 5200 NOK e táxi 350 NOK.'
        )
        self.assertEqual(task.entity, Entity.TRAVEL_EXPENSE)

    def test_travel_NO(self) -> None:
        task = self.parser.parse(
            "Registrer reiseregning med diett for ansatt Kari Nordmann. Reise fra Oslo til Bergen, "
            "avreise 2026-04-10, retur 2026-04-12. Hotellutgift 2500 kr og taxi 450 kr."
        )
        self.assertEqual(task.entity, Entity.TRAVEL_EXPENSE)

    def test_travel_EN(self) -> None:
        task = self.parser.parse(
            "Create a travel expense with per diem for employee John Smith. "
            "Travel from Stavanger to Trondheim, departure 2026-04-10, return 2026-04-12."
        )
        self.assertEqual(task.entity, Entity.TRAVEL_EXPENSE)

    @LLM_ONLY
    def test_travel_DE(self) -> None:
        task = self.parser.parse(
            "Reisekostenabrechnung für Mitarbeiter Klaus Weber erstellen. "
            "Reise von Oslo nach Bergen, Abreise 2026-04-14, Rückkehr 2026-04-17."
        )
        self.assertEqual(task.entity, Entity.TRAVEL_EXPENSE)

    @LLM_ONLY
    def test_travel_ES(self) -> None:
        task = self.parser.parse(
            "Registre una nota de gastos de viaje para Elena García. "
            "Viaje de Oslo a Trondheim, salida 2026-04-20, regreso 2026-04-22."
        )
        self.assertEqual(task.entity, Entity.TRAVEL_EXPENSE)


class PaymentParserTest(unittest.TestCase):
    """Payment registration — search-first patterns from competition."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    def test_payment_existing_invoice_NO(self) -> None:
        task = self.parser.parse(
            'Kunden Havbris AS (org.nr 831357983) har en utestående faktura på 5900 kr eksklusiv '
            'MVA for "Konsulenttimer". Registrer full betaling på denne fakturaen.'
        )
        self.assertEqual(task.entity, Entity.PAYMENT)
        self.assertEqual(task.action, Action.REGISTER)

    @LLM_ONLY
    def test_payment_existing_invoice_FR(self) -> None:
        task = self.parser.parse(
            'Le client Montagne SARL (nº org. 893135979) a une facture impayée de 19050 NOK '
            'hors TVA pour "Service réseau". Enregistrez le paiement intégral de cette facture.'
        )
        self.assertEqual(task.entity, Entity.PAYMENT)
        self.assertEqual(task.action, Action.REGISTER)

    def test_payment_existing_invoice_EN(self) -> None:
        task = self.parser.parse(
            'The customer Greenfield Ltd (org no. 853801941) has an outstanding invoice for '
            '34450 NOK excluding VAT for "Consulting Hours". Register full payment of the invoice.'
        )
        self.assertEqual(task.entity, Entity.PAYMENT)
        self.assertEqual(task.action, Action.REGISTER)

    @LLM_ONLY
    def test_payment_existing_invoice_DE(self) -> None:
        task = self.parser.parse(
            'Der Kunde Waldberg GmbH (Org.-Nr. 811223344) hat eine ausstehende Rechnung über '
            '29500 NOK ohne MwSt für "Softwarelizenz". Registrieren Sie die vollständige Zahlung.'
        )
        self.assertEqual(task.entity, Entity.PAYMENT)
        self.assertEqual(task.action, Action.REGISTER)

    @LLM_ONLY
    def test_payment_existing_invoice_ES(self) -> None:
        task = self.parser.parse(
            'El cliente Montaña SL (org. nº 844556677) tiene una factura pendiente de 38000 '
            'NOK sin IVA por "Desarrollo web". Registre el pago total de la factura.'
        )
        self.assertEqual(task.entity, Entity.PAYMENT)
        self.assertEqual(task.action, Action.REGISTER)


class DimensionParserTest(unittest.TestCase):
    """Dimension + voucher parsing."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    def test_dimension_NO(self) -> None:
        task = self.parser.parse(
            'Opprett en fri regnskapsdimensjon "Prosjekttype" med verdiene "Forskning" og "Utvikling". '
            'Bokfør deretter et bilag på konto 6590 for 10800 kr, knyttet til dimensjonsverdien "Forskning".'
        )
        self.assertEqual(task.entity, Entity.DIMENSION)

    def test_dimension_ES(self) -> None:
        task = self.parser.parse(
            'Cree una dimensión contable personalizada "Prosjekttype" con los valores "Forskning" y '
            '"Utvikling". Luego registre un asiento en la cuenta 7000 por 14550 NOK, vinculado al '
            'valor de dimensión "Forskning".'
        )
        self.assertEqual(task.entity, Entity.DIMENSION)

    def test_dimension_EN(self) -> None:
        task = self.parser.parse(
            'Create accounting dimension "Business Unit" with values "Domestic" and "International". '
            'Post a journal entry on account 6800 for 35000 NOK linked to "International".'
        )
        self.assertEqual(task.entity, Entity.DIMENSION)

    @LLM_ONLY
    def test_dimension_FR(self) -> None:
        task = self.parser.parse(
            'Créez une dimension comptable "Département" avec les valeurs "Marketing" et "Technique". '
            'Enregistrez une écriture sur le compte 7200 de 18000 NOK liée à "Marketing".'
        )
        self.assertEqual(task.entity, Entity.DIMENSION)

    def test_dimension_DE(self) -> None:
        task = self.parser.parse(
            'Erstellen Sie eine benutzerdefinierte Buchhaltungsdimension "Region" mit den Werten '
            '"Vestlandet" und "Midt-Norge". Buchen Sie dann einen Beleg auf Konto 6540 über 8600 NOK.'
        )
        self.assertEqual(task.entity, Entity.DIMENSION)


class FixedPriceProjectParserTest(unittest.TestCase):
    """Fixed-price project + milestone invoice."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    @LLM_ONLY
    def test_fixproj_EN(self) -> None:
        task = self.parser.parse(
            'Set a fixed price of 498050 NOK on the project "CRM Integration" for Ridgepoint Ltd '
            '(org no. 844419856). The project manager is George Walker (george.walker@example.org). '
            'Invoice the customer for 50% of the fixed price as a milestone payment.'
        )
        self.assertEqual(task.entity, Entity.INVOICE)

    @LLM_ONLY
    def test_fixproj_ES(self) -> None:
        task = self.parser.parse(
            'Establezca un precio fijo de 266550 NOK en el proyecto "Seguridad de datos" para '
            'Costa Digital SL (org. nº 891505019). El jefe de proyecto es Clara Navarro '
            '(clara.navarro@example.org). Facture al cliente el 50% del precio fijo.'
        )
        self.assertEqual(task.entity, Entity.INVOICE)

    @LLM_ONLY
    def test_fixproj_NO(self) -> None:
        task = self.parser.parse(
            'Sett fastpris 350000 NOK på prosjektet "Nettside redesign" for Havkyst AS '
            '(org. nr. 911223344). Prosjektleder er Marte Vik. Fakturer kunden for 30% av fastprisen.'
        )
        self.assertEqual(task.entity, Entity.INVOICE)

    @LLM_ONLY
    def test_fixproj_DE(self) -> None:
        task = self.parser.parse(
            'Setzen Sie einen Festpreis von 275000 NOK für das Projekt "ERP-Implementierung" '
            'für Alpenland GmbH. Projektleiter ist Thomas Berger. Meilensteinrechnung für 40%.'
        )
        self.assertEqual(task.entity, Entity.INVOICE)


class CreditNoteParserTest(unittest.TestCase):
    """Credit note detection across languages."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    def test_credit_note_NO(self) -> None:
        task = self.parser.parse(
            'Opprett en kreditnota til kunden Nordlys AS for 15000 kr ekskl. mva.'
        )
        self.assertEqual(task.entity, Entity.INVOICE)

    def test_credit_note_PT(self) -> None:
        task = self.parser.parse(
            'O cliente Luz do Sol Lda reclamou sobre a fatura referente a "Licença de software" '
            '(11400 NOK sem IVA). Emita uma nota de crédito total e envie ao cliente.'
        )
        self.assertEqual(task.entity, Entity.INVOICE)

    def test_credit_note_DE(self) -> None:
        task = self.parser.parse(
            'Gutschrift für Kunden Bergkristall GmbH über 18500 NOK ohne MwSt erstellen.'
        )
        self.assertEqual(task.entity, Entity.INVOICE)

    def test_credit_note_EN(self) -> None:
        task = self.parser.parse(
            'Create a full credit note for customer Seaside Corp for 42000 NOK excluding VAT.'
        )
        self.assertEqual(task.entity, Entity.INVOICE)

    @LLM_ONLY
    def test_credit_note_FR(self) -> None:
        task = self.parser.parse(
            'Émettez une note de crédit pour le client Lumière SARL pour 25000 NOK hors TVA.'
        )
        self.assertEqual(task.entity, Entity.INVOICE)


class OrderToInvoicePaymentParserTest(unittest.TestCase):
    """Order -> Invoice -> Payment multi-step parsing."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    @LLM_ONLY
    def test_o2ip_EN(self) -> None:
        task = self.parser.parse(
            'Create an order for the customer Clearwater Ltd (org no. 812933558) with the products '
            'Web Design (4323) at 6300 NOK and Cloud Storage (1409) at 20400 NOK. Convert the order '
            'to an invoice and register full payment.'
        )
        self.assertIn(task.entity, (Entity.PAYMENT, Entity.INVOICE))

    @LLM_ONLY
    def test_o2ip_NO(self) -> None:
        task = self.parser.parse(
            'Opprett en ordre for kunden Sjøstrand AS (org. nr. 944556677) med produktene '
            'Webdesign (prod.nr 2001) til 15000 NOK og Hosting (prod.nr 2002) til 8000 NOK. '
            'Konverter ordren til faktura og registrer full betaling.'
        )
        self.assertIn(task.entity, (Entity.PAYMENT, Entity.INVOICE))

    @LLM_ONLY
    def test_o2ip_DE(self) -> None:
        task = self.parser.parse(
            'Erstellen Sie einen Auftrag für den Kunden Sonnental GmbH (Org.-Nr. 904562262) '
            'mit den Produkten Netzwerkdienst (5874) zu 9150 NOK und Wartung (8734) zu 22150 NOK. '
            'Wandeln Sie den Auftrag in eine Rechnung um und registrieren Sie die vollständige Zahlung.'
        )
        self.assertIn(task.entity, (Entity.PAYMENT, Entity.INVOICE))

    @LLM_ONLY
    def test_o2ip_FR(self) -> None:
        task = self.parser.parse(
            'Créez une commande pour le client Étoile du Nord SARL (nº org. 945678123) avec les '
            'produits Hébergement web (nº 6001) à 15000 NOK et Support technique (nº 6002) à 8000 NOK. '
            'Convertissez en facture et enregistrez le paiement intégral.'
        )
        self.assertIn(task.entity, (Entity.PAYMENT, Entity.INVOICE))

    @LLM_ONLY
    def test_o2ip_ES(self) -> None:
        task = self.parser.parse(
            'Cree un pedido para el cliente Montaña Digital SL (org. 912345678) con los productos '
            'Consultoría (8001) a 28000 NOK y Testing (8003) a 15000 NOK. Convierta a factura y '
            'registre el pago completo.'
        )
        self.assertIn(task.entity, (Entity.PAYMENT, Entity.INVOICE))

    @LLM_ONLY
    def test_o2ip_PT(self) -> None:
        task = self.parser.parse(
            'Crie um pedido para o cliente Oceano Tech Lda (org. 978123456) com o produto '
            'Implementação ERP (prod. nº 9001) a 185000 NOK. Converta em fatura e registe o pagamento.'
        )
        self.assertIn(task.entity, (Entity.PAYMENT, Entity.INVOICE))


class PurchaseOrderParserTest(unittest.TestCase):
    """Purchase order parsing."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    def test_purchase_order_NO(self) -> None:
        task = self.parser.parse(
            "Opprett en innkjøpsordre til leverandør Bygg og Anlegg AS for 50 sekker sement."
        )
        self.assertEqual(task.entity, Entity.PURCHASE_ORDER)

    def test_purchase_order_EN(self) -> None:
        task = self.parser.parse(
            "Create a purchase order to supplier Office Supplies Ltd for 100 reams of paper."
        )
        self.assertEqual(task.entity, Entity.PURCHASE_ORDER)

    def test_purchase_order_DE(self) -> None:
        task = self.parser.parse(
            "Bestellung an Lieferanten Bürobedarf GmbH erstellen: 200 Kugelschreiber à 25 NOK."
        )
        self.assertEqual(task.entity, Entity.PURCHASE_ORDER)


class MultiLineInvoiceParserTest(unittest.TestCase):
    """Multi-line invoices with different VAT rates."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    def test_multiline_mixed_vat_EN(self) -> None:
        """Real game pattern: 3 lines with 25%, 15%, 0% VAT."""
        task = self.parser.parse(
            'Create an invoice for the customer Windmill Ltd (org no. 994973150) with three product '
            'lines: System Development (6517) at 22950 NOK with 25% VAT, Maintenance (5339) at '
            '8900 NOK with 15% VAT (food), and Consulting Hours (8246) at 10450 NOK with 0% VAT (exempt).'
        )
        self.assertEqual(task.entity, Entity.INVOICE)

    def test_multiline_mixed_vat_NO(self) -> None:
        task = self.parser.parse(
            'Opprett faktura til kunde Havbris AS (org.nr 912345670) med linjene: '
            'Konsulenttimer 25000 kr (25% mva), Reisekostnader 8000 kr (0% mva), '
            'Matservering 4500 kr (15% mva).'
        )
        self.assertEqual(task.entity, Entity.INVOICE)

    @LLM_ONLY
    def test_multiline_mixed_vat_FR(self) -> None:
        task = self.parser.parse(
            'Créez une facture pour le client Prairie SARL (nº org. 834343096) avec trois lignes : '
            'Maintenance (1722) à 11550 NOK avec 15 % TVA, Rapport (1282) à 7500 NOK avec 25 % TVA.'
        )
        self.assertEqual(task.entity, Entity.INVOICE)


class PaymentReversalParserTest(unittest.TestCase):
    """Payment reversal / revert."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    @LLM_ONLY
    def test_reversal_ES(self) -> None:
        task = self.parser.parse(
            'El pago de Costa Brava SL (org. nº 866888426) por la factura "Mantenimiento" '
            '(39250 NOK sin IVA) fue devuelto por el banco. Revierta el pago para que la '
            'factura quede abierta nuevamente.'
        )
        self.assertIn(task.entity, (Entity.PAYMENT, Entity.INVOICE))

    @LLM_ONLY
    def test_reversal_NO(self) -> None:
        task = self.parser.parse(
            'Betalingen fra Havbris AS for fakturaen "IT-support" (28500 NOK ekskl. mva) '
            'ble returnert av banken. Reverser betalingen slik at fakturaen blir åpen igjen.'
        )
        self.assertIn(task.entity, (Entity.PAYMENT, Entity.INVOICE))

    @LLM_ONLY
    def test_reversal_EN(self) -> None:
        task = self.parser.parse(
            'The payment from Nordic Solutions for the invoice "Annual Maintenance" '
            '(52000 NOK excluding VAT) was returned by the bank. Reverse the payment.'
        )
        self.assertIn(task.entity, (Entity.PAYMENT, Entity.INVOICE))


class ModuleActivationParserTest(unittest.TestCase):
    """Module activation parsing."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    def test_module_NO(self) -> None:
        task = self.parser.parse("Aktiver modulen for prosjektstyring i Tripletex.")
        self.assertEqual(task.entity, Entity.COMPANY_MODULE)

    @LLM_ONLY
    def test_module_EN(self) -> None:
        task = self.parser.parse("Activate the project management module in Tripletex.")
        self.assertEqual(task.entity, Entity.COMPANY_MODULE)

    @LLM_ONLY
    def test_module_DE(self) -> None:
        task = self.parser.parse("Aktiviere das Modul für Projektmanagement in Tripletex.")
        self.assertEqual(task.entity, Entity.COMPANY_MODULE)


class DepartmentParserTest(unittest.TestCase):
    """Department creation — including batch."""

    def setUp(self) -> None:
        self.parser = PromptParser()

    def test_departments_batch_PT(self) -> None:
        """Real game pattern: Portuguese batch department creation."""
        task = self.parser.parse(
            'Crie três departamentos no Tripletex: "Økonomi", "Innkjøp" e "Regnskap".'
        )
        self.assertEqual(task.entity, Entity.DEPARTMENT)
        self.assertEqual(task.action, Action.CREATE)


if __name__ == "__main__":
    unittest.main()
