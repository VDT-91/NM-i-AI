import requests
import time
import concurrent.futures
import sys

BASE = "https://tripletex-solver-554556611033.europe-north1.run.app"
CREDS = {
    "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
    "session_token": "eyJ0b2tlbklkIjoyMTQ3NjMyMzc3LCJ0b2tlbiI6ImFlZTQyYWQzLTFhYzYtNDM1NS05NTQxLTBiNWMwYTVjZGVmMSJ9",
}

# =============================================================================
# COMPREHENSIVE TRIPLETEX STRESS TEST
# 30 task types × 7 languages × various styles (typos, bad grammar, informal)
#
# Tripletex is a Norwegian accounting/ERP system. Users do:
# - Employee management (HR)
# - Customer/Contact management (CRM)
# - Supplier management
# - Product catalog
# - Invoicing, payments, credit notes
# - Travel expense reports
# - Project management
# - Department structure
# - Vouchers / journal entries (bookkeeping)
# - Timesheet / hour tracking
# =============================================================================

tests = [
    # =========================================================================
    # 1. CREATE EMPLOYEE — admin role, email, phone
    # =========================================================================
    ("01_EMP_create_NO", "Opprett en ny ansatt som heter Kari Nordmann med e-post kari@firma.no. Hun skal være kontoadministrator."),
    ("01_EMP_create_EN", "Create a new employee named John Smith, email john.smith@company.com. He should be an administrator."),
    ("01_EMP_create_ES", "Crea un nuevo empleado llamado Carlos García con correo carlos@empresa.es. Debe ser administrador."),
    ("01_EMP_create_PT", "Crie um novo funcionário chamado João Silva com email joao@empresa.pt. Ele deve ser administrador."),
    ("01_EMP_create_DE", "Erstelle einen neuen Mitarbeiter namens Hans Müller mit E-Mail hans@firma.de. Er soll Kontoverwalter sein."),
    ("01_EMP_create_FR", "Créez un nouvel employé nommé Pierre Dupont avec email pierre@societe.fr. Il doit être administrateur."),
    ("01_EMP_create_NN", "Opprett ein ny tilsett som heiter Kari Nordmann med e-post kari@firma.no. Ho skal vere kontoadministrator."),
    ("01_EMP_create_typo", "Lag en ny ansat som hetr Per Hansen, epost per.hansen@firma.no. Han er kontoadmnistrator."),
    ("01_EMP_create_informal", "hei kan du lage en ansatt? Per Hansen, per@test.no, han ska vær admin"),

    # =========================================================================
    # 2. CREATE EMPLOYEE — basic, no admin, with phone
    # =========================================================================
    ("02_EMP_basic_NO", "Registrer ny ansatt: Lise Berg, e-post lise.berg@bedrift.no, telefon 98765432."),
    ("02_EMP_basic_EN", "Add new employee Sarah Johnson, email sarah@work.com, phone +47 91234567."),
    ("02_EMP_basic_DE", "Neuen Mitarbeiter anlegen: Max Weber, E-Mail max@arbeit.de, Telefon 44556677."),
    ("02_EMP_basic_FR", "Ajoutez un employé: Marie Martin, email marie@travail.fr, téléphone 55667788."),

    # =========================================================================
    # 3. UPDATE EMPLOYEE — change email, phone
    # =========================================================================
    ("03_EMP_update_NO", "Endre e-postadressen til ansatt Kari Nordmann til ny.kari@firma.no"),
    ("03_EMP_update_EN", "Update employee John Smith's email to new.john@company.com"),
    ("03_EMP_update_ES", "Cambia el correo del empleado Carlos García a nuevo.carlos@empresa.es"),
    ("03_EMP_update_PT", "Atualize o email do funcionário João Silva para novo.joao@empresa.pt"),
    ("03_EMP_update_phone", "Oppdater telefonnummeret til Per Hansen til 99887766"),
    ("03_EMP_update_DE", "Ändere die E-Mail des Mitarbeiters Hans Müller zu neu.hans@firma.de"),

    # =========================================================================
    # 4. DELETE EMPLOYEE
    # =========================================================================
    ("04_EMP_delete_NO", "Slett ansatt Lise Berg fra systemet"),
    ("04_EMP_delete_EN", "Delete employee Sarah Johnson"),
    ("04_EMP_delete_ES", "Eliminar al empleado Carlos García del sistema"),
    ("04_EMP_delete_FR", "Supprimez l'employé Pierre Dupont"),

    # =========================================================================
    # 5. CREATE CUSTOMER — with org number, address, email
    # =========================================================================
    ("05_CUST_create_NO", "Opprett kunden Nordlys AS med organisasjonsnummer 872778330. Adressa er Nygata 45, 6003 Ålesund. E-post: post@nordlys.no."),
    ("05_CUST_create_EN", "Create customer Arctic Solutions Ltd with organization number 987654321. Address: 123 Main Street, 0150 Oslo. Email: info@arctic.com"),
    ("05_CUST_create_ES", "Crea un cliente llamado Sol del Norte SL con número de organización 876543210. Dirección: Calle Mayor 10, 0010 Bergen. Email: contacto@soldelnorte.es"),
    ("05_CUST_create_PT", "Crie o cliente Porto Digital Ltda com número de organização 654321987. Endereço: Rua Principal 5, 5003 Trondheim. Email: contato@portodigital.pt"),
    ("05_CUST_create_DE", "Erstelle den Kunden Nordlicht GmbH mit Organisationsnummer 123456789. Adresse: Hauptstraße 22, 7010 Stavanger. E-Mail: info@nordlicht.de"),
    ("05_CUST_create_FR", "Créez le client Lumière du Nord SARL avec numéro d'organisation 111222333. Adresse: Rue de la Paix 8, 4010 Tromsø. Email: contact@lumiere.fr"),
    ("05_CUST_create_NN", "Opprett kunden Vestland Handel AS med organisasjonsnummer 999888777. Adressa er Strandgata 12, 5003 Bergen. E-post: post@vestland.no"),
    ("05_CUST_create_typo", "Lag kunde Fjordtech AS, orgnr 444555666, adrese Sjøgata 7, 8006 Bodø, epost info@fjordtech.no"),
    ("05_CUST_create_informal", "ny kunde: Havblikk AS, org nr 777888999, adresse Brygga 3, 3015 Drammen, mail post@havblikk.no"),

    # =========================================================================
    # 6. CREATE CUSTOMER — simple, just name and email
    # =========================================================================
    ("06_CUST_simple_NO", "Opprett en ny kunde som heter Solgløtt AS med e-post kontakt@solglett.no"),
    ("06_CUST_simple_EN", "Create a new customer called Sunrise Corp with email hello@sunrise.com"),
    ("06_CUST_simple_ES", "Crear cliente Amanecer SL con email info@amanecer.es"),

    # =========================================================================
    # 7. UPDATE CUSTOMER — change email
    # =========================================================================
    ("07_CUST_update_NO", "Endre e-posten til kunden Nordlys AS til ny@nordlys.no"),
    ("07_CUST_update_EN", "Update the email address of customer Arctic Solutions Ltd to new@arctic.com"),
    ("07_CUST_update_ES", "Cambiar el correo del cliente Sol del Norte SL a nuevo@soldelnorte.es"),
    ("07_CUST_update_NN", "Endre e-posten til kunden Vestland Handel AS til ny@vestland.no"),
    ("07_CUST_update_DE", "Ändere die E-Mail des Kunden Nordlicht GmbH zu neu@nordlicht.de"),

    # =========================================================================
    # 8. DELETE CUSTOMER
    # =========================================================================
    ("08_CUST_delete_NO", "Slett kunden Solgløtt AS"),
    ("08_CUST_delete_EN", "Delete customer Sunrise Corp from the system"),
    ("08_CUST_delete_FR", "Supprimez le client Lumière du Nord SARL"),

    # =========================================================================
    # 9. CREATE PRODUCT — with price excl VAT
    # =========================================================================
    ("09_PROD_create_NO", "Opprett et nytt produkt som heter Konsulenttjeneste med pris 1500 kr ekskl. mva"),
    ("09_PROD_create_EN", "Create a new product called Consulting Service with price 2000 NOK excluding VAT"),
    ("09_PROD_create_ES", "Crea un producto llamado Servicio de Consultoría con precio 1800 NOK sin IVA"),
    ("09_PROD_create_PT", "Crie um produto chamado Serviço de Consultoria com preço 2500 NOK sem IVA"),
    ("09_PROD_create_DE", "Erstelle ein Produkt namens Beratungsleistung mit Preis 1200 NOK ohne MwSt"),
    ("09_PROD_create_FR", "Créez un produit appelé Service de Conseil au prix de 1700 NOK hors TVA"),
    ("09_PROD_create_NN", "Lag eit nytt produkt som heiter Rådgjeving og kostar 3000 kr utan mva"),
    ("09_PROD_create_typo", "Lag produkt Raadgiverpaket, pris 5000 uten mva"),
    ("09_PROD_create_informal", "kan du lage et produkt? Webdesign, koster 8000 ekskl mva"),

    # =========================================================================
    # 10. CREATE PRODUCT — with price incl VAT
    # =========================================================================
    ("10_PROD_incl_NO", "Opprett produkt Programvarelisens med pris 2500 kr inkl. mva"),
    ("10_PROD_incl_EN", "Create product Software License priced at 3125 NOK including VAT"),
    ("10_PROD_incl_ES", "Crear producto Licencia de Software con precio 4000 NOK con IVA incluido"),
    ("10_PROD_incl_DE", "Produkt Softwarelizenz anlegen, Preis 3750 NOK inkl. MwSt"),

    # =========================================================================
    # 11. UPDATE PRODUCT — change price
    # =========================================================================
    ("11_PROD_update_NO", "Endre prisen på produktet Konsulenttjeneste til 2000 kr ekskl. mva"),
    ("11_PROD_update_EN", "Update the price of product Consulting Service to 2500 NOK excluding VAT"),
    ("11_PROD_update_ES", "Cambiar el precio del producto Servicio de Consultoría a 2200 NOK sin IVA"),

    # =========================================================================
    # 12. DELETE PRODUCT
    # =========================================================================
    ("12_PROD_delete_NO", "Slett produktet Programvarelisens"),
    ("12_PROD_delete_EN", "Delete the product Software License"),
    ("12_PROD_delete_DE", "Lösche das Produkt Softwarelizenz"),

    # =========================================================================
    # 13. CREATE DEPARTMENT — single
    # =========================================================================
    ("13_DEPT_single_NO", "Opprett en ny avdeling som heter Salg"),
    ("13_DEPT_single_EN", "Create a new department called Sales"),
    ("13_DEPT_single_ES", "Crear un nuevo departamento llamado Ventas"),
    ("13_DEPT_single_PT", "Crie um novo departamento chamado Vendas"),
    ("13_DEPT_single_DE", "Erstelle eine neue Abteilung namens Vertrieb"),
    ("13_DEPT_single_FR", "Créez un nouveau département appelé Ventes"),
    ("13_DEPT_single_NN", "Opprett ei ny avdeling som heiter Sal"),

    # =========================================================================
    # 14. CREATE DEPARTMENTS — multiple
    # =========================================================================
    ("14_DEPT_multi_NO", "Opprett avdelingene Salg, Markedsføring og Utvikling"),
    ("14_DEPT_multi_EN", "Create the departments Sales, Marketing, and Development"),
    ("14_DEPT_multi_ES", "Crear los departamentos Ventas, Marketing y Desarrollo"),
    ("14_DEPT_multi_NN", "Opprett avdelingane Sal, Marknad og Utvikling i systemet"),
    ("14_DEPT_multi_DE", "Erstelle die Abteilungen Vertrieb, Marketing und Entwicklung"),
    ("14_DEPT_multi_FR", "Créez les départements Ventes, Marketing et Développement"),
    ("14_DEPT_multi_informal", "lag avdelinger: IT, HR, Økonomi"),

    # =========================================================================
    # 15. DELETE DEPARTMENT
    # =========================================================================
    ("15_DEPT_delete_NO", "Slett avdelingen Utvikling"),
    ("15_DEPT_delete_EN", "Delete the department called Development"),
    ("15_DEPT_delete_ES", "Eliminar el departamento Desarrollo"),
    ("15_DEPT_delete_DE", "Lösche die Abteilung Entwicklung"),

    # =========================================================================
    # 16. CREATE PROJECT — with customer and manager
    # =========================================================================
    ("16_PROJ_create_NO", "Opprett et prosjekt som heter Digital Transformasjon for kunden Nordlys AS. Prosjektleder er Kari Nordmann."),
    ("16_PROJ_create_EN", "Create a project called Digital Transformation for customer Arctic Solutions Ltd. Project manager is John Smith, john.smith@company.com."),
    ("16_PROJ_create_ES", "Crea un proyecto llamado Transformación Digital para el cliente Sol del Norte SL. El jefe de proyecto es Carlos García."),
    ("16_PROJ_create_PT", "Crie um projeto chamado Transformação Digital para o cliente Porto Digital Ltda. Gerente de projeto: João Silva."),
    ("16_PROJ_create_DE", "Erstelle ein Projekt Digitale Transformation für den Kunden Nordlicht GmbH. Projektleiter ist Hans Müller, hans@firma.de."),
    ("16_PROJ_create_FR", "Créez un projet Transformation Numérique pour le client Lumière du Nord SARL. Chef de projet: Pierre Dupont."),
    ("16_PROJ_create_NN", "Opprett eit prosjekt som heiter Digital Omstilling for kunden Vestland Handel AS."),
    ("16_PROJ_create_typo", "lag prosjekt Nettbutikk for kunden Fjordtech AS, prosjektledr er Per Hansen"),
    ("16_PROJ_create_informal", "nytt prosjekt: Apputvikling, kunde: Havblikk AS"),

    # =========================================================================
    # 17. UPDATE PROJECT — change name
    # =========================================================================
    ("17_PROJ_update_NO", "Endre navnet på prosjektet Digital Transformasjon til Digital Innovasjon"),
    ("17_PROJ_update_EN", "Rename project Digital Transformation to Digital Innovation"),

    # =========================================================================
    # 18. DELETE PROJECT
    # =========================================================================
    ("18_PROJ_delete_NO", "Slett prosjektet Digital Transformasjon"),
    ("18_PROJ_delete_EN", "Delete the project Digital Transformation"),

    # =========================================================================
    # 19. CREATE INVOICE — for customer with amount
    # =========================================================================
    ("19_INV_create_NO", "Opprett en faktura til kunden Nordlys AS for konsulentarbeid, 25000 kr ekskl. mva, fakturadato 2026-03-15."),
    ("19_INV_create_EN", "Create an invoice for customer Arctic Solutions Ltd, for consulting services, 15000 NOK excluding VAT, invoice date 2026-04-01."),
    ("19_INV_create_ES", "Crea una factura para el cliente Sol del Norte SL, por servicios de consultoría, 20000 NOK sin IVA, fecha 15 de marzo de 2026."),
    ("19_INV_create_PT", "Crie uma fatura para o cliente Porto Digital Ltda, por serviços de consultoria, 18000 NOK sem IVA, data 2026-03-20."),
    ("19_INV_create_DE", "Erstelle eine Rechnung für den Kunden Nordlicht GmbH, Beratungsleistung, 22000 NOK ohne MwSt, Rechnungsdatum 2026-04-01."),
    ("19_INV_create_FR", "Créez une facture pour le client Lumière du Nord SARL, pour services de conseil, 16000 NOK hors TVA, date 2026-03-25."),
    ("19_INV_create_NN", "Lag ein faktura til kunden Vestland Handel AS for rådgjeving, 12000 kr utan mva, dato 2026-03-18."),
    ("19_INV_create_typo", "faktura til Fjordtech AS, konsulenttimer, 30000 ekskl mva, dato 2026-03-20"),
    ("19_INV_create_informal", "lag faktura for Havblikk AS, webutvikling, 45000 uten mva"),
    # Invoice with org number (customer doesn't exist yet)
    ("19_INV_with_org_NO", "Lag faktura til kunde Polar Shipping AS, org.nr 998877665, for frakt, beløp 8000 kr ekskl. mva."),
    ("19_INV_with_org_EN", "Invoice customer Northern Lights Inc, org number 112233445, for web development, amount 12000 NOK excl VAT."),

    # =========================================================================
    # 20. CREATE CREDIT NOTE
    # =========================================================================
    ("20_CREDIT_NO", "Opprett en kreditnota for faktura nummer 10001, dato i dag."),
    ("20_CREDIT_EN", "Create a credit note for invoice number 10001, date today."),
    ("20_CREDIT_ES", "Crear una nota de crédito para la factura número 10001, fecha de hoy."),
    ("20_CREDIT_PT", "Crie uma nota de crédito para a fatura número 10001, data de hoje."),
    ("20_CREDIT_DE", "Erstelle eine Gutschrift für Rechnung Nummer 10001, Datum heute."),
    ("20_CREDIT_FR", "Créez un avoir pour la facture numéro 10001, date d'aujourd'hui."),
    ("20_CREDIT_NN", "Lag ein kreditnota for faktura nummer 10001, dato i dag."),
    ("20_CREDIT_typo", "kreditnota for faktura 10001"),

    # =========================================================================
    # 21. REGISTER PAYMENT — for invoice
    # =========================================================================
    ("21_PAY_NO", "Registrer full betaling for faktura nummer 10001, betalingsdato 2026-03-20."),
    ("21_PAY_EN", "Register full payment for invoice number 10001, payment date 2026-03-20."),
    ("21_PAY_ES", "Registrar el pago total de la factura número 10001, fecha de pago 2026-03-20."),
    ("21_PAY_PT", "Registrar pagamento total da fatura número 10001, data de pagamento 2026-03-20."),
    ("21_PAY_DE", "Zahlung für Rechnung 10001 registrieren, Zahlungsdatum 2026-03-20."),
    ("21_PAY_FR", "Enregistrer le paiement total de la facture numéro 10001, date de paiement 2026-03-20."),
    ("21_PAY_NN", "Registrer full betaling for faktura nummer 10001, betalingsdato 2026-03-20."),
    ("21_PAY_partial", "Registrer delbetaling på 5000 kr for faktura 10001, dato 2026-03-20"),
    ("21_PAY_informal", "betal faktura 10001 i dag"),

    # =========================================================================
    # 22. CREATE TRAVEL EXPENSE
    # =========================================================================
    ("22_TRAVEL_NO", "Registrer en reiseregning for ansatt Kari Nordmann. Reise fra Oslo til Bergen, avreise 2026-04-10, retur 2026-04-12. Formål: kundemøte."),
    ("22_TRAVEL_EN", "Create a travel expense for employee John Smith. Travel from Stavanger to Trondheim, departure 2026-04-10, return 2026-04-12. Purpose: client meeting."),
    ("22_TRAVEL_ES", "Registrar un informe de gastos de viaje para el empleado Carlos García. Viaje de Oslo a Tromsø, salida 2026-04-15, regreso 2026-04-17. Propósito: reunión con cliente."),
    ("22_TRAVEL_PT", "Registre uma despesa de viagem para o funcionário João Silva. Viagem de Bergen a Stavanger, partida 2026-04-20, retorno 2026-04-22. Finalidade: conferência."),
    ("22_TRAVEL_DE", "Reisekostenabrechnung für Mitarbeiter Hans Müller erstellen. Reise von Oslo nach Trondheim, Abfahrt 2026-04-10, Rückkehr 2026-04-12. Zweck: Kundentermin."),
    ("22_TRAVEL_FR", "Créez une note de frais de voyage pour l'employé Pierre Dupont. Voyage de Stavanger à Bergen, départ 2026-04-18, retour 2026-04-20. Objet: réunion client."),
    ("22_TRAVEL_NN", "Registrer ein reiserekneskap for tilsett Kari Nordmann. Reise frå Oslo til Bergen, avreise 2026-04-10, retur 2026-04-12. Formål: kundemøte."),
    ("22_TRAVEL_typo", "lag reiseregning for Per Hansen, fra Stavanger til Tromsø, 2026-04-10 til 2026-04-14, kundebesøk"),
    ("22_TRAVEL_bad_grammar", "make travel expense for employee Jan Berg, he go from Stavanger to Trondheim on 2026-04-10, come back 2026-04-12, purpose is client meeting"),

    # =========================================================================
    # 23. UPDATE TRAVEL EXPENSE
    # =========================================================================
    ("23_TRAVEL_upd_NO", "Endre formålet på reiseregning ID 1 til 'intern konferanse'"),
    ("23_TRAVEL_upd_EN", "Update the purpose of travel expense ID 1 to 'internal conference'"),

    # =========================================================================
    # 24. DELETE TRAVEL EXPENSE
    # =========================================================================
    ("24_TRAVEL_del_NO", "Slett reiseregning med ID 1"),
    ("24_TRAVEL_del_EN", "Delete travel expense with ID 1"),
    ("24_TRAVEL_del_ES", "Eliminar el informe de gastos de viaje con ID 1"),

    # =========================================================================
    # 25. CREATE CONTACT — on customer
    # =========================================================================
    ("25_CONT_create_NO", "Legg til en kontaktperson på kunden Nordlys AS. Navn: Lisa Strand, e-post lisa@nordlys.no, telefon 99887766."),
    ("25_CONT_create_EN", "Add a contact person to customer Arctic Solutions Ltd. Name: Emma Wilson, email emma@arctic.com."),
    ("25_CONT_create_ES", "Añadir una persona de contacto al cliente Sol del Norte SL. Nombre: María López, email maria@soldelnorte.es."),
    ("25_CONT_create_PT", "Adicionar um contato ao cliente Porto Digital Ltda. Nome: Ana Santos, email ana@portodigital.pt."),
    ("25_CONT_create_DE", "Kontaktperson beim Kunden Nordlicht GmbH hinzufügen. Name: Sabine Weber, E-Mail sabine@nordlicht.de."),
    ("25_CONT_create_FR", "Ajouter un contact au client Lumière du Nord SARL. Nom: Sophie Martin, email sophie@lumiere.fr."),
    ("25_CONT_create_NN", "Legg til ein kontaktperson på kunden Vestland Handel AS. Namn: Lisa Strand, e-post lisa@vestland.no."),
    ("25_CONT_create_MIX", "Legg til en contact person pa kunden Fjordtech AS. Navn: Helge Vik, email helge@fjordtech.no"),

    # =========================================================================
    # 26. UPDATE CONTACT
    # =========================================================================
    ("26_CONT_update_NO", "Endre e-posten til kontaktperson Lisa Strand til ny.lisa@nordlys.no"),
    ("26_CONT_update_EN", "Update contact Emma Wilson's email to new.emma@arctic.com"),

    # =========================================================================
    # 27. DELETE CONTACT
    # =========================================================================
    ("27_CONT_delete_NO", "Slett kontaktperson Lisa Strand fra kunden Nordlys AS"),
    ("27_CONT_delete_EN", "Delete contact Emma Wilson from customer Arctic Solutions Ltd"),

    # =========================================================================
    # 28. CREATE SUPPLIER
    # =========================================================================
    ("28_SUPP_create_NO", "Opprett en ny leverandør som heter Kontorrekvisita AS med e-post ordre@kontorrekvisita.no"),
    ("28_SUPP_create_EN", "Create a new supplier called Office Supplies Ltd with email orders@officesupplies.com"),
    ("28_SUPP_create_ES", "Crear un proveedor llamado Suministros Express con email pedidos@suministros.es"),
    ("28_SUPP_create_PT", "Crie um fornecedor chamado Fornecimentos Rápidos com email pedidos@fornecimentos.pt"),
    ("28_SUPP_create_DE", "Erstelle einen Lieferanten namens Bürobedarf GmbH mit E-Mail bestellung@buerobedarf.de"),
    ("28_SUPP_create_FR", "Créez un fournisseur appelé Fournitures Express avec email commandes@fournitures.fr"),
    ("28_SUPP_create_NN", "Opprett ein ny leverandør som heiter Kontorrekvisita AS med e-post ordre@kontorrekvisita.no"),
    ("28_SUPP_create_addr", "Lag leverandør Bygg og Anlegg AS, epost bestilling@bygganlegg.no, adresse Industriveien 10, 2003 Lillestrøm"),

    # =========================================================================
    # 29. UPDATE SUPPLIER
    # =========================================================================
    ("29_SUPP_update_NO", "Endre e-posten til leverandøren Kontorrekvisita AS til ny@kontorrekvisita.no"),
    ("29_SUPP_update_EN", "Update supplier Office Supplies Ltd email to new@officesupplies.com"),

    # =========================================================================
    # 30. CREATE VOUCHER — journal entry
    # =========================================================================
    ("30_VOUCH_create_NO", "Opprett et bilag med beskrivelse Husleie mars, dato 2026-03-01. Debet konto 6300, kredit konto 1920, beløp 15000."),
    ("30_VOUCH_create_EN", "Create a voucher with description Office rent March, date 2026-03-01. Debit account 6300, credit account 1920, amount 15000."),
    ("30_VOUCH_create_ES", "Crear un asiento contable con descripción Alquiler oficina marzo, fecha 2026-03-01. Cuenta debe 6300, cuenta haber 1920, importe 15000."),
    ("30_VOUCH_create_PT", "Crie um lançamento contábil com descrição Aluguel escritório março, data 2026-03-01. Conta débito 6300, conta crédito 1920, valor 15000."),
    ("30_VOUCH_create_DE", "Erstelle einen Buchungsbeleg mit Beschreibung Büromiete März, Datum 2026-03-01. Sollkonto 6300, Habenkonto 1920, Betrag 15000."),
    ("30_VOUCH_create_FR", "Créez une écriture comptable avec description Loyer bureau mars, date 2026-03-01. Compte débit 6300, compte crédit 1920, montant 15000."),
    ("30_VOUCH_create_NN", "Lag eit bilag med skildring Husleige mars, dato 2026-03-01. Debet konto 6300, kredit konto 1920, beløp 15000."),
    ("30_VOUCH_create_informal", "bilag: lønn januar, dato 2026-01-31, debet 5000 kredit 2900, 250000 kr"),
    # Different account combos
    ("30_VOUCH_insurance", "Opprett bilag for forsikring, dato 2026-03-15, debet 6400, kredit 1920, beløp 8000"),
    ("30_VOUCH_supplies", "Bilag for kontorrekvisita, 2026-03-10, debet 6500, kredit 1920, 3500 kr"),

    # =========================================================================
    # 31. DELETE VOUCHER
    # =========================================================================
    ("31_VOUCH_del_NO", "Slett bilag med ID 1"),
    ("31_VOUCH_del_EN", "Delete voucher with ID 1"),

    # =========================================================================
    # 32. CREATE TIMESHEET ENTRY
    # =========================================================================
    ("32_TIME_create_NO", "Registrer 7,5 timer for ansatt Kari Nordmann, aktivitet Administrasjon, dato 2026-03-20."),
    ("32_TIME_create_EN", "Register 8 hours for employee John Smith, activity Administration, date 2026-03-20."),
    ("32_TIME_create_ES", "Registrar 6 horas para el empleado Carlos García, actividad Administración, fecha 2026-03-20."),
    ("32_TIME_create_PT", "Registrar 7 horas para o funcionário João Silva, atividade Administração, data 2026-03-20."),
    ("32_TIME_create_DE", "8 Stunden für Mitarbeiter Hans Müller erfassen, Aktivität Verwaltung, Datum 2026-03-20."),
    ("32_TIME_create_FR", "Enregistrer 6 heures pour employé Pierre Dupont, activité Administration, date 2026-03-20."),
    ("32_TIME_create_NN", "Registrer 7,5 timar for tilsett Kari Nordmann, aktivitet Administrasjon, dato 2026-03-20."),
    ("32_TIME_create_proj", "Før 4 timer for Per Hansen på prosjekt Nettbutikk, aktivitet Utvikling, 2026-03-21"),
    ("32_TIME_create_informal", "loggfør 8 timer for lise berg, administrasjon, i dag"),

    # =========================================================================
    # 33. DELETE TIMESHEET ENTRY
    # =========================================================================
    ("33_TIME_del_NO", "Slett timeregistrering med ID 1"),
    ("33_TIME_del_EN", "Delete timesheet entry with ID 1"),

    # =========================================================================
    # EDGE CASES & TRICKY SCENARIOS
    # =========================================================================

    # Multi-step: Invoice requires creating customer + product first
    ("EDGE_invoice_new_cust", "Lag en faktura til ny kunde Skagerak Energi AS, org.nr 998877110, for strøm, 45000 kr ekskl mva, fakturadato 2026-03-15"),

    # Project requires customer + employee (PM)
    ("EDGE_proj_full", "Opprett prosjekt Smartby for kunde Digital Norge AS, org.nr 776655443, prosjektleder Erik Holm, erik@digitalnorge.no"),

    # Norwegian informal with emoji-like expressions
    ("EDGE_informal_NO", "hei! kan du lage en ny ansatt som heter ole bransen? han jobber som admin og mailen hans er ole@test.no"),

    # Very long prompt with lots of details
    ("EDGE_long_prompt", "Jeg trenger at du oppretter en ny kunde i Tripletex. Kundens navn er Bergens Mekaniske Verksted AS og de har organisasjonsnummer 554433221. Adressen deres er Damsgårdsveien 163, 5160 Laksevåg. E-postadressen er post@bmv.no. Telefonnummeret er 55 34 56 78."),

    # Mixed language (Norwegian + English)
    ("EDGE_mixed_lang", "Create en ny employee som heter Anna Larsen, email anna@firma.no, she should be administrator"),

    # Short/terse prompts
    ("EDGE_terse_dept", "avdeling: Logistikk"),
    ("EDGE_terse_prod", "produkt Vedlikeholdsavtale 12000 ekskl mva"),
    ("EDGE_terse_cust", "kunde: Fjelltopp AS, org 887766554, info@fjelltopp.no"),

    # Common Norwegian accounting terms
    ("EDGE_reiseregning", "Registrer reiseregning for Erik Holm, Oslo-Tromsø, 2026-04-01 til 2026-04-03, prosjektmøte"),
    ("EDGE_timeliste", "Før timeliste: 8 timer, Kari Nordmann, Administrasjon, 2026-03-19"),
    ("EDGE_kreditnota", "Lag kreditnota på faktura 10001"),
    ("EDGE_bilag", "Før bilag: Strøm februar, debet 6340 kredit 2400, 12000"),

    # Addresses in different formats
    ("EDGE_addr_format1", "Opprett kunde Tekno AS, Storgata 1, 0155 Oslo, org.nr 998877665"),
    ("EDGE_addr_format2", "Ny kunde: Havfisk AS, adresse: Sjøgata 22, 8514 Narvik, epost post@havfisk.no"),
    ("EDGE_addr_format3", "Create customer Polar Express Ltd, address Kirkegata 15, 9008 Tromsø, org number 112233998"),

    # Employee with department
    ("EDGE_emp_dept", "Opprett ansatt Erik Solberg i avdeling Salg, epost erik@firma.no, administrator"),
    ("EDGE_emp_dept_EN", "Create employee Maria Garcia in department Marketing, email maria@company.com, role administrator"),

    # Invoice with specific due date
    ("EDGE_inv_due", "Faktura til Nordlys AS for prosjektarbeid, 50000 ekskl mva, fakturadato 2026-04-01, forfallsdato 2026-05-01"),

    # Product with product number
    ("EDGE_prod_num", "Opprett produkt Serviceavtale Premium med produktnummer SA-001, pris 9500 ekskl mva"),
    ("EDGE_prod_num_EN", "Create product Maintenance Plan with product number MP-100, price 7500 NOK excluding VAT"),

    # Delete supplier
    ("EDGE_del_supplier", "Slett leverandøren Kontorrekvisita AS"),

    # Customer with phone
    ("EDGE_cust_phone", "Opprett kunde Vindkraft AS, org.nr 776655332, telefon 22334455, epost post@vindkraft.no"),

    # French with accents
    ("EDGE_FR_accents", "Créez un employé nommé François Bélanger avec l'adresse électronique francois@société.fr. Il est administrateur."),

    # Portuguese formal
    ("EDGE_PT_formal", "Por favor, crie um departamento chamado Recursos Humanos no sistema."),

    # Spanish with ñ
    ("EDGE_ES_special", "Crear empleado José María Muñoz, correo jose.munoz@empresa.es, teléfono +47 98765432, administrador"),

    # German formal
    ("EDGE_DE_formal", "Bitte erstellen Sie einen neuen Kunden mit dem Namen Bergwerk AG, Organisationsnummer 334455667, E-Mail kontakt@bergwerk.de, Adresse Industrieweg 5, 0473 Oslo."),

    # Nynorsk travel expense
    ("EDGE_NN_travel", "Registrer ein reiserekneskap for Kari Nordmann. Ho reiser frå Stavanger til Tromsø den 2026-04-05 og kjem tilbake 2026-04-07. Formål: fagkonferanse."),

    # Multiple products
    ("EDGE_multi_prod", "Lag produktene Grunnpakke, Standardpakke og Premiumpakke"),

    # Customer update with new address
    ("EDGE_cust_upd_addr", "Endre adressen til kunden Nordlys AS til Strandveien 10, 6004 Ålesund"),

    # =========================================================================
    # TIER 2 & TIER 3 — COMPLEX / MULTI-STEP TASKS
    # =========================================================================

    # =========================================================================
    # 34. INCOMING INVOICE (Tier 2)
    # =========================================================================
    ("34_INCINV_NO", "Registrer en innkommende faktura fra leverandør Bygg og Anlegg AS på 120000 kr ekskl. mva, fakturadato 2026-03-10, forfallsdato 2026-04-10. Konto 4300."),
    ("34_INCINV_EN", "Register an incoming invoice from supplier Office Supplies Ltd for 85000 NOK excluding VAT, invoice date 2026-03-12, due date 2026-04-12. Account 4300."),
    ("34_INCINV_ES", "Registrar una factura entrante del proveedor Suministros Express por 95000 NOK sin IVA, fecha de factura 2026-03-14, fecha de vencimiento 2026-04-14. Cuenta 4300."),
    ("34_INCINV_DE", "Eingangsrechnung vom Lieferanten Bürobedarf GmbH über 110000 NOK ohne MwSt erfassen, Rechnungsdatum 2026-03-11, Fälligkeitsdatum 2026-04-11. Konto 4300."),
    ("34_INCINV_FR", "Enregistrer une facture fournisseur de Fournitures Express pour 72000 NOK hors TVA, date de facture 2026-03-13, date d'échéance 2026-04-13. Compte 4300."),
    ("34_INCINV_terse", "innkommende faktura Bygg og Anlegg AS 120000 dato 2026-03-10"),

    # =========================================================================
    # 35. PAY SUPPLIER INVOICE (Tier 2)
    # =========================================================================
    ("35_PAYSUPP_NO", "Betal leverandørfaktura fra Bygg og Anlegg AS på 120000 kr, betalingsdato 2026-03-20. Bruk konto 1920."),
    ("35_PAYSUPP_EN", "Pay supplier invoice from Office Supplies Ltd for 85000 NOK, payment date 2026-03-20. Use account 1920."),

    # =========================================================================
    # 36. MULTI-LINE INVOICE (Tier 2)
    # =========================================================================
    ("36_MULTILINE_NO", "Lag faktura til kunde Nordlys AS med org.nr 872778330: Konsulentarbeid 10000 kr (25% mva), Reisekostnader 5000 kr (0% mva), Programvare 8000 kr (25% mva)."),
    ("36_MULTILINE_EN", "Create an invoice for customer Arctic Solutions Ltd with org number 987654321: Consulting 10000 NOK (25% VAT), Travel expenses 5000 NOK (0% VAT), Software 8000 NOK (25% VAT)."),

    # =========================================================================
    # 37. FIXED-PRICE PROJECT + INVOICE (Tier 2)
    # =========================================================================
    ("37_FIXPRICE_NO", "Opprett et fastprisprosjekt kalt Nettsideredesign for kunde Nordlys AS med totalbeløp 150000 kr ekskl. mva. Prosjektleder er Kari Nordmann. Fakturer kunden for hele beløpet med fakturadato 2026-04-01."),
    ("37_FIXPRICE_EN", "Create a fixed-price project called Website Redesign for customer Arctic Solutions Ltd with total amount 150000 NOK excl VAT. Project manager is John Smith. Invoice the customer for the full amount with invoice date 2026-04-01."),

    # =========================================================================
    # 38. SALARY TRANSACTION (Tier 2)
    # =========================================================================
    ("38_SALARY_NO", "Registrer lønnskjøring for ansatt Kari Nordmann for mars 2026. Bruttolønn 45000 kr, skattetrekk 35%, feriepenger 12%."),
    ("38_SALARY_EN", "Register a salary payment for employee John Smith for March 2026. Gross salary 45000 NOK, tax deduction 35%, holiday pay 12%."),
    ("38_SALARY_FR", "Enregistrer une transaction salariale pour l'employé Pierre Dupont pour mars 2026. Salaire brut 45000 NOK, retenue fiscale 35%, pécule de vacances 12%."),
    ("38_SALARY_DE", "Gehaltsabrechnung für Mitarbeiter Hans Müller für März 2026 erfassen. Bruttogehalt 45000 NOK, Steuerabzug 35%, Urlaubsgeld 12%."),
    ("38_SALARY_ES", "Registrar transacción salarial para el empleado Carlos García de marzo 2026. Salario bruto 45000 NOK, deducción fiscal 35%, paga de vacaciones 12%."),

    # =========================================================================
    # 39. PURCHASE ORDER (Tier 2)
    # =========================================================================
    ("39_PO_NO", "Opprett en innkjøpsordre til leverandør Bygg og Anlegg AS for 50 sekker sement à 250 kr ekskl. mva, leveringsdato 2026-04-15."),
    ("39_PO_EN", "Create a purchase order to supplier Office Supplies Ltd for 100 reams of paper at 150 NOK each excl VAT, delivery date 2026-04-15."),
    ("39_PO_DE", "Bestellung an Lieferanten Bürobedarf GmbH erstellen: 200 Kugelschreiber à 25 NOK ohne MwSt, Lieferdatum 2026-04-15."),

    # =========================================================================
    # 40. TRAVEL WITH PER DIEM (Tier 2)
    # =========================================================================
    ("40_TRAVEL_DIEM_NO", "Registrer reiseregning med diett for ansatt Kari Nordmann. Reise fra Oslo til Bergen, avreise 2026-04-10 kl 08:00, retur 2026-04-12 kl 18:00. Formål: kundemøte. Diett innenlands med overnatting."),
    ("40_TRAVEL_DIEM_EN", "Create a travel expense with per diem for employee John Smith. Travel from Stavanger to Trondheim, departure 2026-04-10 at 08:00, return 2026-04-12 at 18:00. Purpose: client meeting. Domestic per diem with overnight stay."),

    # =========================================================================
    # 41. DIMENSION + VOUCHER (Tier 3)
    # =========================================================================
    ("41_DIM_NO", "Opprett dimensjon Region med verdiene Vestlandet og Midt-Norge. Før et bilag på konto 6860 for 31050 kr knyttet til dimensjonsverdien Vestlandet."),
    ("41_DIM_EN", "Create a dimension called Region with values West Norway and Central Norway. Post a voucher on account 6860 for 31050 NOK linked to dimension value West Norway."),

    # =========================================================================
    # 42. LEAVE OF ABSENCE (Tier 3)
    # =========================================================================
    ("42_LEAVE_NO", "Registrer fravær for ansatt Kari Nordmann. Type: sykmelding. Fra 2026-04-01 til 2026-04-14. 100% fravær."),
    ("42_LEAVE_EN", "Register a leave of absence for employee John Smith. Type: sick leave. From 2026-04-01 to 2026-04-14. 100% absence."),
    ("42_LEAVE_DE", "Abwesenheit für Mitarbeiter Hans Müller erfassen. Art: Krankmeldung. Vom 2026-04-01 bis 2026-04-14. 100% Abwesenheit."),

    # =========================================================================
    # 43. ASSET (Tier 3)
    # =========================================================================
    ("43_ASSET_NO", "Registrer et anleggsmiddel: Kontormaskin, anskaffelsesdato 2026-01-15, anskaffelseskost 85000 kr, avskrivningstid 5 år, lineær avskrivning. Konto 1200."),
    ("43_ASSET_EN", "Register a fixed asset: Office Machine, acquisition date 2026-01-15, acquisition cost 85000 NOK, depreciation period 5 years, straight-line depreciation. Account 1200."),

    # =========================================================================
    # 44. CREDIT NOTE BY CUSTOMER (Tier 2)
    # =========================================================================
    ("44_CREDIT_CUST_NO", "Opprett en kreditnota til kunden Nordlys AS for 15000 kr ekskl. mva. Beskrivelse: retur av defekt vare. Dato 2026-03-18."),
    ("44_CREDIT_CUST_EN", "Create a credit note for customer Arctic Solutions Ltd for 15000 NOK excl VAT. Description: return of defective goods. Date 2026-03-18."),
    ("44_CREDIT_CUST_PT", "Crie uma nota de crédito para o cliente Porto Digital Ltda por 15000 NOK sem IVA. Descrição: devolução de mercadoria com defeito. Data 2026-03-18."),

    # =========================================================================
    # 45. REVERT PAYMENT (Tier 2)
    # =========================================================================
    ("45_REVERT_NO", "Reverser betalingen på faktura 10001. Betalingen var feilregistrert."),
    ("45_REVERT_EN", "Revert the payment on invoice 10001. The payment was registered incorrectly."),

    # =========================================================================
    # 46. MODULE ACTIVATION (Tier 2)
    # =========================================================================
    ("46_MODULE_NO", "Aktiver modulen for prosjektstyring i Tripletex."),
    ("46_MODULE_EN", "Activate the project management module in Tripletex."),
    ("46_MODULE_DE", "Aktiviere das Modul für Projektmanagement in Tripletex."),

    # =========================================================================
    # 47. TIMESHEET + INVOICE (Tier 3)
    # =========================================================================
    ("47_TIME_INV_NO", "Registrer 40 timer for ansatt Kari Nordmann på prosjekt Digital Transformasjon, aktivitet Utvikling, uke 12 2026. Fakturer deretter kunden Nordlys AS for timene med timepris 1200 kr ekskl. mva."),
    ("47_TIME_INV_EN", "Register 40 hours for employee John Smith on project Digital Transformation, activity Development, week 12 2026. Then invoice customer Arctic Solutions Ltd for the hours at hourly rate 1200 NOK excl VAT."),
]


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


if __name__ == "__main__":
    # Allow filtering by prefix: python stress_test.py 19_INV
    prefix = sys.argv[1] if len(sys.argv) > 1 else None
    selected = [(n, p) for n, p in tests if not prefix or n.startswith(prefix)]

    print(f"Running {len(selected)} tests (workers=3)...")
    print(f"{'NAME':40s} | STS |  TIME | RESPONSE")
    print("-" * 100)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_test, n, p): n for n, p in selected}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            results.append(r)
            status = f"\033[92m{r[1]}\033[0m" if r[1] == 200 else f"\033[91m{r[1]}\033[0m"
            print(f"{r[0]:40s} | {status} | {r[2]:5.1f}s | {r[3]}")

    print()
    print("=" * 100)
    ok = sum(1 for r in results if r[1] == 200)
    fail = len(results) - ok
    print(f"Total: {len(results)} | OK: {ok} | Failed: {fail}")
    if fail:
        print(f"\nFailed tests:")
        for r in sorted(results, key=lambda x: x[0]):
            if r[1] != 200:
                print(f"  {r[0]:40s} | {r[1]} | {r[3]}")
