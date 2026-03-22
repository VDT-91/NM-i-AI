from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import unicodedata
from typing import Any

from tripletex_solver.errors import ParsingError
from tripletex_solver.models import Action, Entity, ParsedTask

LLM_TIMEOUT_SECONDS = 55

LOGGER = logging.getLogger(__name__)


def _normalize_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return decomposed.encode("ascii", "ignore").decode("ascii")


def _contains_any_ascii(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _normalize_ascii(text)
    return any(_normalize_ascii(phrase) in normalized for phrase in phrases)


_LLM_CREDIT_NOTE_KEYWORDS = (
    "credit note",
    "credit memo",
    "kreditnota",
    "kreditnote",
    "nota de credito",
    "gutschrift",
    "storno",
    "omgjøring",
    "full reversal",
    "reversal",
    "revert",
    "revertere",
)

SYSTEM_PROMPT = """You are an expert accountant AI that understands how Tripletex (Norwegian accounting ERP) works internally.
Your job: read a task prompt in ANY language and figure out what needs to happen in Tripletex to accomplish it.

## How Tripletex Works (use this knowledge to reason about tasks)

**The account starts EMPTY.** Every entity must be created from scratch.

**Entity dependencies (what requires what):**
- Invoice → needs Customer + Order + Order Lines (with products or descriptions)
- Payment → needs Customer + Invoice (create both first, then pay)
- Project → needs Customer + Project Manager (employee with PM access)
- Project Invoice → needs Customer + Project + Employee + Activity + Timesheet entries, THEN create Order+Invoice
- Timesheet → needs Employee + Activity (linked to Project if specified)
- Salary Transaction → needs Employee + Employment (linked to a Division/virksomhet with org number) + Employment must start BEFORE the salary period
- Travel Expense → needs Employee + travel details (departure, destination, dates)
- Credit Note → needs the original Invoice (find or create it, then credit)
- Order → needs Customer + Order Lines (optionally with Products that have product numbers)
- Contact → needs Customer to attach to
- Incoming Invoice → needs Supplier

**Multi-step workflows (prompt asks for A + B + C):**
When a prompt asks for MULTIPLE things (e.g., "create order, convert to invoice, register payment"), you must detect the FULL workflow. Common patterns:
- Order → Invoice → Payment: entity="order", workflow="orderToInvoiceAndPayment"
- Order → Invoice (no payment): entity="order", workflow="orderToInvoice"
- Timesheet hours → Project Invoice: entity="invoice", workflow="projectInvoice". Extract ALL employees with their hours into "employees" array: [{name, email, hours, role}]. Also extract supplierCosts if any: [{supplierName, supplierOrgNumber, amount}]
- Credit note on invoice: entity="invoice", workflow="creditNote"
- Fixed-price project + milestone invoice: entity="invoice" (we handle the project creation)
- Bank reconciliation with CSV: entity="bank_statement", workflow="reconcile" (we parse the CSV and create payments/vouchers)
- Month-end closing / journal entries / accruals / depreciation: entity="voucher" with postings array. Each posting pair = {debitAccount, creditAccount, amount}
  - Accrual reversal (Rechnungsabgrenzung/periodisering/devengo): debit expense account, credit prepaid account (17xx). The expense account depends on the prepaid type (rent→6300, insurance→6400, etc.)
  - Depreciation (Abschreibung/avskrivning/depreciación/amortissement): debit depreciation expense (6010/7010), credit accumulated depreciation (12x9). Monthly = annual/12, annual = cost/years
  - Salary provision (provisão salarial/lønnsavsetning/Gehaltsrückstellung): debit salary expense (5000), credit accrued salaries (2900). If no amount specified, use a reasonable estimate (e.g. 1/12 of annual salary costs or the monthly amount mentioned elsewhere in the prompt)
  - CRITICAL: If the prompt asks for MULTIPLE journal entries, you MUST include ALL of them as separate items in the postings array. Do NOT skip any, even if the amount must be estimated

**Amounts and pricing:**
- "ekskl. mva"/"sin IVA"/"excluding VAT"/"hors TVA"/"ohne MwSt" → price EXCLUDING VAT
- "inkl. mva"/"con IVA"/"including VAT"/"TTC"/"einschließlich MwSt" → price INCLUDING VAT
- VAT rates in Norway: 0%, 12%, 15%, or 25% (default 25%)
- For invoices with multiple lines at different prices: use "orderLines" array
- For outgoing invoices: "amount" is always the TOTAL excluding VAT
- For incoming invoices: if "including VAT" → use "totalAmountIncludingVat" field, if "excluding VAT" → use "amount" field

**CRITICAL: Do NOT invent or hallucinate data.**
- Only include fields that are EXPLICITLY stated or clearly implied in the prompt.
- Do NOT guess dates — if no date is mentioned, OMIT the date field entirely. The system will use today's date as default.
- Do NOT invent email addresses, phone numbers, or other identifiers not in the prompt.
- NEVER use placeholder strings like "MISSING_EXPENSE_ACCOUNT" or "MISSING_TAXABLE_PROFIT" — if you cannot determine a value, use a reasonable Norwegian accounting default:
  - Prepaid expense reversal from 1700: debit the relevant expense account (e.g. 6300 for rent, 7500 for insurance, or 6800 for office costs). If unclear, use 6300.
  - Tax cost: use account 8300 (betalbar skatt) debit, 2500 (betalbar skatt) credit. Calculate the amount as percentage of the stated taxable result.
  - If you truly cannot calculate an amount, OMIT that posting entirely rather than using a placeholder.

**Roles and access:**
- "administrator"/"kontoadministrator"/"administrador" → role: "administrator"
- "revisor"/"auditor" → role: "auditor"
- "regnskapsfører"/"accountant" → role: "accountant"

## Output Format

Return ONLY valid JSON:
{
  "reasoning": "Think step by step: What is the prompt asking? What entities need creating? What's the full workflow? What data do I need to extract?",
  "action": "create|update|delete|register",
  "entity": "<see entity list below>",
  "workflow": null or "orderToInvoice"|"orderToInvoiceAndPayment"|"projectInvoice"|"creditNote",
  "target_name": "primary entity name or null",
  "identifier": null or integer ID,
  "attributes": { <all extracted data — see field reference below> }
}

## Entity List
employee, customer, department, product, project, invoice, payment, travel_expense, contact, supplier, voucher, timesheet, company_module, incoming_invoice, bank_statement, salary_transaction, purchase_order, dimension, account, order, reminder, activity, division, leave_of_absence, next_of_kin, customer_category, employee_category, asset, product_group, project_category, inventory, inventory_location, stocktaking, goods_receipt, document_archive, event_subscription

**Multi-language entity detection (detect in ANY language):**
- salary/payroll/nómina/lønnsbilag/Gehaltsabrechnung/paie → salary_transaction
- invoice/factura/faktura/Rechnung/fatura → invoice
- order/pedido/ordre/Auftrag/commande → order
- travel expense/reiseregning/viaje/Reisekostenabrechnung → travel_expense
- supplier/leverandør/proveedor/Lieferant/fournisseur → supplier
- purchase order/innkjøpsordre/orden de compra/Bestellung → purchase_order
- credit note/kreditnota/nota de crédito/Gutschrift → invoice with workflow="creditNote"
- timesheet/timeliste/hoja de horas/Stundenzettel → timesheet
- reminder/purring/recordatorio/Mahnung → reminder
- leave/permisjon/permiso/Urlaub/congé → leave_of_absence
- next of kin/pårørende/pariente/Angehörige → next_of_kin
- fixed asset/anleggsmiddel/activo fijo/Anlagevermögen → asset
- dimension/dimensjon/dimensión/dimensão → dimension
- account/konto/cuenta/compte → account
- activity/aktivitet/actividad/Aktivität → activity
- division/divisjon/división/Abteilung → division
- warehouse/lager/almacén/Warenlager → inventory
- stocktaking/varetelling/inventur/recuento → stocktaking
- goods receipt/varemottak/recepción/Wareneingang → goods_receipt
- webhook/event subscription/hendelsesabonnement → event_subscription
- document archive/dokumentarkiv/archivo → document_archive
- module/modul/módulo → company_module
- voucher/bilag/comprobante/Buchung/écriture/journal entry/journal posting → voucher
- month-end closing/Monatsabschluss/månedsslutt/månedsavslutning/cierre mensual/clôture mensuelle → voucher (create journal entries)
- accrual/periodisering/Rechnungsabgrenzung/devengo/régularisation → voucher
- depreciation/avskrivning/Abschreibung/depreciación/amortissement → voucher (unless creating the asset itself)
- payment/betaling/pago/Zahlung/pagamento → payment
- revert payment/reverter/annuler/rückgängig → action: "delete", entity: "payment"
- bank reconciliation/bankavstemming/bankutskrift/extrato bancário/extracto bancario/Kontoauszug/relevé bancaire/reconcilie → bank_statement
- bank statement CSV/kontoutskrift/kontoutdrag → bank_statement

## Field Reference (use these EXACT field names in attributes)

**Employee:** firstName, lastName, email, role, departmentName, phoneNumberWork, phoneNumberMobile, dateOfBirth, startDate, nationalIdentityNumber, employeeNumber, addressLine1, postalCode, city, bankAccountNumber, annualSalary, monthlySalary, hourlyWage, percentageOfFullTimeEquivalent, hoursPerDay (standard working hours per day, e.g. 7.5 or 6.0), employmentForm (PERMANENT for "fast stilling"/"fast ansettelse"/"permanent position", TEMPORARY for "midlertidig"/"temporary"), occupationCode (STYRK/stillingskode/profession code — extract the 4-digit numeric code, e.g. "3323")

**Customer:** name, organizationNumber, email, phoneNumber, phoneNumberMobile, addressLine1, postalCode, city, customerNumber, website, isPrivateIndividual, invoicesDueIn, deliveryAddressLine1, deliveryPostalCode, deliveryCity

**Product:** name, number, priceExcludingVatCurrency, priceIncludingVatCurrency, costExcludingVatCurrency, productUnit, vatRate

**Project:** name, customerName, projectManagerName, projectManagerEmail, orgNumber, endDate, projectNumber, reference, fixedPrice, isFixedPrice

**Invoice:** customerName, customerEmail, organizationNumber, productName, productNumber, quantity, amount (total excl VAT), invoiceDate, invoiceDueDate, invoiceNumber, comment, projectName, rate, hours, employeeName, employeeEmail, activityName, orderLines[{description, productNumber, amount, vatRate, quantity}], employees[{name, email, hours, role}] (for project invoice with MULTIPLE employees — extract ALL), supplierCosts[{supplierName, supplierOrgNumber, amount}] (costs from external suppliers in project), projectBudget (the total project budget/fixed price)

**Order:** customerName, organizationNumber, orderDate, deliveryDate, orderLines[{description, productNumber, amount, quantity}], amount

**Travel Expense:** employeeName, employeeEmail, departureFrom, destination, departureDate, returnDate, purpose, title, projectName, amount, hasPerDiem, perDiemRate, perDiemDailyRate, perDiemDays, kilometers, expenseLines[{description, amount}]

**Contact:** customerName, firstName, lastName, email, phoneNumberWork

**Supplier:** name, organizationNumber, supplierNumber, email, phoneNumber, addressLine1, postalCode, city

**Voucher:** description, date, postings[{debitAccount, creditAccount, amount, description}] (use postings array for MULTIPLE journal entries in one task — e.g. accrual + depreciation). For single posting: debitAccountNumber, creditAccountNumber, amount

**Timesheet:** employeeName, employeeEmail, activityName, projectName, date, hours, comment

**Salary Transaction:** employeeName, employeeEmail, date, baseSalary, bonus, monthlySalary

**Department:** name, departmentNumber, departmentManagerName

**Company Module:** moduleName (SMART, PROJECT, WAGE, LOGISTICS, etc.)

**Incoming Invoice:** supplierName, organizationNumber, invoiceDate, invoiceDueDate, amount (excl VAT), totalAmountIncludingVat (use this when prompt says "including VAT"/"inkl. mva"/"con IVA"/"TTC"/"einschließlich MwSt" OR when prices come from a receipt/kvittering where item prices INCLUDE VAT), vatRate, invoiceNumber, debitAccountNumber, creditAccountNumber, departmentName, description (what the invoice is for). NOTE: Norwegian receipts (kvittering) list prices INCLUDING MVA — use totalAmountIncludingVat for receipt amounts.

**Purchase Order:** supplierName, orderDate, deliveryDate

**Dimension:** dimensionName, dimensionValues[] (IMPORTANT: list of string values like ["Value1","Value2"]), voucherAccountNumber (account number for linked voucher), voucherAmount, voucherDimensionValue (which dimension value to link the voucher to), voucherDate, voucherDescription

**Account:** accountNumber, accountName, vatRate

**Reminder:** invoiceNumber, customerName, reminderDate, dunningFeeAmount, dunningFeeDebitAccount, dunningFeeCreditAccount, partialPaymentAmount, createDunningInvoice (boolean — true if prompt asks to invoice the fee)
IMPORTANT: If the prompt says "find the overdue invoice" or "one of your customers" without naming them, set customerName to null. The system will search for overdue invoices.

**Payment:** customerName, organizationNumber, invoiceNumber, amount, productName, paymentDate, orderLines[{description, productNumber, amount, quantity}], currencyCode, originalExchangeRate, paymentExchangeRate, isPartialPayment (boolean)

**Activity:** activityName, activityNumber, activityType, isChargeable, rate

**Division:** divisionName, startDate, organizationNumber

**Leave of Absence:** employeeName, startDate, endDate, percentage, leaveType (LEAVE_OF_ABSENCE/FURLOUGH/PARENTAL_BENEFITS/MILITARY_SERVICE/EDUCATIONAL/COMPASSIONATE)

**Next of Kin:** employeeName, nextOfKinName, phoneNumber, address, typeOfRelationship (SPOUSE/PARTNER/PARENT/CHILD/SIBLING)

**Customer/Employee Category:** categoryName, categoryNumber

**Asset:** assetName, acquisitionCost, dateOfAcquisition, lifetime (months), depreciationMethod (STRAIGHT_LINE/MANUAL/TAX_RELATED), accountNumber, depreciationAccountNumber

**Inventory:** inventoryName, inventoryNumber, isMainInventory
**Inventory Location:** inventoryName, locationName
**Stocktaking:** inventoryName, date, comment, productLines[{productName, count}]
**Goods Receipt:** supplierName, purchaseOrderId, registrationDate, comment
**Document Archive:** entityType, entityName, fileName
**Bank Statement:** bankAccountNumber, workflow (set to "reconcile" if the prompt asks to reconcile/match the bank statement with invoices)

**Event Subscription:** event, targetUrl, fields, authHeaderName, authHeaderValue

## Critical Rules
- Extract EVERYTHING from the prompt — every name, number, amount, date, org number, address
- Multiple entities of same type → use "names" array: ["Dept A", "Dept B", "Dept C"]
- Dates → YYYY-MM-DD format. No year given → use 2026
- Amounts → numbers, not strings
- Organization numbers → 9-digit strings as "organizationNumber"
- Addresses → split into addressLine1, postalCode (4 digits in Norway), city
- "privatperson"/"private individual" → isPrivateIndividual: true
- "betalingsfrist"/"payment terms"/"due in X days" → invoicesDueIn: X
- Fixed price → fixedPrice (number) + isFixedPrice: true. If prompt mentions X% milestone → amount = fixedPrice * X/100
- target_name: employee="First Last", customer/product/supplier="name", invoice/payment="customerName"
- For delete with no ID, extract enough info to find the entity (name, employee name, etc.)
- Return ONLY the JSON, nothing else
"""


def parse_with_llm(prompt: str, *, few_shot_text: str = "") -> ParsedTask:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ParsingError("GEMINI_API_KEY not set — cannot use LLM parser")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ParsingError("google-genai not installed — cannot use LLM parser")

    # Build system prompt, optionally enriched with few-shot examples
    full_system = SYSTEM_PROMPT
    if few_shot_text:
        full_system = SYSTEM_PROMPT + "\n" + few_shot_text
        LOGGER.info("LLM prompt enriched with %d chars of few-shot examples", len(few_shot_text))

    client = genai.Client(api_key=api_key)

    def _call_gemini():
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(text=full_system),
                        types.Part.from_text(text=f"Task prompt:\n{prompt}"),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_gemini)
            response = future.result(timeout=LLM_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        raise ParsingError(f"Gemini API call timed out after {LLM_TIMEOUT_SECONDS}s")

    raw = response.text or ""
    LOGGER.info("LLM raw response: %s", raw[:2000])

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ParsingError(f"LLM returned invalid JSON: {raw[:300]}")

    # Gemini sometimes wraps the response in a JSON array
    if isinstance(data, list):
        if not data:
            raise ParsingError("LLM returned empty JSON array")
        # Merge multi-element arrays where all tasks are vouchers:
        # combine their postings into a single task so _create_voucher
        # handles all of them (e.g. year-end closing with 3 depreciations).
        if len(data) > 1 and all(
            isinstance(d, dict) and d.get("entity") == "voucher" for d in data
        ):
            merged_postings = []
            for d in data:
                attrs = d.get("attributes", {})
                p_list = attrs.get("postings")
                if p_list and isinstance(p_list, list):
                    merged_postings.extend(p_list)
                elif attrs.get("debitAccountNumber") and attrs.get("creditAccountNumber"):
                    merged_postings.append({
                        "debitAccount": attrs["debitAccountNumber"],
                        "creditAccount": attrs["creditAccountNumber"],
                        "amount": attrs.get("amount"),
                        "description": attrs.get("description"),
                    })
            if merged_postings:
                data[0].setdefault("attributes", {})["postings"] = merged_postings
                LOGGER.info("Merged %d voucher tasks into single task with %d posting pairs",
                            len(data), len(merged_postings))
        data = data[0]

    if not isinstance(data, dict):
        raise ParsingError(f"LLM returned unexpected JSON type: {type(data).__name__}")

    action_str = data.get("action", "")
    entity_str = data.get("entity", "")

    try:
        action = Action(action_str)
    except ValueError:
        raise ParsingError(f"LLM returned unknown action: {action_str}")

    try:
        entity = Entity(entity_str)
    except ValueError:
        raise ParsingError(f"LLM returned unknown entity: {entity_str}")

    attributes = data.get("attributes", {})

    # Store workflow in attributes if LLM provided it
    # Log the LLM's reasoning
    reasoning = data.get("reasoning")
    if reasoning:
        LOGGER.info("LLM reasoning: %s", reasoning[:500])

    workflow = data.get("workflow")
    if workflow:
        attributes["workflow"] = workflow

    # Ensure date objects are strings
    for key, val in list(attributes.items()):
        if val is None:
            del attributes[key]

    task = ParsedTask(
        action=action,
        entity=entity,
        raw_prompt=prompt,
        target_name=data.get("target_name"),
        identifier=data.get("identifier"),
        attributes=attributes,
    )

    # --- Entity correction: fixed-price project + milestone invoice ---
    # When the prompt says "set fixed price on project X + invoice the customer",
    # the LLM sometimes returns entity=project. Redirect to invoice if we have
    # both fixedPrice and an amount (the milestone), OR fixedPrice + invoice keywords.
    _INVOICE_KEYWORDS = ("faktur", "invoice", "rechnung", "factur", "fatura")
    if (
        task.action is Action.CREATE
        and task.entity is Entity.PROJECT
        and task.attributes.get("fixedPrice")
        and (task.attributes.get("amount") or _contains_any_ascii(prompt, _INVOICE_KEYWORDS))
    ):
        task.entity = Entity.INVOICE
        # Copy project name from "name" to "projectName" if not already set
        if not task.attributes.get("projectName") and task.attributes.get("name"):
            task.attributes["projectName"] = task.attributes["name"]
        LOGGER.info("Redirected entity from project to invoice (fixed-price + milestone)")

    if task.action is Action.CREATE and task.entity is Entity.INVOICE:
        if task.attributes.get("workflow") != "creditNote" and _contains_any_ascii(prompt, _LLM_CREDIT_NOTE_KEYWORDS):
            task.attributes["workflow"] = "creditNote"
        if task.attributes.get("workflow") == "creditNote":
            if re.search(
                r"\b(sem|sin|without|excl|excluding|uten)\b.*\b(iva|mva|tax|vat|tva)\b",
                _normalize_ascii(prompt),
            ):
                task.attributes["amountIsVatExclusive"] = True
            elif re.search(
                r"\b(incl|with|inkl|con|avec)\b.*\b(iva|mva|tax|vat|tva)\b",
                _normalize_ascii(prompt),
            ):
                task.attributes["amountIsVatInclusive"] = True

    # --- Entity correction: bank reconciliation ---
    _BANK_RECON_KEYWORDS = (
        "reconcili", "bankavstem", "bankutskrift", "extrato bancario",
        "kontoutskrift", "kontoauszug", "releve bancaire", "extracto bancario",
        "kontoutdrag", "bank statement",
    )
    if _contains_any_ascii(prompt, _BANK_RECON_KEYWORDS) and task.entity is not Entity.BANK_STATEMENT:
        task.entity = Entity.BANK_STATEMENT
        task.action = Action.CREATE
        if not task.attributes.get("workflow"):
            task.attributes["workflow"] = "reconcile"
        LOGGER.info("Redirected to bank_statement (reconciliation keywords detected)")

    # --- Entity correction: month-end closing / journal entries → voucher ---
    _VOUCHER_KEYWORDS = (
        "monatsabschluss", "month-end closing", "month end closing",
        "manedsslutt", "manedsavslutning", "cierre mensual", "cloture mensuelle",
        "rechnungsabgrenzung", "periodisering", "devengo", "regularisation",
        "journal entry", "journal posting", "bilag",
    )
    if (
        _contains_any_ascii(prompt, _VOUCHER_KEYWORDS)
        and task.entity not in (Entity.VOUCHER, Entity.ASSET, Entity.BANK_STATEMENT, Entity.DIMENSION)
    ):
        task.entity = Entity.VOUCHER
        task.action = Action.CREATE
        LOGGER.info("Redirected to voucher (journal entry/month-end keywords detected)")

    # --- Fill employeeName from target_name when the entity needs it ---
    _EMPLOYEE_ENTITIES = (Entity.TRAVEL_EXPENSE, Entity.TIMESHEET, Entity.SALARY_TRANSACTION)
    if task.entity in _EMPLOYEE_ENTITIES and not task.attributes.get("employeeName"):
        if task.target_name:
            task.attributes["employeeName"] = task.target_name
            LOGGER.info("Copied target_name %r to employeeName", task.target_name)

    # --- Regex fallback: catch critical fields the LLM sometimes omits ---
    _apply_regex_fallbacks(task, prompt)

    LOGGER.info("LLM parsed task: %s", task.model_dump())
    return task


def _apply_regex_fallbacks(task: ParsedTask, prompt: str) -> None:
    """Extract critical fields via regex when the LLM missed them."""
    attrs = task.attributes
    norm = _normalize_ascii(prompt)

    # --- Fixed price ---
    if not attrs.get("fixedPrice"):
        fp_match = re.search(
            r"(?:fastpris|fixed\s*price|prix\s*fixe|festpreis|precio\s*fijo|preco\s*fixo|fast\s*pris)"
            r"[^0-9]{0,30}?([\d][\d\s.,]*[\d])\s*(?:nok|kr)?",
            norm,
        )
        if fp_match:
            raw_num = fp_match.group(1).replace(" ", "").replace(",", ".")
            try:
                attrs["fixedPrice"] = float(raw_num)
                attrs["isFixedPrice"] = True
                LOGGER.info("[FALLBACK] Extracted fixedPrice=%s from prompt", attrs["fixedPrice"])
            except ValueError:
                pass

    # --- Project manager name ---
    if not attrs.get("projectManagerName"):
        pm_match = re.search(
            r"(?:prosjektleder|project\s*manager|projektleiter|chef\s*de\s*projet|gerente\s*de\s*projeto|jefe\s*de\s*proyecto)"
            r"\s+(?:er|is|ist|est|e|es)\s+"
            r"([A-Z\u00C0-\u024F][a-z\u00C0-\u024F]+(?:\s+[A-Z\u00C0-\u024F][a-z\u00C0-\u024F]+)+)",
            prompt,
            re.UNICODE,
        )
        if pm_match:
            attrs["projectManagerName"] = pm_match.group(1).strip()
            LOGGER.info("[FALLBACK] Extracted projectManagerName=%s", attrs["projectManagerName"])

    # --- Project manager email ---
    if not attrs.get("projectManagerEmail") and attrs.get("projectManagerName"):
        # Find email near the PM name
        pm_name = attrs["projectManagerName"]
        pm_idx = prompt.find(pm_name)
        if pm_idx >= 0:
            nearby = prompt[pm_idx:pm_idx + 200]
            email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", nearby)
            if email_match:
                attrs["projectManagerEmail"] = email_match.group(0)
                LOGGER.info("[FALLBACK] Extracted projectManagerEmail=%s", attrs["projectManagerEmail"])

    # --- Project name (for invoice workflows that reference a project) ---
    if not attrs.get("projectName"):
        pn_match = re.search(
            r"(?:prosjekt|project|projekt|projet|proyecto|projeto)\s+['\u2018\u2019\u201C\u201D\"]+([^'\u2018\u2019\u201C\u201D\"]+)['\u2018\u2019\u201C\u201D\"]+",
            prompt,
            re.IGNORECASE | re.UNICODE,
        )
        if pn_match:
            attrs["projectName"] = pn_match.group(1).strip()
            LOGGER.info("[FALLBACK] Extracted projectName=%s", attrs["projectName"])

    # --- Milestone amount from percentage of fixed price ---
    if attrs.get("fixedPrice") and not attrs.get("amount"):
        pct_match = re.search(r"(\d+)\s*%", prompt)
        if pct_match:
            pct = float(pct_match.group(1))
            attrs["amount"] = attrs["fixedPrice"] * pct / 100.0
            LOGGER.info("[FALLBACK] Computed milestone amount=%s (%s%% of %s)", attrs["amount"], pct, attrs["fixedPrice"])

    # --- Organization number ---
    if not attrs.get("organizationNumber"):
        org_match = re.search(
            r"(?:org\.?\s*(?:nr|no|num|numero|nº)?\.?\s*:?\s*|organisasjonsnummer\s*:?\s*)"
            r"(\d{9})",
            norm,
        )
        if org_match:
            attrs["organizationNumber"] = org_match.group(1)
            LOGGER.info("[FALLBACK] Extracted organizationNumber=%s", attrs["organizationNumber"])

    # --- Employee name for salary/timesheet ---
    if not attrs.get("employeeName"):
        for kw in ("ansatt", "employee", "mitarbeiter", "employe", "funcionario", "empleado"):
            emp_match = re.search(
                rf"{kw}\s+([A-Z\u00C0-\u024F][a-z\u00C0-\u024F]+(?:\s+[A-Z\u00C0-\u024F][a-z\u00C0-\u024F]+)+)",
                prompt,
                re.UNICODE,
            )
            if emp_match:
                attrs["employeeName"] = emp_match.group(1).strip()
                LOGGER.info("[FALLBACK] Extracted employeeName=%s", attrs["employeeName"])
                break

    # --- Hours ---
    if not attrs.get("hours"):
        hrs_match = re.search(
            r"(\d+[.,]?\d*)\s*(?:timer|hours|heures|horas|stunden|timar)\b",
            norm,
        )
        if hrs_match:
            raw = hrs_match.group(1).replace(",", ".")
            try:
                attrs["hours"] = float(raw)
                LOGGER.info("[FALLBACK] Extracted hours=%s", attrs["hours"])
            except ValueError:
                pass

    # --- STYRK / occupation code ---
    if task.entity is Entity.EMPLOYEE and not attrs.get("occupationCode"):
        styrk_match = re.search(
            r"(?:stillingskode|styrk|occupation\s*code|profession\s*code|codigo\s*de\s*profesion|code\s*profession)"
            r"[^0-9]{0,20}(\d{4})\b",
            norm,
        )
        if styrk_match:
            attrs["occupationCode"] = styrk_match.group(1)
            LOGGER.info("[FALLBACK] Extracted occupationCode=%s", attrs["occupationCode"])

    # --- Annual salary ---
    if task.entity is Entity.EMPLOYEE and not attrs.get("annualSalary"):
        annual_match = re.search(
            r"(?:arslonn|aarsloen|annual\s*salary|salaire\s*annuel|jahresgehalt|salario\s*anual)"
            r"[^0-9]{0,20}([\d][\d\s.,]*[\d])\s*(?:nok|kr)?",
            norm,
        )
        if annual_match:
            raw = annual_match.group(1).replace(" ", "").replace(",", ".")
            try:
                attrs["annualSalary"] = float(raw)
                LOGGER.info("[FALLBACK] Extracted annualSalary=%s", attrs["annualSalary"])
            except ValueError:
                pass

    # --- Employment form ---
    if task.entity is Entity.EMPLOYEE and not attrs.get("employmentForm"):
        if re.search(r"(?:fast\s*stilling|permanent\s*position|fest\s*anstellung|poste\s*permanent|puesto\s*fijo)", norm):
            attrs["employmentForm"] = "PERMANENT"
            LOGGER.info("[FALLBACK] Extracted employmentForm=PERMANENT")
        elif re.search(r"(?:midlertidig|temporary|befristet|temporaire|temporal)", norm):
            attrs["employmentForm"] = "TEMPORARY"
            LOGGER.info("[FALLBACK] Extracted employmentForm=TEMPORARY")

    # --- Percentage of full-time equivalent ---
    if task.entity is Entity.EMPLOYEE and not attrs.get("percentageOfFullTimeEquivalent"):
        pct_match = re.search(
            r"(?:stillingsprosent|percentage|pourcentage|prozent|porcentaje)"
            r"[^0-9]{0,20}(\d+[.,]?\d*)\s*%",
            norm,
        )
        if pct_match:
            raw = pct_match.group(1).replace(",", ".")
            try:
                attrs["percentageOfFullTimeEquivalent"] = float(raw)
                LOGGER.info("[FALLBACK] Extracted percentageOfFullTimeEquivalent=%s", attrs["percentageOfFullTimeEquivalent"])
            except ValueError:
                pass

    # --- Hours per day ---
    if task.entity is Entity.EMPLOYEE and not attrs.get("hoursPerDay"):
        hpd_match = re.search(
            r"(?:arbeidstid|working\s*hours|heures\s*de\s*travail|arbeitszeit|horas\s*de\s*trabajo)"
            r"[^0-9]{0,20}(\d+[.,]?\d*)\s*(?:timer|hours|heures|stunden|horas)\s*(?:per|pr|par|pro|por)\s*(?:dag|day|jour|tag|dia)",
            norm,
        )
        if hpd_match:
            raw = hpd_match.group(1).replace(",", ".")
            try:
                attrs["hoursPerDay"] = float(raw)
                LOGGER.info("[FALLBACK] Extracted hoursPerDay=%s", attrs["hoursPerDay"])
            except ValueError:
                pass

    # --- Base salary ---
    if not attrs.get("baseSalary"):
        sal_match = re.search(
            r"(?:grunnlonn|base\s*salary|salaire\s*de\s*base|grundgehalt|salario\s*base|sueldo\s*base)"
            r"[^0-9]{0,20}?([\d][\d\s.,]*[\d])\s*(?:nok|kr)?",
            norm,
        )
        if sal_match:
            raw = sal_match.group(1).replace(" ", "").replace(",", ".")
            try:
                attrs["baseSalary"] = float(raw)
                LOGGER.info("[FALLBACK] Extracted baseSalary=%s", attrs["baseSalary"])
            except ValueError:
                pass

    # --- Bonus ---
    if not attrs.get("bonus"):
        bonus_match = re.search(
            r"(?:bonus|tillegg|prime|zuschlag)"
            r"[^0-9]{0,20}?([\d][\d\s.,]*[\d])\s*(?:nok|kr)?",
            norm,
        )
        if bonus_match:
            raw = bonus_match.group(1).replace(" ", "").replace(",", ".")
            try:
                attrs["bonus"] = float(raw)
                LOGGER.info("[FALLBACK] Extracted bonus=%s", attrs["bonus"])
            except ValueError:
                pass
