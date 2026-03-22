from __future__ import annotations

from datetime import date, timedelta
import unicodedata
import re
from typing import Iterable

from tripletex_solver.errors import ParsingError
from tripletex_solver.models import Action, Entity, ParsedTask

ACTION_KEYWORDS: dict[Action, tuple[str, ...]] = {
    # IMPORTANT: REGISTER must be checked BEFORE CREATE because "register" is ambiguous.
    # The parser checks longer/more-specific phrases first.
    Action.REGISTER: (
        "register full payment",
        "register payment",
        "register invoice payment",
        "registrer full betaling",
        "registrer betaling",
        "registrer fakturabetaling",
        "registrar pago",
        "registrar pagamento",
        "enregistrer le paiement",
        "enregistrer paiement",
        "zahlung registrieren",
        "zahlung erfassen",
        "registrer innbetaling",
        "pay supplier invoice",
        "pay incoming invoice",
        "betal leverandørfaktura",
        "betal leverandorfaktura",
        "betal innkommende faktura",
        "payer facture fournisseur",
        "lieferantenrechnung bezahlen",
        "pagar factura del proveedor",
    ),
    Action.CREATE: (
        "create",
        "opprett",
        "lag",
        "registrer",
        "register",
        "add",
        "ny",
        "nuevo",
        "nouveau",
        "novo",
        "erstelle",
        "anlegen",
        "crie",
        "criar",
        "cree",
        "crea",
        "enable",
        "aktiver",
        "activate",
        "activar",
        "activer",
        "aktivieren",
        "emit",
        "emita",
        "emite",
        "emitir",
        "emitir",
        "reverte",
        "reverter",
        "revertere",
        "reverta",
        "storno",
        "stornoer",
        "anull",
        "anule",
        "anular",
        "anuller",
        "annuler",
        "run",
        "ejecutar",
        "exécutez",
        "executez",
        "pay",
        "betal",
        "utfør",
        "utfor",
        "gjennomfør",
        "gjennomfor",
        "gjer",
        "führen",
        "fuhren",
        "durchführen",
        "buchen",
        "bokfør",
        "bokfor",
        "realize",
        "realice",
        "reconcili",
        "encontre",
        "encuentre",
        "finden",
        "beregn",
        "rekn",
        "efectúe",
        "efectue",
        "effectuez",
        "réalisez",
        "réaliser",
    ),
    Action.UPDATE: (
        "update",
        "endre",
        "oppdater",
        "change",
        "modify",
        "rename",
        "legg til",
        "add phone",
        "add email",
        "ajouter",
        "mettre à jour",
        "actualizar",
        "cambiar",
        "atualizar",
        "ändern",
    ),
    Action.DELETE: (
        "delete",
        "remove",
        "slett",
        "fjern",
        "eliminar",
        "borrar",
        "supprimer",
        "löschen",
        "apagar",
    ),
}

ENTITY_KEYWORDS: dict[Entity, tuple[str, ...]] = {
    Entity.EMPLOYEE: (
        "employee",
        "ansatt",
        "empleado",
        "empregado",
        "mitarbeiter",
        "employé",
        "employe",
    ),
    Entity.CUSTOMER: (
        "customer",
        "kunde",
        "cliente",
        "client",
        "kund",
    ),
    Entity.DEPARTMENT: (
        "departments",
        "department",
        "avdelingene",
        "avdelingar",
        "avdelinger",
        "avdeling",
        "departamentos",
        "departamento",
        "departement",
        "abteilungen",
        "abteilung",
    ),
    Entity.PRODUCT: (
        "products",
        "product",
        "produktene",
        "produkter",
        "produkt",
        "productos",
        "producto",
        "produtos",
        "produto",
        "produits",
        "produit",
    ),
    Entity.PROJECT: (
        "project",
        "prosjekt",
        "proyecto",
        "projeto",
        "projet",
    ),
    Entity.TRAVEL_EXPENSE: (
        "travel expense",
        "reiseutgift",
        "travel report",
        "expense report",
        "reiseregning",
        "frais de déplacement",
        "gastos de viagem",
    ),
    Entity.INVOICE: (
        "invoice",
        "faktura",
        "factura",
        "fatura",
        "facture",
        "rechnung",
    ),
    Entity.PAYMENT: (
        "payment",
        "betaling",
        "pago",
        "pagamento",
        "paiement",
        "zahlung",
    ),
    Entity.CONTACT: (
        "contact",
        "contact person",
        "kontaktperson",
        "kontakt",
        "contacto",
        "contato",
        "ansprechpartner",
    ),
    Entity.SUPPLIER: (
        "supplier",
        "leverandør",
        "leverandor",
        "vendor",
        "proveedor",
        "fornecedor",
        "fournisseur",
        "lieferant",
    ),
    Entity.VOUCHER: (
        "voucher",
        "bilag",
        "journal entry",
        "comprobante",
        "comprovante",
        "bon",
        "beleg",
        "buchungsbeleg",
        "monatsabschluss",
        "månedsslutt",
        "månedsavslutning",
        "month-end closing",
        "årsoppgjør",
        "arsoppgjor",
        "årsoppgjer",
        "year-end closing",
        "jahresabschluss",
        "encerramento anual",
        "cierre anual",
        "cierre mensual",
        "clôture annuelle",
        "clôture mensuelle",
    ),
    Entity.TIMESHEET: (
        "hours",
        "timesheet",
        "time entry",
        "timer",
        "timeliste",
        "tidregistrering",
        "horas",
        "heures",
        "stunden",
        "zeiterfassung",
    ),
    Entity.COMPANY_MODULE: (
        "module",
        "modul",
        "avdelingsregnskap",
        "department accounting",
        "enable module",
        "aktiver modul",
        "activate module",
        "sales module",
    ),
    Entity.INCOMING_INVOICE: (
        "incoming invoice",
        "supplier invoice",
        "innkommende faktura",
        "leverandørfaktura",
        "leverandorfaktura",
        "factura del proveedor",
        "eingangsrechnung",
        "facture fournisseur",
    ),
    Entity.PURCHASE_ORDER: (
        "purchase order",
        "innkjøpsordre",
        "innkjopsordre",
        "bestilling",
        "orden de compra",
        "bestellung",
        "commande",
    ),
    Entity.SALARY_TRANSACTION: (
        "salary transaction",
        "lønnsbilag",
        "lonnsbilag",
        "payroll",
        "run payroll",
        "lønnskjøring",
        "gehaltsabrechnung",
        "nómina",
        "nomina",
        "salaire",
        "la paie",
        "exécutez la paie",
        "ejecutar nómina",
        "ejecutar nomina",
    ),
    Entity.BANK_STATEMENT: (
        "bank statement",
        "kontoutdrag",
        "bank reconciliation",
        "bankavstemming",
        "kontoauszug",
        "extracto bancario",
        "extrato bancário",
        "extrato bancario",
        "relevé bancaire",
        "reconcili",
    ),
    Entity.DIMENSION: (
        "dimension",
        "dimensjon",
        "accounting dimension",
        "Buchhaltungsdimension",
        "dimensión contable",
        "dimensão",
    ),
    Entity.ACCOUNT: (
        "ledger account",
        "chart of accounts",
        "konto",
        "kontoplan",
        "Konto",
        "compte",
        "cuenta",
        "account number",
        "kontonummer",
    ),
    Entity.ORDER: (
        "sales order",
        "ordre",
        "order",
        "auftrag",
        "pedido",
        "commande",
    ),
    Entity.REMINDER: (
        "reminder",
        "purring",
        "inkasso",
        "mahnung",
        "mahngebühr",
        "mahngebuhr",
        "rappel",
        "recordatorio",
        "dunning",
        "overdue invoice",
        "überfällige rechnung",
        "uberfallige rechnung",
        "factura vencida",
        "fatura vencida",
        "forfalt faktura",
        "forfallen faktura",
    ),
    Entity.ACTIVITY: (
        "activity",
        "aktivitet",
        "aktivität",
        "activité",
        "actividad",
    ),
    Entity.DIVISION: (
        "division",
        "divisjon",
        "abteilung",
        "división",
    ),
    Entity.LEAVE_OF_ABSENCE: (
        "leave of absence",
        "permisjon",
        "fravær",
        "urlaub",
        "congé",
        "permiso",
        "sick leave",
        "sykefravær",
    ),
    Entity.NEXT_OF_KIN: (
        "next of kin",
        "pårørende",
        "angehörige",
        "proche",
        "pariente",
        "emergency contact",
        "nødkontakt",
    ),
    Entity.CUSTOMER_CATEGORY: (
        "customer category",
        "kundekategori",
        "kundenkategorie",
        "catégorie client",
        "categoría de cliente",
    ),
    Entity.EMPLOYEE_CATEGORY: (
        "employee category",
        "ansattkategori",
        "mitarbeiterkategorie",
        "catégorie employé",
        "categoría de empleado",
    ),
    Entity.ASSET: (
        "fixed asset",
        "anleggsmiddel",
        "anlagevermögen",
        "immobilisation",
        "activo fijo",
        "asset",
        "eiendel",
    ),
    Entity.PRODUCT_GROUP: (
        "product group",
        "produktgruppe",
        "produktgruppe",
        "groupe de produits",
        "grupo de productos",
    ),
    Entity.PROJECT_CATEGORY: (
        "project category",
        "prosjektkategori",
        "projektkategorie",
        "catégorie de projet",
        "categoría de proyecto",
    ),
    Entity.INVENTORY: (
        "inventory",
        "warehouse",
        "lager",
        "varelager",
        "almacén",
        "inventário",
        "inventaire",
        "Lager",
        "Warenlager",
    ),
    Entity.INVENTORY_LOCATION: (
        "inventory location",
        "warehouse location",
        "lagerlokasjon",
        "lagerplassering",
        "ubicación de almacén",
        "emplacement d'inventaire",
        "Lagerstandort",
    ),
    Entity.STOCKTAKING: (
        "stocktaking",
        "stock count",
        "varetelling",
        "inventur",
        "recuento de inventario",
        "comptage de stock",
        "Inventur",
        "Bestandsaufnahme",
    ),
    Entity.GOODS_RECEIPT: (
        "goods receipt",
        "varemottak",
        "goods received",
        "recepción de mercancías",
        "réception de marchandises",
        "Wareneingang",
    ),
    Entity.DOCUMENT_ARCHIVE: (
        "document archive",
        "dokumentarkiv",
        "upload document",
        "last opp dokument",
        "archivo de documentos",
        "archive de documents",
        "Dokumentenarchiv",
    ),
    Entity.EVENT_SUBSCRIPTION: (
        "event subscription",
        "webhook",
        "hendelsesabonnement",
        "event notification",
        "abonnement",
        "suscripción de evento",
        "abonnement d'événement",
        "Ereignisabonnement",
    ),
}

NAME_LABELS = (
    "named",
    "called",
    "med navn",
    "som heter",
    "name",
    "navn",
    "nombre",
    "nome",
    "nom",
    "genannt",
)
RENAME_KEYWORDS = (
    "rename",
    "change name",
    "new name",
    "endre navn",
    "nytt navn",
    "renommer",
)

ADMIN_KEYWORDS = (
    "account administrator",
    "administrator",
    "kontoadministrator",
    "admin",
    "administrador",
    "administrateur",
)

NO_ACCESS_KEYWORDS = (
    "no access",
    "ingen tilgang",
    "sin acceso",
    "sans accès",
    "kein zugang",
)

PRICE_KEYWORDS = (
    "price",
    "pris",
    "precio",
    "preço",
    "prix",
    "preis",
    "amount",
    "beløp",
    "montant",
)

CUSTOMER_LINK_KEYWORDS = (
    "for customer",
    "kunden",
    "kunde",
    "customer",
    "cliente",
    "client",
    "al cliente",
    "au client",
    "para o cliente",
    "para a cliente",
    "ao cliente",
    "a cliente",
    "for o cliente",
    "dem kunden",
)

DEPARTMENT_LINK_KEYWORDS = (
    "department",
    "avdeling",
    "departamento",
    "departement",
    "abteilung",
)

PRODUCT_LINK_KEYWORDS = (
    "products",
    "product",
    "produktene",
    "produkter",
    "produkt",
    "productos",
    "producto",
    "produtos",
    "produto",
    "produits",
    "produit",
    "item",
    "line item",
    "service",
    "tjeneste",
)

PROJECT_MANAGER_KEYWORDS = (
    "project manager",
    "prosjektleder",
    "jefe de proyecto",
    "gerente de projeto",
    "chef de projet",
    "projektleiter",
    "projektleiar",
)

ORG_NUMBER_PATTERN = re.compile(
    r"(?:org\.?\s*(?:nr|n[ºo°]|number|no|nummer)\.?|organization\s*number|organisasjonsnummer)\s*[:=]?\s*(\d{9})",
    re.IGNORECASE,
)

CREDIT_NOTE_KEYWORDS = (
    "credit note",
    "credit memo",
    "kreditnota",
    "kreditnote",
    "credit-note",
    "nota de cr\u00e9dito",
    "nota de credito",
    "gutschrift",
    "credito",
    "storno",
    "omgj\u00f8ring",
    "full reversal",
    "reversal",
)

INVOICE_DATE_KEYWORDS = (
    "invoice date",
    "fakturadato",
    "invoice dated",
)

DUE_DATE_KEYWORDS = (
    "due date",
    "forfallsdato",
    "due",
    "falls due",
    "pay by",
)

PAYMENT_DATE_KEYWORDS = (
    "payment date",
    "betalingsdato",
    "paid on",
    "betalt",
    "on",
)

PAYMENT_TYPE_KEYWORDS = (
    "payment type",
    "betalingsmåte",
    "using",
    "via",
)

ACTIVITY_LINK_KEYWORDS = (
    "activity",
    "aktivitet",
    "actividad",
    "atividade",
    "activité",
    "aktivität",
)

HOURS_KEYWORDS = (
    "hours",
    "timer",
    "horas",
    "heures",
    "stunden",
)

DEBIT_KEYWORDS = ("debit", "debet", "débito", "soll")
CREDIT_KEYWORDS = ("credit", "kredit", "crédito", "haben")
DESCRIPTION_KEYWORDS = ("description", "beskrivelse", "descripción", "descrição", "beschreibung")

TODAY_KEYWORDS = ("today", "i dag", "idag", "hoy", "hoje", "aujourd'hui", "heute")
PHONE_LABELS = (
    "phone",
    "phone number",
    "mobile",
    "mobile phone",
    "telefon",
    "telefonnummer",
    "mobil",
    "mobilnummer",
    "tlf",
    "tel",
)

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\d)(\+?\d[\d\s()-]{6,}\d)")
IDENTIFIER_PATTERN = re.compile(r"(?:\bid\b|\bnummer\b|\bnumber\b|\bnr\b|#)\s*[:=]?\s*(\d+)", re.IGNORECASE)
QUOTED_PATTERN = re.compile(r'"([^"\n]{2,100})"')
DATE_TOKEN_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b")


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _normalize_ascii(text: str) -> str:
    # Remove diacritics so matching works for both "credito" and "crédito".
    decomposed = unicodedata.normalize("NFKD", _normalize(text))
    return decomposed.encode("ascii", "ignore").decode("ascii")


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    normalized_text = _normalize_ascii(text)
    return any(_normalize_ascii(phrase) in normalized_text for phrase in phrases)


def _clean_segment(value: str) -> str:
    segment = value.strip().strip(",.;: ")
    separators = (
        ",",
        ".",
        ";",
        " in ",
        " with ",
        " med ",
        " som ",
        " who ",
        " for ",
        " til ",
        " fra ",
        " from ",
        " on ",
        " para ",
        " pour ",
        " mit ",
        " con ",
        " y ",
        " og ",
        " and ",
        " und ",
    )

    lowered = segment.casefold()
    cut = len(segment)
    for separator in separators:
        index = lowered.find(separator)
        if index >= 0:
            cut = min(cut, index)

    cleaned = segment[:cut].strip().strip(",.;: ")
    return cleaned.strip('"')


def _strip_parenthetical(text: str) -> str:
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"\s*\(.*$", "", text)
    return text.strip()


def _extract_raw_after_keywords(prompt: str, keywords: Iterable[str]) -> str | None:
    """Like _extract_after_keywords but returns raw text without _clean_segment trimming."""
    for keyword in keywords:
        pattern = re.compile(
            rf"\b{re.escape(keyword)}\b\s*(?:is|er|est|ist|es|=|:)?\s+(?P<value>[^.\n]+)",
            re.IGNORECASE,
        )
        match = pattern.search(prompt)
        if match:
            return match.group("value").strip()
    return None


def _extract_after_keywords(prompt: str, keywords: Iterable[str]) -> str | None:
    for keyword in keywords:
        pattern = re.compile(
            rf"\b{re.escape(keyword)}\b\s*(?:is|er|est|ist|es|=|:)?\s+(?P<value>[^.\n]+)",
            re.IGNORECASE,
        )
        match = pattern.search(prompt)
        if match:
            raw_value = match.group("value")
            quoted = QUOTED_PATTERN.search(raw_value)
            if quoted:
                return quoted.group(1).strip()
            raw_value = _strip_parenthetical(raw_value)
            value = _clean_segment(raw_value)
            if value:
                return value
    return None


def _extract_quoted_values(prompt: str) -> list[str]:
    return [value.strip() for value in QUOTED_PATTERN.findall(prompt)]


def _extract_rename_values(prompt: str) -> tuple[str, str] | None:
    normalized = _normalize(prompt)
    quoted_values = _extract_quoted_values(prompt)
    if len(quoted_values) >= 2 and _contains_any(normalized, RENAME_KEYWORDS):
        return quoted_values[0], quoted_values[1]
    return None


def _extract_name(prompt: str, entity_keywords: Iterable[str]) -> str | None:
    after_entity = _extract_after_keywords(prompt, entity_keywords)
    if after_entity:
        return after_entity

    labeled = _extract_after_keywords(prompt, NAME_LABELS)
    if labeled:
        return labeled

    quoted = QUOTED_PATTERN.search(prompt)
    if quoted:
        return quoted.group(1).strip()

    return None


def _extract_linked_name(prompt: str, relation_keywords: Iterable[str]) -> str | None:
    return _extract_after_keywords(prompt, relation_keywords)


def _trim_linked_entity_name(value: str | None) -> str | None:
    if not value:
        return None
    parts = re.split(
        r"\s+(?:amount|bel\øp|beløp|invoice|faktura|factura|org|iva|sem|inkl|excluding|including|include)\b|[;()]|\n",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    trimmed = _clean_segment(parts[0])
    return trimmed or None


def _extract_email(prompt: str) -> str | None:
    match = EMAIL_PATTERN.search(prompt)
    return match.group(0) if match else None


def _extract_phone(prompt: str) -> str | None:
    label_pattern = "|".join(re.escape(label) for label in PHONE_LABELS)
    labeled_match = re.search(
        rf"(?:{label_pattern})\s*(?:is|=|:)?\s*(\+?\d[\d\s()-]{{6,}}\d)",
        prompt,
        re.IGNORECASE,
    )
    if labeled_match:
        candidate = labeled_match.group(1).strip()
        digits = re.sub(r"\D", "", candidate)
        if 8 <= len(digits) <= 15:
            return candidate

    for match in PHONE_PATTERN.finditer(prompt):
        candidate = match.group(1).strip()
        digits = re.sub(r"\D", "", candidate)
        if candidate.startswith("+") and 8 <= len(digits) <= 15:
            return candidate
    return None


def _extract_identifier(prompt: str) -> int | None:
    match = IDENTIFIER_PATTERN.search(prompt)
    return int(match.group(1)) if match else None


def _extract_number(prompt: str) -> str | None:
    match = re.search(
        r"(?:product number|produktnummer|product no|product number is|number)\s*[:=]?\s*([A-Z0-9_-]+)",
        prompt,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _extract_invoice_number(prompt: str) -> str | None:
    patterns = (
        r"(?:invoice number|invoice no|invoice nr|fakturanummer)\s*[:=]?\s*([A-Z0-9_-]+)",
        r"(?:fatura|factura|facture)\s*(?:n[ºo°]\s*)?(?:[:#]?\s*)?(\d+)",
        r"(?:invoice|faktura|rechnung)\s*#?\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_price(prompt: str) -> float | None:
    keyword_pattern = "|".join(re.escape(keyword) for keyword in PRICE_KEYWORDS)
    match = re.search(
        rf"(?:{keyword_pattern})\s*(?:is|=|:)?\s*(?:nok|kr|eur|€)?\s*(\d+(?:[.,]\d+)?)",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _extract_currency_amount(prompt: str) -> float | None:
    patterns = (
        r"(?:nok|kr|eur|€|usd|dkk)\s*(\d+(?:[.,]\d+)?)",
        r"(\d+(?:[.,]\d+)?)\s*(?:nok|kr|eur|€|usd|dkk)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def _extract_amount_scope(prompt: str) -> bool | None:
    normalized = _normalize(prompt)
    if _contains_any(
        normalized,
        (
            "sem iva",
            "sin iva",
            "uten mva",
            "excl",
            "excluding vat",
            "excluding tax",
            "without iva",
            "hors tva",
            "sin mva",
            "sem mva",
        ),
    ):
        return True
    if _contains_any(
        normalized,
        (
            "inkl",
            "inkludert",
            "med iva",
            "con iva",
            "avec tva",
            "including vat",
            "with vat",
        ),
    ):
        return False
    return None


def _extract_amount_after_keywords(prompt: str, keywords: Iterable[str]) -> float | None:
    keyword_pattern = "|".join(re.escape(keyword) for keyword in keywords)
    match = re.search(
        rf"(?:{keyword_pattern})[^0-9]{{0,20}}(?:nok|kr|eur|€)?\s*(\d+(?:[.,]\d+)?)",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _parse_date_token(token: str) -> date:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        return date.fromisoformat(token)

    separator = "." if "." in token else "/" if "/" in token else "-"
    parts = token.split(separator)
    if len(parts) != 3:
        raise ValueError(token)

    first, second, third = (int(part) for part in parts)
    if first > 31:
        year, month, day = first, second, third
    else:
        day, month, year = first, second, third
    if year < 100:
        year += 2000
    return date(year, month, day)


def _extract_date_for_keywords(prompt: str, keywords: Iterable[str]) -> date | None:
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), prompt, re.IGNORECASE):
            window = prompt[match.end() : match.end() + 40]
            date_match = DATE_TOKEN_PATTERN.search(window)
            if date_match:
                return _parse_date_token(date_match.group(1))
            normalized_window = _normalize(window)
            if any(today_word in normalized_window for today_word in TODAY_KEYWORDS):
                return date.today()
    return None


def _extract_due_in_days(prompt: str) -> int | None:
    match = re.search(r"(?:due in|forfaller om)\s*(\d{1,3})\s*(?:days|dager)", prompt, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_payment_type(prompt: str) -> str | None:
    match = re.search(
        r"(?:payment type|betalingsmåte|using|via)\s+(.+?)(?=(?:\s+on\s+\d|\s+\d{4}-\d{2}-\d{2}|[.,;]|$))",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip().strip('"')


def _extract_all_dates(prompt: str) -> list[date]:
    dates: list[date] = []
    for token in DATE_TOKEN_PATTERN.findall(prompt):
        parsed = _parse_date_token(token)
        if parsed not in dates:
            dates.append(parsed)
    return dates


def _extract_route(prompt: str) -> tuple[str | None, str | None]:
    patterns = (
        r"(?:from|fra)\s+(.+?)\s+(?:to|til)\s+(.+?)(?=[,.;]|\s+on\s+\d|\s+\d{4}-|\s+\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|$)",
        r"(?:departure from)\s+(.+?)\s+(?:destination)\s+(.+?)(?=[,.;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return _clean_segment(match.group(1)), _clean_segment(match.group(2))
    return None, None


def _extract_comment(prompt: str) -> str | None:
    return _extract_after_keywords(prompt, ("comment", "reason", "begrunnelse", "forklaring"))


def _extract_hours(prompt: str) -> float | None:
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:hours|timer|horas|heures|stunden|h)\b",
        prompt,
        re.IGNORECASE,
    )
    return float(match.group(1).replace(",", ".")) if match else None


def _extract_account_number(prompt: str, keywords: Iterable[str]) -> int | None:
    keyword_pattern = "|".join(re.escape(k) for k in keywords)
    match = re.search(
        rf"(?:{keyword_pattern})\s+(?:account|konto)?\s*(?:number|nr|#)?\s*[:=]?\s*(\d{{4}})",
        prompt,
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def _split_person_name(name: str) -> tuple[str, str]:
    parts = [part for part in name.split() if part]
    if len(parts) < 2:
        raise ParsingError("Employee name must include both first and last name")
    return parts[0], " ".join(parts[1:])


class PromptParser:
    def parse(self, prompt: str) -> ParsedTask:
        normalized = _normalize(prompt)
        norm_ascii = _normalize_ascii(prompt)
        action = self._detect_action(normalized)
        if action is Action.CREATE and _contains_any(normalized, CREDIT_NOTE_KEYWORDS):
            entity = Entity.INVOICE
        else:
            entity = self._detect_entity(normalized)

        # Priority overrides: certain keyword combos should force entity type
        _REMINDER_FORCE_KEYWORDS = (
            "mahngebuhr", "overdue invoice", "forfalt faktura",
            "factura vencida", "fatura vencida", "uberfallige rechnung",
            "purring", "inkasso", "reminder fee", "cargo por recordatorio",
            "forfallen faktura",
        )
        if entity not in (Entity.REMINDER, Entity.VOUCHER, Entity.BANK_STATEMENT) and any(
            kw in norm_ascii for kw in _REMINDER_FORCE_KEYWORDS
        ):
            entity = Entity.REMINDER
            action = Action.CREATE

        # REGISTER action targets PAYMENT unless an incoming-invoice keyword was found
        if action is Action.REGISTER:
            if entity is not Entity.INCOMING_INVOICE:
                entity = Entity.PAYMENT

        task = ParsedTask(action=action, entity=entity, raw_prompt=prompt)
        task.identifier = _extract_identifier(prompt)

        if entity is Entity.EMPLOYEE:
            return self._parse_employee(prompt, task)
        if entity is Entity.CUSTOMER:
            return self._parse_customer(prompt, task)
        if entity is Entity.DEPARTMENT:
            return self._parse_department(prompt, task)
        if entity is Entity.PRODUCT:
            return self._parse_product(prompt, task)
        if entity is Entity.PROJECT:
            return self._parse_project(prompt, task)
        if entity is Entity.TRAVEL_EXPENSE:
            return self._parse_travel_expense(prompt, task)
        if entity is Entity.INVOICE:
            return self._parse_invoice(prompt, task)
        if entity is Entity.PAYMENT:
            return self._parse_payment(prompt, task)
        if entity is Entity.CONTACT:
            return self._parse_contact(prompt, task)
        if entity is Entity.SUPPLIER:
            return self._parse_supplier(prompt, task)
        if entity is Entity.VOUCHER:
            return self._parse_voucher(prompt, task)
        if entity is Entity.TIMESHEET:
            return self._parse_timesheet(prompt, task)

        # Generic handler for Tier 2/3 entities without dedicated parsers
        return self._parse_generic(prompt, task)

    def _detect_action(self, normalized_prompt: str) -> Action:
        hits: list[tuple[int, int, Action]] = []
        for action, phrases in ACTION_KEYWORDS.items():
            for phrase in phrases:
                index = normalized_prompt.find(phrase)
                if index >= 0:
                    hits.append((index, -len(phrase), action))
        if hits:
            hits.sort()
            return hits[0][2]
        raise ParsingError("Could not determine the requested action from the prompt")

    def _detect_entity(self, normalized_prompt: str) -> Entity:
        hits: list[tuple[int, int, Entity]] = []
        for entity, phrases in ENTITY_KEYWORDS.items():
            for phrase in phrases:
                lowered = phrase.casefold()
                index = normalized_prompt.find(lowered)
                if index >= 0:
                    # Use negative length so longer matches sort first at same position
                    hits.append((index, -len(lowered), entity))
                    break
        if hits:
            hits.sort()
            return hits[0][2]
        raise ParsingError("Could not determine the target entity from the prompt")

    def _parse_employee(self, prompt: str, task: ParsedTask) -> ParsedTask:
        rename_values = _extract_rename_values(prompt)
        name = _extract_name(prompt, ENTITY_KEYWORDS[Entity.EMPLOYEE])
        email = _extract_email(prompt)
        phone = _extract_phone(prompt)
        department_name = _extract_linked_name(prompt, DEPARTMENT_LINK_KEYWORDS)
        normalized = _normalize(prompt)

        effective_name = rename_values[0] if rename_values else name
        if effective_name:
            task.target_name = effective_name

        if rename_values:
            first_name, last_name = _split_person_name(rename_values[1])
            task.attributes["firstName"] = first_name
            task.attributes["lastName"] = last_name
        elif name:
            first_name, last_name = _split_person_name(name)
            task.attributes["firstName"] = first_name
            task.attributes["lastName"] = last_name

        if email:
            task.attributes["email"] = email
        if phone:
            task.attributes["phoneNumberWork"] = phone
        if department_name and _normalize(department_name) not in {_normalize(effective_name or ""), "department", "avdeling"}:
            task.attributes["departmentName"] = department_name

        if _contains_any(normalized, ADMIN_KEYWORDS):
            task.attributes["userType"] = "EXTENDED"
            task.notes.append("Mapped administrator-style wording to userType=EXTENDED")
        elif _contains_any(normalized, NO_ACCESS_KEYWORDS):
            task.attributes["userType"] = "NO_ACCESS"
        elif task.action is Action.CREATE:
            task.attributes["userType"] = "STANDARD"

        if task.action is Action.CREATE and "firstName" not in task.attributes:
            raise ParsingError("Could not extract employee name from prompt")

        return task

    def _parse_customer(self, prompt: str, task: ParsedTask) -> ParsedTask:
        rename_values = _extract_rename_values(prompt)
        quoted_names = _extract_quoted_values(prompt)
        name = _extract_name(prompt, ENTITY_KEYWORDS[Entity.CUSTOMER])
        email = _extract_email(prompt)
        phone = _extract_phone(prompt)

        effective_name = rename_values[0] if rename_values else name
        if task.action is Action.CREATE and len(quoted_names) > 1 and not email:
            task.target_name = quoted_names[0]
            task.attributes["name"] = quoted_names[0]
            task.attributes["names"] = quoted_names
            return task
        if effective_name:
            task.target_name = effective_name
        if task.action is Action.CREATE and name:
            task.attributes["name"] = name
        elif rename_values:
            task.attributes["name"] = rename_values[1]
        if email:
            task.attributes["email"] = email
        if phone:
            task.attributes["phoneNumberMobile"] = phone

        if task.action is Action.CREATE and "name" not in task.attributes:
            raise ParsingError("Could not extract customer name from prompt")

        return task

    def _parse_department(self, prompt: str, task: ParsedTask) -> ParsedTask:
        rename_values = _extract_rename_values(prompt)
        quoted_names = _extract_quoted_values(prompt)
        name = _extract_name(prompt, ENTITY_KEYWORDS[Entity.DEPARTMENT])
        effective_name = rename_values[0] if rename_values else name
        if task.action is Action.DELETE and task.identifier is not None and not effective_name:
            return task
        if task.action is Action.UPDATE and task.identifier is not None and not effective_name and not rename_values:
            return task
        if task.action is Action.CREATE and len(quoted_names) > 1:
            task.target_name = quoted_names[0]
            task.attributes["name"] = quoted_names[0]
            task.attributes["names"] = quoted_names
            return task
        if not effective_name:
            raise ParsingError("Could not extract department name from prompt")
        task.target_name = effective_name
        if task.action is Action.CREATE:
            task.attributes["name"] = effective_name
        elif rename_values:
            task.attributes["name"] = rename_values[1]
        return task

    def _parse_product(self, prompt: str, task: ParsedTask) -> ParsedTask:
        rename_values = _extract_rename_values(prompt)
        quoted_names = _extract_quoted_values(prompt)
        name = _extract_name(prompt, ENTITY_KEYWORDS[Entity.PRODUCT])
        effective_name = rename_values[0] if rename_values else name
        if task.action is Action.DELETE and task.identifier is not None and not effective_name:
            return task
        price = _extract_price(prompt)
        product_number = _extract_number(prompt)
        if task.action is Action.CREATE and len(quoted_names) > 1 and not price and not product_number:
            task.target_name = quoted_names[0]
            task.attributes["name"] = quoted_names[0]
            task.attributes["names"] = quoted_names
            return task
        # Handle unquoted comma/and-separated names like "X, Y og Z" or "X, Y and Z"
        if task.action is Action.CREATE and not price and not product_number:
            raw_after = _extract_raw_after_keywords(prompt, ENTITY_KEYWORDS[Entity.PRODUCT])
            if raw_after:
                parts = re.split(r",\s*(?:og|and|und|y|et|e)?\s*|\s+og\s+|\s+and\s+|\s+und\s+|\s+y\s+|\s+et\s+", raw_after)
                parts = [p.strip().strip(",.;: ") for p in parts if p.strip().strip(",.;: ")]
                if len(parts) > 1:
                    task.target_name = parts[0]
                    task.attributes["name"] = parts[0]
                    task.attributes["names"] = parts
                    return task
        if not effective_name and task.action is Action.CREATE:
            raise ParsingError("Could not extract product name from prompt")
        if effective_name:
            task.target_name = effective_name
        if task.action is Action.CREATE:
            task.attributes["name"] = effective_name
        elif rename_values:
            task.attributes["name"] = rename_values[1]

        if product_number:
            task.attributes["number"] = product_number
        if price is not None:
            task.attributes["priceExcludingVatCurrency"] = price

        return task

    @staticmethod
    def _is_ledger_analysis_prompt(prompt: str) -> bool:
        """Detect ledger analysis prompts where project names come from the ledger, not the prompt."""
        norm = _normalize_ascii(prompt)
        analysis_kw = ("analice", "analyze", "analyser", "analysier", "analysere",
                        "identifique", "identify", "identifiser", "identifisere",
                        "libro mayor", "hovedbok", "hovudboka", "ledger", "hauptbuch", "grand livre",
                        "finn dei", "finn de")
        expense_kw = ("gastos", "expense", "utgift", "kostnad", "aufwand", "charge",
                       "despesa", "cuentas de gastos", "expense account",
                       "kostnadskonto", "kostnadskon", "costos", "costo")
        project_kw = ("proyecto", "project", "prosjekt", "projekt", "projet", "projeto")
        has_analysis = any(_normalize_ascii(kw) in norm for kw in analysis_kw)
        has_expense = any(_normalize_ascii(kw) in norm for kw in expense_kw)
        has_project = any(_normalize_ascii(kw) in norm for kw in project_kw)
        return has_analysis and has_expense and has_project

    def _parse_project(self, prompt: str, task: ParsedTask) -> ParsedTask:
        rename_values = _extract_rename_values(prompt)
        name = _extract_name(prompt, ENTITY_KEYWORDS[Entity.PROJECT])
        effective_name = rename_values[0] if rename_values else name
        if task.action is Action.DELETE and task.identifier is not None and not effective_name:
            return task
        if not effective_name and task.action is Action.CREATE:
            # Check if this is a ledger analysis task (names come from ledger, not prompt)
            if self._is_ledger_analysis_prompt(prompt):
                task.attributes["isLedgerAnalysis"] = True
                return task
            raise ParsingError("Could not extract project name from prompt")
        if effective_name:
            task.target_name = effective_name
        if task.action is Action.CREATE:
            task.attributes["name"] = effective_name
        elif rename_values:
            task.attributes["name"] = rename_values[1]

        customer_name = _extract_linked_name(prompt, CUSTOMER_LINK_KEYWORDS)
        if customer_name and _normalize(customer_name) not in {_normalize(effective_name or ""), "customer", "kunde", "kunden"}:
            task.attributes["customerName"] = customer_name

        department_name = _extract_linked_name(prompt, DEPARTMENT_LINK_KEYWORDS)
        if department_name and _normalize(department_name) not in {_normalize(effective_name or ""), "department", "avdeling"}:
            task.attributes["departmentName"] = department_name

        pm_name = _extract_linked_name(prompt, PROJECT_MANAGER_KEYWORDS)
        if pm_name:
            task.attributes["projectManagerName"] = pm_name
        email = _extract_email(prompt)
        if email:
            task.attributes["projectManagerEmail"] = email

        org_match = ORG_NUMBER_PATTERN.search(prompt)
        if org_match:
            task.attributes["orgNumber"] = int(org_match.group(1))

        return task

    def _parse_travel_expense(self, prompt: str, task: ParsedTask) -> ParsedTask:
        if task.action is Action.DELETE:
            if task.identifier is None:
                raise ParsingError("Travel expense deletion needs an explicit ID in the prompt")
            return task
        if task.action is Action.UPDATE:
            departure_from, destination = _extract_route(prompt)
            all_dates = _extract_all_dates(prompt)
            purpose = _extract_after_keywords(prompt, ("purpose", "formål", "reason", "because"))
            if departure_from:
                task.attributes["departureFrom"] = departure_from
            if destination:
                task.attributes["destination"] = destination
            if all_dates:
                task.attributes["departureDate"] = all_dates[0]
            if len(all_dates) > 1:
                task.attributes["returnDate"] = all_dates[1]
            if purpose:
                task.attributes["purpose"] = purpose
            return task
        if task.action is not Action.CREATE:
            raise ParsingError("Travel expense support currently covers create, update, and delete workflows")

        employee_name = _extract_linked_name(prompt, ENTITY_KEYWORDS[Entity.EMPLOYEE])
        employee_email = _extract_email(prompt)
        departure_from, destination = _extract_route(prompt)
        all_dates = _extract_all_dates(prompt)
        departure_date = all_dates[0] if all_dates else None
        return_date = all_dates[1] if len(all_dates) > 1 else departure_date
        purpose = _extract_after_keywords(prompt, ("purpose", "formål", "reason", "because")) or "Business travel"
        amount = _extract_amount_after_keywords(prompt, ("amount", "beløp", "cost", "expense", "utgift", "kostnad"))

        if not employee_name:
            raise ParsingError("Could not extract employee name from travel expense prompt")
        if departure_date is None:
            raise ParsingError("Could not extract travel date from prompt")

        task.attributes["employeeName"] = employee_name
        if employee_email:
            task.attributes["employeeEmail"] = employee_email
        task.attributes["departureDate"] = departure_date
        task.attributes["returnDate"] = return_date or departure_date
        task.attributes["departureFrom"] = departure_from or "Unknown"
        task.attributes["destination"] = destination or "Unknown"
        task.attributes["purpose"] = purpose
        if amount is not None:
            task.attributes["amount"] = amount
        lowered = _normalize(prompt)
        task.attributes["isForeignTravel"] = "foreign" in lowered or "utenland" in lowered
        return task

    def _parse_invoice(self, prompt: str, task: ParsedTask) -> ParsedTask:
        if task.action is not Action.CREATE:
            raise ParsingError("Invoice support currently only covers create-invoice workflows")

        normalized = _normalize(prompt)
        if _contains_any(normalized, CREDIT_NOTE_KEYWORDS):
            invoice_number = _extract_invoice_number(prompt)
            all_dates = _extract_all_dates(prompt)
            credit_date = _extract_date_for_keywords(prompt, INVOICE_DATE_KEYWORDS) or (all_dates[0] if all_dates else date.today())
            customer_name = _trim_linked_entity_name(_extract_linked_name(prompt, CUSTOMER_LINK_KEYWORDS))
            amount = _extract_price(prompt)
            if amount is None:
                amount = _extract_currency_amount(prompt)
            if amount is None:
                amount = _extract_amount_after_keywords(prompt, ("for", "p\u00e5", "amount", "bel\u00f8p", "sem", "iva", "excl"))
            amount_scope = _extract_amount_scope(normalized)
            org_match = ORG_NUMBER_PATTERN.search(prompt)

            task.attributes["workflow"] = "creditNote"
            task.attributes["creditNoteDate"] = credit_date
            if invoice_number:
                task.attributes["invoiceNumber"] = invoice_number
            if customer_name:
                task.attributes["customerName"] = customer_name
            if amount is not None:
                task.attributes["amount"] = amount
                if amount_scope is True:
                    task.attributes["amountIsVatExclusive"] = True
                elif amount_scope is False:
                    task.attributes["amountIsVatInclusive"] = True
            if org_match:
                task.attributes["organizationNumber"] = org_match.group(1)
            comment = _extract_comment(prompt)
            if comment:
                task.attributes["comment"] = comment
            if task.identifier is None and not invoice_number and not customer_name:
                raise ParsingError("Credit note prompt must contain an invoice ID, invoice number, or customer reference")
            return task

        customer_name = _trim_linked_entity_name(_extract_linked_name(prompt, CUSTOMER_LINK_KEYWORDS))
        customer_email = _extract_email(prompt)
        product_name = _extract_linked_name(prompt, PRODUCT_LINK_KEYWORDS)
        product_number = _extract_number(prompt)
        amount = _extract_price(prompt)
        if amount is None:
            amount = _extract_currency_amount(prompt)
        if amount is None:
            amount = _extract_amount_after_keywords(prompt, ("for", "på", "amount", "beløp"))

        invoice_date = _extract_date_for_keywords(prompt, INVOICE_DATE_KEYWORDS) or date.today()
        due_date = _extract_date_for_keywords(prompt, DUE_DATE_KEYWORDS)
        due_in_days = _extract_due_in_days(prompt)
        if due_date is None and due_in_days is not None:
            due_date = invoice_date + timedelta(days=due_in_days)
        if due_date is None:
            due_date = invoice_date + timedelta(days=30)
            task.notes.append("Defaulted invoice due date to 30 days after invoice date")

        quantity = _extract_amount_after_keywords(prompt, ("quantity", "qty", "antall", "count"))
        if quantity is None:
            quantity = 1.0

        if not customer_name:
            raise ParsingError("Could not extract customer name from invoice prompt")
        if amount is None and not product_name and not product_number:
            raise ParsingError("Could not extract invoice amount from prompt")

        task.attributes["customerName"] = customer_name
        if customer_email:
            task.attributes["customerEmail"] = customer_email
        if product_name:
            task.attributes["productName"] = product_name
        if product_number:
            task.attributes["productNumber"] = product_number
        task.attributes["lineDescription"] = product_name or f"Invoice for {customer_name}"
        if amount is not None:
            task.attributes["amount"] = amount
        task.attributes["quantity"] = quantity
        task.attributes["invoiceDate"] = invoice_date
        task.attributes["invoiceDueDate"] = due_date
        return task

    def _parse_payment(self, prompt: str, task: ParsedTask) -> ParsedTask:
        if task.action is Action.DELETE:
            # Revert / delete payment
            invoice_number = _extract_invoice_number(prompt)
            if invoice_number:
                task.attributes["invoiceNumber"] = invoice_number
            customer_name = _extract_linked_name(prompt, CUSTOMER_LINK_KEYWORDS)
            if customer_name:
                task.attributes["customerName"] = customer_name
            return task

        if task.action is not Action.REGISTER:
            raise ParsingError("Payment support currently only covers register-payment and delete workflows")

        invoice_number = _extract_invoice_number(prompt)
        payment_date = _extract_date_for_keywords(prompt, PAYMENT_DATE_KEYWORDS) or date.today()
        amount = _extract_amount_after_keywords(
            prompt,
            ("amount", "beløp", "payment of", "betaling på", "paid", "betalt"),
        )
        payment_type = _extract_payment_type(prompt)
        customer_name = _extract_linked_name(prompt, CUSTOMER_LINK_KEYWORDS)

        if task.identifier is None and not invoice_number and not customer_name:
            raise ParsingError("Payment prompt must contain an invoice ID, invoice number, or customer reference")

        if invoice_number:
            task.attributes["invoiceNumber"] = invoice_number
        if amount is not None:
            task.attributes["amount"] = amount
        if payment_type:
            task.attributes["paymentTypeDescription"] = payment_type
        if customer_name:
            task.attributes["customerName"] = customer_name
        task.attributes["paymentDate"] = payment_date
        return task

    def _parse_contact(self, prompt: str, task: ParsedTask) -> ParsedTask:
        name = _extract_name(prompt, ENTITY_KEYWORDS[Entity.CONTACT])
        email = _extract_email(prompt)
        phone = _extract_phone(prompt)
        customer_name = _extract_linked_name(prompt, CUSTOMER_LINK_KEYWORDS)

        if name:
            task.target_name = name
            if task.action in (Action.CREATE, Action.UPDATE):
                first_name, last_name = _split_person_name(name)
                task.attributes["firstName"] = first_name
                task.attributes["lastName"] = last_name
        if email:
            task.attributes["email"] = email
        if phone:
            task.attributes["phoneNumberMobile"] = phone
        if customer_name and _normalize(customer_name) not in {"customer", "kunde", "contact", "kontakt"}:
            task.attributes["customerName"] = customer_name

        if task.action is Action.CREATE and "firstName" not in task.attributes:
            raise ParsingError("Could not extract contact name from prompt")

        return task

    def _parse_supplier(self, prompt: str, task: ParsedTask) -> ParsedTask:
        rename_values = _extract_rename_values(prompt)
        name = _extract_name(prompt, ENTITY_KEYWORDS[Entity.SUPPLIER])
        email = _extract_email(prompt)
        phone = _extract_phone(prompt)

        effective_name = rename_values[0] if rename_values else name
        if effective_name:
            task.target_name = effective_name
        if task.action is Action.CREATE and name:
            task.attributes["name"] = name
        elif rename_values:
            task.attributes["name"] = rename_values[1]
        if email:
            task.attributes["email"] = email
        if phone:
            task.attributes["phoneNumber"] = phone

        if task.action is Action.CREATE and "name" not in task.attributes:
            raise ParsingError("Could not extract supplier name from prompt")

        return task

    def _parse_voucher(self, prompt: str, task: ParsedTask) -> ParsedTask:
        if task.action is Action.DELETE:
            if task.identifier is None:
                raise ParsingError("Voucher deletion needs an explicit ID")
            return task

        description = _extract_after_keywords(prompt, DESCRIPTION_KEYWORDS)
        if not description:
            quoted = _extract_quoted_values(prompt)
            description = quoted[0] if quoted else None
        amount = _extract_price(prompt)
        if amount is None:
            amount = _extract_amount_after_keywords(prompt, ("amount", "beløp", "sum"))
        debit_account = _extract_account_number(prompt, DEBIT_KEYWORDS)
        credit_account = _extract_account_number(prompt, CREDIT_KEYWORDS)
        all_dates = _extract_all_dates(prompt)
        voucher_date = all_dates[0] if all_dates else date.today()

        if description:
            task.attributes["description"] = description
        task.attributes["voucherDate"] = voucher_date
        if amount is not None:
            task.attributes["amount"] = amount
        if debit_account is not None:
            task.attributes["debitAccountNumber"] = debit_account
        if credit_account is not None:
            task.attributes["creditAccountNumber"] = credit_account

        if task.action is Action.CREATE and not description:
            # Use a default description for month-end/year-end closing tasks
            task.attributes["description"] = "Journal entry"

        return task

    def _parse_timesheet(self, prompt: str, task: ParsedTask) -> ParsedTask:
        task.action = Action.CREATE
        employee_name = _extract_linked_name(prompt, ENTITY_KEYWORDS[Entity.EMPLOYEE])
        employee_email = _extract_email(prompt)
        activity_name = _extract_linked_name(prompt, ACTIVITY_LINK_KEYWORDS)
        hours = _extract_hours(prompt)
        all_dates = _extract_all_dates(prompt)
        entry_date = all_dates[0] if all_dates else date.today()
        project_name = _extract_linked_name(prompt, ENTITY_KEYWORDS[Entity.PROJECT])
        comment = _extract_comment(prompt)

        if not employee_name:
            raise ParsingError("Could not extract employee name from timesheet prompt")
        if hours is None:
            raise ParsingError("Could not extract number of hours from timesheet prompt")

        task.attributes["employeeName"] = employee_name
        if employee_email:
            task.attributes["employeeEmail"] = employee_email
        if activity_name:
            task.attributes["activityName"] = activity_name
        task.attributes["hours"] = hours
        task.attributes["date"] = entry_date
        if project_name and _normalize(project_name) not in {"project", "prosjekt"}:
            task.attributes["projectName"] = project_name
        if comment:
            task.attributes["comment"] = comment

        return task

    def _parse_generic(self, prompt: str, task: ParsedTask) -> ParsedTask:
        """Generic handler for Tier 2/3 entities without dedicated parsers.

        Extracts common fields (name, email, identifier, amounts, dates, quoted
        values) so the LLM-based service layer has something to work with.
        """
        entity_kws = ENTITY_KEYWORDS.get(task.entity, ())
        name = _extract_name(prompt, entity_kws)
        if name:
            task.target_name = name
            task.attributes["name"] = name

        email = _extract_email(prompt)
        if email:
            task.attributes["email"] = email

        amount = _extract_price(prompt)
        if amount is None:
            amount = _extract_currency_amount(prompt)
        if amount is not None:
            task.attributes["amount"] = amount

        all_dates = _extract_all_dates(prompt)
        if all_dates:
            task.attributes["date"] = all_dates[0]
        if len(all_dates) > 1:
            task.attributes["endDate"] = all_dates[1]

        description = _extract_after_keywords(prompt, DESCRIPTION_KEYWORDS)
        if description:
            task.attributes["description"] = description

        employee_name = _extract_linked_name(prompt, ENTITY_KEYWORDS[Entity.EMPLOYEE])
        if employee_name:
            task.attributes["employeeName"] = employee_name

        supplier_name = _extract_linked_name(prompt, ENTITY_KEYWORDS[Entity.SUPPLIER])
        if supplier_name:
            task.attributes["supplierName"] = supplier_name

        customer_name = _extract_linked_name(prompt, CUSTOMER_LINK_KEYWORDS)
        if customer_name and _normalize(customer_name) not in {"customer", "kunde", "kunden"}:
            task.attributes["customerName"] = customer_name

        invoice_number = _extract_invoice_number(prompt)
        if invoice_number:
            task.attributes["invoiceNumber"] = invoice_number

        return task
