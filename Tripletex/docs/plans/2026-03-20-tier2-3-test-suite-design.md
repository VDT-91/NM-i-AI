# Tier 2/3 Test Suite Design

## Problem

Tier 1 (simple CRUD) is well-covered by existing tests. Tier 2 (multi-step workflows, x2 multiplier) and Tier 3 (complex scenarios, x3 multiplier) have no test coverage. A single Tier 3 fix is worth three Tier 1 fixes in competition scoring.

## Approach: Risk-weighted testing

Focus tests where points are lost: Tier 2/3 entities, multi-step workflows, file processing, and multi-language edge cases. Use `pytest.mark.parametrize` for language matrices where it adds value.

## Test Files

| File | Tests | Focus |
|---|---|---|
| `tests/test_parser_tier2_3.py` | ~45 | Parser correctness for Tier 2/3 entities, multi-language, edge cases |
| `tests/test_service_tier2_3.py` | ~15 | Service workflow call sequences for multi-step handlers |
| `tests/test_csv_processing.py` | ~9 | File handling, CSV formats, encoding, attachment pipeline |
| `stress_test.py` (additions) | ~40 | Live Tier 2/3 prompts against deployed endpoint |

## 1. Parser Tests (`test_parser_tier2_3.py`)

### 1a. Multi-step workflow parsing (highest risk)

| Test group | Languages | What we verify |
|---|---|---|
| Invoice + payment (fresh account) | NO, EN, ES, PT, DE, FR, NN | action=register, entity=payment, extracts customerName + amount + productName |
| Credit note by customer (no invoice #) | NO, EN, PT | workflow=creditNote, customerName, amount, amountIsVatExclusive |
| Fixed-price project + milestone invoice | NO, EN, DE | entity redirects to invoice, fixedPrice extracted, amount = milestone |
| Timesheet + invoice (register hours then bill) | NO, EN | hours, activityName, employeeName, projectName, customerName, amount all extracted |
| Multi-line invoice (different VAT rates) | NO, EN | orderLines array with per-line vatRate |

### 1b. Tier 2/3 entity identification

| Test group | Languages | Key assertion |
|---|---|---|
| Incoming invoice (not sales invoice) | NO, EN, ES, DE, FR | entity=incoming_invoice, supplierName, amount |
| Bank statement import | NO, EN, DE | entity=bank_statement |
| Salary transaction | NO, EN, FR, DE, ES | entity=salary_transaction, employeeName, baseSalary, bonus |
| Purchase order | NO, EN, DE | entity=purchase_order, supplierName |
| Dimension + voucher | NO, EN | entity=dimension, dimensionName, dimensionValues, voucherAmount |
| Account (ledger) | NO, EN | entity=account, accountNumber, accountName |
| Leave of absence | NO, EN, DE | entity=leave_of_absence, employeeName, startDate, leaveType |
| Asset / fixed asset | NO, EN | entity=asset, assetName, acquisitionCost |

### 1c. Language edge cases

| Test | What it catches |
|---|---|
| Mixed NO+EN | Parser doesn't choke on language mixing |
| Typos in keywords | Rule-based parser resilience |
| Terse prompts | Minimal prompt still extracts entity + key fields |
| Date format variations | Dates parsed correctly |
| Amount format variations | Amounts parsed as floats |
| Norwegian special characters | Names with ae/oe/aa preserved |

## 2. Service Workflow Tests (`test_service_tier2_3.py`)

### Extended FakeTripletexClient

Adds tracking for: deliver/approve travel expense, approve incoming invoice, bank reconciliation flow (import, create reconciliation, suggest matches, close), salary transaction, module activation.

### Test cases

| Test | Workflow verified | Key assertions |
|---|---|---|
| Travel expense deliver+approve | create -> deliver -> approve | Both called with expense ID |
| Travel expense with per diem | create -> per diem -> deliver -> approve | Per diem payload correct, deliver/approve still called |
| Incoming invoice + approve | ensure supplier -> create -> approve | approve called with returned invoice ID |
| Bank statement full reconciliation | import -> find account -> create reconciliation -> suggest matches -> close | All 5 steps called in order |
| Bank statement no file | raises ParsingError | Error mentions "attached file" |
| Invoice + payment (fresh account) | create customer -> create order -> create invoice -> pay | Payment uses amountCurrencyOutstanding |
| Credit note on existing invoice | find invoice -> create credit note | Correct invoice ID |
| Credit note by customer | find customer -> find invoice -> create credit note | Falls back to customer search |
| Salary transaction | ensure employee -> ensure DOB -> create salary | Employee has DOB, salary type resolved |
| Company module activation | activate module | correct module name |
| Multi-line invoice | create customer -> order with N lines -> invoice | Multiple orderLines with different VAT types |
| Fixed-price project + invoice | create customer -> project (fixedPrice) -> order -> invoice | fixedPrice and isFixedPrice=true set |
| Dimension + voucher | create dimension -> values -> voucher | Voucher posting linked to dimension value |
| Incoming invoice payment | find supplier invoice -> pay | pay_supplier_invoice called |

## 3. CSV/File Processing Tests (`test_csv_processing.py`)

### Bank statement CSV

| Test | What it verifies |
|---|---|
| DNB CSV format | Correct file path and format passed to import |
| Nordea CSV format | Format detection works |
| CSV with encoding issues | Norwegian chars in ISO-8859-1 don't crash |
| Empty CSV | Graceful error or import still attempted |
| Non-CSV attachment | .xml extension still picked up |
| Multiple attachments | CSV selected over PDF for bank import |

### Attachment text extraction

| Test | What it verifies |
|---|---|
| CSV content in prompt | Text extracted and available to parser |
| Invoice-like text | Parser extracts fields from attachment content |
| Large attachment truncation | >100KB file truncated, no OOM |

## 4. Stress Test Additions (`stress_test.py`)

~40 new prompts covering:

- Incoming invoice (NO, EN, DE, ES, FR)
- Incoming invoice payment (NO, EN)
- Multi-line invoice (NO, EN)
- Invoice + payment single prompt (NO, EN)
- Fixed-price project + invoice (NO, EN)
- Salary transaction (NO, EN, FR, DE, ES)
- Purchase order (NO, EN)
- Travel expense with per diem (NO, EN)
- Bank statement with CSV file attachment (NO, EN)
- Dimension + voucher (NO, EN)
- Leave of absence (NO, EN, DE)
- Asset / fixed asset (NO, EN)
- Credit note by customer (NO, EN)
- Revert payment (NO, EN)

Bank statement tests include a base64-encoded CSV in the `files` array. Stress test runner updated to support dict entries with files alongside plain string prompts.

## Conscious Trade-offs

- No real Gemini API calls in unit tests (LLM tested via stress tests)
- No Tripletex API contract tests (real API issues surface in stress tests)
- No exhaustive Tier 1 x 7 language matrix (already works, not worth maintenance cost)
- No new pip dependencies (unittest, tempfile, pathlib, base64)
