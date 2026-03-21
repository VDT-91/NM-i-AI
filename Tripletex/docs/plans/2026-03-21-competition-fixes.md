# Competition Emergency Fixes — 49th → Top 10

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs and improve scoring from 58.48 to 70+ points (top 10) in the NM i AI Tripletex competition.

**Architecture:** Fix 5 critical bugs in service.py that cause 0-score tasks, improve efficiency by leveraging free GETs before writes, and harden the LLM parser against edge cases.

**Tech Stack:** Python 3.12, FastAPI, Gemini 2.5 Flash, Tripletex REST API v2

---

## Context

**Current:** 49th place, 58.48 points, 225 submissions, 30/30 tasks seen
**Target:** Top 10 (~70+ points)
**Gap analysis:** T1=14.1 (good), T2=23.4 (missing 6.1), T3=21.0 (missing 20.7)

### Biggest Point Losses (from production logs)

| Bug | Tasks Affected | Points Lost |
|-----|---------------|-------------|
| Project module activation fails (422: "Du mangler tilgang") | Task 11 (T2), others | ~4-6 pts |
| `_ensure_account` uses `id=number` fallback (wrong!) | Voucher tasks | ~3-5 pts |
| Incoming invoice POST returns 403 (wrong module) | T3 incoming invoice tasks | ~3-4 pts |
| Voucher postings include `"date"` field (causes 422) | Incoming invoice vouchers | ~2-3 pts |
| LLM returns `"MISSING_EXPENSE_ACCOUNT"` string | Voucher tasks | ~1-2 pts |
| No GET validation before writes (4xx errors tank efficiency) | All tasks | ~3-5 pts efficiency |

---

### Task 1: Fix Project Module Activation

**Files:**
- Modify: `tripletex_solver/service.py:1272-1285` (`_activate_project_module`)

**Problem:** `_activate_project_module` tries `KOMPLETT`, `PROJECT`, `PROSJEKT` but none work. The `_activate_module` method maps "prosjekt" → `SMART_PROJECT`. Cloud Run logs show repeated `422: Du mangler tilgang til å opprette nye prosjekter`.

**Step 1: Fix the module name list**

Change `_activate_project_module` (line 1272) to try the correct module names in priority order. The `_activate_module` method already knows `SMART_PROJECT` is the right one. Also try `SMART` as fallback since it enables most features.

```python
def _activate_project_module(self) -> None:
    """Activate the module required for project creation, trying multiple options."""
    for mod in ("SMART_PROJECT", "SMART", "KOMPLETT", "PROJECT", "PROSJEKT"):
        try:
            self.client.activate_sales_module(mod)
            LOGGER.info("Activated %s module for project creation", mod)
            return
        except TripletexAPIError as e:
            if e.status_code == 409:
                LOGGER.info("%s module already active", mod)
                return
            LOGGER.warning("%s activation failed (status=%s): %s", mod, e.status_code, e)
    LOGGER.warning("All project module activations failed, proceeding anyway")
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_service.py -v -x -k project 2>&1 | head -40`

**Step 3: Commit**

---

### Task 2: Fix `_ensure_account` Fallback Bug

**Files:**
- Modify: `tripletex_solver/service.py:4103-4124` (`_ensure_account`)

**Problem:** When account creation fails, the fallback does `{"id": number, ...}` — using the account NUMBER as the ID. This is wrong because Tripletex account IDs are internal auto-incremented integers, not the 4-digit account numbers. This causes `422: account_number field doesn't exist` on voucher creation.

**Step 1: Fix the fallback to search again instead of using number as ID**

```python
def _ensure_account(self, number: int) -> dict[str, Any]:
    """Find account by number. If not found, create it using NS 4102 name."""
    cache_key = f"account:{number}"
    cached = self._cache_get(cache_key)
    if cached:
        return cached
    results = self.client.search_accounts_by_number(number)
    if results:
        self._cache_set(cache_key, results[0])
        return results[0]
    # Account not found — create it
    name = self._NS4102_NAMES.get(number, f"Konto {number}")
    LOGGER.info("Account %d not found, creating as %r", number, name)
    try:
        acct = self.client.create("/ledger/account", {"number": number, "name": name})
        self._cache_set(cache_key, acct)
        return acct
    except Exception as e:
        LOGGER.warning("Could not create account %d: %s — searching again", number, e)
        # Search again in case it was created concurrently or the creation
        # partially succeeded
        results = self.client.search_accounts_by_number(number)
        if results:
            self._cache_set(cache_key, results[0])
            return results[0]
        raise EntityNotFoundError(f"Account {number} not found and could not be created: {e}")
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_service.py -v -x -k voucher 2>&1 | head -40`

**Step 3: Commit**

---

### Task 3: Fix Incoming Invoice Module Activation + Voucher Postings

**Files:**
- Modify: `tripletex_solver/service.py:3781-3787` (module activation in `_create_incoming_invoice`)
- Modify: `tripletex_solver/service.py:3949-4052` (remove `"date"` from postings in `_create_incoming_invoice_via_voucher`)

**Problem 1:** `_create_incoming_invoice` activates `SMART`, `AGRO`, `APPROVE_VOUCHER` but `POST /incomingInvoice` still returns 403. Need to try more module names.

**Problem 2:** Individual voucher postings have `"date": invoice_date.isoformat()` which the API rejects with 422. The date belongs ONLY on the voucher header, not on individual postings.

**Step 1: Fix module activation — try comprehensive list**

At line 3781, change:
```python
for mod in ("SMART", "AGRO", "APPROVE_VOUCHER"):
```
to:
```python
for mod in ("SMART", "APPROVE_VOUCHER", "AGRO", "KOMPLETT"):
```

**Step 2: Remove `"date"` from all individual posting dicts in `_create_incoming_invoice_via_voucher`**

In the method starting at line 3949, remove ALL occurrences of `"date": invoice_date.isoformat()` from posting dicts. There are ~5 occurrences. The date should ONLY be on the voucher payload (which already has it at the end of the method).

Find and remove these lines (approximate locations in postings):
- `"row": row, "date": invoice_date.isoformat(), "description": description,` → `"row": row, "description": description,`

Do this for ALL posting dicts in the method (expense_posting, vat posting, credit_posting, expense_posting_nv, and the no-VAT credit_posting).

**Step 3: Run tests**

Run: `python -m pytest tests/ -v -x -k "incoming" 2>&1 | head -40`

**Step 4: Commit**

---

### Task 4: Harden LLM Parser Against Placeholder Strings

**Files:**
- Modify: `tripletex_solver/service.py` (multiple methods)
- Modify: `tripletex_solver/llm_parser.py:90-93` (system prompt already says don't use placeholders, but add validation)

**Problem:** The LLM sometimes returns `"MISSING_EXPENSE_ACCOUNT"` or similar strings instead of valid account numbers, causing `ValueError: invalid literal for int()`. Already partially fixed in `_create_voucher` but needs hardening everywhere accounts are used.

**Step 1: Add defensive int conversion helper**

Add at top of service.py (after imports):
```python
def _safe_int(value: Any, fallback: int | None = None) -> int | None:
    """Safely convert a value to int. Returns fallback if not numeric."""
    if value is None:
        return fallback
    try:
        return int(value)
    except (ValueError, TypeError):
        LOGGER.warning("Non-numeric value %r, using fallback %s", value, fallback)
        return fallback
```

**Step 2: Use `_safe_int` in `_create_incoming_invoice`**

At line ~3825 where `debit_account_num` is used:
```python
debit_account_num = _safe_int(task.attributes.get("debitAccountNumber"), fallback=4000)
```

**Step 3: Use `_safe_int` in `_create_voucher` fallback section**

Already handled with try/except blocks, but ensure consistency.

**Step 4: Run tests**

Run: `python -m pytest tests/ -v -x 2>&1 | tail -20`

**Step 5: Commit**

---

### Task 5: Add Free GET Validation Before Write Calls

**Files:**
- Modify: `tripletex_solver/service.py` (multiple `_ensure_*` methods and `_create_*` methods)

**Problem:** The competition scores efficiency on WRITE calls only. GETs are FREE. But our fresh-account optimization skips validation GETs, causing write failures (4xx errors) that tank efficiency scores. Adding pre-write GET validation costs nothing but prevents costly 4xx errors.

**Step 1: In `_ensure_customer`, add a search-first check**

Currently creates directly without searching. Add a quick GET search first (free) to avoid duplicate creation (409):

In the `_ensure_customer` method, before the `self.client.create("/customer", payload)` call, add:
```python
# Free GET to check for existing customer (avoids 409 on write)
try:
    existing = self.client.search_customers(name=name)
    if existing:
        LOGGER.info("Found existing customer %r (id=%s)", name, existing[0].get("id"))
        self._cache_set(cache_key, existing[0])
        return existing[0]
except Exception:
    pass  # Non-critical, proceed with create
```

**Step 2: Same pattern for `_ensure_employee`**

Add a search-first GET before creating:
```python
try:
    first, last = name.rsplit(" ", 1) if " " in name else (name, "")
    existing = self.client.search_employees(firstName=first, lastName=last)
    if existing:
        self._cache_set(cache_key, existing[0])
        return existing[0]
except Exception:
    pass
```

**Step 3: Same pattern for `_ensure_department` and `_ensure_supplier`**

Add search-first GETs for these too.

**Step 4: Run tests**

Run: `python -m pytest tests/ -v -x 2>&1 | tail -20`

**Step 5: Commit**

---

### Task 6: Improve Module Activation Coverage

**Files:**
- Modify: `tripletex_solver/service.py` (add module activation calls before entity creation in more places)

**Problem:** Several entity creation methods don't activate required modules. Fresh accounts need modules activated before certain entities can be created.

**Step 1: Add module activation to `_create_incoming_invoice`**

Already done in Task 3, but also activate in the bank reconciliation path.

**Step 2: Add SMART module activation in `_reconcile_bank_statement`**

Already activates SMART but also needs `APPROVE_VOUCHER` for incoming invoice search:
```python
for mod in ("SMART", "APPROVE_VOUCHER"):
    try:
        self.client.activate_sales_module(mod)
    except Exception:
        pass
```

**Step 3: Add module activation to `_create_salary_transaction`**

Before creating salary transactions, activate WAGE module:
```python
try:
    self.client.activate_sales_module("WAGE")
except Exception:
    pass
```

**Step 4: Add module activation in `_create_timesheet_entry`**

Before creating timesheet entries, activate time tracking:
```python
try:
    self.client.activate_sales_module("SMART_TIME_TRACKING")
except Exception:
    pass
```

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v 2>&1 | tail -30`

**Step 6: Commit**

---

### Task 7: Fix Voucher Account Number Field in Postings

**Files:**
- Modify: `tripletex_solver/service.py` (`_create_voucher` method, around line 2870)

**Problem:** Voucher postings include a `_account_number` metadata field that leaks into the API payload and causes `422: _account_number: Feltet eksisterer ikke i objektet` (field doesn't exist).

**Step 1: Ensure `_account_number` is stripped before sending**

Find where the voucher payload is built and ensure `_account_number` is popped from each posting before the API call. Check if there's already a stripping step and verify it runs for ALL posting types.

Search for `_account_number` in the file and ensure every posting that includes it has it removed before the API call.

The existing code may already pop it, but verify the pop happens BEFORE the create call for all code paths (the LLM postings path AND the rule-based fallback path).

**Step 2: Run tests**

Run: `python -m pytest tests/ -v -x -k voucher 2>&1 | head -30`

**Step 3: Commit**

---

### Task 8: Improve Incoming Invoice Voucher Type Resolution

**Files:**
- Modify: `tripletex_solver/service.py` (`_create_incoming_invoice_via_voucher`)

**Problem:** The voucher fallback for incoming invoices tries to use voucher type "Leverandørfaktura" or "Supplier invoice". If neither exists, it creates a generic voucher that won't score correctly for incoming invoice checks.

**Step 1: Add more voucher type name variants**

```python
try:
    vt_candidates = self.client.search_voucher_types(name="Leverandørfaktura")
    if not vt_candidates:
        vt_candidates = self.client.search_voucher_types(name="Supplier invoice")
    if not vt_candidates:
        vt_candidates = self.client.search_voucher_types(name="Inngående faktura")
    if not vt_candidates:
        # Try partial match on all voucher types
        all_types = self.client.list("/ledger/voucherType", fields="id,name", params={"count": 100})
        vt_candidates = [vt for vt in all_types if any(kw in (vt.get("name") or "").lower() for kw in ("leverandør", "supplier", "inngående", "incoming"))]
    if vt_candidates:
        voucher_payload["voucherType"] = {"id": vt_candidates[0]["id"]}
except Exception as e:
    LOGGER.warning("Could not resolve supplier invoice voucher type: %s", e)
```

**Step 2: Commit**

---

### Task 9: Add Pre-Dispatch Module Activation

**Files:**
- Modify: `tripletex_solver/service.py` (`_pre_process` method)

**Problem:** Many task types need specific modules activated but each handler does it independently (or not at all). Add a centralized module activation step in `_pre_process` based on entity type.

**Step 1: Add entity-to-module mapping in `_pre_process`**

Find `_pre_process` and add a module activation step at the beginning:

```python
# Activate modules based on entity type (GETs are free, prevents 403/422 on writes)
entity_modules = {
    Entity.PROJECT: ("SMART_PROJECT", "SMART"),
    Entity.TIMESHEET: ("SMART_TIME_TRACKING", "SMART"),
    Entity.SALARY_TRANSACTION: ("WAGE",),
    Entity.INCOMING_INVOICE: ("SMART", "APPROVE_VOUCHER"),
    Entity.PURCHASE_ORDER: ("SMART",),
    Entity.TRAVEL_EXPENSE: ("SMART",),
    Entity.INVOICE: ("BASIS",),
    Entity.BANK_STATEMENT: ("SMART", "APPROVE_VOUCHER"),
    Entity.DIVISION: ("SMART",),
    Entity.INVENTORY: ("LOGISTICS",),
    Entity.STOCKTAKING: ("LOGISTICS",),
    Entity.GOODS_RECEIPT: ("LOGISTICS",),
    Entity.ASSET: ("FIXED_ASSETS_REGISTER",),
    Entity.DIMENSION: ("SMART",),
}
modules_needed = entity_modules.get(task.entity, ())
for mod in modules_needed:
    try:
        self.client.activate_sales_module(mod)
    except Exception:
        pass
```

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -v 2>&1 | tail -30`

**Step 3: Commit**

---

### Task 10: Deploy and Test

**Step 1: Deploy to Cloud Run**

Run deploy.ps1 or:
```powershell
& 'C:\Users\sondree\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd' run deploy tripletex-solver --source . --region europe-north1 --allow-unauthenticated --memory 1Gi --timeout 300 --min-instances 1 --set-env-vars "GEMINI_API_KEY=<key>,GCS_BUCKET=tripletex-solver-data"
```

**Step 2: Verify health**

```bash
curl https://tripletex-solver-554556611033.europe-north1.run.app/health
```

**Step 3: Submit to competition**

Submit the endpoint URL on the competition platform and let tasks flow in. Monitor Cloud Run logs for new errors.

**Step 4: Monitor and iterate**

Watch for new 4xx errors in logs and fix iteratively.

---

## Expected Impact

| Fix | Est. Points |
|-----|------------|
| Project module activation | +4-6 |
| Account ID fallback | +3-5 |
| Incoming invoice module + voucher fix | +3-4 |
| LLM placeholder hardening | +1-2 |
| Free GET validation (efficiency) | +3-5 |
| Module pre-activation | +2-3 |
| **Total estimated** | **+16-25** |

**Projected score: 58.48 + 16-25 = 74-83 → Top 5-10**
