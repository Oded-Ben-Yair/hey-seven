# Component 10: CMS + SMS — Architecture Audit

**Auditor**: auditor-api
**Date**: 2026-03-05
**Scope**: Google Sheets CMS integration, Telnyx SMS client, TCPA compliance, webhook handlers

---

## 1. Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/cms/sheets_client.py` | 176 | Google Sheets API v4 wrapper. Lazy import. `read_category()`, `read_all_categories()`, content hash for change detection. |
| `src/cms/webhook.py` | 248 | CMS webhook handler: HMAC-SHA256 signature verification, replay protection, TTLCache-backed content hash change detection, live re-indexing via `reingest_item()`. |
| `src/cms/validation.py` | 104 | Content validation: `REQUIRED_FIELDS` per category (8 categories), `validate_item()`, `validate_details_json()`. |
| `src/sms/telnyx_client.py` | 200 | Telnyx SMS client: httpx.AsyncClient, `send_message()`, `check_delivery_status()`, GSM-7/UCS-2 encoding detection, word-boundary message segmentation. |
| `src/sms/webhook.py` | 305 | SMS webhook: Ed25519 signature verification, replay protection, `WebhookIdempotencyTracker` (TTL + hard cap), delivery receipt handling, `handle_inbound_sms()`. |
| `src/sms/compliance.py` | 763 | TCPA compliance: STOP/HELP/START keywords (English + Spanish), 280+ area code to timezone mappings, `is_quiet_hours()`, `ConsentHashChain` (SHA-256/HMAC-SHA256), tiered consent hierarchy. |

**Total**: 6 files, 1,796 lines

---

## 2. Wiring Verification

All modules are **actively wired** from the API entry point:

- `src/cms/webhook.py` — imported by `src/api/app.py` for `POST /cms/webhook` endpoint
- `src/cms/sheets_client.py` — imported by `src/cms/webhook.py` for live re-indexing
- `src/cms/validation.py` — imported by `src/cms/webhook.py` for content validation
- `src/sms/webhook.py` — imported by `src/api/app.py` for `POST /sms/webhook` endpoint
- `src/sms/telnyx_client.py` — imported by `src/sms/webhook.py` for outbound SMS
- `src/sms/compliance.py` — imported by `src/sms/webhook.py` for TCPA checks

**Grep proof**:
```
src/api/app.py:from src.cms.webhook import handle_cms_webhook, verify_webhook_signature
src/api/app.py:from src.sms.webhook import handle_inbound_sms
src/cms/webhook.py:from src.cms.sheets_client import SheetsClient
src/cms/webhook.py:from src.cms.validation import validate_item
src/sms/webhook.py:from src.sms.telnyx_client import TelnyxSMSClient
src/sms/webhook.py:from src.sms.compliance import check_consent, is_quiet_hours
```

**Verdict**: All 6 files are REAL production code, fully wired.

---

## 3. Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `tests/test_cms.py` | 58 | Webhook verification, content validation, re-indexing, signature replay, hash change detection |
| `tests/test_sheets_client.py` | 16 | Google Sheets API reads, category parsing, content hashing |
| `tests/test_sms.py` | 81 | TCPA compliance, consent chain, quiet hours, STOP/HELP/START, timezone lookup, message segmentation, webhook signature verification, idempotency tracker |

**Total**: 155 tests covering the CMS + SMS layer

---

## 4. Live vs Mock Assessment

| Test File | Mock Count | Live Calls | Assessment |
|-----------|-----------|------------|------------|
| `test_cms.py` | 46 | 0 | **Appropriately mocked** — Google Sheets API is external. Mocks validate integration logic. |
| `test_sheets_client.py` | ~10 | 0 | **Appropriately mocked** — cannot call Google Sheets in CI. |
| `test_sms.py` | 32 | 0 | **Appropriately mocked** — Telnyx is external. TCPA logic is deterministic (quiet hours, consent, keyword detection) and doesn't need mocks for those paths. |

**Summary**: Mocking is appropriate here. CMS and SMS are external service integrations — you SHOULD mock the HTTP clients. The deterministic logic (TCPA compliance, content validation, signature verification, consent chain) is tested without mocks where possible. No live Telnyx or Google Sheets calls in tests, which is correct.

---

## 5. Known Gaps

### GAP-1: SMS Feature Flag Off by Default (BY DESIGN)
`SMS_ENABLED: bool = False` in config.py and `sms_enabled: False` in feature_flags.py. Requires Telnyx setup (API key, messaging profile, public key, from number).

**Impact**: None — this is intentional. SMS is wired and tested but gated behind configuration.
**Status**: Ready to enable when Telnyx credentials are provisioned.

### GAP-2: Consent Store is In-Memory (MEDIUM)
`ConsentHashChain` stores consent records in a dict. No persistence to Firestore or Redis. Lost on container restart.

**Impact**: After a restart, all consent records are lost. Outbound SMS would require re-consent from every guest.
**Mitigation**: `ConsentHashChain` is designed for Firestore backing (HMAC-SHA256 hash chain is serializable). Wiring to Firestore is straightforward.

### GAP-3: Area Code Coverage is US-Only (LOW)
280+ area codes mapped to US timezones. No international support.

**Impact**: None for current market (US casinos only). Would need expansion for international properties.

### GAP-4: CMS Re-Indexing is Synchronous (LOW)
`reingest_item()` in the webhook handler runs synchronously in the request path. For bulk content updates, this could cause webhook timeout.

**Impact**: Slow webhook responses during bulk CMS updates. Single-item updates are fine.
**Mitigation**: Could move to background task, but volume is low (casino content changes infrequently).

### GAP-5: No Outbound SMS Campaign Orchestrator (SCAFFOLDED)
`outbound_campaigns_enabled: False` in feature flags. No campaign scheduling, audience targeting, or send throttling exists. Only inbound SMS handling and single outbound send are implemented.

**Impact**: Outbound marketing campaigns not available.
**Status**: Scaffolded — flag exists, client exists, compliance exists. Missing: campaign scheduler, audience segmentation, send queue.

---

## 6. Confidence: 82%

**Strengths**:
- TCPA compliance is thorough: STOP/HELP/START in English + Spanish, quiet hours with timezone-aware area code lookup, consent hash chain with HMAC-SHA256
- Ed25519 webhook signature verification for SMS (Telnyx) + HMAC-SHA256 for CMS (Google Sheets)
- Replay protection on both webhook handlers (timestamp window)
- Idempotency tracker on SMS webhook (prevents duplicate processing)
- Content validation with required fields per category (8 categories)
- GSM-7 vs UCS-2 encoding detection with word-boundary segmentation
- Production secret validation: CONSENT_HMAC_SECRET and TELNYX_PUBLIC_KEY hard-fail in production if using defaults

**Weaknesses**:
- Consent store not persistent (in-memory only)
- Outbound campaigns not implemented (only flag + client exist)
- CMS re-indexing is synchronous

---

## 7. Verdict: PRODUCTION-READY (with consent store caveat)

CMS integration is fully production-ready. SMS is production-ready for inbound handling and single outbound sends. The consent store persistence gap (GAP-2) must be addressed before enabling SMS in production — losing consent records on restart would create TCPA liability. Outbound campaigns are scaffolded but not implemented. Overall, the compliance code is impressively thorough for a regulated domain.
