# Round 8 — Batch 1 Applied (FIX-1 through FIX-7, Priority 1: CRITICAL)

**Applied**: 2026-02-13
**File**: assignment/architecture.md

## Fixes Applied

### FIX-1: Rafi Ashkenazi deal values corrected [line 3291]
- Changed "$4.7B PokerStars acquisition" to "$4.7B Sky Betting & Gaming acquisition"
- Changed "$12B Flutter merger" to "~$6B Flutter Entertainment merger creating the world's largest online gambling company"

### FIX-2: NV NGC -> NV NGCB [lines 3293, 3297]
- Fixed both occurrences: "NV NGC" (numismatic grading) to "NV NGCB" (Nevada Gaming Control Board)
- Line 3293: Restructured market bullet to include regulatory layers with correct abbreviation
- Line 3297: Fixed in regulatory complexity section

### FIX-3: .format() crash vector [lines 272, 492]
- Replaced `ROUTER_PROMPT.format()` with `Template(ROUTER_PROMPT).safe_substitute()`
- Replaced `VALIDATION_PROMPT.format()` with `Template(VALIDATION_PROMPT).safe_substitute()`
- Added comments explaining the DoS vector (user input with curly braces causes KeyError)

### FIX-4: state["property_id"] KeyError [line 2768]
- Changed `state["property_id"]` to `get_property_config()["id"]`
- property_id was removed from state in R6; monitoring code was stale

### FIX-5: Monitoring metric inverts cosine distance [line 2777]
- Changed "top result score > 0.8" to "top result distance < 0.4"
- Added clarification "(cosine distance; lower = more similar)"

### FIX-6: Stale validation state across turns [lines 278-281]
- Router node now returns reset values: `validation_result: None, retry_count: 0, retry_feedback: None`
- Prevents stale retry state from leaking between multi-turn conversation turns

### FIX-7: Streaming corruption on RETRY path [lines 1753, 1777-1787]
- Added `retry_replace_sent = False` tracking variable in stream generator
- Before streaming retry tokens, emits `event: replace` with empty content to clear frontend buffer
- Prevents [failed][corrected] token concatenation in the UI

## Verification

All 7 critical fixes applied surgically. No surrounding code disrupted. Document style preserved.
