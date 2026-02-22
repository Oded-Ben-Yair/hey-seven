# R25 Hostile Review: Domain Intelligence & Persona Quality

**Reviewer**: Claude Opus 4.6 (hostile)
**Date**: 2026-02-22
**Scope**: Knowledge-base accuracy, casino config, HEART escalation, test coverage
**Files Reviewed**: 6 target files + production wiring trace

---

## Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Domain Intelligence | 7/10 | Solid casino ops knowledge with real programs and tier structures. Multiple factual errors and a major knowledge gap (no entertainment guide). |
| Persona Excellence | 6/10 | HEART framework is well-structured but NOT WIRED to production code. Escalation language in _base.py is hardcoded, not using HEART_ESCALATION_LANGUAGE dict. |
| Multi-Property Readiness | 7/10 | 3-property config architecture is clean. TypedDict/profile schema drift and hardcoded CT helplines limit true multi-state readiness. |

---

## CRITICAL Findings (Must Fix)

### C-001: HEART_ESCALATION_LANGUAGE is DEAD CODE (Severity: CRITICAL)

**File**: `/home/odedbe/projects/hey-seven/src/agent/prompts.py:317-335`

`HEART_ESCALATION_LANGUAGE` is defined in `prompts.py` but is NEVER imported or used by any production code path. It is only referenced in `tests/test_r24_domain.py`.

The actual escalation language in `_base.py:206-213` is hardcoded inline:
```python
system_prompt += (
    "\n\n## Escalation Guidance\n"
    "The guest has expressed frustration across multiple messages. "
    "After addressing their current question, gently offer to connect "
    "them with a human host..."
)
```

This hardcoded string does NOT use the HEART framework steps (hear, empathize, apologize, resolve, thank). The HEART dict is scaffolded, tested, and never called. This is the exact anti-pattern from Rule 4 (dead code) and the Documentation Honesty rule (scaffolded != implemented).

**Fix**: Import `HEART_ESCALATION_LANGUAGE` in `_base.py` and use it to construct the escalation guidance dynamically based on frustration level (e.g., 2 consecutive = "hear" + "empathize", 3+ = full HEART sequence).

### C-002: The Mirage Listed as Active MGM Property (Severity: CRITICAL)

**File**: `/home/odedbe/projects/hey-seven/knowledge-base/casino-operations/loyalty-programs.md:23`

> "Valid across all MGM Resorts properties: Bellagio, ARIA, MGM Grand, Mandalay Bay, Park MGM, Vdara, **The Mirage**, Luxor, Excalibur..."

The Mirage permanently closed on July 17, 2024, and is being demolished for the Hard Rock Las Vegas. Including a closed property in the knowledge base means the AI agent could recommend a non-existent venue to guests. This is a factual accuracy failure that directly impacts guest trust.

**Fix**: Remove "The Mirage" from the MGM Rewards network list. Optionally add a note about Hard Rock Las Vegas (opening ~2027).

### C-003: `get_responsible_gaming_helplines()` is Hardcoded to CT (Severity: CRITICAL)

**File**: `/home/odedbe/projects/hey-seven/src/agent/prompts.py:32-38`

```python
def get_responsible_gaming_helplines() -> str:
    return RESPONSIBLE_GAMING_HELPLINES_DEFAULT  # Always CT
```

For the NJ property (`hard_rock_ac`), the function returns Connecticut helplines (1-888-789-7777 CT Council) instead of New Jersey helplines (1-800-GAMBLER). The `RegulationConfig` in `CASINO_PROFILES` correctly stores per-state helplines, but this function ignores them entirely.

This is a regulatory compliance risk. Directing NJ guests to CT helplines is both confusing and potentially non-compliant with NJ DGE requirements.

**Fix**: Accept `casino_id` parameter, look up the casino's `regulations` config, and return state-appropriate helplines. The data already exists in `CASINO_PROFILES[casino_id]["regulations"]`.

---

## HIGH Findings

### H-001: TypedDict / CASINO_PROFILES Schema Drift (Severity: HIGH)

**File**: `/home/odedbe/projects/hey-seven/src/casino/config.py`

`OperationalConfig` TypedDict (lines 61-70) defines 8 fields. But `CASINO_PROFILES` operational sections include 4 extra fields NOT in the TypedDict:
- `property_type` (line 220)
- `property_size_gaming_sqft` (line 221)
- `dining_venues` (line 222)
- `hotel_towers` (line 223)

Similarly, `RegulationConfig` TypedDict (lines 48-58) is missing:
- `self_exclusion_authority`
- `self_exclusion_url`
- `self_exclusion_options`

`total=False` on TypedDicts means this won't crash, but it defeats the purpose of type checking. Any code using `profile["operational"]["property_type"]` gets no type safety, and IDE autocompletion won't show these fields.

**Fix**: Add missing fields to `OperationalConfig` and `RegulationConfig` TypedDicts. Add a parity assertion test (per code-quality.md: `assert set(TypedDict.__annotations__) >= set(profile_keys)`).

### H-002: CASINO_PROFILES and DEFAULT_CONFIG Are Mutable (Severity: HIGH)

**File**: `/home/odedbe/projects/hey-seven/src/casino/config.py:110, 174`

Both `DEFAULT_CONFIG` and `CASINO_PROFILES` are plain mutable dicts at module level. Per code-quality.md ("Immutable Module-Level Defaults"), these should be wrapped in `MappingProxyType` to prevent accidental mutation in async LangGraph applications.

`get_casino_profile()` returns a direct reference to the profile dict (line 384: `return CASINO_PROFILES.get(casino_id, DEFAULT_CONFIG)`). Any caller mutating the returned dict corrupts the shared state for all subsequent requests.

**Fix**: Wrap in `MappingProxyType` or return `copy.deepcopy()` from `get_casino_profile()`. Note: `get_casino_config()` (the async version) already does `copy.deepcopy(DEFAULT_CONFIG)` on the fallback path (line 530), but not for Firestore-loaded configs.

### H-003: No Entertainment/Shows Knowledge Base File (Severity: HIGH)

**File**: Missing entirely

The dining guide exists. The hotel guide exists. The loyalty programs guide exists. But there is NO `entertainment-guide.md` in the knowledge base. Mohegan Sun has:
- 10,000-seat Mohegan Sun Arena (major concerts)
- Comedy club (Comix Roadhouse)
- Ultra nightclub
- Various bars and lounges

Foxwoods has:
- The Grand Theater (4,000 seats)
- The Premier Theater
- Foxwoods clubs and bars

An AI casino host that cannot answer "What shows are playing this weekend?" or "Where can I see live music?" is missing a core competency. Entertainment is one of the 3 pillars of casino hospitality (gaming, dining, entertainment).

**Fix**: Create `knowledge-base/casino-operations/entertainment-guide.md` covering arenas, theaters, comedy clubs, nightclubs, bars, and how entertainment comps work.

### H-004: `get_casino_profile()` Not Used by Any Production Code (Severity: HIGH)

**File**: `/home/odedbe/projects/hey-seven/src/casino/config.py:371`

`get_casino_profile()` (the sync function) is only called from tests. Production code uses `get_casino_config()` (async) or directly imports `DEFAULT_CONFIG`. The `CASINO_PROFILES` dict is essentially a static data store that is never queried at runtime.

The production code in `_base.py:185` does:
```python
from src.casino.config import DEFAULT_CONFIG
branding = DEFAULT_CONFIG.get("branding", {})
```

It always reads `DEFAULT_CONFIG`, never the property-specific profile. This means Hard Rock AC gets Seven's persona instead of Ace's, and Foxwoods gets Seven instead of Foxy.

**Fix**: Wire `get_casino_profile(casino_id)` into the production code path. The `casino_id` should come from the request context or session state.

---

## MEDIUM Findings

### M-001: Foxwoods Amenities Section Missing from Hotel Operations (Severity: MEDIUM)

**File**: `/home/odedbe/projects/hey-seven/knowledge-base/casino-operations/hotel-operations.md`

Mohegan Sun has a detailed "On-Property Amenities" table (lines 28-35) listing spa, fitness, pool, and golf. Foxwoods has NO equivalent amenities section. Foxwoods has:
- Norwich Spa at Foxwoods
- G Spa
- Foxwoods Golf Course
- Indoor pool
- Fitness center

A guest asking "Does Foxwoods have a spa?" would get no RAG context.

**Fix**: Add a Foxwoods "On-Property Amenities" table parallel to Mohegan Sun's.

### M-002: Dining Guide Missing Hard Rock AC Restaurants (Severity: MEDIUM)

**File**: `/home/odedbe/projects/hey-seven/knowledge-base/casino-operations/dining-guide.md`

The dining guide covers Mohegan Sun (17 venues) and Foxwoods (9 venues) but has ZERO entries for Hard Rock Atlantic City, which is a configured property in `CASINO_PROFILES`. Hard Rock AC has:
- Hard Rock Cafe
- Council Oak Fish
- Council Oak Steaks & Seafood
- Il Mulino New York
- Kuro Japanese
- Sugar Factory

A property in the config with no knowledge base content is a data gap.

**Fix**: Add Hard Rock AC dining section to dining-guide.md (or create a separate per-property file).

### M-003: Hotel Operations Missing Hard Rock AC Rooms (Severity: MEDIUM)

**File**: `/home/odedbe/projects/hey-seven/knowledge-base/casino-operations/hotel-operations.md`

Same gap as M-002: hotel operations covers Mohegan Sun and Foxwoods but not Hard Rock AC. The config says `hotel_towers: 1` for Hard Rock AC, but the knowledge base has no room types, amenities, or pricing guidance.

**Fix**: Add Hard Rock AC hotel section.

### M-004: Spanish Greeting Missing Accent Marks (Severity: MEDIUM)

**File**: `/home/odedbe/projects/hey-seven/src/casino/config.py:131, 227, 291, 357`

All Spanish greeting templates use:
```
"Como puedo ayudarte hoy?"
```

Correct Spanish requires: **"Como puedo ayudarte hoy?"** -- actually, the correct form is "**\u00bfC\u00f3mo puedo ayudarte hoy?**" with opening inverted question mark and accent on the 'o'. This is basic Spanish orthography. For a bilingual AI host, incorrect Spanish signals low quality to native speakers.

**Fix**: Change to `"\u00bfC\u00f3mo puedo ayudarte hoy?"` in all greeting templates.

### M-005: Loyalty Programs Missing Wynn "Red Card" Rename (Severity: MEDIUM)

**File**: `/home/odedbe/projects/hey-seven/knowledge-base/casino-operations/loyalty-programs.md:95-107`

Wynn Rewards was rebranded and restructured. The tier names and thresholds should be verified against the current (2025-2026) program structure. Additionally, the Wynn network statement (line 106) says "Wynn regional properties" -- Wynn does not operate regional casinos in the traditional sense; they operate Encore Boston Harbor and Wynn Macau.

**Fix**: Verify current Wynn Rewards tier structure and correct the network description.

### M-006: HEART Framework Missing Scenario-Specific Variants (Severity: MEDIUM)

**File**: `/home/odedbe/projects/hey-seven/src/agent/prompts.py:317-335`

The HEART language is generic. Casino-specific escalation scenarios need tailored language:
- **Wait time complaint**: "I know waiting isn't how you want to spend your evening..."
- **Comp dispute**: "I understand that feels like your loyalty isn't being recognized..."
- **Room issue**: "Your comfort is our top priority, and I'm sorry we fell short..."
- **Service failure**: "That's not the experience we want for any of our guests..."

Generic "I completely understand how frustrating that must be" sounds scripted for an AI that should feel like a knowledgeable insider.

**Fix**: Add scenario-keyed variants to `HEART_ESCALATION_LANGUAGE` (e.g., `"empathize_wait"`, `"empathize_comp"`, `"empathize_room"`), or at minimum include 2-3 variant phrases per step.

---

## LOW Findings

### L-001: Test Coverage Gaps in test_r24_domain.py

Missing test cases:
1. **No test for `get_casino_config()` async path** -- only `get_casino_profile()` (sync) is tested
2. **No negative test for invalid section access** -- what happens when accessing `profile["nonexistent_section"]`?
3. **No test that CASINO_PROFILES keys match OperationalConfig TypedDict** (parity check)
4. **No test that DEFAULT_CONFIG contains all the same keys as CASINO_PROFILES entries** (coverage parity)
5. **No test for `_deep_merge()` edge cases** (nested dict merge, key collision, empty override)
6. **No test for `clear_config_cache()`**
7. **No test that knowledge-base content is consistent with CASINO_PROFILES** (e.g., profile says 40 dining venues, dining guide should have ~40 entries for that property)

### L-002: Mohegan Sun Loyalty Program Name Ambiguity

**File**: `/home/odedbe/projects/hey-seven/knowledge-base/casino-operations/loyalty-programs.md:55`

The section header says "Mohegan Sun Momentum" but the actual program branding uses just "Momentum" or "Mohegan Sun Momentum Rewards". Verify the current official program name.

### L-003: `ai_disclosure_law` Is Empty String for All Properties

**File**: `/home/odedbe/projects/hey-seven/src/casino/config.py:146, 201, 265, 329`

Every property has `"ai_disclosure_law": ""`. If AI disclosure is required (`ai_disclosure_required: True`), the specific law reference should be populated. For CT, this would be the relevant tribal-state compact provisions. For NJ, the DGE's AI guidance.

### L-004: Quiet Hours Same for All Properties (21:00-08:00)

All 3 properties have identical quiet hours. NJ may have different TCPA quiet hours restrictions than CT. Verify per-state regulations.

---

## Dead Code Summary

| Item | Status | Evidence |
|------|--------|----------|
| `HEART_ESCALATION_LANGUAGE` | DEAD (scaffolded) | Zero imports from `src/` (only tests) |
| `get_casino_profile()` | DEAD (scaffolded) | Zero imports from `src/` (only tests) |
| `CASINO_PROFILES` | DEAD (scaffolded) | Only used by `get_casino_profile()` which is itself dead |

**Impact**: The entire multi-property config system (`CASINO_PROFILES`, `get_casino_profile`) and the HEART escalation framework are scaffolded but not wired. Tests validate the data structure exists, but no production code path reads it.

---

## Factual Accuracy Issues

| Issue | File | Line | Problem |
|-------|------|------|---------|
| The Mirage listed as active | loyalty-programs.md | 23 | Closed July 2024, being demolished |
| Wynn "regional properties" | loyalty-programs.md | 106 | Wynn has Encore Boston Harbor, not "regional" casinos |
| Spanish missing accents | config.py | 131+ | "Como" should be "\u00bfC\u00f3mo" |
| CT helplines served to NJ guests | prompts.py | 38 | Regulatory non-compliance risk |

---

## Top 3 Fixes (Priority Order)

1. **Wire HEART_ESCALATION_LANGUAGE and CASINO_PROFILES to production code** -- Both are high-quality data structures that are completely dead. This is the single highest-impact change.
2. **Fix `get_responsible_gaming_helplines()` to be property-aware** -- Regulatory compliance risk for multi-state deployment.
3. **Remove The Mirage from MGM network list** -- Factual error that would embarrass the product in any demo or review.

---

## Summary

The knowledge base demonstrates genuine domain expertise in casino operations -- loyalty program structures, comp math, ADT formulas, and dietary accommodations show real research. However, the implementation has a critical gap between what is scaffolded and what is wired. Three entire subsystems (`CASINO_PROFILES`, `get_casino_profile`, `HEART_ESCALATION_LANGUAGE`) exist only in tests, not in production code paths. The multi-property promise is undermined by a hardcoded CT helpline function and production code that always reads `DEFAULT_CONFIG` regardless of which casino is active. Adding Hard Rock AC to the config without corresponding knowledge base content creates a property with a persona but no knowledge.

**Overall Assessment**: Strong foundation, weak wiring. The data quality is 8/10 but the production integration is 5/10.
