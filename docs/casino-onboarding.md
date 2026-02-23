# Casino Onboarding Checklist

Step-by-step checklist for adding a new casino property to the Hey Seven platform.

## Prerequisites

- Casino regulatory information (state, gaming age, self-exclusion authority)
- Property operational data (restaurants, entertainment, hotel, hours)
- Branding guidelines (persona name, tone, formality level)
- Helpline phone numbers (state gambling helpline, property contact)

## Onboarding Steps

### 1. Casino Profile Configuration

**File**: `src/casino/config.py` — `CASINO_PROFILES` dict

Add a new entry with casino_id as key. Required fields:

| Section | Required Fields |
|---------|----------------|
| `branding` | persona_name, tone, formality_level, emoji_allowed, exclamation_limit |
| `regulations` | state, gaming_age_minimum, ai_disclosure_required, ai_disclosure_law, quiet_hours_start/end, responsible_gaming_helpline, state_helpline |
| `operational` | timezone, contact_phone |
| `prompts` | casino_name_display, greeting_template, fallback_message |

### 2. Feature Flags

**File**: `src/casino/feature_flags.py`

Per-casino feature flag overrides (optional — defaults from `DEFAULT_FEATURES` apply).

### 3. Property Data File

**File**: `data/<casino_id>.json`

Create a JSON file with structured property data for RAG ingestion:
- `property`: name, location, website, phone
- `restaurants`: array of dining venues
- `entertainment`: shows, events, amenities, spa
- `hotel`: room types, towers
- `gaming`: table games, slots, poker room

### 4. Environment Variables

Set for the deployment:
- `CASINO_ID=<casino_id>` — must match CASINO_PROFILES key
- `PROPERTY_NAME=<display name>`
- `PROPERTY_DATA_PATH=data/<casino_id>.json`
- `PROPERTY_WEBSITE=<url>`
- `PROPERTY_PHONE=<phone>`
- `PROPERTY_STATE=<state name>`

### 5. RAG Ingestion

Run ingestion for the new property data:
```bash
CASINO_ID=<casino_id> PROPERTY_DATA_PATH=data/<casino_id>.json python -m src.rag.pipeline
```

### 6. Guardrail Verification

Verify state-specific guardrails cover the new property's state:
- Self-exclusion authority in `config.py` `regulations.self_exclusion_authority`
- State helpline number in `regulations.state_helpline`
- Responsible gaming helpline in `regulations.responsible_gaming_helpline`
- Check `src/agent/guardrails.py` for state-specific regex patterns

### 7. Validation

```bash
# Run tests with the new casino_id
CASINO_ID=<casino_id> python -m pytest tests/ -x -q

# Verify profile loads correctly
python -c "from src.casino.config import get_casino_profile; import json; print(json.dumps(get_casino_profile('<casino_id>'), indent=2))"
```

## Startup Validation

The application logs a CRITICAL warning at startup if `CASINO_ID` is not found
in `CASINO_PROFILES` and falls back to `DEFAULT_CONFIG` (Mohegan Sun defaults).
Check logs for: `Casino profile not found for <casino_id>, using DEFAULT_CONFIG`.

## Common Mistakes

- **Missing CASINO_PROFILES entry**: Silent fallback to Mohegan Sun defaults.
  The guest sees CT helplines for a NJ property.
- **Forgetting to update PROPERTY_STATE**: Age verification and helplines use
  the wrong state's regulations.
- **Data file not matching CASINO_ID**: RAG ingestion creates chunks with wrong
  property_id metadata, causing cross-tenant retrieval.
