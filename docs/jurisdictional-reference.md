# Jurisdictional Reference: US Casino Regulations

## State Comparison

| Requirement | Connecticut | New Jersey | Nevada | Pennsylvania |
|-------------|-------------|------------|--------|--------------|
| Gaming age minimum | 21 | 21 | 21 | 21 |
| Self-exclusion authority | Tribal gaming commissions (Mohegan, Pequot) | NJ DGE | NGCB | PGCB |
| Responsible gaming helpline | 1-800-MY-RESET | 1-800-GAMBLER | 1-800-GAMBLER (NCPG) | 1-800-GAMBLER |
| State helpline | 1-888-789-7777 | 1-800-GAMBLER | 1-800-GAMBLER | 1-800-848-1880 |
| CTR threshold | $10,000 (federal) | $10,000 (federal) | $10,000 (federal) | $10,000 (federal) |
| SAR filing | BSA/FinCEN | BSA/FinCEN | BSA/FinCEN + NGC | BSA/FinCEN |
| Alcohol service | Until 2 AM | Until 2 AM | 24 hours | Until 2 AM |
| Smoking | Designated areas only | Indoor ban | Permitted in gaming areas | Designated areas only |

## Self-Exclusion Programs

### Connecticut (Mohegan Sun, Foxwoods)
- **Authority**: Tribal gaming commissions (NOT CT DCP -- tribal casinos are sovereign)
  - Mohegan Sun: Mohegan Tribal Gaming Commission
  - Foxwoods: Mashantucket Pequot Tribal Nation Gaming Commission
- **Duration**: 1-year minimum (longer durations available upon request)
- **URL**: Contact property directly (tribal jurisdiction — not ct.gov/selfexclusion)
- **Phone**: Contact property directly (tribal gaming commission)
- **Process**: In-person enrollment at casino or tribal gaming commission office

### New Jersey (Hard Rock AC, Resorts World)
- **Authority**: NJ Division of Gaming Enforcement (DGE)
- **Duration**: 1, 5 years, or lifetime
- **URL**: njportal.com/dge/selfexclusion
- **Phone**: 1-833-788-4DGE
- **Process**: In-person or online enrollment

### Nevada (Wynn Las Vegas)
- **Authority**: Nevada Gaming Control Board (NGCB)
- **Regulation**: NGC Regulation 5.170 (voluntary self-exclusion). Note: NRS 463.368 governs involuntary exclusions for cheating — different program.
- **Duration**: 1-year minimum (removal requires written petition to NGCB per Reg. 5.170), or lifetime (irrevocable)
- **URL**: gaming.nv.gov
- **Phone**: 1-702-486-2000
- **Process**: In-person at NGCB office

### Pennsylvania (Parx Casino)
- **Authority**: PA Gaming Control Board (PGCB)
- **Duration**: 1 year, 5 years, or lifetime
- **URL**: gamingcontrolboard.pa.gov
- **Phone**: 1-855-405-1429
- **Process**: In-person at PGCB office or casino

## Notes
- Federal BSA/AML requirements apply uniformly ($10K CTR, suspicious activity)
- State-specific regulations apply to self-exclusion, alcohol, smoking
- AI host must defer to human host for all self-exclusion requests
- Casino profiles in `src/casino/config.py` must match this reference
- CT tribal casinos self-exclude through their own gaming commissions, NOT CT DCP (R39 fix D10-M003)
