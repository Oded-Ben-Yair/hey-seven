# State-by-State Regulatory Requirements for AI Casino Communication

## Universal Requirements (All US Jurisdictions)

### Self-Exclusion Compliance
- Check self-exclusion database before EVERY outbound communication
- Strict liability: system failures are not a defense
- Existing accounts must be identified and suspended
- Self-excluded players cannot receive any targeted marketing
- Audit trail required for every check

### BSA/AML Compliance
- Casinos with GAGR > $1M are financial institutions under BSA
- CTR required for cash transactions > $10,000 (file within 15 days)
- SAR required for suspicious transactions ≥ $5,000
- AI must flag suspicious patterns for human review
- Cannot autonomously refuse service for suspected structuring

### TCPA Compliance (Federal)
- Written consent required before marketing SMS/calls
- Opt-out must be honored within 5 minutes (effective April 2025)
- Opt-out can be expressed informally (not just "STOP")
- One-to-one consent required per entity (effective January 26, 2026)
- Quiet hours: respect state-specific restrictions
- Liability: $500-$1,500 per violation

### Responsible Gaming Disclosures
- Age restriction (21+) must appear prominently
- Risk statement: "Gambling involves financial risk"
- Problem gambling helpline (state-specific number)
- Licensed operator name displayed
- Required in ALL promotional communications

---

## Nevada

**Regulator**: Nevada Gaming Control Board / Nevada Gaming Commission
**Key Regulation**: Regulation 5A (Interactive Gaming)

### AI-Specific Rules
- Notice 2026-04 (Jan 2026): Due diligence required for each jurisdiction
- Regulation 5A.070: "Robust and redundant identification methods" required
- AI cannot autonomously verify player identities
- All monitoring documented and available to Board on request

### Self-Exclusion
- Regulation 5A.110: Verify against self-exclusion list within 30 days of registration
- Must check NRS 463.151 and Regulation 28 excluded persons list
- Must record player physical location while logged in

### Marketing
- Regulation 5A.135: "Reasonable steps" to prevent marketing to self-excluded
- No explicit push notification ban (unlike NJ) but self-exclusion check mandatory
- No explicit AI disclosure requirement yet

### AML Enforcement
- 2025: ~$27M in penalties across major Strip operators
- Citations for failing to identify/respond to high-risk patrons
- Expectation: AI-assisted monitoring must flag anomalies for human review

### Penalties
- License suspension or revocation under NRS 463.310
- Multi-million dollar fines for repeated AML failures

---

## New Jersey

**Regulator**: Division of Gaming Enforcement (DGE)
**Key Legislation**: Senate Bills 3401, 3419, 3420, 3461 (Feb 2026)

### AI-Specific Rules
- SB 3401: BANS push notifications/texts soliciting wagers or deposits
- Definition covers automatic messages when platform is not actively open
- Penalty: not less than $500 per offense per violation
- SB 3420: BANS promotional offers to players using responsible gaming tools

### Payment Restrictions
- SB 3461: BANS credit card payments for online casino games
- AI payment systems must reject credit card transactions
- AI cannot suggest credit cards as payment method

### Account Management
- SB 3419: Published rules required for account limitations
- Patrons must be notified when accounts are limited
- AI cannot unilaterally enforce limitations without human review

### Enforcement Track Record
- BetMGM: $260,905 fine (152 self-excluded individuals gambled)
- Strict liability applied regardless of system error defense

---

## Pennsylvania

**Regulator**: Pennsylvania Gaming Control Board (PGCB)
**Key Development**: Enhanced KYC Requirements (effective Sept 30, 2025)

### AI-Specific Rules
- Enhanced KYC MFA fraud prevention applies to: account creation, MFA, payment methods, geolocation
- Exact match required: birth date, SSN, last name (scraped from uploaded ID)
- Manual KYC process required for ANY anomalies
- Deceased individual detection required
- Multi-jurisdiction account checks required
- Device account limits enforced

### Third-Party AI
- Operators cannot delegate verification to unlicensed AI vendors
- All platform services must be reviewed through PGCB licensing process

### Enforcement
- Valley Forge Casino: $30,000 fine (self-excluded player gambled 13 months)

### Upcoming
- Senate Bill 666 (April 2025): Potential gaming regulatory changes

---

## Data Privacy Requirements

### California (CCPA, effective Jan 1, 2026)
- Casino player data = sensitive personal information
- Collection must be "reasonably necessary and proportionate"
- ADMT framework applies to significant decisions
- Risk assessment required before AI deployment
- ADMT compliance deadline: January 1, 2027

### Nevada
- Regulation 5A.070: Protect authorized player PII
- Designated senior privacy officer required
- No comprehensive state privacy law equivalent to CCPA

---

## AI Autonomy Limits

### What AI CAN Do
- Generate comp recommendations based on player data
- Auto-approve low-value comps within pre-defined parameters
- Assist monitoring for AML/BSA compliance
- Analyze player behavior for churn prediction
- Send communications to consented, non-excluded players with required disclosures
- Track and report on player engagement metrics

### What AI CANNOT Do Without Human Oversight
- Approve high-value comps
- Make responsible gaming intervention decisions
- Override self-exclusion protocols
- File SAR/CTR reports (final decision must be human)
- Restrict or suspend player accounts
- Send communications without required disclosures
- Process transactions near $10,000 threshold without compliance review

### Audit Requirements
- Every AI decision must maintain audit trail
- Decision basis, timestamp, player ID, outcome logged
- Available for regulatory review on demand
- Human review documented for flagged decisions

---

## AI Disclosure Laws (Emerging)

### Maine (2025)
- Must disclose AI chatbot use when it could be mistaken for human
- "Clear and conspicuous" disclosure required

### New York (2025)
- AI companion safety features required
- Sustained interaction interruption notifications
- Self-harm detection protocols

### FTC (Federal)
- Three AI-washing cases since April 2025
- Cannot misrepresent AI capabilities
- Must substantiate performance claims
- Proposed orders bar misrepresenting AI effectiveness

---

## Compliance Architecture for AI Casino Host

```
Pre-Communication Checks:
1. Self-exclusion database query (logged)
2. Consent verification (channel-specific opt-in)
3. Age verification (21+)
4. Responsible gaming mechanism check (no promos if active)
5. AML/BSA risk flag review
6. TCPA quiet hours check (recipient timezone)
7. Communication frequency cap check

Communication Requirements:
1. AI nature disclosure (where required by state)
2. 21+ age statement
3. Risk disclosure
4. State-specific helpline number
5. Licensed operator name

Post-Communication:
1. Log interaction with audit trail
2. Process opt-outs within 5 minutes
3. Archive for regulatory review
4. Update communication history
```
