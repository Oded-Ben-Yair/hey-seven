# US Gaming Regulations for AI Communication — 2025-2026

**Date**: 2026-02-12
**Confidence**: High (specific regulatory citations included)

---

## Core Regulatory Principle

**AI systems can enhance casino operations but must operate under rigorous human oversight**, particularly in areas affecting player protection, financial reporting, and responsible gaming compliance. No jurisdiction currently permits fully autonomous AI decision-making for player-facing interactions without human oversight.

---

## State-by-State Requirements

### Nevada

**Governing Body**: Nevada Gaming Control Board (NGCB) / Nevada Gaming Commission
**Key Regulation**: Regulation 5A (Interactive Gaming)

**AI Communication Rules**:
- Notice 2026-04 (Jan 16, 2026): Licensees must conduct independent due diligence on legality in each jurisdiction
- Reg 5A.070: "Robust and redundant identification methods" required — AI cannot autonomously verify identity
- Reg 5A.110: Player registration must verify self-exclusion status within 30 days
- Reg 5A.135: "Reasonable steps" to prevent marketing to self-excluded individuals
- All monitoring activities must be "fully documented" and available to the Board

**Self-Exclusion**: Must check against NRS 463.151 and Regulation 28 lists before ANY communication
**AML/BSA**: 2025 enforcement wave: ~$27M in penalties across major Strip operators
**Penalties**: License suspension or revocation under NRS 463.310

---

### New Jersey

**Governing Body**: Division of Gaming Enforcement (DGE)
**Key Development**: Four Senate Bills introduced February 2026

**Senate Bill 3401** (Push Notification Ban):
- Prohibits push notifications/text messages soliciting wagers or deposits
- Covers: casino licensees, Internet gaming affiliates, sports wagering licensees
- Penalty: Not less than $500 per offense (potential millions for mass campaigns)

**Senate Bill 3461** (Credit Card Ban):
- Prohibits credit card payments for online casino games and sports wagering
- AI payment systems must be configured to reject credit card transactions

**Senate Bill 3419** (Account Limitations):
- Requires published rules governing account limitations
- Patrons must be notified when accounts are limited
- AI cannot unilaterally enforce limitations without human review

**Senate Bill 3420** (Responsible Gaming):
- Prohibits incentive promotions to players using responsible gaming mechanisms
- AI cannot auto-generate offers to players with deposit/loss limits active

**Enforcement**: BetMGM fined $260,905 for allowing 152 self-excluded individuals to gamble

---

### Pennsylvania

**Governing Body**: Pennsylvania Gaming Control Board (PGCB)
**Key Development**: Enhanced KYC multi-factor authentication (Jan 2025)

**Enhanced KYC Requirements** (effective Sept 30, 2025):
- Identification and liveness check
- Data accuracy verification
- Manual KYC process required for anomalies
- Deceased individual detection
- Multi-jurisdiction account checks
- Device account limits
- Account creation requires exact match: birth date, SSN, last name

**AI Constraints**: Discrepancies between provided info and ID documents must trigger manual review. Third-party AI systems require PGCB licensing review.

**Senate Bill 666** (April 2025): Proposes gaming regulatory framework changes (monitor for AI-specific provisions)

---

## Self-Exclusion Requirements (All Jurisdictions)

### Universal Rules
1. Check self-exclusion database **before EVERY communication** (not just account creation)
2. Checks must be logged for audit purposes
3. Self-excluded players cannot receive: targeted mailings, telemarketing, player club materials, promotional materials
4. Any existing accounts must be identified and suspended
5. Remaining balances must be refunded
6. AI system failures are NOT a defense — strict liability applies

### Penalties for Violations
| Jurisdiction | Example | Fine |
|-------------|---------|------|
| Pennsylvania | Valley Forge Casino (13-month violation) | $30,000 |
| New Jersey | BetMGM (152 self-excluded gambled) | $260,905 |
| Nevada | Major Strip operator (years of violations) | Multi-million dollar |

### Implementation for AI Systems
- Query self-exclusion database before EVERY outbound communication
- Real-time or near-real-time database synchronization (not just daily batch)
- Redundant systems — single-point failures not acceptable
- Audit trail: requesting system, timestamp, PlayerID, result

---

## BSA/AML (Bank Secrecy Act / Anti-Money Laundering)

### Casino Classification
Casinos with GAGR > $1M are classified as **financial institutions** under the BSA.

### Currency Transaction Reports (CTR)
- **Required for**: Cash-in/cash-out transactions > $10,000
- **Trigger transactions**: Chip purchases, front money, safekeeping deposits, marker payments, currency bets, wire transfers, casino check purchases, currency exchanges
- **Filing deadline**: Within 15 calendar days
- AI systems must identify and flag reportable transactions for human review

### Suspicious Activity Reports (SAR)
- **Required for**: Transactions ≥ $5,000 where institution suspects evasion or suspicious activity
- Must implement risk-based monitoring (AI-assisted monitoring acceptable)
- Final SAR filing decisions must involve human judgment
- No ongoing account review mandate post-SAR, but risk-based monitoring required

### Structuring Detection
- AI must review complete transaction history to detect deliberate transaction-splitting
- Cannot autonomously refuse service — must flag for human review
- 2025 Nevada enforcement: $27M in penalties for AML control failures

### What This Means for AI Casino Hosts
- Any player communication involving financial transactions must comply with BSA
- AI cannot process deposits/withdrawals without CTR compliance checks
- Suspicious patterns must be flagged for compliance team review
- Host-equivalent AI must maintain awareness of AML obligations

---

## TCPA (Telephone Consumer Protection Act)

### New Rules (2025-2026)

**Effective April 11, 2025**:
- Consumers can revoke consent via ANY method (email, voicemail, informal messages like "Leave me alone")
- AI must interpret wide range of opt-out requests (not just "STOP")
- Opt-out confirmation must be sent within 5 minutes
- Confirmation cannot contain promotional content
- Liability: $500-$1,500 per violation

**Effective January 26, 2026** (One-to-One Consent):
- Each casino entity must obtain separate consent
- Affiliate consent sharing prohibited
- Consent cannot be bundled across multiple businesses

**Quiet Hours**:
- Connecticut: 9 AM - 8 PM local time (stricter than federal)
- AI must determine recipient's local time zone
- Default to most restrictive zone if location uncertain

**Liability**: Extends to any entity that "benefits from or directs" the communication (including AI platform providers)

---

## Responsible Gaming Disclosures

### Required in ALL Promotional Communications
1. **Age restriction**: Bold "21+" (or 18+ where applicable) in headlines
2. **Risk statement**: "Gambling involves financial risk"
3. **Problem gambling helpline**: State-specific number required
4. **Licensed operator name**: Must be prominently displayed
5. **State-specific language**: Varies by jurisdiction

### Impact on AI Systems
- No AI-generated promotional communication can omit required disclosures
- Disclosures must appear consistently, not dynamically generated
- Templates must include all legally required elements
- AI content generation constrained by disclosure requirements

---

## Data Privacy (CCPA and State Laws)

### CCPA Requirements (effective Jan 1, 2026)
- Casino player data = "sensitive personal information" (financial accounts, geolocation, etc.)
- Collection must be "reasonably necessary and proportionate"
- ADMT (Automated Decision-Making Technology) framework applies to significant decisions
- Risk assessment with attestation required before processing
- ADMT compliance deadline: January 1, 2027

### AI-Specific Implications
- AI comp decisions may constitute "significant decisions" under CCPA
- Risk assessments required before deploying AI for player decisions
- Player data used for AI training must comply with proportionality requirements

---

## Can AI Make Comp Decisions Autonomously?

**Short answer: Not fully. Human oversight required.**

### Current Regulatory Consensus
1. AI can **recommend** comp amounts based on player data
2. AI can **auto-approve** within pre-defined parameters (low-value comps)
3. AI **cannot** autonomously approve high-value comps without human review
4. AI **cannot** make responsible gaming intervention decisions autonomously
5. AI **cannot** override self-exclusion protocols
6. All AI decisions must maintain audit trails

### Best Practice Architecture
```
AI System generates recommendation →
Human reviews for high-value/flagged decisions →
Approval documented with audit trail →
Communication sent with required disclosures →
Response tracked and logged
```

---

## AI Disclosure Requirements (Emerging)

### Maine AI Chatbot Disclosure Law (2025)
- Must disclose use of AI chatbots that could be mistaken for humans
- Disclosure must be "clear and conspicuous"

### New York AI Companion Safeguards (2025)
- Safety features required for sustained AI interactions
- Detection of suicidal ideation/self-harm required

### FTC AI-Washing Enforcement (2025-2026)
- Three major cases since April 2025
- Cannot misrepresent AI capabilities
- Must substantiate claims about AI performance
- Proposed orders bar misrepresenting AI effectiveness

### Implications for Hey Seven
- Any AI host interaction must clearly disclose AI nature
- Cannot claim capabilities the system doesn't actually have
- Performance claims must be substantiated with evidence

---

## Compliance Framework Summary for AI Casino Host

| Requirement | Pre-Communication | During Communication | Post-Communication |
|------------|-------------------|---------------------|-------------------|
| Self-exclusion | Check database | N/A | Log result |
| Consent verification | Verify opt-in | N/A | Honor opt-out within 5 min |
| Age verification | Confirm 21+ | Include 21+ disclosure | N/A |
| Responsible gaming | Check RG mechanism status | Include helpline/disclosures | N/A |
| AML/BSA | Review player risk flags | Monitor for structuring | File CTR/SAR as needed |
| TCPA | Verify consent, check quiet hours | Comply with channel rules | Process opt-outs |
| Data privacy | Purpose limitation check | Minimize data collection | Retention limits |
| AI disclosure | N/A | Disclose AI nature | N/A |
| Audit trail | Log decision basis | Log interaction | Archive for review |

---

## Sources
- Nevada Gaming Control Board: Regulation 5A, Notice 2026-04
- New Jersey Legislature: SB 3401, SB 3419, SB 3420, SB 3461
- PGCB: Enhanced KYC Guidance (Jan 2025)
- FinCEN: SAR FAQ (Oct 2025), BSA Reporting Requirements
- AGA: Responsible Gaming Regulations Guide, AML Best Practices (2025)
- FTC: AI enforcement actions (2025-2026)
- Virginia Admin Code: Title 11, Agency 5, Chapter 60
- WilmerHale: Legal Developments in Gaming Industry (H2 2025)