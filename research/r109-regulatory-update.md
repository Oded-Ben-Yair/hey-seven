# R109 Regulatory & Compliance Update

**Date**: 2026-03-09
**Author**: Research Specialist (automated)
**Scope**: US gaming regulation changes affecting Hey Seven AI casino host agent
**Period covered**: H2 2025 through March 2026

## Sources

### Primary Sources
- [WilmerHale: Legal Developments in the Gaming Industry H2 2025](https://www.wilmerhale.com/en/insights/client-alerts/20260205-legal-developments-in-the-gaming-industry-second-half-of-2025)
- [AGA Responsible Gaming Code of Conduct (Updated Sept 2025)](https://www.americangaming.org/wp-content/uploads/2025/12/Responsible-Gaming-Code-of-Conduct-2026.pdf)
- [AGA Responsible Gaming Regulations and Statutes Guide](https://www.americangaming.org/resources/responsible-gaming-regulations-and-statutes-guide/)
- [FinCrime Central: AML Compliance Guide for Casinos 2025](https://fincrimecentral.com/aml-compliance-guide-casinos-2025-updates/)
- [Congress.gov: SAFE Bet Act S.1033 (119th Congress)](https://www.congress.gov/bill/119th-congress/senate-bill/1033)
- [Congress.gov: SAFE Bet Act H.R.2087 (119th Congress)](https://www.congress.gov/bill/119th-congress/house-bill/2087/text)
- [CPPA: California Finalizes CCPA ADMT Regulations](https://cppa.ca.gov/announcements/2025/20250923.html)
- [Nevada Gaming Control Board Industry Notices](https://www.gaming.nv.gov/)
- [NJ DGE: Responsible Gaming Task Force Final Report](https://www.nj.gov/oag/ge/2025news/RGTF_Final_Report_to_GO_3.31.25.pdf)
- [NVSEP: National Voluntary Self-Exclusion Program](https://www.nvsep.org/)

### Secondary Sources
- [Gambling Insider: 2026 US Gambling Bill Tracker](https://www.gamblinginsider.com/in-depth/112652/us-gambling-bill-tracker)
- [CasinoBeats: US Online Casino Legal Tracker 2026](https://casinobeats.com/features/us-online-casino-gambling-legal-tracker/)
- [Drata: Artificial Intelligence Regulations 2026](https://drata.com/blog/artificial-intelligence-regulations-state-and-federal-ai-laws-2026)
- [BDO: 2026 Is a Pivotal Year for Privacy](https://www.bdo.com/insights/advisory/2026-is-a-pivotal-year-for-privacy)
- [Locance: CES 2026 Insights on AI Gaming Compliance](https://www.locance.com/blog/ces-2026-insights-ai-compliance/)
- [Ballard Spahr: Nevada Online Gaming Compliance Requirements](https://www.ballardspahr.com/insights/alerts-and-articles/2026/01/why-nevada-latest-compliance-requirements-for-online-gaming-matter-far-beyond-state-lines)
- [GamblingHarm.org: AI Legislation Tracker 2025](https://gamblingharm.org/sports-betting-artificial-intelligence-ai-legislation-tracker/)
- [Mayer Brown: CCPA ADMT Requirements](https://www.mayerbrown.com/en/insights/publications/2026/01/updates-to-the-ccpa-regulations-what-businesses-need-to-know-now-about-automated-decision-making-cybersecurity-audits-and-risk-assessments)

---

## Section 1: New State Legalizations (2025-2026)

### Current Landscape
As of March 2026, **seven states** have legalized real-money online casinos: Connecticut, Delaware, Michigan, New Jersey, Pennsylvania, Rhode Island, and West Virginia.

### Maine — First New iGaming State (2025)
- **LD 1164** granted Wabanaki tribes exclusive rights to offer online casino-style gaming
- Launch expected 2026 with operators like DraftKings and Caesars
- **18% tax rate** on gross gaming revenue
- First tribal-exclusive iGaming framework in the US

### Pending Legislation (Active 2026 Bills)

| State | Bill | Status | Key Details |
|-------|------|--------|-------------|
| **New York** | Sen. Addabbo bill | Introduced | Online slots, poker, table games, live dealer. 30.5% tax. Commercial + tribal + licensed sportsbooks |
| **Virginia** | Internet gaming authorization | Requires 2-session passage (2026 + 2027) | Digital slots, table games, poker |
| **Illinois** | Multiple bills | Committee | AI restrictions, deposit limits, microbetting ban |
| **Nebraska** | Various | Active | Unified self-exclusion list enacted (HB 2) |

### Sweepstakes Casino Crackdown
Multiple states enacted bans on dual-currency sweepstakes systems in 2025:
- **California** (effective Jan 2026): Banned dual-currency systems
- **New York** (Dec 2025): Similar ban
- **Connecticut, New Jersey, Nevada, Montana**: Comparable legislation
- **Tennessee**: Cease-and-desist letters to 38 operators
- **Los Angeles**: First governmental civil suit against Stake.us

### Impact on Hey Seven
- **Maine tribal launch** = potential new client market (Wabanaki tribal operations)
- **New York legalization** (if passed) = massive market expansion opportunity
- **Multi-state expansion** increases need for jurisdiction-aware guardrails and state-specific regulatory compliance per property

---

## Section 2: TCPA Enforcement Actions (Casino/Gaming Specific)

### Litigation Surge
- **2,788 TCPA cases** filed in 2024 (up 67% from 2023)
- **57.9% year-to-date increase** through September 2025
- **78% of all TCPA filings** are class actions (vs 5.1% for FDCPA)
- Monthly class action filings hit **172 in January 2025** (268% increase over Jan 2024)

### Key Rule Changes (Effective April 11, 2025)
1. **Consent revocation**: Consumers can revoke consent to be contacted by **any reasonable means** (not just specific opt-out mechanisms)
2. **10 business day deadline**: Businesses must remove a consumer from contact lists within 10 business days after consent revocation
3. **Broader "any reasonable means"**: Previously required specific STOP keyword; now any clear statement of revocation qualifies

### Casino/Gaming Specific Concerns
No casino-specific TCPA enforcement actions were identified in the search period, but the general TCPA litigation surge creates heightened risk for:
- SMS outreach programs (Hey Seven's Telnyx integration)
- Promotional communications via digital channels
- Re-engagement campaigns after player inactivity

### Hey Seven Current State
Hey Seven's `src/sms/compliance.py` already implements:
- STOP/HELP/START keyword processing (English + Spanish)
- Quiet-hours enforcement per guest timezone
- HMAC-SHA256 consent hash chain
- Area-code-to-timezone mapping

### Gaps Identified
1. **"Any reasonable means" revocation**: Current STOP_KEYWORDS is a fixed frozenset. Under the new rule, a message like "please don't text me anymore" or "remove me from your list" would legally constitute consent revocation but would NOT match any keyword in the current set.
2. **10 business day compliance**: No explicit 10-business-day processing deadline is enforced in the current codebase.
3. **Consent revocation audit trail**: The HMAC hash chain tracks consent grants but the revocation path needs the same tamper-evident logging.

---

## Section 3: Self-Exclusion Program Changes

### National Voluntary Self-Exclusion Program (NVSEP) Expansion
The NVSEP, operated by idPair (CEO Jonathan Aiwazian), is creating a **national cross-state self-exclusion system**:

| Phase | Jurisdictions | Timeline |
|-------|---------------|----------|
| Phase 1 | Colorado, Iowa, Michigan, Tennessee, California | 2024 |
| Phase 2 | **Nebraska** (Dec 2025), **Wyoming** (Dec 2025) | Late 2025 |
| Phase 3 | Additional states expected | 2026 |

### Key Features
- Single online registration for multi-state self-exclusion
- Participants choose exclusion duration
- Identity verification + digital signature
- Covers both land-based casinos and online gaming platforms

### State-Level Updates

| State | Change | Details |
|-------|--------|---------|
| **Nebraska** | HB 2 signed into law (June 2025) | Unified single statewide self-exclusion list under State Lottery and Gaming Commission |
| **New Hampshire** | Platform under construction | Expected launch 2026 |
| **New Jersey** | DGE enforcement | Fined Digital Gaming Corporation (Super Group) for self-exclusion failures |
| **Nevada** | Facial recognition for self-excluded | Slot machines to incorporate facial recognition to identify self-excluded players (2026) |
| **Colorado** | Anniversary milestone | One year of successful centralized self-exclusion list |

### States with Centralized Programs
Arizona, California, Colorado, Iowa, Michigan, Nebraska, New Jersey, New Mexico, New York, Tennessee, Wyoming

### Impact on Hey Seven
- **NVSEP API integration**: As the national program expands, Hey Seven should integrate with the NVSEP/idPair API for real-time self-exclusion verification across states
- **Current self-exclusion detection** (`detect_responsible_gaming()` and `self-exclu` pattern) handles text-based detection but does NOT verify against external registries
- **Cross-state operation**: A guest self-excluded in Michigan but visiting Connecticut should be flagged -- requires registry lookup, not just keyword detection

---

## Section 4: BSA/AML Digital Channel Updates

### AGA Best Practices Guide 2025 Update

The American Gaming Association's updated AML compliance guide addresses several areas directly relevant to Hey Seven:

#### Digital Wallets and Online Payments
- Wallets must be **restricted to a single user**
- Withdrawals must **return to the original deposit method**
- Accounts showing **deposit-withdrawal cycles without meaningful wagering** must be flagged
- These rules blur the boundary between casinos and conventional financial institutions

#### Enhanced KYC for Online Onboarding
- Names, addresses, SSNs when applicable, government-issued ID required before BSA-reportable transactions
- For online channels: **non-documentary verification** using third-party databases, photo IDs, or selfies is encouraged
- This applies to any digital channel where transactions may occur

#### New Typology: Human Trafficking
- AGA guide now has a **dedicated section** on human trafficking indicators
- Training must cover this as an emerging AML typology

### Nevada Enforcement Wave
- Nevada regulators issued approximately **$27 million in penalties** against major operators for systemic AML control failures
- Failures included inadequate customer due diligence and source-of-funds verification
- **Nevada Notice #2026-04**: Licensees in out-of-state online gaming must report jurisdictions by March 17, 2026, with enhanced due diligence

### FinCEN Modernization
- Director Andrea Gacki emphasizes **modernizing reporting requirements** to reduce compliance burdens while maintaining risk-based focus
- FinCEN designated **10 Mexican gambling establishments** as primary money laundering concerns linked to cartel operations
- **Stablecoin and digital wallet guidance** expected: AML/CFT obligations for stablecoin issuers, Travel Rule compliance, transaction monitoring

### 2026 Outlook
- Final AML/CFT reform rule anticipated sometime in 2026
- Focus on modernizing reporting while maintaining risk-based approach
- Digital channels increasingly treated as equivalent to in-person transactions for compliance purposes

### Impact on Hey Seven
- Hey Seven's BSA/AML guardrails (`_BSA_AML_PATTERNS`) correctly detect structuring, smurfing, and CTR avoidance queries
- **New pattern needed**: Digital wallet abuse indicators (deposit-withdrawal cycling mentions, "separate wallet" queries)
- **Guest profiling data**: If Hey Seven tracks financial behavior patterns (comp history, spending indicators), this data may have BSA/AML reporting implications

---

## Section 5: AI-Specific Gaming Regulations

### Federal Level

#### SAFE Bet Act (S.1033 / H.R.2087, 119th Congress)
The most significant pending federal legislation for Hey Seven:

**Section 105 would prohibit:**
1. Using AI to **track individual bettor behavior**
2. Creating **personalized offers or promotions targeting a specific bettor**
3. Developing **AI-generated betting products** (including microbets)
4. Limiting customer deposits to **5 per 24-hour period**

**Status**: Introduced March 2025 by Sen. Blumenthal (D-CT) and Rep. Tonko (D-NY). Pending in congressional committees. Prospects unclear. Significant industry opposition from iDEA (iDevelopment & Economic Association).

**CRITICAL for Hey Seven**: If passed, this would directly impact:
- Guest profiling enrichment node (tracks guest preferences across turns)
- Incentive engine (personalized comp offers based on guest profile)
- Behavior tools (CompStrategy, LTV Nudge)
- The entire "relationship builder" paradigm of learning guest preferences

**Risk Assessment**: Low-medium probability of passage in current Congress, but signals regulatory direction. The HOST use case (concierge services, dining recommendations) is distinct from the BETTING use case the bill targets.

#### Executive Order: Federal AI Preemption (Dec 11, 2025)
- President Trump signed "Ensuring a National Policy Framework for Artificial Intelligence"
- Directs AG to establish AI Litigation Task Force to challenge state AI laws deemed unconstitutional
- Tasks administration with developing national AI legislative framework that would **preempt conflicting state rules**
- This creates **regulatory uncertainty**: comply with state laws that may be preempted, or wait?

### State Level

#### Colorado AI Act (SB 24-205) — Effective June 30, 2026
- Requires developers and deployers of **high-risk AI systems** to use reasonable care to prevent **algorithmic discrimination**
- Impact assessments required
- Consumer disclosures mandatory
- **Attorney General has exclusive enforcement authority**
- Implementation date extended from Feb 1 to June 30, 2026

**Applicability to Hey Seven**: If Hey Seven's AI routing (Flash vs Pro model selection), guest profiling, or comp strategy decisions are considered "consequential decisions," the algorithmic discrimination provisions could apply. The bill covers decisions with "significant effects" on consumers.

#### Illinois SB 2398
- Would prohibit sportsbooks from using AI to:
  - Track individual wagering activity
  - Create personalized offers targeting specific bettors
  - Create AI-generated gambling products (microbets)
- Referred to State Assignments Committee
- Mirrors SAFE Bet Act provisions at state level

#### California SB 53 (Transparency in Frontier AI Act) — Signed Sept 29, 2025
- Requires "frontier developers" of large AI models to:
  - Publish risk frameworks
  - Report critical safety incidents
  - Implement whistleblower protections
- **Applicability**: Likely applies to foundation model providers (Google, Anthropic) rather than deployers like Hey Seven, but could affect model availability or require documentation

### AI in Gaming Compliance — Emerging Requirements
From CES 2026 and industry reports:

| Requirement | Description | Timeline |
|-------------|-------------|----------|
| **Explainability** | AI systems must demonstrate WHY a decision was made | Emerging (no hard deadline in US) |
| **Audit trails** | Traceable logs of inputs, decision paths, outputs | 2026+ (MGA framework, US state laws) |
| **Bias monitoring** | Drift, bias, and adversarial resilience testing | 2026+ |
| **Annual risk assessments** | For AI systems used in compliance or player safety | MGA 2026, US states TBD |

### Impact on Hey Seven

**Immediate (low risk)**:
- No US federal AI gaming law is currently in effect
- Colorado AI Act (June 2026) is the closest hard deadline

**Medium-term (monitor closely)**:
- SAFE Bet Act: Distinguish HOST use case from BETTING use case. Hey Seven recommends restaurants, not bets
- State AI bills: Track Illinois SB 2398 and copycat legislation

**Architecture preparation**:
- Hey Seven already has audit logging in `compliance_gate_node` for all guardrail triggers
- LangSmith/Langfuse tracing provides decision audit trails
- Add: explicit documentation of AI decision-making logic (model routing, profiling, comp strategy) for potential regulatory review
- Add: opt-out mechanism for AI-driven personalization (proactive compliance with SAFE Bet Act direction)

---

## Section 6: Privacy & Data Collection Changes

### CCPA/CPRA Expansions (Effective January 1, 2026)

#### Automated Decision-Making Technology (ADMT) Requirements
The California Privacy Protection Agency finalized ADMT regulations effective January 1, 2026:

1. **Pre-use notice**: Businesses must provide notice BEFORE using ADMT for "significant decisions"
2. **Opt-out right**: Consumers can opt out of ADMT decisions (subject to exceptions)
3. **Access right**: Consumers can request information about ADMT logic and how outputs are used
4. **Appeal right**: Consumers can appeal results of ADMT decisions
5. **Risk assessments**: Mandatory for six categories including "automated profiling"

**Compliance deadline**: Businesses using ADMT before Jan 1, 2027 must comply by Jan 1, 2027. New ADMT use after that date must comply immediately.

**Casino relevance**: "Significant decisions" are defined as those affecting financial services, employment, housing, education, or healthcare. Casino comp decisions, tier assignments, and personalized offers MAY qualify depending on interpretation. Automated profiling for marketing purposes is explicitly covered.

#### Cybersecurity Audit Requirements
- Mandatory cybersecurity audits for qualifying businesses
- Risk assessments for data processing activities
- Enhanced data broker obligations (DELETE Act compliance)

### Multi-State Privacy Landscape

**19 states now have comprehensive consumer privacy laws in effect** as of January 1, 2026.

New laws effective January 1, 2026:
- **Indiana**: Data protection impact assessments, opt-out for targeted advertising
- **Kentucky**: Similar to Indiana framework
- **Rhode Island**: Applies to entities processing data of 35,000+ residents

**Key principle**: Data collected on a California customer at a Connecticut casino must meet CCPA requirements. Data on a European traveler must meet GDPR standards. **Jurisdiction follows the consumer, not the property.**

### State Enforcement Trends
- State attorneys general are increasingly active in privacy enforcement
- Focus shifting from new legislation to **enforcing existing laws**
- Cross-state data portability requests increasing

### Casino-Specific Privacy Considerations

| Data Type | Regulatory Concern | Hey Seven Relevance |
|-----------|-------------------|---------------------|
| Guest name, party size, occasion | PII under all state laws | Extracted by profiling enrichment node |
| Gaming preferences, spend indicators | Sensitive data under some state laws | Used by comp strategy + incentive engine |
| Conversation history | Contains PII + behavioral data | Stored via LangGraph checkpointer |
| SMS phone numbers | TCPA + CCPA coverage | Telnyx integration |
| Guest location (timezone) | Geolocation data under CCPA | Area-code timezone mapping |

### Impact on Hey Seven

**CRITICAL for guest profiling**: Hey Seven's profiling enrichment node (`src/agent/profiling.py`) extracts guest information across conversation turns:
- Guest name, party size, occasion, food preferences, hometown
- This constitutes **automated profiling** under CCPA

**Required changes**:
1. **ADMT disclosure**: System must be able to explain what data it collects and why
2. **Opt-out mechanism**: Guests must be able to opt out of profiling
3. **Data minimization**: Only collect what's needed for the current interaction
4. **Retention policy**: Define how long conversation/profile data is retained
5. **Cross-state compliance**: Property state != guest home state; apply strictest applicable law

---

## Section 7: Responsible Gaming Technology Requirements

### NJ DGE Mandatory Responsible Gaming Protocols (Proposal PRN 2025-130)

The most significant development for casino digital channels:

#### Three-Tier Intervention System
| Phase | Trigger | Action |
|-------|---------|--------|
| **Phase 1** | Account flags risky behavior | Contact player with RG info, highlight deposit limits, time restrictions, self-exclusion tools |
| **Phase 2** | Continued concerning behavior | Block wagering until player views state-approved responsible gaming video tutorial |
| **Phase 3** | Persistent concern | Direct outreach from licensed RG Lead, offer access to paid RG professional |

#### Trigger Thresholds
- Deposits **over $10,000 in a single day**
- Deposits **over $100,000 in three months**
- **Withdrawal reversal patterns** (changing mind about cashing out)
- High-frequency late-night sessions

#### Additional Requirements
- Every operator must designate a **Responsible Gaming Lead**
- **3-day cooling period**: Ban operators from soliciting players to reverse withdrawal requests for at least 3 days
- Applies to all licensed online gaming operators in NJ

**Status**: Public comments closed November 2025. Adoption expected mid-2026.

### AGA Responsible Gaming Code of Conduct (Updated September 2025)
- Covers operational conduct + advertising/marketing across all gaming forms
- Governance provisions with chairs having relevant RG/regulatory background
- Complaint process with CCRB Liaison
- All AGA members engaged in US gaming must adhere

### Nevada Responsible Gaming Technology
- **Facial recognition in slot machines**: Starting 2026, Nevada slots will incorporate facial recognition to identify self-excluded players and underage individuals
- **On-screen responsible gaming tools**: First-of-its-kind program integrating directly into slot machines:
  - Budget tools
  - Time limit settings
  - Direct access to responsible gambling resources
- **Surveillance Standards update** (effective March 15, 2026): Updated digital recording standards, equipment, staffing

### Michigan MGCB
- "Don't Regret the Bet" campaign with high school booster clubs (March 2026)
- Partnering education with enforcement

### Industry Technology Trends (CES 2026)
- AI transitioning from optional to **mandatory infrastructure** for:
  - Identity verification
  - AML/fraud detection
  - Responsible gaming monitoring
  - Geolocation anomaly detection
- Regulators will soon **require AI**, not merely permit it
- Drift, bias, and adversarial resilience will become **compliance obligations**
- Annual AI risk assessments emerging as standard

### FanDuel "Play With A Plan"
- Major operator RG campaign focusing on informed decision-making
- Industry trend toward proactive responsible gaming communication

### Impact on Hey Seven

**Hey Seven's current RG infrastructure is strong**:
- 5-layer guardrail system with 214 compiled patterns across 11 languages
- 4-level crisis escalation (none, concern, urgent, immediate)
- Self-harm detection with 988 Lifeline routing
- Responsible gaming keyword detection (English, Spanish, Portuguese, Mandarin, French, Vietnamese, Hindi, Tagalog, Japanese, Korean)

**Gaps identified**:
1. **Threshold-based monitoring**: NJ's trigger thresholds ($10K/day, $100K/3mo) require financial transaction awareness. Hey Seven currently does keyword detection only -- it does not track guest financial behavior across turns or sessions.
2. **Withdrawal reversal detection**: No pattern for "I want to reverse my withdrawal" or "cancel my cashout"
3. **Proactive RG intervention**: Current system is reactive (detects RG keywords). NJ requires proactive monitoring and intervention.
4. **RG Lead routing**: Phase 3 requires routing to a human RG professional. Hey Seven's handoff system routes to host team, not specifically RG staff.
5. **Facial recognition integration**: Not applicable to digital channel (text-based), but signals regulatory direction toward identity-linked responsible gaming.

---

## Section 8: Impact on Hey Seven — Required Code Changes

### Priority 1: CRITICAL (Legal compliance risk)

#### 8.1 TCPA "Any Reasonable Means" Revocation
**File**: `src/sms/compliance.py`
**Current**: Fixed `STOP_KEYWORDS` frozenset (10 keywords)
**Required**: Add NLP-based intent detection for consent revocation. Messages like "please stop texting me", "take me off your list", "don't contact me anymore" must be treated as valid STOP requests under the April 2025 TCPA rule change.
**Approach**: Add a `detect_revocation_intent()` function with regex patterns for common revocation phrases, failing safe (if in doubt, treat as STOP).

#### 8.2 CCPA ADMT Opt-Out for Guest Profiling
**Files**: `src/agent/profiling.py`, `src/agent/extraction.py`, `src/api/app.py`
**Current**: No opt-out mechanism for automated profiling
**Required**: Add a `profiling_opt_out` flag per guest session/profile. When set, the profiling enrichment node must skip extraction entirely. Expose via API endpoint. Must be implementable by January 1, 2027 (compliance deadline for existing ADMT use).

#### 8.3 Data Retention Policy
**Files**: `src/agent/memory.py`, `src/data/guest_profile.py`
**Current**: No explicit retention policy. Firestore checkpointer retains conversation history indefinitely.
**Required**: Define and implement data retention periods. CCPA gives consumers the right to deletion. Must support "delete my data" requests per guest.

### Priority 2: HIGH (Regulatory compliance, near-term deadlines)

#### 8.4 NVSEP Integration Readiness
**Files**: New file `src/agent/self_exclusion.py` or extend `src/agent/guardrails.py`
**Current**: Self-exclusion detected via keyword regex (`self-exclu` pattern). No registry verification.
**Required**: Architecture for external self-exclusion registry lookup (NVSEP/idPair API). Initially can be a stub with feature flag, but the interface must be designed for per-property registry configuration.
**Timeline**: As NVSEP expands to more states through 2026

#### 8.5 NJ Responsible Gaming Threshold Patterns
**File**: `src/agent/guardrails.py`
**Current**: No patterns for financial threshold language
**Required**: Add patterns for:
- Withdrawal reversal requests: `r"\b(?:reverse|cancel|undo)\s+(?:my\s+)?(?:withdrawal|cashout|cash[- ]out)\b"`
- Deposit limit awareness: `r"\bdeposit\s+limit\b"`, `r"\bspending\s+limit\b"`
- Session time limit requests: `r"\b(?:time|session)\s+limit\b"`
- These should route to a responsible gaming specialist response, not the generic off_topic handler

#### 8.6 BSA/AML Digital Wallet Patterns
**File**: `src/agent/guardrails.py`
**Current**: BSA/AML patterns cover structuring, smurfing, CTR avoidance
**Required**: Add patterns for:
- Digital wallet abuse: `r"\b(?:multiple|separate|different)\s+(?:wallet|account)s?\s+(?:to|for)\s+(?:deposit|cash|move)\b"`
- Deposit-withdrawal cycling: `r"\bdeposit\s+(?:and|then)\s+(?:immediately\s+)?withdraw\b"`
- Cryptocurrency structuring: `r"\b(?:crypto|bitcoin|ethereum)\s+(?:to|for)\s+(?:avoid|bypass|hide)\b"`

### Priority 3: MEDIUM (Proactive compliance, competitive advantage)

#### 8.7 AI Decision Audit Trail Enhancement
**Files**: `src/agent/nodes.py`, `src/agent/profiling.py`, `src/agent/incentives.py`
**Current**: LangSmith/Langfuse tracing provides basic audit trail
**Required**: Structured audit log entries for every AI decision that affects guest experience:
- Model routing decision (Flash vs Pro) + reason
- Profiling extraction decisions + confidence scores
- Comp strategy recommendations + policy basis
- Handoff decisions + triggering factors
This prepares for Colorado AI Act (June 2026) and potential federal AI gaming requirements.

#### 8.8 SAFE Bet Act Compliance Architecture
**Files**: Architecture-level, affects `src/agent/profiling.py`, `src/agent/incentives.py`, `src/agent/behavior_tools/`
**Current**: Profiling and personalization are core to the product
**Required**: Design a feature flag architecture that can disable personalization features per jurisdiction if SAFE Bet Act or state equivalents pass. The HOST use case (concierge, dining, entertainment) should be architecturally separable from the GAMING use case (comp offers, betting engagement, personalized promotions).
**Note**: This is architecture preparation, not immediate implementation. The bill's prospects are unclear, but the direction is clear.

#### 8.9 Cross-State Jurisdiction-Aware Compliance
**Files**: `src/casino/config.py`, `src/agent/compliance_gate.py`
**Current**: `PROPERTY_STATE` is a single config value per property
**Required**: When operating in multiple states, compliance rules must consider BOTH:
1. The property's state (where the casino is located)
2. The guest's home state (where privacy and RG laws follow the consumer)
This requires extending the per-property config to include guest-state-aware rule overrides.

#### 8.10 Responsible Gaming Documentation Compliance
**Files**: `docs/adr/`, `docs/responsible-gaming.md` (new)
**Current**: ADRs document technical decisions but not RG policy
**Required**: Create a Responsible Gaming Policy document that can be provided to regulators, covering:
- How the AI identifies at-risk players (guardrail layers)
- How crisis escalation works (4 levels)
- Self-exclusion handling procedures
- Data used for player assessment
- Human oversight mechanisms (host handoff)
This addresses the emerging requirement for AI systems to document decision-making processes.

### Priority 4: LOW (Future-proofing)

#### 8.11 Facial Recognition Integration Interface
Not immediately needed (Hey Seven is text-based), but Nevada's move to facial recognition in slot machines signals a trend toward identity-linked responsible gaming across ALL channels. Design the guest identity verification interface to accommodate future biometric integration.

#### 8.12 Multi-Language RG Resource Updates
**File**: `src/agent/guardrails.py`, response templates
**Current**: 11 languages covered for RG detection
**Required**: As new states legalize (Maine tribal, potentially New York), review language coverage for dominant patron demographics. Maine Wabanaki operations may need specific cultural sensitivity in crisis responses.

---

## Key Takeaways

### Regulatory Direction (Clear Trends)

1. **Responsible gaming is becoming MANDATORY, not voluntary** -- NJ's DGE proposal is the leading edge. Other states will follow. Hey Seven's strong RG infrastructure is a competitive advantage, but needs enhancement for threshold-based monitoring.

2. **AI in gaming is accelerating toward regulation** -- The SAFE Bet Act, Illinois SB 2398, Colorado AI Act, and Trump's federal preemption order all signal that AI gaming regulation is coming. The question is whether it comes from states or federal government.

3. **Privacy laws follow the CONSUMER, not the property** -- 19 states now have comprehensive privacy laws. A California resident visiting a Connecticut casino is protected by CCPA. Hey Seven must implement jurisdiction-aware privacy compliance.

4. **Self-exclusion is going national** -- NVSEP is building cross-state infrastructure. Text-based keyword detection is insufficient; registry integration is the future standard.

5. **AML compliance is expanding to digital channels** -- Digital wallets, cryptocurrency, and online payments are being brought under the same BSA/AML framework as cash transactions. Hey Seven's BSA/AML guardrails need digital-channel-specific patterns.

### Competitive Advantage Opportunities

1. **First-mover on ADMT compliance** -- Building opt-out and audit trail infrastructure now puts Hey Seven ahead of the January 2027 CCPA deadline.

2. **RG technology leadership** -- Hey Seven's 5-layer guardrail system with 214 patterns across 11 languages is already industry-leading. Adding threshold-based monitoring and proactive intervention would be best-in-class.

3. **AI governance documentation** -- Creating transparent AI decision documentation now positions Hey Seven as a responsible AI operator in a market that will soon require it.

4. **NVSEP integration** -- Being among the first AI casino hosts to integrate with the national self-exclusion registry is a strong differentiator for casino clients.

### Timeline Summary

| Deadline | Regulation | Action Required |
|----------|-----------|-----------------|
| **Already effective** | TCPA "any reasonable means" (April 2025) | Update SMS consent revocation detection |
| **Already effective** | CCPA ADMT pre-use notice (Jan 2026) | Implement profiling disclosure |
| **June 30, 2026** | Colorado AI Act | Prepare AI impact assessments |
| **Mid-2026 (expected)** | NJ DGE mandatory RG protocols | Implement threshold monitoring |
| **Jan 1, 2027** | CCPA ADMT opt-out compliance | Implement profiling opt-out |
| **Pending** | SAFE Bet Act (federal) | Monitor; design separable architecture |
| **Ongoing** | NVSEP expansion | Plan registry integration |
| **2026+** | Nevada surveillance standards | Monitor for digital channel extensions |