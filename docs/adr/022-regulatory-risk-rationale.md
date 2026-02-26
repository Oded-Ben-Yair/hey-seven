# ADR-022: Regulatory Risk Rationale for Guardrail Categories

## Status
Accepted (2026-02-26)

## Context
Hey Seven operates in a regulated US casino environment where AI-generated responses carry
legal liability. Each guardrail category maps to specific federal and state regulations.
This ADR documents the legal basis for each guardrail and the risk of not enforcing it.

## Decision

### Guardrail-to-Regulation Mapping

| Guardrail Category | Regulation | Jurisdiction | Risk if Missing |
|-------------------|------------|--------------|-----------------|
| Prompt Injection | N/A (security) | All | Complete system compromise; attacker controls AI output |
| Responsible Gaming | NGC Reg. 5.170 (NV), CT SB 2, PA Act 71 | State-specific | Regulatory fines; license revocation risk |
| Age Verification | 21+ gaming age (all US states) | Federal + State | License suspension; underage gambling liability |
| BSA/AML | Bank Secrecy Act (31 USC 5311), FinCEN regs | Federal | Federal fines up to $1M/violation; criminal liability |
| Patron Privacy | State privacy laws, CCPA (CA), tribal law | State-specific | Lawsuits; regulatory action; trust erosion |
| Self-Harm/Crisis | Duty of care, premises liability | State-specific | Wrongful death liability; media/reputation damage |

### Enforcement Precedent

| Date | Entity | Violation | Penalty | Relevance |
|------|--------|-----------|---------|-----------|
| 2023 | BetMGM (NJ) | Self-exclusion violations | $260K fine | DGE enforcement of exclusion list compliance |
| 2022 | Valley Forge Casino (PA) | AML reporting failures | $30K fine | PGCB enforcement of BSA/AML obligations |
| 2023 | Wynn Resorts (NV) | Inadequate patron dispute resolution | $5.5M settlement | NGC enforcement of patron protection standards |
| 2024 | Multiple NJ casinos | CCPA compliance failures | $50K-$200K | AG enforcement of patron data privacy rights |

### Liability Exposure Analysis

1. **BSA/AML**: Highest risk. Federal penalties, no state cap. AI system that fails to redirect
   AML-related queries to compliance team could be construed as facilitating money laundering.
2. **Responsible Gaming**: High risk. State-specific fines + license revocation. AI that provides
   gambling advice or fails to surface helplines violates state gaming commission requirements.
3. **Self-Harm**: High risk (reputational). No specific gaming regulation, but premises liability
   and duty-of-care standards apply. AI that ignores crisis language creates wrongful death exposure.
4. **Age Verification**: Medium-high risk. License suspension risk. AI must not facilitate underage
   access to gaming information.
5. **Patron Privacy**: Medium risk. State privacy laws vary. AI sharing guest information violates
   privacy policies and potentially CCPA.
6. **Prompt Injection**: Security risk (not regulatory). But successful injection could compromise
   ANY of the above categories, making it the highest-priority technical guardrail.

## Consequences
- Every guardrail has a documented legal basis and enforcement precedent
- Regulatory counsel can review and approve the mapping
- New jurisdictions require adding their specific regulations to this mapping
- Guardrail bypass bugs are classified by regulatory risk severity

## Review
Last reviewed: 2026-02-26
Next review: Before any new jurisdiction onboarding
