# Spanish Casino Conventions — Research for Phase 1 Multilingual

**Date**: 2026-02-28
**Source**: Perplexity research (6 queries, primary regulatory sources verified)
**Confidence**: High

## 1. US Casino Spanish Dialect Conventions

- **Register**: Neutral Latin American Spanish ("espanol neutro"), NOT Castilian/vosotros
- **Formality**: Default to **usted** in casino hospitality contexts (professional service register). Mirror **tu** only if the guest uses it first.
- **Key pattern**: English loan words retained for prestige terms (jackpot, spin, comp, high roller). Spanish for standard game mechanics (apostar, fichas, plantarse). Code-switching verbs common (spinear, cashear, blofear).
- **Avoid**: "tragaperras" (Spain-only term for slots — use "las maquinas" or "tragamonedas" in US context)
- **Property names**: Always in English (Mohegan Sun, Hard Rock, Wynn)

## 2. Spanish Responsible Gaming Helplines by State

| State | Primary Helpline | Spanish Access | Notes |
|-------|-----------------|----------------|-------|
| National | 1-800-GAMBLER (1-800-426-2537) | Language Line Solutions (160+ languages) | Professional interpreter, not native staff |
| National | 1-800-522-4700 (NCPG) | Language Line Solutions (240+ languages) | Being phased into 1-800-GAMBLER branding |
| CT | 1-888-789-7777 (CCPG) | Spanish resources page + interpreter | ccpg.org/get-help/resourcesinspanish/ |
| NV | 1-800-GAMBLER | Via interpreter services | Same national line |
| PA | 1-800-848-1880 | Via interpreter services | PA state-specific |
| NJ | 1-800-GAMBLER | Via interpreter services | NJ DGE primary |

**Note**: CT's CCPG has the most robust direct Spanish resources including downloadable Spanish-language brochures, warning signs docs, and self-exclusion guides.

## 3. 988 Suicide & Crisis Lifeline — Spanish Support

- **Call 988**: Press **2** for Spanish-speaking counselors (direct, NOT interpreters)
- **Text 988**: Text **"AYUDA"** to 988 for Spanish text support
- **Legacy Spanish line**: **1-888-628-9454** (still active, connects directly to Spanish-speaking counselors)
- **988 Spanish Subnetwork**: ~67,000 calls in 2022, 50% growth since launch
- **Chat**: Available in Spanish at suicidepreventionlifeline.org

**CRITICAL**: Use "AYUDA" not "HOLA" for 988 text. "HOLA" is for general Crisis Text Line.

## 4. Crisis Text Line — Spanish Support

- **Text "HOLA" to 741741** for Spanish crisis text support
- **WhatsApp**: Text "442-AYUDAME" on WhatsApp (alternative for those without SMS)
- Staffed by hundreds of bilingual volunteer counselors
- First-of-its-kind Spanish crisis text service in the US

**Note**: Research found conflicting info — "HOLA" for 741741, "AYUDA" for 988. Use both correctly per service.

## 5. TCPA/Regulatory — Multilingual AI Disclosure

- **No federal law** explicitly requires AI disclosure in Spanish
- **NJ Bot Disclosure Law (NJ Rev Stat 56:18-2)**: Requires "clear and conspicuous" disclosure when bot communicates with NJ residents for commercial purposes. Logically requires Spanish disclosure when communicating in Spanish. Penalties: $2,500 first offense, $5,000 second, $10,000 subsequent.
- **CT SB 2**: Did NOT pass (Gov. Lamont veto threat). CT does not currently have an AI disclosure law.
- **FCC NPRM (Aug 2024)**: Proposed AI-specific disclosure rules, status unclear (may be finalized by now)
- **Recommendation**: Include AI disclosure in Spanish for all states as best practice. NJ requires it legally.

## 6. Spanish Gambling Slang Corpus (Key Terms)

### Game Terms
| English | US Casino Spanish | Notes |
|---------|-------------------|-------|
| Slot machine | las maquinas, tragamonedas | NOT "tragaperras" (Spain only) |
| Blackjack/21 | blackjack, veintiuno | English term dominant |
| Table games | juegos de mesa | Standard |
| Poker | poker | English loan word |
| Craps | dados, craps | Both used |
| Roulette | ruleta | Standard |

### Action Terms
| English | US Casino Spanish |
|---------|-------------------|
| To bet | apostar |
| To win | ganar |
| To lose | perder |
| To spin | spinear (code-switch), girar |
| To cash out | cashear, cobrar |
| To bluff | blofear, farolear |
| Chips | fichas |
| Jackpot | jackpot (English loan) |
| High roller | high roller, apostador fuerte |
| Comp | comp (English loan), cortesia |

### Emotional/Gambling States
| English | US Casino Spanish |
|---------|-------------------|
| On a hot streak | estar en racha |
| On tilt | estar en tilt, estar frustrado |
| Lost everything | perdi todo, lo perdi todo |
| Need to win it back | necesito recuperarlo |
| Lucky | tener suerte, con suerte |
| Unlucky | mala suerte, sin suerte |

## Implementation Notes

1. **Usted vs Tu**: The plan specified tu (informal), but research suggests usted as default in US casino hospitality. **Decision needed** — but since Hey Seven targets a younger, digital-first audience, tu may be appropriate. Use usted for crisis/compliance, tu for casual conversation.
2. **988 keywords**: "AYUDA" for 988 text, "HOLA" for Crisis Text Line 741741
3. **AI disclosure in Spanish**: Add to all casino profiles for NJ compliance (other states as best practice)
4. **Slang normalization**: Add Spanish gambling slang to `src/agent/slang.py` for RAG search normalization (Phase 2 scope)
