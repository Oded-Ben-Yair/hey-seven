# ADR-018: Confusable Homoglyph Coverage (145 entries)

## Status
Accepted (bounded scope)

## Context
Unicode TR39 defines thousands of confusable character pairs. Our guardrail normalization maps 145 highest-risk confusables to Latin equivalents.

## Decision
Cover the 8 most common attack scripts only:
1. **Cyrillic** (23 entries) — highest risk, most common bypass
2. **Greek** (18 entries) — mathematical/academic contexts
3. **Fullwidth Latin** (52 entries) — CJK contexts (lowercase + uppercase)
4. **Armenian** (16 entries) — visual Latin lookalikes
5. **Cherokee** (10 entries) — visual Latin lookalikes
6. **IPA/Latin Extended** (7 entries) — survive NFKD
7. **Mathematical/Letterlike symbols** (9 entries) — partial differential, script l, Weierstrass p, etc.
8. **Georgian Mkhedruli** (10 entries) — R69 fix D7: visual Latin lookalikes (ა≈a, ო≈o, ე≈e, etc.)

## Rationale
- Full TR39 mapping adds ~8K entries and ~50KB to memory. For 204 regex patterns, the marginal security benefit diminishes rapidly after the top scripts.
- NFKD normalization already handles many confusables (e.g., fi -> fi).
- The confusable table is a defense-in-depth layer, not the primary defense. Regex patterns match normalized Latin text.
- Attack sophistication: adversaries using Georgian or Bengali confusables are rare in casino guest interactions.

## Consequences
- Positive: O(n) single-pass replacement via str.maketrans (C-speed)
- Positive: Covers the 7 most likely attack scripts
- Negative: Exotic scripts (Georgian, Bengali, extended Cyrillic) not covered
- **Upgrade path**: Replace with `confusable_homoglyphs` library if attack surface expands

## Last Reviewed
2026-02-25
