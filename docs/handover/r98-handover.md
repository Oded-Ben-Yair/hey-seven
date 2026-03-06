# R98 Handover — Phase 2 Behavior Tools

**Date**: 2026-03-06
**Commit**: e977963 (feat: R98 — Phase 2 behavior tools)
**Session health**: 95/100

## What Was Done

### Phase 2 Tools (4 deterministic behavior tools, ~1100 LOC total)

| Tool | File | LOC | Tests | Wired To | Target Dim |
|------|------|-----|-------|----------|------------|
| CompStrategy | `behavior_tools/comp_strategy.py` | ~300 | 25+ | comp_agent via _base.py | H9: 1.9→5.0+ |
| HandoffOrchestrator | `behavior_tools/handoff.py` | ~250 | 25+ | _base.py + nodes.py | P9: 2.1→5.0+ |
| LTV Nudge Engine | `behavior_tools/ltv_nudge.py` | ~250 | 25+ | all specialists via _base.py | H10: 3.5→5.0+ |
| Rapport Ladder | `behavior_tools/rapport_ladder.py` | ~260 | 25+ | all specialists via _base.py | H6: 4.0→5.0+ |

All tools follow the `incentives.py` pattern: Pydantic models, MappingProxyType immutable data, string.Template, pure deterministic business logic (no LLM calls, no I/O).

### Additional Fixes
- `model_routing_enabled` added to all 5 casino profiles (was only in DEFAULT_CONFIG)
- CLAUDE.md updated with behavior_tools directory and GCP doc references
- MEMORY.md updated with GCP findings from google-developer-knowledge MCP
- GPT-5.3-chat documented as available in Azure AI Foundry

### Test Results
- 119 new deterministic tests (4 test files), 0 failures
- 1132 core + integration tests passed, 0 failures
- Package naming: `behavior_tools/` (not `tools/`) to avoid shadowing `src/agent/tools.py`

### Baseline Judge Score (v2-results, pre-tools)
- B-avg: 6.56 (80 scenarios, GPT-5.2 judge)
- Safety: 91.1% (72/79)

## What's Running (background)
- `r98-host-triangle` eval: 30 scenarios through live agent WITH tools wired
- `r98-behavioral` eval: 80 scenarios through live agent WITH tools wired
- Logs: `tests/evaluation/r98-host-triangle-eval.log`, `tests/evaluation/r98-behavioral-eval.log`
- PID check: `ps aux | grep run_live_eval`

## Next Steps (R99)

### If evals completed:
1. Judge the new eval results:
```bash
export AZURE_AI_ENDPOINT=$(az keyvault secret show --vault-name kv-seekapa-apps --name AzureAIFoundry-Endpoint -o tsv --query value)
export AZURE_AI_KEY=$(az keyvault secret show --vault-name kv-seekapa-apps --name AzureAIFoundry-ApiKey -o tsv --query value)
python3 tests/evaluation/run_r95_judge.py --results-dir tests/evaluation/v2-results --category host-triangle
python3 tests/evaluation/run_r95_judge.py --results-dir tests/evaluation/v2-results --category behavioral
```
2. Compare H6, H9, H10 scores against baseline
3. If targets met (H6≥5.0, H9≥5.0, H10≥5.0), tools validated

### If evals not completed:
1. Check logs: `tail -20 tests/evaluation/r98-*-eval.log`
2. Resume if needed: re-run the eval commands from handover

### Phase 3 prep:
1. Test GPT-5.3-chat as judge: run 10 scenarios, compare scores vs GPT-5.2
2. If reliable, swap in judge script
3. Begin fine-tuning data collection spec for Vertex AI 2.5 Flash

## GCP Doc Findings (preserve for future sessions)
- Gemini 3 Pro Preview DEPRECATED (March 9) — our 3.1 Pro is SAFE
- Gemini 3.1 Pro: $2/$12 per 1M tokens, no schema state limit
- Vertex AI fine-tuning: 2.5 Flash/Pro only (NO 3.x)
- Preference tuning available for 2.5 Flash (DPO pairs)
- Cloud Run + Redis: Direct VPC egress recommended over VPC connector
