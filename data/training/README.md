# Hey Seven Fine-Tuning Training Data

Generated: 2026-03-09T06:05:28Z
Min score threshold: 7.0

## Files

| File | Count | Source |
|------|-------|--------|
| `gold-traces-r106.jsonl` | 47 | R103-R105 eval results (judge score >= 7.0) |
| `gold-traces-manual.jsonl` | 4 | Manually crafted gold traces (score >= 6/10) |

## Format

Vertex AI supervised fine-tuning JSONL format:
- `systemInstruction.role`: `"system"`
- `contents[].role`: `"user"` or `"model"`
- `contents[].parts[].text`: message text
- Last message is always `"model"` role

**NOT OpenAI messages format.** Uses `"model"` not `"assistant"`.

## Target Model

**Gemini 2.5 Flash** (via Vertex AI). Gemini 3.x does not support fine-tuning as of March 2026.

### Deprecation Warning (verified 2026-03-09 via google-developer-knowledge MCP)

- `gemini-2.5-flash` GA shuts down **June 17, 2026** (~3 months)
- `gemini-2.0-flash` shuts down **June 1, 2026** — do NOT use
- `gemini-3-pro-preview` shut down **March 9, 2026** (replaced by `gemini-3.1-pro-preview`)
- Gemini 3.x models are all **preview** — no GA, no fine-tuning support

**Strategy**: Tune 2.5 Flash now for immediate gains. When 3.x GA launches with tuning, re-tune with same JSONL data (same format expected).

## Total

**51 training conversations**

## Usage

```bash
# Upload to GCS
gsutil cp data/training/*.jsonl gs://hey-seven-training/

# Start tuning job
python -c "
from vertexai.tuning import sft
job = sft.train(
    source_model='gemini-2.5-flash',
    train_dataset='gs://hey-seven-training/gold-traces-r106.jsonl',
    epochs=4, adapter_size=4,
    tuned_model_display_name='hey-seven-host-v1',
)
"
```
