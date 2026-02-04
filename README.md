# Legal CPT Data Pipeline

Fully automated pipeline to download, clean, chunk and balance legal text for **continued pre-training** (CPT) of language models.

**Target**: 50% Dutch · 25% UK English · 25% US English  
**Output**: Single JSONL file with `{"text": "..."}` records, ready for training.

---

## Quick Start

```bash
chmod +x run.sh
./run.sh              # full run: ~20M tokens
./run.sh --test       # API smoke test (tiny download)
```

Or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python legal_cpt_pipeline.py
```

## Sources & Licences

| Source | Language | Content | Licence |
|---|---|---|---|
| Rechtspraak.nl | Dutch | Court decisions (800k+ available) | Public domain |
| wetten.overheid.nl (BWB) | Dutch | Legislation (45k+ regulations) | CC-0 |
| legislation.gov.uk | UK English | Acts of Parliament | Open Government Licence v3 |
| eCFR | US English | Code of Federal Regulations | Public domain (US gov work) |

**All sources are unrestricted for commercial use.**

## Output Structure

```
legal_cpt_data/
├── rechtspraak/          # raw Dutch court decisions
├── wetten_nl/            # raw Dutch legislation
├── uk_legislation/       # raw UK Acts
├── ecfr/                 # raw US federal regulations
├── progress.json         # resume checkpoint
├── legal_cpt_train.jsonl       # ← TRAINING FILE (text only)
└── legal_cpt_train.meta.jsonl  # same + metadata (for debugging)
```

### Training file format

```jsonl
{"text": "De rechtbank overweegt als volgt. Artikel 6:162 BW vereist..."}
{"text": "The Secretary of State may by regulations make provision..."}
{"text": "Each Federal banking agency shall prescribe regulations..."}
```

### Metadata file format (for reference only, not for training)

```jsonl
{"text": "...", "meta": {"lang": "nl", "source": "rechtspraak", "tokens": 1024}}
```

## CLI Options

```
python legal_cpt_pipeline.py [OPTIONS]

  --target-tokens N    Total target tokens (default: 20,000,000)
  --output-dir DIR     Output directory (default: ./legal_cpt_data)
  --test               Quick test: minimal download per source
  --verbose / -v       Debug logging
```

## How It Works

1. **Download**: Fetches legal text from each source via their public APIs
2. **Clean**: Strips boilerplate, normalises whitespace, removes page numbers
3. **Chunk**: Splits into 512–2048 token segments on paragraph boundaries
4. **Balance**: Samples to hit 50/25/25 language ratio
5. **Export**: Writes shuffled JSONL

The pipeline is **resume-capable**: `progress.json` tracks what's been downloaded.  
Re-run safely after any interruption.

## Extending the Pipeline

### Add more Dutch laws
Edit `CURATED_BWB` in the script — add any `BWBR` identifier from wetten.overheid.nl.

### Add more UK Acts
Edit `UK_ACTS` — add tuples of `(type, year, number)`. Find these in legislation.gov.uk URLs.

### Add more US CFR titles
Edit `ECFR_PRIORITY_TITLES` — there are 50 titles total (1–50).

## Rate Limits

The pipeline respects all source APIs:
- Rechtspraak.nl: < 10 req/s (we use ~6/s)
- wetten.overheid.nl: 1 req/s
- EUR-Lex: 1 req/s
- legislation.gov.uk: ~1 req/s
- eCFR: ~0.5 req/s (large downloads)

## Notes for CPT Training

- The output is **plain text only** — no instructions, no chat formatting
- Chunk size (1024 target tokens) is tunable via constants at top of script
- For Qwen2.5-14B QLoRA CPT: use `max_seq_length=2048`, `lr=1e-4`, pack sequences
- The metadata JSONL is useful for debugging but should NOT be fed to training
- Consider running deduplication if combining with other corpora

## Troubleshooting

| Issue | Solution |
|---|---|
| `ConnectionError` on Rechtspraak | Their API has occasional downtime; re-run (pipeline resumes) |
| Empty wetten.overheid.nl pages | Some BWB IDs may be outdated; the script skips them gracefully |
| eCFR timeout | Large titles (7, 40, 42) can be 100MB+; increase timeout in config |
| Low token count for a source | The pipeline downloads extra (1.3× headroom) to ensure enough |
