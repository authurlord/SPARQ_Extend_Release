# SPARQ-Extend — Data Hand-off (for code release)

Two data blobs are **not in git** (too large). They live on the PI's Quark netdisk
under `/quark/SPARQ_Extend/data/`. This doc says what they are, how the code consumes
them, and exactly where to put them after download.

| Blob (on Quark `/quark/SPARQ_Extend/data/`) | Size (bytes) | What it is |
|---|---:|---|
| `ottqa_all_passages/all_passages.json` | 3,385,460,255 (3.2 GB) | OTT-QA official Wikipedia passage corpus — **5,963,520** passages |
| `data_misc.tar.gz` | 237,772,271 (227 MB) | Everything else under `data/` (HybridQA raw + IBM TableIR + reranker training sets) |

Both were uploaded byte-for-byte verified (Quark size == local size).

---

## 1. `all_passages.json` — the 5.96M OTT-QA open-domain passage pool

**Format.** A single JSON **array** of 5,963,520 objects:

```json
[
  {"chunk_id": "Anarchism", "title": "Anarchism", "text": "Anarchism is a radical political movement ..."},
  ...
]
```

- `chunk_id` == `title` == the Wikipedia article title (this is the join key).
- This is the open-domain leg: table cell hyperlinks (`/wiki/<Title>`) are resolved to
  a passage by matching `url_to_title(url) == chunk_id`.

**How the code uses it — STREAM, never `json.load`.** The file is 3.2 GB; loading it whole
will OOM. The one consumer streams it with `ijson`:

- `scripts/export_sparqx_evidence_for_helios.py` (`--passages`)
  ```python
  import ijson
  with open(passages_path) as f:
      for obj in ijson.items(f, "item"):     # streaming, O(1) memory
          if obj["chunk_id"] in needed_titles:
              pool[obj["chunk_id"]] = obj["text"].strip()
  ```
  It first collects the set of cell-link titles needed by the retrieved top-1 tables, then
  makes a single streaming pass to pull only those passages. Reproduces the exact evidence
  behind the full-dev SPARQ-X result `analysis/ottqa_full2214/e2e_5m_reranker_top1`
  (EM 0.6762 / F1 0.7315).

**⚠️ Path gotcha.** The script's default is `data/ottqa_5m/all_passages.json`, but the file
ships at `data/ottqa_all_passages/all_passages.json`. Either pass the real path, or symlink:

```bash
# option A: pass the path explicitly
python scripts/export_sparqx_evidence_for_helios.py \
    --passages data/ottqa_all_passages/all_passages.json ...

# option B: make the default path resolve
mkdir -p data/ottqa_5m && ln -s ../ottqa_all_passages/all_passages.json data/ottqa_5m/all_passages.json
```

> Note: an old line in `CLAUDE.md` calls this corpus "deferred / not downloaded". That is
> stale — it **is** present and was used for the full-dev 2214 numbers.

> The Helios export also needs OTT-QA tables at `data/ottqa_repo/data/traindev_tables.json`
> (`--ottqa-tables`), which is **not** part of this hand-off (fetch from the OTT-QA repo if
> you rerun that specific export). The table pool the main pipeline uses is the IBM TableIR
> corpus inside `data_misc.tar.gz` (below), not this file.

---

## 2. `data_misc.tar.gz` — HybridQA raw + IBM TableIR + training sets

**Restore (extract at repo root — paths inside are already `data/...`):**

```bash
cd SPARQ_Extend
tar -xzf data_misc.tar.gz          # recreates data/data/, data/hybridqa_raw, data/ottqa_raw_full, ...
```

Top-level contents:

| Path after extract | Size | What it is / who reads it |
|---|---:|---|
| `data/data/hybridqa_raw/` + symlink `data/hybridqa_raw` → it | 208 MB | **Official HybridQA released_data** — see below |
| `data/ottqa_raw_full/` | 62 MB | **IBM TableIR corpus** — `corpus_structure.jsonl` (8,891 tables) + train/val parquet + qrels/queries. This is the OTT-QA table pool (100% dev coverage). |
| `data/hybridqa_table_reranker_training/` | 42 MB | Table-reranker fine-tune data |
| `data/passage_reranker_training/` | 16 MB | Passage-reranker fine-tune data |
| `data/ibm_tableir/` | 1.4 MB | IBM TableIR metadata |
| `data/ottqa_raw/` | 788 KB | OTT-QA dev/train question splits |

### 2a. `data/hybridqa_raw/` — the HybridQA table+text corpus

Contents (identical files at both `data/hybridqa_raw/` and its real dir `data/data/hybridqa_raw/`):

- `train.json` / `dev.json` / `test.json` — question annotations (HybridQA released_data).
- `WikiTables-WithLinks-<sha>.zip` (193 MB) — tables **and** their hyperlinks, kept zipped.

**How the code uses it.** `src/utils/hybridqa_local.py :: load_hybridqa_dataset(split, raw_dir)`
returns `LocalHybridQADataset`, which opens the zip **in place** (no extraction) and reads
per-table JSON from inside it:

```python
self._zip = zipfile.ZipFile(self.zip_path)         # WikiTables-WithLinks*.zip
url_data = self._read_json(f"{root}/request_tok/{table_id}.json")   # hyperlinks
# tables_tok/... for the table body
```

Default path is `data/hybridqa_raw` — used by ~30 scripts (`run_hybridqa_exp_a_pilot.py`,
`train_query_table_gnn.py`, `eval_hybridqa_stage1_3leg.py`, all the reranker-prep scripts,
etc.), each exposing `--hybridqa-dir`.

**⚠️ Symlink note.** `data/hybridqa_raw` is a **relative symlink** → `data/data/hybridqa_raw`
(a historical double-nesting). The tar preserves both the symlink and the real dir, so after
extraction both `data/hybridqa_raw/dev.json` (default) and `data/data/hybridqa_raw/dev.json`
resolve to the same files — nothing to fix. If you prefer a clean layout for the public
release, replace the symlink with the real dir:

```bash
rm data/hybridqa_raw && mv data/data/hybridqa_raw data/hybridqa_raw && rmdir data/data 2>/dev/null
```

---

## 3. Download from Quark (three options — see `Coreset-Tab/README.md §Reproduction bundle`)

The Quark folder is not a public URL. To pull:

- **(A)** Ask the PI (yanmy) for a Quark share link / the files directly. No alist needed.
- **(B)** On the lab server where alist already runs (`127.0.0.1:5246`, Quark mounted at
  `/quark`): get the alist token from the PI, then
  ```bash
  TOKEN=<ask-PI>
  curl -s --noproxy '*' -X POST 127.0.0.1:5246/api/fs/get -H "Authorization: $TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"path":"/quark/SPARQ_Extend/data/ottqa_all_passages/all_passages.json"}'
  # -> .data.raw_url is a direct download link; wget/curl it. Repeat for data_misc.tar.gz.
  ```
- **(C)** Your own server: install alist, add a Quark storage with the cookie/share the PI
  gives you, mount at `/quark`, then use the same `/api/fs/get` call.

> Always strip proxy env vars for localhost alist calls (`--noproxy '*'` or
> `unset http_proxy https_proxy all_proxy`).

---

## 4. Final layout the code expects

```
SPARQ_Extend/
└── data/
    ├── ottqa_all_passages/all_passages.json      # 3.2 GB, 5.96M passages (Quark)
    │   (optional symlink data/ottqa_5m/all_passages.json for the default path)
    ├── hybridqa_raw/            -> data/data/hybridqa_raw   (train/dev/test.json + WikiTables zip)
    ├── ottqa_raw_full/          # IBM TableIR: corpus_structure.jsonl (8891 tables) + parquet/qrels
    ├── hybridqa_table_reranker_training/
    ├── passage_reranker_training/
    ├── ibm_tableir/
    └── ottqa_raw/
```
