# Commit history from our private working repo

We worked in a private course repository that also holds course material and large datasets, so it can't be made public. This public repo was created from its `assignment2/` folder with fresh history, so the commit hashes cited in the reports refer to that private repo. The relevant commits are listed below; we can share the full history on request.

| commit | date (CDT) | content |
|--|--|--------|
| `0e0629a` | 2026-10-19 18:32 | Phase 1–3 code for all three papers (phase1.py, phase3_v1.py, phase3.py), Phase 1 runs, Keskar F1/C1 Phase 3 results |
| `674bce2` | 2026-10-20 09:33 | Phase 3 results for Zhang (Inception) and Keskar C2; β-VAE Phase 1 runs |
| `63069f7` | 2026-10-20 09:44 | Phase 4 comparisons (phase4.md), Keskar BN ablation, notebooks |
| `1eb1dd4` | 2026-10-20 13:15 | β-VAE Phase 3 seeds 0–2 (unchanged phase1.py) |

The rubric's requirement that Phase 3 be committed before Phase 4 is met: `0e0629a` and `674bce2` precede `63069f7`. For β-VAE, Phase 4 (`63069f7`) found no authors' code to compare against, and Phase 3 only re-ran the unchanged `phase1.py` with more seeds (`1eb1dd4`).

**Checking that the code is unchanged.** Git identifies file contents by a content hash (the blob ID), which is the same in any repository. Running `git hash-object <file>` in this repo gives the IDs below, which match the blobs committed in `0e0629a` in our working repo; none of these files changed afterwards.

| file | blob ID |
|--|--------|
| `zhang/phase1.py` | `6f9e25c3eebfa3bbf0020dbc4ba39d18611bbf02` |
| `zhang/phase3_v1.py` | `f5e33a84af2fa8c0ade71cf10e213face31b1f1b` |
| `zhang/phase3.py` | `46fa37da955153ba58a57b75a88051cc115e984a` |
| `keskar/phase1.py` | `a87ae8e9d73c860c130fd89751c3f476b574b2e4` |
| `keskar/phase3.py` | `5b684d6edb3e9be23c7141705186c64983af5ce9` |
| `betavae/phase1.py` | `0c9febc80dd2c697b6c375c99d0b54fdec061cd9` |
