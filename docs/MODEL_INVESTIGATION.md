# GeoRAG — Investigation of the Two Proposed Transformer Architectures

Date: 2026-10-22. Method: literature/implementation research (sources cited) plus
direct inspection of the Google Drive project data. **Verified** = checked against a
source or the actual project files. **Assumed** = reasonable but not confirmed.
**Recommendation** = our judgement for GeoRAG.

---

## 1. Model A — "Multi-Temporal Swin Transformer (MTS)"

### 1.1 Can the exact architecture be identified?

**Partly. Verified:** a remote-sensing method explicitly called the
"multi-temporal Swin Transformer (MTS)" exists inside:

> Voelsen, Rottensteiner & Heipke (2024): *Transformer models for Land Cover
> Classification with Satellite Image Time Series*, PFG – Journal of Photogrammetry,
> Remote Sensing and Geoinformation Science 92(5), 547–568.
> DOI: https://doi.org/10.1007/s41064-024-00299-7
> Record: https://research.uni-hannover.de/en/publications/transformer-models-for-land-cover-classification-with-satellite-i/

**Verified:** "Multi-Temporal Swin Transformer" is **not** the paper title, the name is
**not** a standardised architecture name, and **no official implementation or
pretrained weights** were found for it (checked paper/institutional records; absence of
a repo found is not proof that none exists). A closely related predecessor is
Voelsen et al. (2023), ISPRS Annals (Swin + spatio-temporal transformer block):
https://isprs-annals.copernicus.org/articles/X-1-W1-2023/981/2023/

**Therefore, per the brief's rule, we did NOT invent "the MTS implementation".**
Other distinct multi-temporal models exist with real code and must not be confused
with MTS (verified repos):
- TSViT (CVPR 2023) — https://github.com/michaeltrs/DeepSatModels
- U-TAE (ICCV 2021) — https://github.com/VSainteuf/utae-paps
- SatMAE (NeurIPS 2022) — https://github.com/sustainlab-group/SatMAE (8×V100 used)
- SatMAE++ (CVPR 2024) — https://github.com/techmn/satmae_pp
- Microsoft Swin backbone — https://github.com/microsoft/Swin-Transformer

### 1.2 Suitability questions (answers for GeoRAG)

| Question | Answer | Status |
|---|---|---|
| Official implementation? | None found | Verified (search) |
| Supports multispectral imagery? | Paper is Sentinel-2 based; band count/layout details not accessible in the free text | Partially verified |
| Accepts multiple temporal observations? | Yes — that is the point of the method (per-date label maps from a time series) | Verified (abstract) |
| Works with the existing CSVs? | No: it needs image patches / time-series image tensors, not isolated per-pixel CSV rows | Verified (task type) |
| Additional data/weights needed? | No official weights exist; training must be from scratch | Verified |
| Hardware in the literature | Multi-temporal transformers are trained on GPUs (SatMAE: 8×V100; U-TAE: ~12 h/fold on V100) | Verified for related models; MTS paper's own hardware unverified |
| Training here (2 CPU cores, no GPU) | A full published-scale training is impractical; a **small adapted window-attention model on one district** is practical (~7 min/epoch measured) | Measured here |
| 4-class support? | Any dense encoder can end in a 4-logit head | Verified (standard) |
| GeoTIFF conversion | Predict on the shared 693×669 grid and copy CRS/transform from Member 1's label grid | Verified (standard rasterio practice) |

**MTS multi-temporal sequences for GeoRAG's experiments — important finding:**
with only three dates (2015/2020/2025) a leakage-free temporal sequence can only be
constructed for experiment 3 (train 2015+2020 → test 2025):
train inputs [masked, 2015] and [2015, 2020]; test input [2020, 2025].
Experiments 1 and 2 (train 2015 only → test 2020/2025) have no usable history slot
without test-year information, so those runs use single-date input. This constraint is
inherent to the data, not to our code.

## 2. Model B — "Dual-Level Attention Lightweight Vision Transformer"

### 2.1 Can the exact architecture be identified?

**A near-exact published paper exists (verified via Crossref):**

> Bansal, K. & Tripathi, A. K. (2024): *Dual level attention based lightweight vision
> transformer for streambed land use change classification using remote sensing.*
> Computers & Geosciences 191, 105676.
> DOI: https://doi.org/10.1016/j.cageo.2024.105676
> Crossref: https://api.crossref.org/works/10.1016/j.cageo.2024.105676

**Verified:** title, authors, journal, year, DOI. **Not verified:** the internal
architecture (publisher full text is paywalled), an official acronym, and any code
repository — **no official implementation was found.** So an exact reproduction is
impossible from available information; anything we build is an *adaptation*.

Do not confuse it with (all verified, different models):
DaViT (Dual Attention Vision Transformers, https://github.com/dingmyu/davit),
DAT = *Deformable* Attention Transformer (https://github.com/LeapLabTHU/DAT),
XCiT (https://github.com/facebookresearch/xcit), MobileViT, PiT.

### 2.2 Suitability questions (answers for GeoRAG)

| Question | Answer | Status |
|---|---|---|
| Official implementation? | None found; full text paywalled | Verified (search) |
| Classification or segmentation? | Title says "classification" for streambed land-use *change*; per-pixel mapping protocol unverified | Partially verified |
| Multispectral support? | Search-indexed abstract mentions multispectral satellite imagery; band count unverified | Partially verified |
| Requires spatial image patches? | Any ViT does — patch tokens are its native input | Verified (architecture class) |
| Works with existing CSVs? | No — CSV rows are isolated pixels without spatial patches | Verified |
| Additional data/weights needed? | Unknown (no code); training from scratch required | Verified (absence of code) |
| Hardware (literature) | Comparable lightweight ViTs are trained multi-GPU (DaViT/DAT/XCiT examples: 8 GPUs) | Verified for related models; this paper's hardware unverified |
| Training here (CPU only) | A compact patch model of this width trains in minutes per epoch on this grid | Measured here |
| 4-class support? | Yes via a 4-logit dense head | Verified (standard) |
| GeoTIFF conversion | Same shared-grid write as above | Verified (standard) |

## 3. Comparison table (as requested)

| Criterion | MTS | Dual-Level Attention Lightweight Vision Transformer |
|---|---|---|
| Intended task | Pixel-wise land-cover classification from satellite image **time series** | Streambed land-use **change classification** from remote sensing (paper); internal protocol unverified |
| Required input format | Patchified multi-date image tensors (sequence of images) | Spatial image patches (tokens) |
| Compatibility with existing CSVs | None (needs images/sequences) | None (needs spatial patches) |
| Compatibility with GeoTIFF imagery | Yes — scenes exist for all 3 years | Yes — same |
| Multi-year temporal capability | **Native** (its defining feature); limited to exp. 3 here (only leakage-free sequence) | Not indicated in the title; single-date |
| Spatial-context capability | Yes (window/patch attention) | Yes (token + channel attention) |
| Four-class classification support | Yes (dense head) | Yes (dense head) |
| Training requirements | From scratch (no weights); paper-scale runs used GPUs | From scratch (no weights); comparable models use multi-GPU |
| GPU requirements | GPU expected at publication scale; impractical here; small adaptation feasible on CPU | Same |
| Implementation complexity | High (window attention + shift masks + temporal slots) | High (dual attention blocks) |
| Main limitations | No official code; ambiguous name; temporal sequences barely supported by 3 dates | No official code; paywalled details; exactness impossible |
| Suitability for GeoRAG | Possible only as a **documented adaptation** | Possible only as a **documented adaptation** |

## 4. Decision: is a model change practical? (Case A / Case B)

**Case A applies — original satellite GeoTIFFs are available and verified:**

- 7-band Landsat scenes 2015/2020 on the shared 693×669 grid, 2025 scene (743×743)
  windowed onto the same grid, plus Member 1's label rasters for all three years.
- Band order verified: blue, green, red, nir, swir1, swir2, thermal.
- Member 1's index formulas reproduced exactly (max |Δ| = 3e-8 vs their Index and
  2025-ML-Features rasters), so our 9-feature stacks match their preprocessing.
- Labels exist as **spatially complete label rasters** (not just CSV rows): the
  aligned 30 m label rasters cover the grid with classes 1–4 + background.
- Spatial alignment verified: same CRS/transform for Landsat, Indices, aligned labels.
  The 2025 scene sits on a shifted origin; window (col 19, row 200) aligns it exactly.
- **2025 coverage limit:** grid rows 543–668 (87,318 px) lie outside the 2025 scene →
  excluded from 2025 training/evaluation and written as NoData in 2025 outputs.

**Case B (CSV-only) would have blocked both models** — the CSV has 3 index columns and
no spatial context; forcing a ViT onto tabular rows would be invalid. Because Case A
holds, we built **image patches from the real GeoTIFFs**, not fabricated data.

**What we did NOT do:** modify Member 1's datasets, invent an "official" MTS or
DLAT-ViT implementation, or claim any published accuracy.

## 5. What was actually implemented (honest labelling)

Two compact architectures in `src/models.py`, both ~0.42 M parameters, patch 4×4,
window 7×7, depth 6, width 64, per-pixel head — trained from scratch on CPU:

- `mts-lite-adapted` — Swin-style shifted-window dense encoder **+ optional second
  temporal slot (history/current/mask flag)**, i.e. the MTS direction.
- `dla-lite-adapted` — token-level window attention **+ channel-level cross-covariance
  attention** per block (a genuine "dual-level" attention scheme), i.e. the
  lightweight-ViT direction.

Both are **adaptations, not reproductions**, and are labelled that way in every result
file (`model.architecture_source`).

## 6. Sources

- Voelsen et al. 2024 (MTS): https://doi.org/10.1007/s41064-024-00299-7 ;
  https://research.uni-hannover.de/en/publications/transformer-models-for-land-cover-classification-with-satellite-i/
- Voelsen et al. 2023 (predecessor): https://isprs-annals.copernicus.org/articles/X-1-W1-2023/981/2023/
- Bansal & Tripathi 2024 (DLA lightweight ViT): https://doi.org/10.1016/j.cageo.2024.105676 ;
  https://api.crossref.org/works/10.1016/j.cageo.2024.105676
- TSViT: https://arxiv.org/abs/2301.04944 ; https://github.com/michaeltrs/DeepSatModels
- U-TAE: https://github.com/VSainteuf/utae-paps
- SatMAE: https://github.com/sustainlab-group/SatMAE ; https://arxiv.org/abs/2207.08051
- SatMAE++: https://github.com/techmn/satmae_pp
- Swin: https://github.com/microsoft/Swin-Transformer
- DaViT: https://github.com/dingmyu/davit ; DAT: https://github.com/LeapLabTHU/DAT ;
  XCiT: https://github.com/facebookresearch/xcit ; MobileViT: https://github.com/apple/ml-cvnets
- Rasterio georeferencing: https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html
