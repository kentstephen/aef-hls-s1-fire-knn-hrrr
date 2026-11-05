# aef-hls-s1-fire-knn

Burnable ground from satellite embeddings, checked against real fires, on a web map that reads everything from
public cloud stores at view time.

The core claim: a small network on [AlphaEarth Foundations](https://source.coop/tge-labs/aef) embeddings (10 m,
yearly) tells burnable fuel from ground that will not burn. Everything else in the repo is either a test of that
claim or context around it.

## What is here

### The fuel head (`scripts/fit_fuel.py`, `fuel_lab.py`, `fuel_scenarios.py`, `export_fuel_head.py`)

Five small networks (128 inputs, 64 hidden, 11 fuel kinds), averaged. Input per pixel: the newest year's AlphaEarth
embedding and its spread over that year and the two before. Labels: LANDFIRE LF2024 FBFM40 fuel models over 580
windows across California. The map draws P(fuel), the summed chance of the six burnable kinds.

- Held out by 50 km block: AUC 0.956; at a cutoff that hides 1 % of fuel, 58 % of non-fuel is caught
  (`results/fuel_lab/LEADERBOARD.md`).
- A blind NAIP photo check of its errors (`results/naip_check.md`): much of the fuel it hides is sparse desert scrub
  that is mostly bare sand on the photo; much of the non-fuel it misses is yards, orchards and dry fields that do
  look burnable.

Weights: `models/fuel_head.json` (the browser runs it on the GPU, WebGPU, or the CPU without it).

### The fire test (`scripts/fire_test_build.py`, `fire_test_grids.py`, `fire_test_score.py`)

The head never saw a fire, so fires are an outside check. Each fire is scored from the embeddings of the year
before it, so the map cannot see the burn scar. California, 2020 to 2026.

| Test | Result |
| --- | --- |
| Burned ground vs the ground just outside each burn (200 NIFC perimeters, 100 acres and up) | AUC 0.575 [0.559, 0.591] |
| Same, 2025 fires only (26) | head 0.586, LANDFIRE 0.539 |
| WFIGS start points vs random ground nearby (600 starts) | AUC 0.31 at the point, 0.44 within 100 m |

Burns land on fuel a bit more than the unburned ground beside them, and the head separates the two a little better
than LANDFIRE on fires after LANDFIRE's map was made. Start points land on less fuel than the ground around them:
most are reported on roads and at the edges of towns, where people start fires (LANDFIRE puts them on roads too).
The start points used here are WFIGS's initial reported locations; the web map uses each record's own point, which
can differ by kilometers.

`results/fire_test.json`, `results/fire_test_2025.json`.

### Live fuel moisture (`scripts/make_samples.py`, `sample_*.py`, `fit_knn.py`)

The repo's first question: does AlphaEarth help predict live fuel moisture (National Fuel Moisture Database field
samples) on top of HLS and Sentinel-1? A little: kNN RMSE 34.2 to 33.4 moisture points statewide, GBM 34.0 to 31.9,
with modest skill overall (R2 0.2 to 0.5). `results/knn_aef_*.json`.

### The web map (`web/`)

React, MapLibre and deck.gl. Nothing is precomputed but the model weights and small file indexes; the embeddings,
the fire records and the Sentinel-2 imagery are read from their public stores when the view opens. Two pages:

**GOES triage** (the default page, `web/src/triage/`): map left, table right. Every fire GOES-West saw over
California in the Park Fire week (Jul 22 to Aug 2, 2024) fills the table as its scans are read, ranked by a
GOES-only score (`models/goes_score.json`, fit by `scripts/goes_score_fit.py`). Then each event's fuel under and
around it is read from the embeddings, top row first. Each event reads up to 24 mosaic chunks of 3 years, so the
whole week is tens of GB; the bottom rows may never fill.

**The fuel map** (`?app=fuel`):

- **The ground**: P(fuel) from the 10 m AlphaEarth Zarr mosaic, from zoom 11 in. The panel shows the embeddings
  year large (2019 to 2025); `[` and `]` step it. The fuel model reads that year and the two before it.
- **Burned areas** (`web/src/perimeters.ts`): every California wildfire of 100 acres and up since 2020 (about 590),
  read live from NIFC's WFIGS perimeter service, outlines simplified to about 30 m (about 4 MB). Bright cyan, which
  none of the fuel ramps holds; outlines only. `B` shows and hides them. Hovering lists every fire that burned the
  spot (name, year, acres), newest first.
- **A fire's fuel by year** (`web/src/fuel/burnHist.ts`, `web/src/ui/FireHist.tsx`): clicking inside an outline
  selects that fire (the newest where they overlap; Esc clears). The panel shows one bar per embeddings year, 2019 to
  2025: the average P(fuel) over the outline's pixels in the mosaic chunks in view. The year on screen is read first,
  then the others chunk by chunk, bars growing faint until complete. Park Fire near Cohasset, zoom 14: 94, 94, 95, 96,
  94, 95 (fire year), 91 % (2025), read in about 9 s. P(fuel) asks whether ground can burn, not how much fuel is on
  it, and burned wildland is still fuel to LANDFIRE, so a fire moves it little.
- **Sentinel-2** (`web/src/s2.ts`): Earth Genome's yearly mosaics, 2022 to 2025, found with Earth Genome's STAC
  search and read from each footprint's TCI COG on Source Cooperative, decoded in workers. Hold Space to see the
  year matching the embeddings (2022 for earlier ones) in place of the fuel, drawn bare.
- **Reading the mosaic** (`web/src/aef/aef.worker.ts`, `pool.ts`): chunks come from the S3 host, kept on disk in
  IndexedDB between visits. Each worker also keeps the chunk-years it decoded lately in memory (192 MB each, least
  recently used dropped), and a chunk always goes to the same worker, so the next year over needs one new read, not
  three. Measured on the Mac's GPU for the fire bars above: 15 s before, 9 s after. Headless Chromium scores on a
  software GPU (SwiftShader), about 20 times slower, so time it headed.
- Behind `FUEL_ONLY` in `App.tsx` (off while it is true): the 2025 season of starts and burns, Overture buildings
  colored by nearby fuel, GOES-West fires as boxes, one fire hour by hour.

```bash
npm --prefix web install
npm --prefix web run dev        # http://localhost:5374
```

### GOES-West fire pixels (`scripts/goes_fire_refs.py`)

[VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr) opens each ABI-L2-FDCC file in `s3://noaa-goes18`
as byte-range references (headers only) and keeps the chunks of `Mask` and `Power` over one fire's box. The browser
fetches and decodes those ranges itself (zlib and shuffle). Every file's form (variables, chunks, codecs, projection,
grid) is fingerprinted and the scans are split into eras of one form each, because the files change form over the
years: GOES-18 gained a Shuffle filter on 2023-03-27 and moved from 250 x 250 tiles to full-width strips on
2026-07-09.

```bash
uv run scripts/goes_fire_refs.py --fire Green --year 2025 scout        # fire pixels an hour, to pick the window
uv run scripts/goes_fire_refs.py --fire Green --year 2025 build --from 2025-07-02T00:00 --to 2025-07-20T00:00
```

### GOES + fuel test (`scripts/goes_ca_detections.py`, `goes_events.py`, `goes_fuel.py`, `goes_fuel_score.py`)

GOES finds the heat; the embeddings say what ground is under and around it. Every GOES-18 FDCC scan (one per 15 min)
gives California's fire pixels, joined into events (pixels touching on the same or the next day). An event is
matched if a NIFC perimeter or WFIGS wildfire start lies near it, unmatched otherwise. For each event's first box,
the fuel share inside the hot pixels ("in") and in the ring 1.5 GOES pixels out ("around") is read three ways from
the 10 m pixels of the year before: kNN on the AlphaEarth embeddings, the fuel head, and LANDFIRE. No COG overviews.
Four years, 2023 to 2026 (2026 through Oct 27, where WFIGS ends): 13,795 events, 1,675 matched. The GOES-18 grid is
the same over all four years.

AUC matched vs unmatched (`results/goes_fuel_<year>.json`, pooled in `results/goes_fuel_check_2023_2024_2025_2026.json`;
95 % intervals resample 0.1 degree places, gains are paired):

| | matched / unmatched | kNN in | LANDFIRE in | GOES alone | GOES + kNN | gain |
|---|---|---|---|---|---|---|
| 2023 | 349 / 2,994 | 0.628 | 0.564 | 0.666 | 0.720 | +0.054 [0.026, 0.084] |
| 2024 | 525 / 3,441 | 0.642 | 0.573 | 0.660 | 0.746 | +0.086 [0.058, 0.113] |
| 2025 | 382 / 2,632 | 0.582 | 0.541 | 0.705 | 0.741 | +0.036 [0.013, 0.058] |
| 2026 | 419 / 3,053 | 0.662 | 0.599 | 0.664 | 0.751 | +0.086 [0.057, 0.116] |
| pooled | 1,675 / 12,120 | 0.630 | 0.570 | 0.666 | 0.738 | +0.072 [0.054, 0.090] |
| pooled, 2+ scans | 1,124 / 4,335 | 0.693 | 0.635 | 0.648 | 0.757 | +0.108 [0.083, 0.134] |
| pooled, single scan | 551 / 7,785 | 0.552 | 0.499 | 0.635 | 0.655 | +0.021 [-0.004, 0.047] |
| pooled, first scan class 6 | 863 / 2,669 | 0.757 | 0.687 | 0.673 | 0.793 | +0.121 [0.088, 0.152] |
| pooled, perimeter vs unmatched | 501 / 12,120 | 0.768 | 0.694 | 0.643 | 0.840 | +0.197 [0.166, 0.227] |
| pooled, start vs unmatched | 1,174 / 12,120 | 0.571 | 0.517 | 0.680 | 0.711 | +0.032 [0.017, 0.046] |

- Fuel adds to GOES every year. 2025, the first year run, is the weakest; whether that is the year or its 2024
  embeddings is not measured.
- The embeddings read fire ground better than LANDFIRE; kNN and the head are even (head gain +0.070 pooled).
- The gain lives in events seen in 2+ scans, the strongest first-scan class, and fires that grew into a perimeter.
  Single-scan events get little (+0.021, interval crosses 0).
- Single-scan matched events are mostly small starts near people: 83 % starts, median 11.7 acres, 57 % within
  0.5 km of a building (unmatched single scan 18 %). Away from roads the fuel signal is there: single-scan events over
  0.5 km from any road, kNN in alone 0.732 (75 matched of 3,204); within 0.5 km of a road, 0.500.
- Remoteness (km to the nearest road, major road and building, Overture) lifts GOES alone from 0.666 to 0.768:
  WFIGS records fires where people are. Fuel still adds with remoteness held fixed: +0.038 [0.025, 0.050] pooled,
  +0.076 on 2+ scans, +0.155 on perimeters.
- Unmatched is not the same as a false alarm. Fuel and remoteness score how a fire gets into WFIGS as much as
  whether it is a fire.

#### Following the fire (`scripts/goes_fuel_track.py`, `results/goes_fuel_track_2023_2024_2025_2026.json`)

Fuel per GOES cell, flat and with terrain parallax corrected (Mapterhorn DEM; median shift 0.21 GOES pixels, p90
0.65). Terrain correction changes nothing (0.738 flat, 0.739 corrected). Scored at 0 to 24 h after the first scan:

| hours after first scan | events still burning (matched) | GOES so far | + fuel | gain |
|---|---|---|---|---|
| 0 | 5,459 (1,124) | 0.648 | 0.757 | +0.109 [0.084, 0.134] |
| 1 | 3,227 (703) | 0.763 | 0.806 | +0.043 [0.027, 0.059] |
| 6 | 1,725 (311) | 0.766 | 0.832 | +0.065 [0.043, 0.090] |
| 12 | 1,578 (265) | 0.773 | 0.849 | +0.075 [0.050, 0.100] |
| 24 | 1,223 (207) | 0.780 | 0.843 | +0.063 [0.039, 0.087] |

- GOES alone climbs fast in the first hour (more scans, more pixels); fuel keeps adding on fires that last.
- Fuel ahead of the newest pixels adds nothing over fuel so far. Against final acres it tracks the pixel count
  (1 h: Spearman 0.335 vs 0.300 for cells so far, 380 fires).

#### Will it grow (`scripts/goes_growth.py`, `results/goes_growth_2023_2024_2025_2026.json`)

Matched fires, first event per fire (1,305 fires), out of fold:

| | Spearman with log acres | AUC past 100 ac | AUC past 1,000 ac | AUC perimeter vs start |
|---|---|---|---|---|
| GOES alone | 0.182 | 0.604 | 0.542 | 0.558 |
| GOES + kNN fuel | 0.388 | 0.723 | 0.708 | 0.724 |
| GOES + remoteness | 0.434 | 0.793 | 0.821 | 0.816 |
| GOES + remoteness + kNN fuel | 0.479 | 0.805 | 0.828 | 0.827 |
| GOES + HRRR weather | 0.210 | 0.602 | 0.581 | 0.589 |

- Fuel adds +0.206 [0.152, 0.262] to GOES for final acres, but with remoteness held fixed only +0.045
  [0.020, 0.070], and nothing past 1,000 acres. For growth, much of "fuel" is "far from roads and homes".
- HRRR weather at the first scan adds almost nothing; only past 1,000 acres does it add (+0.033 over GOES + fuel).

#### Inside "unmatched" (`scripts/goes_unmatched.py`, `results/goes_unmatched_<year>.json`)

| | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| unmatched | 2,994 | 3,441 | 2,632 | 3,053 |
| nothing else saw it | 1,963 | 2,193 | 1,753 | 2,167 |
| VIIRS heat, no record | 362 | 327 | 300 | (no VIIRS) |
| fixed hot spot (5+ events or VIIRS static) | 296 | 560 | 236 | 557 |
| farmland (half the box or more) | 223 | 226 | 231 | 209 |
| prescribed burn record (WFIGS Rx, FACTS) | 145 | 93 | 92 | 81 |
| CAL FIRE incident | 5 | 42 | 20 | 39 |
| near-zero fuel (kNN under 10 % in and around) | 68 | 120 | 80 | 130 |

- Each source is checked against a chance rate (the same rules with event times moved by 45 and 90 days). Loosening
  the WFIGS match finds few beyond chance (2023: 62 vs 39; 2025: 70 vs 58).
- VIIRS heat with no record is mostly October to June (likely Rx and pile burns missing from WFIGS and FACTS).
- Near-zero fuel is bare hot ground (Imperial Valley, Death Valley, dry lakes, mostly July and August), Sacramento
  Valley rice (September to November), and industrial heat (landfills, refineries, cogeneration plants).
- The FIRMS yearly S-NPP file is keyless for 2023 to 2025 only, so 2026 has no VIIRS column; 2023, 2024 and 2026 ran
  without the OSM landfill / industry lookup (`--no-osm`).

```bash
Y=2025
uv run scripts/goes_ca_detections.py --year $Y    # ~15 min, 9.5 GB pulled
uv run scripts/goes_events.py --year $Y
uv run scripts/goes_fuel.py plan --year $Y --max-events 10000   # every event, no sampling
uv run scripts/goes_fuel_track.py pixels --year $Y
uv run scripts/goes_fuel_track.py plan --year $Y   # chunks along each fire's later scans
uv run scripts/goes_fuel.py grids --year $Y --workers 32 --max-gb 8 --extra-chunks data/goes/track_chunks_$Y.parquet
uv run scripts/goes_fuel.py shares --year $Y      # ~4 min, peak 5.5 GB
uv run scripts/goes_fuel_score.py --year $Y
uv run scripts/goes_fuel_track.py cells --year $Y --threads 8
uv run scripts/goes_fuel_track.py steps --year $Y
uv run scripts/goes_unmatched.py --year $Y
# pooled
uv run scripts/goes_fuel_check.py overture        # roads and buildings, once
uv run scripts/goes_fuel_check.py run --years 2023 2024 2025 2026 --distances
uv run scripts/goes_fuel_track.py score --years 2023 2024 2025 2026
uv run scripts/goes_growth.py --years 2023,2024,2025,2026
```

Grids are network-bound at about 460 chunks a minute (~56 MB/s) however many runs share the line; a year takes
1 to 4.5 hours, about 2.2 GB peak per run.

## Data

All public, no credentials: AlphaEarth Foundations embeddings (tge-labs on Source Cooperative), LANDFIRE LF2024,
NIFC WFIGS incident locations and perimeters, NOAA GOES-18 ABI on AWS, Overture Maps buildings and roads, NAIP, the
National Fuel Moisture Database, HLS and Sentinel-1 (for the moisture work), NOAA HRRR via dynamical.org, NASA FIRMS
VIIRS S-NPP, CAL FIRE incidents, USFS FACTS prescribed burns, Mapterhorn terrain tiles.

Python scripts run with `uv run` (dependencies inline in each script).

## Inspirations

- [goes-fdc-catalog](https://github.com/mmidzik/goes-fdc-catalog): GOES fire pixels (ABI FDC) as virtual Zarr
  references into NOAA's bucket. This repo builds its own references the same way (`scripts/goes_fire_refs.py`).
- [CTrees FuelWatch](https://fuelwatch.ctrees.org/): a live fuel moisture map on fuel groups. The fuel map here
  starts from the same idea, with the fuel read from embeddings.
