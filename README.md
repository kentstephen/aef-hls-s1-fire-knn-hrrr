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
the fire records and the satellite fire pixels are read from their public stores when the view opens.

- **The ground**: P(fuel) from the 10 m AlphaEarth Zarr mosaic, from zoom 11 in.
- **Buildings** (zoom 14 in): Overture footprints read with DuckDB-WASM, each outline colored by the most burnable
  ground within a reach of it.
- **The 2025 season**: every California wildfire start and burn perimeter of 2025 (NIFC WFIGS services), day by
  day, over the fuel of 2024. A start whose circle (30 m by default) holds fuel lights the fuel inside it; burns light
  the fuel inside them; the rest stays dim.
- **One fire from orbit**: the Green Fire (19,022 acres, lightning, July 2025, northeast of Redding) as GOES-West's
  fire product saw it, hour by hour, over the 2024 fuel. The fuel inside the final burn outline lights up where the
  satellite has already seen fire.

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

## Data

All public, no credentials: AlphaEarth Foundations embeddings (tge-labs on Source Cooperative), LANDFIRE LF2024,
NIFC WFIGS incident locations and perimeters, NOAA GOES-18 ABI on AWS, Overture Maps buildings, NAIP, the National
Fuel Moisture Database, HLS and Sentinel-1 (for the moisture work), NOAA HRRR via dynamical.org (in the map's code,
switched off).

Python scripts run with `uv run` (dependencies inline in each script).
