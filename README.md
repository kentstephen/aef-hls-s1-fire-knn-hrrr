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
the fire records and the Sentinel-2 imagery are read from their public stores when the view opens.

- **The ground**: P(fuel) from the 10 m AlphaEarth Zarr mosaic, from zoom 11 in. The panel picks the embeddings
  year (2019 to 2025); the fuel model reads that year and the two before it.
- **Sentinel-2** (`web/src/s2.ts`): Earth Genome's yearly mosaics, 2022 to 2025, found with Earth Genome's STAC
  search and read from each footprint's TCI COG on Source Cooperative, decoded in workers. Drawn bare, over the
  basemap's streets and labels: hold Space to see it in place of the fuel, `[` and `]` (or the wheel while Space
  is held) to step its year, P for imagery and fuel side by side, moving together. S turns the 2025 season on
  and off.
- **Buildings** (zoom 14 in): Overture footprints read with DuckDB-WASM, each outline colored by the most burnable
  ground within a reach of it.
- **The 2025 season**: every California wildfire start and burn perimeter of 2025 (NIFC WFIGS services), day by
  day, over the fuel of 2024. A start whose circle (30 m by default) holds fuel lights the fuel inside it; burns light
  the fuel inside them; the rest stays dim. Clicking a start or a burn reads that fire's whole NIFC record (dates,
  size, cause, place, land, agency, NIFC's fuel notes, personnel, cost) into the panel.
- **One fire from orbit** (off in the panel, code kept): the Green Fire (19,022 acres, lightning, July 2025) as
  GOES-West's fire product saw it, hour by hour. GOES pixels are 2 to 4 km across, so over a 10 m fuel map they
  cover the whole burn and say little about the ground.

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

GOES finds the heat; the embeddings say what ground is under and around it. Every GOES-18 FDCC scan of 2025 (one per
15 min) gives California's fire pixels, joined into 3,014 events (pixels touching on the same or the next day).
An event is matched if a NIFC perimeter or WFIGS wildfire start lies near it (382), unmatched otherwise (2,632).
For each event's first box, the fuel share inside the hot pixels ("in") and in the ring 1.5 GOES pixels out
("around") is read three ways from the 10 m pixels of 2024: kNN on the AlphaEarth embeddings, the fuel head, and
LANDFIRE. No COG overviews; 35,777 mosaic chunks.

Results (`results/goes_fuel_2025.json`, AUC matched vs unmatched, 95 % intervals resampling 0.1 degree places):

| | all events | 2+ scans | single scan |
|---|---|---|---|
| kNN fuel in | 0.582 [0.544, 0.618] | 0.640 | 0.497 |
| head fuel in | 0.585 [0.549, 0.621] | 0.645 | 0.491 |
| LANDFIRE fuel in | 0.541 [0.505, 0.574] | 0.610 | 0.456 |
| GOES alone (power, class, pixels) | 0.705 [0.672, 0.738] | 0.653 | 0.634 |
| GOES + kNN fuel | 0.741 [0.709, 0.773] | 0.737 | 0.632 |

- The embeddings read fire ground better than LANDFIRE does; kNN and the head are even.
- Fuel adds to GOES on events seen in 2+ scans and on the strongest first-scan class (0.597 to 0.752), and adds
  nothing on single-scan or low-probability events. Fuel inside the box is a little stronger than the ring.
- Under 10 % fuel in and around: 0.5 % of matched events, 3 % of unmatched (0 of 261 matched among 2+ scan events).
- The gain over GOES alone is not yet tested as a paired difference. Unmatched is not the same as a false alarm
  (ag and prescribed burns, fires missing from WFIGS).

```bash
uv run scripts/goes_ca_detections.py --year 2025    # ~15 min, 9.5 GB pulled
uv run scripts/goes_events.py --year 2025
uv run scripts/goes_fuel.py plan --year 2025 --max-events 10000   # every event, no sampling
uv run scripts/goes_fuel.py grids --year 2025 --workers 32 --max-gb 8   # ~60 min at 32 workers, peak 2.3 GB
uv run scripts/goes_fuel.py shares --year 2025      # ~4 min, peak 5.5 GB
uv run scripts/goes_fuel_score.py --year 2025
```

## Data

All public, no credentials: AlphaEarth Foundations embeddings (tge-labs on Source Cooperative), LANDFIRE LF2024,
NIFC WFIGS incident locations and perimeters, NOAA GOES-18 ABI on AWS, Overture Maps buildings, NAIP, the National
Fuel Moisture Database, HLS and Sentinel-1 (for the moisture work), NOAA HRRR via dynamical.org (in the map's code,
switched off).

Python scripts run with `uv run` (dependencies inline in each script).
