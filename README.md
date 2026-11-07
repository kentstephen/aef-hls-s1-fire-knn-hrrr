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

### What P(fuel) measures, and what it does not

P(fuel) answers one question: is this ground burnable land (any FBFM40 burnable model) or not (urban and roads,
snow and ice, farmland, water, bare, codes 91 to 99). It is a map of land type, not of fuel load, dryness or
readiness to burn.

It barely moves across a fire. The Park Fire (July 2024), averaged inside its outline in the map's fire bars
(near Cohasset, zoom 14, 12 mosaic chunks): 94, 94, 95, 96, 94, 95, 91 % for 2019 to 2025. Zoomed out
(151 chunks): 95, 96, 97, 97, 96, 96, 94 %.

That follows from the labels. LANDFIRE has no "burned" class: after a fire it changes the fuel model (timber to
grass-shrub, say) but keeps it burnable, so by the definition the head learned, a burned slope is still fuel. The
yearly embedding of a burned hillside also still looks like wildland. Summing the six burnable kinds into one
number throws away the part a fire changes.

What this means:
- On grass, high P(fuel) a year after a fire is close to right: grass is fuel again the next season.
- On timber, it hides most of what matters: less load and a different kind of fuel for decades.
- The map's strength is drawing the line between wildland and everything else at 10 m, each year. Its edge over
  LANDFIRE in the fire test comes from that line, not from knowing which wildland is more dangerous.
- The head's kinds do not shift after timber fires either (next section), so the change is not hiding in the kinds.

### The embeddings see the burn; the head does not (`scripts/fire_change.py`)

Five fires, eight mosaic chunks mostly inside each perimeter and eight 1 to 6 km outside (pixels inside any other
perimeter on file left out of both). Per year, the cosine of each pixel's embedding with its own embedding the year
before the fire (1 = unchanged; ordinary ground drifts to about 0.95 from year to year):

| Fire | Fuel | Year after | 2025 | Outside |
| --- | --- | --- | --- | --- |
| Caldor 2021 | timber | 0.71 | 0.69 | ~0.96 |
| Dixie 2021 | timber | 0.78 | 0.74 | ~0.94 |
| Creek 2020 | timber | 0.71 | 0.72 | ~0.93 |
| Bobcat 2020 | chaparral | 0.75 | 0.88 | ~0.92 |
| Park 2024 | grass and oak | 0.83 | 0.83 | ~0.92 |

- Burned timber stays far from its old embedding. Bobcat's chaparral climbs back every year (0.75, 0.79, 0.82,
  0.85, 0.88): regrowth shows in the embeddings alone.
- The head keeps calling burned timber timber (Caldor: 83 to 85 % of pixels led by a timber kind before, 76 to
  91 % after) and P(fuel) stays flat. It calls burned Bobcat chaparral timber-litter, which is wrong. The head's
  spread input (the pixel's spread over three years) jumps after a fire, a likely cause.
- Fires before 2020 are not on file, so older scars inside the samples are not removed.

`results/fire_change.json`.

### A head that sees the burn (`scripts/sample_change.py`, `fit_change.py`)

LANDFIRE's own fuel models barely change after these fires (burned forest stays a timber model), but its canopy
cover does (inside the sampled chunks: Caldor 42 % in LF2016 to 22 % in LF2022, Dixie 38 to 22, Creek 35 to 19,
Park 22 to 11 to 5 for LF2023, LF2024, LF2025). So the new head learns canopy cover (CC), canopy height (CH) and
the fuel kind from four LANDFIRE editions (2016, 2022, 2024, 2025), each paired with the embedding of its own year
(LF2016 with 2017). Pixels: 537 of the fuel lab's random windows plus 191 windows inside 2020 to 2023 burns, 1,500
each; windows within about 10 km of the five test fires above are left out.

One network (in -> 64 -> CC, CH, 11 kinds), five input sets; 5 folds by 0.5 degree block. All five score within a
hair of each other; the best is the year's own embedding alone, with all four editions (`e_all`):

| | CC | CH |
| --- | --- | --- |
| All windows, MAE / R^2 | 6.6 points / 0.80 | 3.8 m / 0.72 |
| Burn windows | 9.5 / 0.73 | 5.8 / 0.63 |

Pixels that lost 20 or more points of CC between LF2016 and LF2022 (39.7 on average): predicted loss 26.0, out of
fold. Pixels LANDFIRE kept steady: predicted change -0.9. Correlation of predicted with true change 0.70.

The five held-out fires, mean predicted CC inside the burn (outside in parentheses):

| Fire | Year before | Fire year | 2025 | LANDFIRE before -> after |
| --- | --- | --- | --- | --- |
| Caldor 2021 | 40 (28) | 27 (29) | 19 (28) | 42 -> 22 |
| Dixie 2021 | 37 (22) | 29 (24) | 22 (24) | 38 -> 22 |
| Creek 2020 | 34 (27) | 28 (28) | 15 (29) | 35 -> 19 |
| Park 2024 | 21 (24) | 13 (23) | 10 (24) | 23 -> 5 |
| Bobcat 2020 | 24 (5) | 12 (5) | 14 (6) | 26 -> 18 |

Canopy height follows (Caldor 20 m to 11 m). The kinds move slowly, as LANDFIRE's do: in burned Creek forest
shrub leads on 0 % of pixels before the fire and 22 % in 2025.

`results/fit_change.json`.

**Every 2024 fire** (`scripts/change_test.py`): 65 fires of 100 acres and up, 134 mosaic chunks their outlines
cross, none within 3 km of a training window. Per fire, predicted CC change 2023 -> 2025 inside the burn minus the
change on the unburned ground beside it (150 m back from the line on both sides):

| Canopy before | Fires | Predicted | LANDFIRE (LF2023 -> LF2025) | Fires with canopy down |
| --- | --- | --- | --- | --- |
| 25 % and up | 14 | -9.2 points | -11.5 | 86 % |
| 10 to 25 % | 19 | -3.7 | -8.9 | 95 % |
| under 10 % | 32 | -0.4 | 0.0 | 66 % |

Fire by fire the predicted drop follows LANDFIRE's (correlation 0.79; Crozier -30 vs -28, Mill -25 vs -34). Grass
and brush fires barely move canopy cover, which counts trees only: the reason for the volume head below.
`results/change_test.json`.

### Vegetation volume (`scripts/fit_volume.py`, `export_volume_head.py`)

The map's value for "how much is here to burn". LANDFIRE's existing vegetation cover and height (EVC, EVH) give each
pixel's leading lifeform (tree, shrub or herb) with its cover % and height; their product, cover x height in meters,
is the vegetation volume: about 10 m in closed forest, 0.5 m in chaparral, 0.2 m in grass, 0 on roads and water.
A proxy for fuel amount, not a load in tons. One network on the year's embedding alone (64 -> 64 -> volume, cover,
height, CC, CH, 11 kinds, 4 lifeforms), the same rows and folds as the change head.

- Out of block: log volume R^2 0.79 (median error a factor of 1.6), lifeform right on 80 % of pixels.
- Pixels whose LANDFIRE volume fell by a factor of e or more from LF2016 to LF2022 (on average by a factor of 18):
  predicted fall a factor of 4. Steady pixels: predicted change 6 %.

The five held-out fires, mean volume inside (outside):

| Fire | Year before | Fire year | Year after | 2025 | LANDFIRE 2016 -> 2025 |
| --- | --- | --- | --- | --- | --- |
| Caldor 2021 | 5.8 m (2.4) | 2.0 (2.4) | 1.4 (2.4) | 0.6 (2.1) | 6.1 -> 0.7 |
| Dixie 2021 | 4.9 (0.9) | 2.5 (1.0) | 1.6 (1.0) | 0.9 (1.0) | 5.8 -> 0.9 |
| Creek 2020 | 4.7 (2.1) | 2.9 (2.3) | 1.0 (2.9) | 0.6 (2.3) | 5.0 -> 0.5 |
| Park 2024 | 1.3 (1.3) | 0.5 (1.3) | 0.3 (1.3) | 0.3 (1.3) | 1.8 -> 1.4 |
| Bobcat 2020 | 1.7 (0.2) | 0.6 (0.2) | 0.5 (0.2) | 0.6 (0.3) | 2.2 -> 0.6 |

Unlike the fuel kinds, the lifeforms change after a fire, in the order the ground recovers: burned Creek forest is
91 % tree before, 51 % herb in 2022, 56 % shrub in 2025; Bobcat's chaparral goes to 36 % herb in 2022 and back to
66 % shrub from 2023. `results/fit_volume.json`; weights `models/fuel_volume.json`.

### The risk backtest (`scripts/risk_backtest.py`)

The fire risk map's value (below) recomputed in Python, from what was known the year before each fire: fuel from
that year's embeddings, slope and facing from Mapterhorn, fires on record only from earlier years (leak check: 0).
The Python value matches the map's click card at the two spots compared. Same 200 fires and 600 starts as the fire
test; controls are all of the fire's own year, so only the ground differs. AUC per fire, mean over fires, 95 %
bootstrap over fires.

| Burned ground vs | Value | Fuel alone | Slope | Fuel within 500 m | Facing | Years since a fire |
| --- | --- | --- | --- | --- | --- | --- |
| unburned ground across the line | 0.599 [0.579, 0.620] | 0.575 [0.559, 0.592] | 0.571 | 0.576 | 0.518 | 0.495 |
| a chunk 5 to 10 km away, no fire that year | 0.586 [0.551, 0.624] | 0.575 [0.542, 0.607] | 0.561 | 0.538 | 0.515 | 0.487 |
| a chunk 20 to 30 km away, no fire that year | 0.652 [0.614, 0.691] | 0.610 [0.574, 0.645] | 0.583 | 0.595 | 0.524 | 0.512 |

- Across the line, value minus fuel on the same fires: +0.024 [+0.007, +0.040]. The value is above fuel alone in
  every fire year, 2020 to 2026 (by 0.02 to 0.04; each year's interval is wide).
- Slope and fuel nearby carry the gain; facing adds little and years since the last fire nothing. The four parts
  without fuel, averaged, score 0.578; multiplying by fuel adds 0.021 over that average.
- Older years of embeddings (2017 to the year before the fire) add nothing here: mean and lowest P(fuel) over the
  years score as one year does (0.577, 0.578), and the year-to-year embedding change (largest, total, last three
  years, years since the largest) sits at 0.49 to 0.53.
- Start points: the value scores 0.33 at the point and 0.46 within 100 m (fuel alone 0.31, 0.44), the same
  pattern as the fire test. Starts land on roads and town edges, where the value is low, so it says where fire
  spreads, not where it starts.

`results/risk_backtest.json`, `results/risk_backtest_starts.json`. Needs the fire test's data and
`fire_test_grids.py --need <parquet> --no-lf`; the steps are in the script's docstring.

### The agency comparison (`scripts/agency_compare.py`)

The backtest's fires, starts and controls, scored again with three agency maps beside the value:

- USFS Wildfire Risk to Communities burn probability (2024 edition, 30 m, simulated at 270 m)
- USFS Wildfire Hazard Potential (2023 edition, 270 m, continuous)
- CAL FIRE Fire Hazard Severity Zones (State Responsibility Area zones in force April 2024, Local zones recommended
  2024 to 2025)

Each is cut to the backtest's 10 m chunk grid. Every layer is scored on the same pixels as the value. The CAL FIRE
zones are scored only on State and Local ground (federal land is never zoned).

Both USFS layers describe the landscape at the end of 2020, so fires from 2021 on are after them. The 2020 fires
are already in their fuels, and there they score lower: burn probability 0.508 across the line for 2020 to 2022,
0.559 for 2023 to 2026. The CAL FIRE zones were drafted in 2022 (State) and 2024 (Local). The rows below are fires
from 2021 on.

| Burned ground vs | Value | Fuel alone | USFS burn probability | USFS hazard potential | CAL FIRE zones |
| --- | --- | --- | --- | --- | --- |
| unburned ground across the line (155 fires) | 0.594 [0.572, 0.618] | 0.570 | 0.563 [0.535, 0.589] | 0.542 | |
| the same, State and Local ground only (106 fires) | 0.604 [0.572, 0.638] | 0.583 | 0.580 [0.538, 0.626] | | 0.530 [0.515, 0.547] |
| a chunk 5 to 10 km away (152 fires) | 0.572 [0.531, 0.613] | 0.565 | 0.521 [0.459, 0.588] | 0.494 | |
| a chunk 20 to 30 km away (154 fires) | 0.635 [0.588, 0.682] | 0.597 | 0.626 [0.561, 0.695] | 0.575 | |

- Across the line, value minus burn probability on the same fires: +0.031 [-0.001, +0.065] from 2021 on;
  +0.062 [+0.031, +0.093] over all years, the 2020 fires included. Value minus the CAL FIRE zones: +0.074
  [+0.040, +0.111]. Value minus hazard potential: +0.052 [+0.029, +0.079].
- Across the line is a 10 m test, and a map simulated at 270 m, or a zone map, can't separate ground a few hundred
  meters apart. Against a chunk 20 to 30 km away, burn probability scores the same as the value (difference +0.008
  [-0.051, +0.068]).
- Start points: no map does better than chance against ground nearby or 5 to 30 km away. Within 100 m, burn
  probability 0.50 to 0.51, CAL FIRE zones 0.49 to 0.51, the value 0.44 to 0.46 (lower where starts sit on roads).
- Agreement (Spearman, 2,039 chunks of about 2.3 km): value and burn probability 0.63, value and CAL FIRE zone 0.65,
  burn probability and CAL FIRE zone 0.72. The value's median by zone: unzoned 13, moderate 55, high 60, very high
  61, federal ground 63. It separates zoned from unzoned ground, but hardly high from very high.
- Where they disagree most, by chunk rank:
  - The value is high and burn probability low on the wet north coast (Humboldt, Mendocino: full fuel, few fires
    on record) and on the 2020 fires (burned before the USFS landscape date).
  - Burn probability is high and the value low in Southern California grass and chaparral (San Diego border,
    Cajon Pass, the San Gabriel foothills), where P(fuel) is 0.4 to 0.6. Those places are CAL FIRE very high too.
  - The value has no wind or fire weather; both agency maps model them, and USFS also models where fires have
    started and how far they ran.

`results/agency_compare.json`. Needs the backtest's data; `uv run scripts/agency_compare.py fetch` reads the
layers per chunk (5 minutes), then `score`.

### Live fuel moisture (`scripts/make_samples.py`, `sample_*.py`, `fit_knn.py`)

The repo's first question: does AlphaEarth help predict live fuel moisture (National Fuel Moisture Database field
samples) on top of HLS and Sentinel-1? A little: kNN RMSE 34.2 to 33.4 moisture points statewide, GBM 34.0 to 31.9,
with modest skill overall (R2 0.2 to 0.5). `results/knn_aef_*.json`.

### The web map (`web/`)

React, MapLibre and deck.gl. Nothing is precomputed but the model weights and small file indexes; the embeddings,
the fire records and the Sentinel-2 imagery are read from their public stores when the view opens. Three pages:

**GOES triage** (the default page, `web/src/triage/`): map left, table right. Every fire GOES-West saw over
California in the Park Fire week (Jul 22 to Aug 2, 2024) fills the table as its scans are read, ranked by a
GOES-only score (`models/goes_score.json`, fit by `scripts/goes_score_fit.py`). Then each event's fuel under and
around it is read from the embeddings, top row first. Each event reads up to 24 mosaic chunks of 3 years, so the
whole week is tens of GB; the bottom rows may never fill.

**The fuel map** (`?app=fuel`):

- **The ground**: vegetation volume (the volume head, `web/src/fuel/volume.ts`, `volume-gpu.ts`) from the 10 m
  AlphaEarth Zarr mosaic, from zoom 11 in, on a log scale 0 to 50 m and not stretched, so years compare as drawn.
  `V` switches to P(fuel) (the first fuel head, stretched to the view). The panel shows the embeddings year large
  (2019 to 2025); `[` and `]` step it. The volume head reads that year alone; the fuel head that year and the two
  before. Hovering reads the volume, the leading lifeform and its height, canopy cover, the fuel kind and the volume
  two years before.
- **Burned areas** (`web/src/perimeters.ts`): every California wildfire of 100 acres and up since 2020 (about 590),
  read live from NIFC's WFIGS perimeter service, outlines simplified to about 30 m (about 4 MB). Bright cyan, which
  none of the fuel ramps holds; outlines only. `B` shows and hides them. `M` (or "Only inside burned areas") draws
  the fuel only inside them: the outlines filled as a deck.gl MaskExtension mask on the GPU, the rest left to the
  basemap. Hovering lists every fire that burned the
  spot (name, year, acres), newest first.
- **A fire's fuel by year** (`web/src/fuel/burnHist.ts`, `web/src/ui/FireHist.tsx`): clicking inside an outline
  selects that fire (the newest where they overlap; Esc clears). The panel shows one bar per embeddings year, 2019 to
  2025: the average volume (or P(fuel)) over the outline's pixels in the mosaic chunks in view. The year on screen is
  read first, then the others chunk by chunk, bars growing faint until complete. Caldor near Grizzly Flats, zoom 13,
  volume: 11, 13 m (2019, 2020), 4.0 (fire year), 4.2, 3.1, 2.8, 2.5 m (2025). The same in P(fuel), Park Fire near
  Cohasset: 94, 94, 95, 96, 94, 95 (fire year), 91 %: P(fuel) asks whether ground can burn, not how much is on it.
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

**The fire risk map** (`?app=risk`, `web/src/risk/`): where fuel is ready to burn, and why. From zoom 12, a value
per 10 m pixel for the area in view, drawn in viridis:

    value = fuel x (fuel within 500 m + slope + facing + years since a fire) / 4

- Fuel is the fuel head on the embeddings; fuel within 500 m the share of fuel around the pixel. Slope (full at 30
  degrees) and facing (south-west most, in step with slope up to 10 degrees) come from Mapterhorn's GLO-30 tiles,
  cleaned with a 5 x 5 median (drops single-cell tree and building spikes) and a Gaussian, then blended to 10 m.
  Years since a fire (full at 20) come from NIFC's all-years perimeter history before 2020 and WFIGS from 2020.
- Clicking a spot lists the parts behind its value. Overture draws over the picture by kind: roads, water, power
  lines and infrastructure, parks and protected land (hover anywhere inside for its name and the value there), and
  from zoom 14 buildings. CARTO's base map sits under the picture, so it shows only where the value has not loaded;
  its labels stay on top.
- Checked against past fires in the risk backtest above.

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
NIFC WFIGS incident locations and perimeters, NIFC InterAgency Fire Perimeter History, NOAA GOES-18 ABI on AWS,
Overture Maps (buildings, roads, water, land use, infrastructure), CARTO's dark base map, NAIP, the National Fuel
Moisture Database, HLS and Sentinel-1 (for the moisture work), NOAA HRRR via dynamical.org, NASA FIRMS VIIRS S-NPP,
CAL FIRE incidents, USFS FACTS prescribed burns, Mapterhorn terrain tiles, USFS Wildfire Risk to Communities burn
probability and Wildfire Hazard Potential (imagery.geoplatform.gov), CAL FIRE Fire Hazard Severity Zones and
Responsibility Areas.

Python scripts run with `uv run` (dependencies inline in each script).

## Inspirations

- [goes-fdc-catalog](https://github.com/mmidzik/goes-fdc-catalog): GOES fire pixels (ABI FDC) as virtual Zarr
  references into NOAA's bucket. This repo builds its own references the same way (`scripts/goes_fire_refs.py`).
- [CTrees FuelWatch](https://fuelwatch.ctrees.org/): a live fuel moisture map on fuel groups. The fuel map here
  starts from the same idea, with the fuel read from embeddings.
