# NAIP check of the fuel head (mlp64_e_sd_kinds_2yr_ens5, 1 % fuel hidden vs LANDFIRE)

475 of 475 picks labeled. NAIP 2022 vs LANDFIRE 2024 / AEF 2024. Labeler agreement: c10_burn: same call 76%, never yes-vs-no 86% (n 51); c30_burn: same call 78%, never yes-vs-no 84% (n 51).

## 10 m AEF cell

| group             |   n |   yes |   mixed |   no |   unsure |   naip_burnable |
|:------------------|----:|------:|--------:|-----:|---------:|----------------:|
| hidden_grass_near |  50 |    16 |       8 |   24 |        2 |            0.42 |
| hidden_grass_far  |  50 |    17 |       7 |   21 |        5 |            0.46 |
| hidden_woody_near |  50 |     3 |      11 |   32 |        4 |            0.18 |
| hidden_woody_far  |  50 |     2 |       2 |   45 |        1 |            0.06 |
| missed_urban      |  50 |    18 |      15 |   16 |        1 |            0.52 |
| missed_farm       |  50 |    13 |      14 |   22 |        1 |            0.41 |
| missed_bare       |  50 |     4 |       4 |   41 |        1 |            0.12 |
| missed_water_snow |  25 |    13 |       5 |    4 |        3 |            0.7  |
| agree_fuel        |  50 |    34 |       4 |   10 |        2 |            0.75 |
| agree_nonfuel     |  50 |     8 |       6 |   31 |        5 |            0.24 |

- Head's hidden fuel that NAIP shows is NOT burnable (LANDFIRE wrong): 67%
- Head's missed non-fuel that NAIP shows IS burnable (LANDFIRE wrong): 31%
- Controls: both-say-fuel that NAIP calls not burnable 25%; both-say-not-fuel that NAIP calls burnable 24%
- Re-scored on NAIP truth at the same cutoff: fuel hidden 7.8% (90 % 5.0% to 10.8%), non-fuel caught 35% (29% to 42%). Against LANDFIRE: 1.1 % and 58 %.

## 30 m LANDFIRE cell

| group             |   n |   yes |   mixed |   no |   unsure |   naip_burnable |
|:------------------|----:|------:|--------:|-----:|---------:|----------------:|
| hidden_grass_near |  50 |    14 |      10 |   25 |        1 |            0.39 |
| hidden_grass_far  |  50 |    16 |       8 |   20 |        6 |            0.45 |
| hidden_woody_near |  50 |     5 |      13 |   29 |        3 |            0.24 |
| hidden_woody_far  |  50 |     2 |       3 |   44 |        1 |            0.07 |
| missed_urban      |  50 |    19 |      15 |   16 |        0 |            0.53 |
| missed_farm       |  50 |    13 |      12 |   25 |        0 |            0.38 |
| missed_bare       |  50 |     3 |       4 |   42 |        1 |            0.1  |
| missed_water_snow |  25 |    15 |       5 |    3 |        2 |            0.76 |
| agree_fuel        |  50 |    32 |       9 |    7 |        2 |            0.76 |
| agree_nonfuel     |  50 |     7 |       9 |   29 |        5 |            0.26 |

- Head's hidden fuel that NAIP shows is NOT burnable (LANDFIRE wrong): 67%
- Head's missed non-fuel that NAIP shows IS burnable (LANDFIRE wrong): 31%
- Controls: both-say-fuel that NAIP calls not burnable 24%; both-say-not-fuel that NAIP calls burnable 26%
- Re-scored on NAIP truth at the same cutoff: fuel hidden 8.1% (90 % 5.5% to 10.9%), non-fuel caught 36% (30% to 42%). Against LANDFIRE: 1.1 % and 58 %.

## Reading it (10 m cell)

- Labelers: 6 blind helper agents, 51 ids labeled twice: same call 76 %, never yes-vs-no 86 %. Three Salton Sea / Mono Lake
  chips came back as flat fill and were set to water by location (labels/aa_fix.csv).
- Hidden fuel: 67 % not burnable on the photo. Most of that is creosote desert scrub (EVT 7087, FBFM SH1): 9 % of fuel
  but 28 % of hidden fuel; the head hides 3.2 % of it vs 0.84 % of other fuel. On the photo it is mostly bare sand with
  scattered shrubs. Counting sparse desert as burnable instead: 41 % of hidden fuel is not burnable.
- Hidden grass is about half real: 42 to 46 % burnable on the photo. This is the head's own miss.
- Missed non-fuel: bare is mostly a head error (12 % burnable); urban 52 % and farmland 41 % burnable (yard trees,
  fallow / dry fields, orchards with grassy rows); LANDFIRE water 70 % burnable (shore grass, marsh, riparian; 2022 was
  a drought year with low reservoirs).
- Re-scored hidden fuel on NAIP (7.8 %) is almost all from the "both say not-fuel" group: 24 % of it is burnable on the
  photo (dry fields, orchards, yards). LANDFIRE counts all farmland and urban as not burnable, and the head learned that.
  With sparse desert counted burnable: hidden 7.5 %, caught 54 %.
