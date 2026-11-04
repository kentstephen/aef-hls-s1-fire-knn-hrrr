# NAIP check: how to label a panel

Each panel (data/naip_check/chips/<id>.png) shows one spot in California from the 2022 NAIP aerial photos:

1. Close-up, natural color, 120 m across. Yellow outline = the 10 m AEF cell (the one the head scores).
   Blue outline = the 30 m LANDFIRE cell around it.
2. The same close-up as greenness (NDVI): white = no live plants, darker green = more live plants.
   Dry grass and bare soil both look pale here; use the color photo and texture to tell them apart.
3. Context, natural color, 600 m across. White outline = the close-up area, yellow = the LANDFIRE cell.

Labels are blind: don't guess what LANDFIRE or the model said. Judge only what the photo shows.

## Fields, one row per panel

| field | values |
|---|---|
| id | the chip id printed top left (n000 ...) |
| c10_cover | dominant cover inside the yellow 10 m cell (see list) |
| c10_veg | share of the yellow cell covered by plants, live or dry: 0, low (<25 %), mid (25 to 75 %), high (>75 %) |
| c10_burn | yes / no / mixed / unsure |
| c30_cover | dominant cover inside the blue 30 m cell |
| c30_veg | as c10_veg, for the blue cell |
| c30_burn | yes / no / mixed / unsure |
| conf | high / med / low: how sure you are of c10_burn |
| note | a few words: what is there (e.g. "dry grass verge beside road", "orchard rows", "roof and lawn") |

Cover list: grass, shrub, trees, crop_green, crop_dry, orchard_vineyard, wetland, bare, built, water, snow, mixed.
"built" = roofs, pavement, gravel lots, rail. "bare" = rock, sand, bare soil, dirt roads, quarries.
"crop_dry" = harvested stubble, fallow or dry field. "crop_green" = growing irrigated crop.

## Burnable or not (c10_burn, c30_burn)

The question: could a wildfire spread through this cell?

- yes: plants (live or dry) cover most of the cell and could carry fire: grass (green or cured), shrubs, trees,
  dry fields, stubble, weedy lots, overgrown verges, yards that are mostly trees or brush.
- no: most of the cell is roofs, pavement, bare rock / soil / sand, water, snow, or a green irrigated crop,
  irrigated lawn or flooded field.
- mixed: roughly half burnable, half not (a road edge, a roof half in the cell, a field boundary).
- unsure: the photo can't tell (shadow, cloud, blur, can't tell dry grass from bare soil).

Orchards and vineyards: "yes" if there is grass or brush between the rows, "mixed" if bare tilled soil between rows.
Wetland or marsh vegetation: "yes" if it is standing reeds or grass above water, "no" if mostly open water.

Output: CSV with header `id,c10_cover,c10_veg,c10_burn,c30_cover,c30_veg,c30_burn,conf,note`, one row per id, the
note in double quotes.
