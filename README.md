# Chhaya: HeatMap-to-Tree

A decision-support prototype that recommends where a city should plant trees first, so a limited planting budget gives the most cooling to the most people. Built around Ghaziabad, Uttar Pradesh.

**Live demo:** https://claude.ai/artifact/2gSd17jQ4nrMF76bno768G

## Problem

Parts of Delhi-NCR get much hotter than others because of concrete, heavy traffic and little shade, and the region also suffers severe winter air pollution. Tree-planting drives are often placed wherever land is easy to get, not where trees help most.

## What it does

1. Builds a grid of the city with vegetation (NDVI), built-up area, water proximity, population and surface temperature.
2. Trains a regression model that predicts surface temperature from those layers.
3. Simulates "what if this pixel had tree cover?" and measures the predicted cooling.
4. Scores each pixel as cooling x people benefiting, then greedily picks the best sites under a tree budget, keeping a minimum spacing between them.

Extras in the web prototype:
- Extra priority near schools and hospitals
- An "After planting" layer with assumed spillover cooling to nearby pixels
- A zone report showing the hottest zones before and after the plan
- Clean-air mode: favours sites beside busy roads and recommends a hedge-plus-tree mix there
- A button that asks Claude to write a short brief for a municipal commissioner, and a CSV export

## Files

- `heatmap-to-tree.html`: the interactive prototype (open it in a browser)
- `heatmap_to_tree.py`: Python pipeline (random forest, greedy site selection, PNG/CSV/map output). Run `python heatmap_to_tree.py` for the synthetic demo. The `--real` option reads Landsat and WorldPop data from Google Earth Engine and has not been tested yet.

## Limits (please read)

- **The data is simulated.** The city in the prototype is generated around Ghaziabad. Results show how the method works, not real conditions in the city. Real Earth Engine data is the next step.
- Cooling is land surface temperature in the planted pixel, which is larger than the air temperature change people feel.
- CO2 figures assume 25 trees per site and 21 kg per tree per year. They are rough estimates.
- Spillover cooling, road effects and school/hospital priority are modelling assumptions, not measured findings.
- Trees reduce roadside particle exposure only modestly and depend on design. This project does not claim to lower city-wide AQI.

## Related work

Inspired by American Forests' Tree Equity Score and Google's Tree Canopy Lab. This prototype adds budgeted, spaced site selection, a before-and-after view and a clean-air roadside mode, aimed at an Indian city.

## AI assistance

Built with assistance from Claude (Anthropic) and adapted by the author.
