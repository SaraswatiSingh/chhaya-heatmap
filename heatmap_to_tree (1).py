"""
HeatMap-to-Tree: where should a city plant trees first?

Pipeline
  1. Get a grid of the city: vegetation (NDVI), built-up index, water index,
     population, and land surface temperature (LST).
  2. Train a Random Forest to predict LST from the other layers.
  3. "What if this pixel had tree cover?" -> predict LST again with NDVI raised
     and measure the cooling.
  4. Score = cooling x population, then greedily pick the best N pixels
     (with a minimum spacing so picks are spread out).
  5. Save a ranked CSV, a PNG overview, and (if folium is installed) an
     interactive HTML map.

Run:   python heatmap_to_tree.py            -> synthetic demo, works offline
       python heatmap_to_tree.py --real     -> real Landsat/WorldPop data
                                               (needs Google Earth Engine, see
                                               load_real_data below)
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------------- settings
CENTER_LAT, CENTER_LON = 28.6692, 77.4538   # Ghaziabad (change to your city)
GRID = 60                                   # 60 x 60 pixels
PIXEL_M = 100                               # each pixel ~100 m
N_TREES_TO_PICK = 40                        # budget: number of planting sites
MIN_SPACING = 3                             # pixels between picked sites
TARGET_NDVI = 0.6                           # NDVI a planted pixel would reach
TREES_PER_PIXEL = 25                        # ASSUMPTION: trees per site
CO2_KG_PER_TREE_YR = 21                     # ASSUMPTION: typical urban tree
FEATURES = ["ndvi", "built_up", "water_index"]
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- data
def make_synthetic_city(n=GRID):
    """Fake but realistic-looking city so the project runs without internet."""
    def field(sigma):
        f = gaussian_filter(RNG.normal(size=(n, n)), sigma)
        return (f - f.min()) / (f.max() - f.min())

    built_up = np.clip(field(6) * 1.3 - 0.1, 0, 1)
    # a river running diagonally
    yy, xx = np.mgrid[0:n, 0:n]
    river = np.abs((yy - 0.6 * xx - 8)) < 1.5
    dist = distance_transform_edt(~river)
    water_index = np.exp(-dist / 6)
    ndvi = np.clip(0.75 * (1 - built_up) + RNG.normal(0, 0.05, (n, n)), 0.02, 0.9)
    ndvi[river] = 0.0
    built_up[river] = 0.0
    population = built_up * 800 * (0.6 + 0.8 * field(3))   # people per pixel
    population[river] = 0
    # "true" physics used to fake satellite temperature
    lst = (28 + 11 * built_up - 9 * ndvi - 3 * water_index
           + gaussian_filter(RNG.normal(0, 1.2, (n, n)), 2))
    return pd.DataFrame({
        "row": yy.ravel(), "col": xx.ravel(),
        "ndvi": ndvi.ravel(), "built_up": built_up.ravel(),
        "water_index": water_index.ravel(), "population": population.ravel(),
        "lst": lst.ravel(), "is_water": river.ravel(),
    })


def load_real_data(n=GRID):
    """
    Real data from Google Earth Engine (UNTESTED here: I had no internet).
    Setup in Google Colab:
        !pip install earthengine-api
        import ee; ee.Authenticate(); ee.Initialize(project="YOUR_PROJECT")
    Then run:  python heatmap_to_tree.py --real   (or paste into a notebook)
    """
    import ee
    ee.Initialize()
    half = n * PIXEL_M / 2 / 111_000                      # degrees, rough
    region = ee.Geometry.Rectangle([CENTER_LON - half, CENTER_LAT - half,
                                    CENTER_LON + half, CENTER_LAT + half])
    # hot-season composite (April-June 2024) from Landsat 9 Collection 2 L2
    col = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
           .filterBounds(region).filterDate("2024-04-01", "2024-06-30")
           .filter(ee.Filter.lt("CLOUD_COVER", 20)).median())
    sr = col.select(["SR_B3", "SR_B4", "SR_B5", "SR_B6"]).multiply(0.0000275).add(-0.2)
    green, red, nir, swir = (sr.select(b) for b in ["SR_B3", "SR_B4", "SR_B5", "SR_B6"])
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
    ndbi = swir.subtract(nir).divide(swir.add(nir)).rename("built_up")
    mndwi = green.subtract(swir).divide(green.add(swir)).rename("water_index")
    lst = col.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("lst")
    pop = (ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(region)
           .filter(ee.Filter.eq("year", 2020)).mosaic().rename("population"))
    stack = ee.Image.cat([ndvi, ndbi, mndwi, lst, pop]).unmask(0).clip(region)
    arr = stack.reproject("EPSG:4326", scale=PIXEL_M).sampleRectangle(region, defaultValue=0).getInfo()
    props = arr["properties"]
    h, w = np.array(props["ndvi"]).shape
    yy, xx = np.mgrid[0:h, 0:w]
    df = pd.DataFrame({k: np.array(props[k]).ravel() for k in
                       ["ndvi", "built_up", "water_index", "lst", "population"]})
    df["row"], df["col"] = yy.ravel(), xx.ravel()
    df["is_water"] = df["water_index"] > 0.1
    df.attrs["shape"] = (h, w)
    return df


# ---------------------------------------------------------------- model
def train_model(df):
    X, y = df[FEATURES], df["lst"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                  n_jobs=-1, random_state=0).fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"Model test R2 = {r2_score(yte, pred):.3f} | "
          f"MAE = {mean_absolute_error(yte, pred):.2f} deg C")
    imp = dict(zip(FEATURES, model.feature_importances_.round(3)))
    print("Feature importance:", imp)
    return model


def simulate_cooling(df, model):
    """Predicted drop in LST if each pixel were turned into tree cover."""
    before = model.predict(df[FEATURES])
    after_X = df[FEATURES].copy()
    after_X["ndvi"] = np.maximum(after_X["ndvi"], TARGET_NDVI)
    after = model.predict(after_X)
    df = df.copy()
    df["cooling_c"] = np.clip(before - after, 0, None)
    return df


# ---------------------------------------------------------------- optimiser
def pick_sites(df, n_pick=N_TREES_TO_PICK, spacing=MIN_SPACING):
    """Greedy: highest (cooling x people benefiting) first, keep sites apart."""
    cand = df[(~df["is_water"]) & (df["ndvi"] < 0.3)].copy()
    cand["score"] = cand["cooling_c"] * cand["population"]
    cand = cand.sort_values("score", ascending=False)
    chosen = []
    for _, r in cand.iterrows():
        if all(max(abs(r.row - c.row), abs(r.col - c.col)) >= spacing for c in chosen):
            chosen.append(r)
        if len(chosen) == n_pick:
            break
    out = pd.DataFrame(chosen).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out["co2_kg_per_year"] = TREES_PER_PIXEL * CO2_KG_PER_TREE_YR
    return out


# ---------------------------------------------------------------- outputs
def to_latlon(row, col, shape):
    h, w = shape
    dlat = PIXEL_M / 111_000
    dlon = PIXEL_M / (111_000 * np.cos(np.radians(CENTER_LAT)))
    return (CENTER_LAT + (h / 2 - row) * dlat, CENTER_LON + (col - w / 2) * dlon)


def save_outputs(df, picks, shape):
    h, w = shape
    grid = lambda c: df.sort_values(["row", "col"])[c].to_numpy().reshape(h, w)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    im0 = ax[0].imshow(grid("lst"), cmap="inferno")
    ax[0].set_title("Land surface temperature (deg C)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    sc = grid("cooling_c") * grid("population")
    im1 = ax[1].imshow(sc, cmap="YlGn")
    ax[1].set_title("Planting benefit score\n(cooling x people)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    ax[2].imshow(grid("lst"), cmap="inferno", alpha=0.55)
    ax[2].scatter(picks["col"], picks["row"], c="lime", edgecolor="k", s=60)
    for _, r in picks.head(10).iterrows():
        ax[2].text(r.col + 0.6, r.row, int(r["rank"]), color="white", fontsize=8)
    ax[2].set_title(f"Top {len(picks)} sites to plant first")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig("heatmap_to_tree_overview.png", dpi=150)

    picks = picks.copy()
    picks[["lat", "lon"]] = [to_latlon(r, c, shape) for r, c in zip(picks.row, picks.col)]
    picks[["rank", "lat", "lon", "lst", "ndvi", "population", "cooling_c",
           "score", "co2_kg_per_year"]].round(4).to_csv("planting_sites.csv", index=False)

    try:
        import folium
        m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=14, tiles="CartoDB positron")
        for _, r in picks.iterrows():
            folium.CircleMarker(
                [r.lat, r.lon], radius=7, color="darkgreen", fill=True, fill_opacity=0.8,
                popup=(f"Rank {int(r['rank'])}<br>Cooling: {r.cooling_c:.2f} C<br>"
                       f"People nearby: {int(r.population)}<br>"
                       f"CO2 absorbed: ~{int(r.co2_kg_per_year)} kg/yr")).add_to(m)
        m.save("planting_map.html")
        print("Saved planting_map.html")
    except ImportError:
        print("folium not installed -> skipped HTML map (pip install folium)")

    total_cool = picks["cooling_c"].mean()
    print(f"\nTop {len(picks)} sites: avg local cooling ~{total_cool:.2f} C, "
          f"~{int(picks['co2_kg_per_year'].sum()):,} kg CO2/yr absorbed, "
          f"~{int(picks['population'].sum()):,} people in the pixels benefiting")
    print("Saved heatmap_to_tree_overview.png and planting_sites.csv")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use Google Earth Engine data")
    args = ap.parse_args()
    df = load_real_data() if args.real else make_synthetic_city()
    shape = df.attrs.get("shape", (GRID, GRID))
    model = train_model(df)
    df = simulate_cooling(df, model)
    picks = pick_sites(df)
    save_outputs(df, picks, shape)
