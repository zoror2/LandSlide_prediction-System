"""
Data Preparation Script — Real Sikkim Landslide Dataset
========================================================
Combines the two real Zenodo CSVs (point + polygon inventories),
generates realistic non-landslide samples (Class 0) for the same
region, and outputs a single training-ready CSV.

Run this once:
    python prepare_real_data.py

Output:
    landslide_real_sikkim.csv   (ready to use in train_landslide_models.py)
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
FOLDER   = r"c:\Users\Admin\OneDrive\Desktop\Landslide_1\Real_data(NOT SYNTHETIC)"
POINT    = os.path.join(FOLDER, "Google_Earth_landslides_point_21Dec2021.csv")
POLYGON  = os.path.join(FOLDER, "Google_Earth_landslides_polygon_21Dec2021.csv")
OUT_CSV  = os.path.join(FOLDER, "landslide_real_sikkim.csv")

# ==============================================================================
# 1. LOAD BOTH REAL CSVs (Class 1 — actual landslide events)
# ==============================================================================
print("Loading real landslide data ...")

point   = pd.read_csv(POINT)
polygon = pd.read_csv(POLYGON)

# Keep only the shared useful columns
shared_cols = ["Name", "Slope", "Aspect", "Curvature", "Elevation", "Geology", "Extent"]
point_clean   = point[shared_cols].copy()
polygon_clean = polygon[shared_cols].copy()

# Combine
landslide_df = pd.concat([point_clean, polygon_clean], ignore_index=True)
landslide_df["Landslide"] = 1          # All real events → Class 1

print(f"  Real landslide events (Class 1): {len(landslide_df)} rows")

# ==============================================================================
# 2. UNDERSTAND THE REAL DATA DISTRIBUTIONS
# ==============================================================================
slope_mean  = landslide_df["Slope"].mean()
slope_std   = landslide_df["Slope"].std()
elev_mean   = landslide_df["Elevation"].mean()
elev_std    = landslide_df["Elevation"].std()
aspect_mean = landslide_df["Aspect"].mean()
aspect_std  = landslide_df["Aspect"].std()

print(f"\n  Real data statistics:")
print(f"    Slope     : mean={slope_mean:.1f}°  std={slope_std:.1f}°")
print(f"    Elevation : mean={elev_mean:.0f}m   std={elev_std:.0f}m")
print(f"    Aspect    : mean={aspect_mean:.1f}°  std={aspect_std:.1f}°")

geology_values = landslide_df["Geology"].dropna().unique().tolist()
extent_values  = landslide_df["Extent"].dropna().unique().tolist()
name_values    = landslide_df["Name"].dropna().unique().tolist()

print(f"    Geology   : {geology_values}")
print(f"    Extents   : {extent_values}")

# ==============================================================================
# 3. GENERATE REALISTIC NON-LANDSLIDE SAMPLES (Class 0)
#
#    Non-landslide areas in Sikkim tend to have:
#      - Lower slope angles (flat to gentle — valley floors, plains)
#      - Similar elevation range (same region)
#      - Same geology types (same region)
#    We sample slope from a lower distribution (mean ~12°, capped at 20°)
# ==============================================================================
n_safe = len(landslide_df)   # Match class sizes (balanced dataset)
print(f"\nGenerating {n_safe} non-landslide samples (Class 0) ...")

safe_slope     = np.clip(np.random.normal(loc=12.0, scale=5.0, size=n_safe), 0, 22)
safe_elevation = np.clip(
    np.random.normal(loc=elev_mean * 0.75, scale=elev_std * 0.8, size=n_safe),
    200, elev_mean * 1.1
)
safe_aspect    = np.random.uniform(0, 360, size=n_safe)
safe_curvature = np.random.normal(loc=0, scale=5e6, size=n_safe)  # flatter terrain
safe_geology   = np.random.choice(geology_values, size=n_safe)
safe_extent    = np.random.choice(extent_values,  size=n_safe)
safe_name      = np.random.choice(["No landslide", "Stable slope"], size=n_safe)

safe_df = pd.DataFrame({
    "Name"      : safe_name,
    "Slope"     : safe_slope,
    "Aspect"    : safe_aspect,
    "Curvature" : safe_curvature,
    "Elevation" : safe_elevation,
    "Geology"   : safe_geology,
    "Extent"    : safe_extent,
    "Landslide" : 0,
})

print(f"  Non-landslide samples (Class 0): {len(safe_df)} rows")

# ==============================================================================
# 4. COMBINE AND ENCODE
# ==============================================================================
df = pd.concat([landslide_df, safe_df], ignore_index=True)

# Encode Geology as one-hot (replaces Soil_Type columns)
geology_dummies = pd.get_dummies(df["Geology"], prefix="Geology")
df = pd.concat([df, geology_dummies], axis=1)

# Encode Extent as one-hot
extent_dummies = pd.get_dummies(df["Extent"], prefix="Extent")
df = pd.concat([df, extent_dummies], axis=1)

# Drop original text columns
df.drop(columns=["Name", "Geology", "Extent"], inplace=True)

# Clean up curvature (very large numbers — normalise to readable range)
df["Curvature"] = df["Curvature"] / 1e6

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset:")
print(f"  Total rows : {len(df)}")
print(f"  Columns    : {df.columns.tolist()}")
print(f"  Class dist : {df['Landslide'].value_counts().to_dict()}")
print(f"\n  Sample rows:")
print(df.head(5).to_string())

# ==============================================================================
# 5. SAVE
# ==============================================================================
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")
print("\nDone! Now update train_landslide_models.py with the new column names.")
print("See the column list printed above.")
