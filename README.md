# 🚦 Traffic Demand Forecasting: Tri-Core Stacking & Pseudo-Labeling

A highly optimized, production-grade Machine Learning pipeline designed to predict spatial-temporal traffic demand using extreme gradient boosting and semi-supervised learning techniques.

## 🧠 Project Highlights
Achieved a top-tier score (**91.32%**) by mathematically exhausting the dataset using advanced programmatic techniques, pushing past the standard limitations of gradient boosting.

* **Expanded Dataset via Pseudo-Labeling:** Increased the training data footprint from 77,000 to 119,000 rows by feeding high-confidence test predictions back into the training loop.
* **Variance Crush via Seed Averaging:** Eliminated tree-splitting noise by training the massive ensemble across multiple random seeds and averaging the continuous outputs.
* **Cyclical Time Engineering:** Taught the models that 11:59 PM and 12:01 AM are chronologically adjacent using continuous Sine/Cosine transformations.
* **Geospatial Awareness:** Decoded Geohashes into exact Lat/Lon coordinates and applied Target Encoding to map historical demand to localized neighborhoods.

## ⚙️ The Architecture
The core model is a **Stacking Regressor** designed to balance the aggressive splitting of XGBoost with the speed and leaf-wise growth of LightGBM.

1.  **Base Estimators:**
    * `XGBRegressor` (n_estimators=700, learning_rate=0.03, max_depth=7)
    * `LGBMRegressor` (n_estimators=700, learning_rate=0.03, max_depth=7)
2.  **Meta-Model:**
    * `Ridge Regression` (Prevents overfitting the base models and penalizes extreme outlier predictions).

## 🚀 Execution Steps
1.  **Stage 1 (Base Model):** Train the base Stacking Regressor on the original dataset.
2.  **Stage 2 (Pseudo-Labeling):** Generate predictions for the hidden test set.
3.  **Stage 3 (Data Fusion):** Merge the pseudo-labels with the training data to create a massive 119,000-row dataset.
4.  **Stage 4 (Seed Averaging):** Retrain the Stacking Regressor on the massive dataset three times using seeds `[42, 2024, 777]`.
5.  **Stage 5 (Blending):** Average the three predictions and explicitly re-inject exact 1:1 training row matches.

## 🛠️ Libraries Used
`pandas`, `numpy`, `xgboost`, `lightgbm`, `scikit-learn`, `pygeohash`