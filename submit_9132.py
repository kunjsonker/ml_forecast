import pandas as pd
import numpy as np
import pygeohash as pgh
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

print("--- INITIATING THE FINAL 91.32 SEED AVERAGING PIPELINE ---")

# 1. Load Data & The Best Pseudo-Truth
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# IMPORTANT: Make sure this file exists in your folder!
best_sub = pd.read_csv('pseudo_labeled_submission.csv') 

# 2. Build Massive Dataset (119,000 Rows)
print("Expanding dataset with high-confidence pseudo-labels...")
pseudo_test = test.copy()
pseudo_test['demand'] = best_sub['demand']
massive_train = pd.concat([train, pseudo_test], axis=0).reset_index(drop=True)
n_train = massive_train.shape[0]

combined = pd.concat([massive_train.drop('demand', axis=1), test], axis=0).reset_index(drop=True)

# --- 3. FEATURE ENGINEERING ---
print("Applying Spatial-Temporal features...")
combined['lat'] = combined['geohash'].apply(lambda x: pgh.decode(x)[0])
combined['lon'] = combined['geohash'].apply(lambda x: pgh.decode(x)[1])

geohash_map = massive_train.groupby('geohash')['demand'].mean().to_dict()
combined['historical_local_demand'] = combined['geohash'].map(geohash_map).fillna(massive_train['demand'].mean())

def time_to_minutes(t_str):
    try:
        h, m = t_str.split(':')
        return int(h) * 60 + int(m)
    except:
        return 0

combined['time_in_minutes'] = combined['timestamp'].apply(time_to_minutes)
combined['sin_time'] = np.sin(2 * np.pi * combined['time_in_minutes'] / 1440)
combined['cos_time'] = np.cos(2 * np.pi * combined['time_in_minutes'] / 1440)
combined['day_of_week'] = combined['day'] % 7

combined['Temperature'] = combined['Temperature'].fillna(combined['Temperature'].median())
combined['RoadType'] = combined['RoadType'].fillna('Unknown')
combined['Weather'] = combined['Weather'].fillna('Unknown')

categorical_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

features = ['lat', 'lon', 'historical_local_demand', 'day', 'day_of_week',
            'sin_time', 'cos_time', 'RoadType', 'NumberofLanes',
            'LargeVehicles', 'Landmarks', 'Temperature', 'Weather']

X_massive = combined.iloc[:n_train][features]
y_massive = massive_train['demand']
X_final_test = combined.iloc[n_train:][features]

# --- 4. THE SEED LOOP (Crushing Variance) ---
seeds = [42, 2024, 777] # Three different random states
all_predictions = []

for i, seed in enumerate(seeds):
    print(f"\nTraining Stack {i+1}/3 with Random Seed: {seed}...")
    
    estimators = [
        ('xgb', xgb.XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=7, random_state=seed, n_jobs=-1)),
        ('lgb', lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, max_depth=7, random_state=seed, n_jobs=-1, verbose=-1))
    ]
    
    stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=5, n_jobs=-1)
    stacking_model.fit(X_massive, y_massive)
    
    pred = stacking_model.predict(X_final_test)
    all_predictions.append(pred)

# --- 5. AVERAGE THE PREDICTIONS ---
print("\nBlending all seed predictions together...")
final_averaged_predictions = np.mean(all_predictions, axis=0)
test['prediction'] = final_averaged_predictions

# --- 6. RE-INJECT EXACT LEAKS ---
print("Applying the 268 perfect training matches...")
features_to_match = ['geohash', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Temperature', 'Weather']
leak_mapping = train.groupby(features_to_match)['demand'].mean().reset_index()
leak_mapping = leak_mapping.rename(columns={'demand': 'perfect_demand'})
test_with_leak = pd.merge(test, leak_mapping, on=features_to_match, how='left')

perfect_mask = test_with_leak['perfect_demand'].notna()
test.loc[perfect_mask, 'prediction'] = test_with_leak.loc[perfect_mask, 'perfect_demand']

# --- 7. SAVE ---
submission = pd.DataFrame({
    'Index': test['Index'].astype(int), 
    'demand': test['prediction'].clip(lower=0.0)
})

submission.to_csv('OFFICIAL_9132_SUBMISSION.csv', index=False)
print("\n[SUCCESS] Saved 'OFFICIAL_9132_SUBMISSION.csv'.")