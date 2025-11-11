import pandas as pd
import nfl_data_py as nfl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

os.chdir(r'C:\Users\rfo7799\Desktop\Git\TetheredAI\NFL\PROD')

pd.set_option('display.max_columns', None)

import sqlite3
import pandas as pd
DATABASE_NAME = 'odds.db'
#SOURCE_DB = r'C:\Users\rfo7799\Downloads\odds.db'

conn = sqlite3.connect(DATABASE_NAME)
table_name = 'player_props'
prop_df = pd.read_sql(f"""
                 SELECT * FROM {table_name} WHERE market_type = 'player_pass_yds'
                 --AND lower(event_name) like '%tit%'
                 """, conn)
#prop_df

# Select only the relevant columns
columns = ['passer_player_name', 'game_id', 'posteam', 'defteam', 'season', 'week', 'home_team', 'away_team', 'play_type', 'air_yards', 
           'yards_after_catch', 'epa', 'complete_pass', 'incomplete_pass', 'interception', 'qb_hit', 'sack', 'pass_touchdown',
           'passing_yards', 'cpoe', 'roof', 'surface','drive','game_seconds_remaining']

# Loading in the NFL pbp data
data = nfl.import_pbp_data(range(2010,2026), downcast=True)
data = data[columns]

# nfl-data-py still loads other columns, so we again need to set our data equal to only the columns we want
data = data[columns]

# Drop all rows that are not a pass
data = data[(data['play_type'] == 'pass')]

# Drop the play type column
passer_data = data.drop(columns=['play_type'])

# Group the data together by passer, week, season and aggregate
passer_df = passer_data.groupby(['passer_player_name', 'game_id', 'week', 'season'], as_index=False).agg(
    {'posteam' : 'first',
     'defteam' : 'first',
     'home_team' : 'first',
     'away_team' : 'first',
     'air_yards' : 'sum',
     'yards_after_catch' : 'sum',
     'epa' : 'sum',
     'complete_pass' : 'sum',
     'incomplete_pass' : 'sum',
     'interception' : 'sum',
     'qb_hit' : 'sum',
     'sack' : 'sum',
     'pass_touchdown' : 'sum',
     'passing_yards' : 'sum',
     'cpoe' : 'mean',
     'roof' : 'first',
     'surface' : 'first'
     }
)

# Create a new column that is completion percentage
passer_df['completion_percentage'] = passer_df['complete_pass'] / (passer_df['complete_pass'] + passer_df['incomplete_pass'])

# Create a new column that is the number of pass attempts
passer_df['pass_attempts'] = passer_df['complete_pass'] + passer_df['incomplete_pass']

# Drop the complete_pass and incomplete_pass columns
passer_df = passer_df.drop(columns=['complete_pass', 'incomplete_pass'])

# Create a new column that equals 1 if the passer is the home team and 0 if the passer is the away team
passer_df['home_flag'] = passer_df['home_team'] == passer_df['posteam']

passer_df = passer_df.drop(columns=['home_team','away_team'])


# Assuming 'pbp' is your play-by-play DataFrame from nfl_data_py
drives = data[['game_id', 'drive', 'game_seconds_remaining', 'posteam']].copy()

# Sort the data by game and drive to ensure correct ordering
drives = drives.sort_values(by=['game_id', 'drive', 'game_seconds_remaining'], ascending=[True, True, False])

# Drop duplicates to get the start and end of each drive
drive_starts = drives.groupby(['game_id', 'drive']).first().reset_index()
drive_ends = drives.groupby(['game_id', 'drive']).last().reset_index()

# Merge the start and end times to calculate the duration
drive_durations = pd.merge(drive_starts, drive_ends, on=['game_id', 'drive', 'posteam'], suffixes=('_start', '_end'))

# Calculate the duration of each drive in seconds
drive_durations['drive_duration_seconds'] = drive_durations['game_seconds_remaining_start'] - drive_durations['game_seconds_remaining_end']

# Group by game and team to get total TOP
game_top = drive_durations.groupby(['game_id', 'posteam']).agg(
    total_top_seconds=('drive_duration_seconds', 'sum')
).reset_index()

passer_df_merged = passer_df.merge(game_top, on=['game_id', 'posteam'], how='inner')
schedule_df = pd.read_csv(r'2025_schedule.csv')
passer_df_merged = passer_df_merged.merge(schedule_df[['game_id','gameday']], on=['game_id'], how='inner')

passer_df_merged.drop('game_id', axis=1, inplace=True)
passer_df_merged.head()

ewma_cols = ['completion_percentage', 'pass_attempts', 'air_yards', 'yards_after_catch', 'epa', 'interception', 
             'qb_hit', 'sack', 'pass_touchdown', 'passing_yards', 'cpoe','total_top_seconds']

passer_df = passer_df_merged.sort_values(by=['passer_player_name', 'season', 'week']).reset_index(drop=True)

for col in ewma_cols:
    new_col_name = f'{col}_ewma'
    
    # Group by player, apply EWM (which is calculated cumulatively), and then shift
    passer_df[new_col_name] = passer_df.groupby('passer_player_name')[col]\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean().shift(1))

columns_to_drop = [col for col in ewma_cols if col != 'passing_yards']
# 3. Drop the non-ewma columns
passer_df = passer_df.drop(columns=columns_to_drop)

# Select only the relevant columns
defense_columns = ['defteam', 'season', 'week', 'home_team', 'away_team', 'play_type', 'air_yards',
                   'yards_after_catch', 'epa', 'complete_pass', 'incomplete_pass', 'interception', 'qb_hit', 'sack', 'pass_touchdown',
                   'passing_yards', 'cpoe', 'roof', 'surface']


# nfl-data-py still loads other columns, so we again need to set our data equal to only the columns we want
defense_data = data[defense_columns]

# Drop the play type column
defense_data = defense_data.drop(columns=['play_type'])



# Group the data together by passer, week, season and aggregate
defense_df = defense_data.groupby(['defteam', 'week', 'season'], as_index=False).agg(
    {'home_team': 'first',
     'away_team': 'first',
     'air_yards': 'sum',
     'yards_after_catch': 'sum',
     'epa': 'sum',
     'complete_pass': 'sum',
     'incomplete_pass': 'sum',
     'interception': 'sum',
     'qb_hit': 'sum',
     'sack': 'sum',
     'pass_touchdown': 'sum',
     'passing_yards': 'sum',
     'cpoe': 'mean',
     'roof': 'first',
     'surface': 'first'
     }
)

# Create a new column that is completion percentage
defense_df['completion_percentage'] = defense_df['complete_pass'] / (defense_df['complete_pass'] + defense_df['incomplete_pass'])

# Create a new column that is the number of pass attempts
defense_df['pass_attempts'] = defense_df['complete_pass'] + defense_df['incomplete_pass']

# Drop the complete_pass and incomplete_pass columns
defense_df = defense_df.drop(columns=['complete_pass', 'incomplete_pass'])

# Create a new column that equals 1 if the defense is the home team and 0 if the defense is the away team
defense_df['home_flag'] = defense_df['home_team'] == defense_df['defteam']

# Drop the home_team and away_team columns
defense_df = defense_df.drop(columns=['home_team', 'away_team'])

# Reorder the columns
defense_df = defense_df[['defteam', 'season', 'week', 'home_flag', 'passing_yards', 'completion_percentage', 'pass_attempts',
                       'air_yards',  'yards_after_catch', 'epa', 'interception', 'qb_hit', 'sack', 'pass_touchdown', 
                       'cpoe', 'roof', 'surface']]
                       
# Create a list of columns to generate EWMA features for
ewma_cols = ['completion_percentage', 'pass_attempts', 'air_yards', 'yards_after_catch', 
             'epa', 'interception', 'qb_hit', 'sack', 'pass_touchdown', 'passing_yards', 'cpoe']

# Step 1: Sort the DataFrame chronologically by team, season, and week
# This is crucial for accurate time-series feature generation.
defense_df = defense_df.sort_values(by=['defteam', 'season', 'week']).reset_index(drop=True)

# Step 2: Calculate the EWMA for each feature, applying a .shift(1)
# The .shift(1) ensures the EWMA for the current week is based on data up to the PREVIOUS week.
for col in ewma_cols:
    new_col_name = f'{col}_ewma'
    defense_df[new_col_name] = defense_df.groupby('defteam')[col].transform(
        lambda x: x.ewm(min_periods=1, span=10).mean().shift(1)
    )

# Step 3: Drop the original, non-EWMA columns to prevent data leakage
defense_df = defense_df.drop(columns=ewma_cols)

# Merge the defense and passer dataframes together
df = passer_df.merge(defense_df, how='inner', on=['defteam', 'season', 'week', 'roof', 'surface'], suffixes=('_passer', '_defense'))
df = df[df['pass_attempts_ewma_passer'] > 5]

df.isnull().sum()
final_df = df.dropna()

from rapidfuzz import process, fuzz
import pandas as pd
import re

# 1. Prepare Name Lists
final_names = final_df['passer_player_name'].unique().tolist()
prop_names = prop_df['player_name'].unique().tolist()
search_choices = final_names # List to search against


def clean_name_minimal(name):
    """Standardizes names for better partial ratio matching: Lowercase, remove periods/suffixes."""
    name = name.lower()
    # Only remove the period, keep the initial separate (e.g., 'J.Hurts' -> 'j hurts')
    name = name.replace('.', ' ') 
    # Remove common suffixes
    name = re.sub(r'\s+(jr|sr|ii|iii|iv)\.?$', '', name)
    # Remove any extra spaces created
    name = re.sub(r'\s+', ' ', name).strip()
    return name


# 2. Create the Cleaned Search List and a Map to Retrieve Original Names
# NOTE: We clean prop_names (the queries) but keep final_names (the choices) as they are for the original map
cleaned_choices = [clean_name_minimal(n) for n in search_choices]

# Map from CLEANED final_name back to ORIGINAL final_name (for merge key)
clean_to_original_map = {clean_name_minimal(n): n for n in final_names}


# 3. Perform the Fuzzy Matching and Build the Match List (for ALL names)
matched_names = []
# partial_ratio requires a lower threshold than token_set, try 80
SCORE_THRESHOLD = 80

for name_prop in prop_names:
    # Clean the current name_prop for the query
    cleaned_query = clean_name_minimal(name_prop)

    best_match = process.extractOne(
        query=cleaned_query,
        choices=cleaned_choices,
        # --- KEY CHANGE: Use partial_ratio ---
        scorer=fuzz.partial_ratio,
        score_cutoff=SCORE_THRESHOLD
    )

    if best_match:
        # best_match is (cleaned_match, score, index)
        cleaned_match_name = best_match[0]
        score = best_match[1]
        
        # Get the original (uncleaned) name from final_df using the map
        original_final_name = clean_to_original_map.get(cleaned_match_name)

        matched_names.append({
            'prop_name': name_prop,
            'final_name_match': original_final_name,
            'similarity_score': score
        })
    else:
        # No match found above the threshold
        matched_names.append({
            'prop_name': name_prop,
            'final_name_match': None,
            'similarity_score': 0.0
        })

# 4. Create Match DataFrame and Map for Merging
match_df = pd.DataFrame(matched_names)

# Display sample results
print("--- Sample Name Matches (prop_df -> final_df) ---")
print(match_df[['prop_name', 'final_name_match', 'similarity_score']].sort_values(by='similarity_score', ascending=False))
print("-" * 50)
print("\n--- Unmatched Players (Score < {}) ---".format(SCORE_THRESHOLD))
print(match_df[match_df['final_name_match'].isna()])


# 5. Perform the Merge
# Create the map from prop_name (full name) to final_name (abbreviated name)
valid_matches = match_df.dropna(subset=['final_name_match'])
name_map = valid_matches.set_index('prop_name')['final_name_match'].to_dict()

# Create the common key column in prop_df using the map
prop_df['passer_player_name'] = prop_df['player_name'].map(name_map)
prop_df['gameday'] = pd.to_datetime(prop_df['event_commence_time']).dt.strftime('%Y-%m-%d')
# Merge the DataFrames on the newly created common key
merged_df = final_df.merge(
    prop_df.drop(columns=['player_name','outcome_type']).drop_duplicates(), # Drop the redundant name column from prop_df
    on=['passer_player_name','gameday'],
    how='left'
)

merged_df.dropna(subset='sport_key', inplace=True)

merged_df.drop(columns=['event_id','event_name','sport_key','event_commence_time','updated_dttm'], axis=1, inplace=True)

# Identify categorical columns to be encoded
categorical_cols = ['passer_player_name', 'posteam', 'defteam', 'roof', 'surface']

# Perform one-hot encoding
final_df = pd.get_dummies(merged_df, columns=categorical_cols, drop_first=True)
final_df.drop(['gameday','market_type'], axis=1, inplace=True)
final_df['Status'] = (final_df['passing_yards'] > final_df['point_value']).astype(int)

X = final_df.drop(['passing_yards','week','season','point_value','Status'], axis=1)  # Features
y = final_df['Status']              # Target

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve
)

# Instantiate the models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LR': LogisticRegression(random_state=42),
}

roc_data = {}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    
    # 1. Get predicted class labels
    y_pred = model.predict(X_test)
    
    # 2. Get predicted probabilities for the positive class (required for AUC-ROC and ROC Curve)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback (though all models above have predict_proba)
        y_proba = y_pred 
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc_roc = np.nan 

    # Store metrics
    results[name] = {
        'Accuracy': accuracy, 
        'Precision': precision, 
        'Recall': recall, 
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }
    
    # Calculate ROC Curve points (FPR, TPR, thresholds)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_roc}


# -----------------------------------------------
# 1. Print the comparison results
# -----------------------------------------------
results_df = pd.DataFrame(results).T.sort_values(by='AUC-ROC', ascending=False)
print("--- Classification Model Comparison ---")
print(results_df)

best_model_name = results_df.index[0]
print(f"\nThe best performing model based on AUC-ROC is: {best_model_name}")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Instantiate the RandomForest model
rf = RandomForestClassifier(random_state=42)

# Perform Grid Search with cross-validation
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid,
                              cv=5, scoring='neg_root_mean_squared_error',
                              n_jobs=-1, verbose=1)

grid_search_rf.fit(X_train, y_train)

# Get the best model and its parameters
tuned_rf = grid_search_rf.best_estimator_
rf_best_params = grid_search_rf.best_params_
print(f"Best parameters found for RandomForest: {grid_search_rf.best_params_}")

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

rf_tuned_model = RandomForestClassifier(**rf_best_params, random_state=42)

rf_tuned_model.fit(X_train, y_train)
y_pred = rf_tuned_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
y_proba = rf_tuned_model.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

# --- 1. ADD FEATURE IMPORTANCE SCORING ---
feature_importances_df = pd.DataFrame(
    rf_tuned_model.feature_importances_, 
    index=X_train.columns, 
    columns=['Importance'] # Only one column name at this step
)

feature_importances_df = feature_importances_df.sort_values(
    by='Importance', 
    ascending=False
).reset_index()
# Rename the columns to your requested names
feature_importances_df.columns = ['Feature', 'Importance']
sns.barplot(data=feature_importances_df[:15], x='Importance', y='Feature', palette='magma')
plt.title("Most Important Features in Dataset")
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.savefig(r'Images/Most Impactful Features.png')

top_feature_1 = feature_importances_df.loc[0, 'Feature']
top_feature_2 = feature_importances_df.loc[1, 'Feature']
plt.figure(figsize=(10, 7))
sns.scatterplot(data=final_df, x=top_feature_1, y=top_feature_2, hue='Status', palette='viridis', s=70, alpha=0.7, legend='full')
plt.title(
    f'Relationship Between Top Features: {top_feature_1} vs {top_feature_2}',
    fontsize=16, 
    fontweight='bold', 
    pad=20
)
plt.xlabel(top_feature_1, fontsize=12)
plt.ylabel(top_feature_2, fontsize=12)

# 4. Enhance the appearance and legend
plt.grid(True, linestyle='--', alpha=0.5) # Add a light grid
plt.legend(title='Outcome (Status)', loc='best', labels=['Failure (0)', 'Success (1)'])

plt.tight_layout()
plt.savefig(r'Images/Top 2 Features Plotted.png')

top_n = 18
min_entries = 3
target_season = 2025
season_df = passer_df[passer_df['season'] == target_season].copy()
qb_stats = season_df.groupby('passer_player_name').agg(
    Avg_Passing_Yards=('passing_yards', 'mean'), 
    Game_Count=('passing_yards', 'count') # Use a distinct name for the count column
).reset_index() # Promote passer_player_name from index to column
qb_stats
sns.barplot(data=qb_stats.sort_values(by='Avg_Passing_Yards', ascending=False)[:top_n].dropna(), x='passer_player_name', y='Avg_Passing_Yards', palette='rocket')
plt.xticks(rotation=45)
plt.xlabel('QB Name')
plt.ylabel('Average Passing Yards')
plt.title(f'Top {top_n} QBs by Avg Passing Yards - 2025 Season')
plt.savefig(r'Images/Avg Pass Yds by QB.png')

min_entries = passer_df[passer_df['season'] == 2025]['week'].max()-2
target_season = 2025
season_df = passer_df[passer_df['season'] == target_season][['passing_yards','passer_player_name']].copy().dropna(subset='passing_yards')
min_df = passer_df[passer_df['season'] == target_season][['passing_yards','passer_player_name']].copy().dropna(subset='passing_yards').groupby('passer_player_name')['passing_yards'].count().reset_index()
min_df.rename({'passing_yards':'count'}, axis=1, inplace=True)
season_df = season_df.merge(min_df, how='inner', on='passer_player_name')
sns.boxplot(data=season_df[season_df['count'] >= min_entries].sort_values(by='passing_yards', ascending=False), x='passer_player_name', y='passing_yards')
plt.xticks(rotation=45)
plt.xlabel('QB Name')
plt.ylabel('Passing Yards')
plt.title(f'Top QBs by Passing Yards - 2025 Season')
plt.savefig(r'Images/BoxPlot PassYds.png')


analysis_df = merged_df.copy()
analysis_df['Status'] = (analysis_df['passing_yards'] > analysis_df['point_value']).astype(int)
cols = analysis_df.select_dtypes(exclude='object')
cols = cols.drop(['week','season','home_flag_defense'], axis=1)
cols = cols.columns
X = analysis_df[cols].drop(['Status'], axis=1)
y = analysis_df['Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)
coefs = model.coef_
coefs = model.coef_[0] # coef_ is a 2D array, extract the first (and only) row
magnitude_weights = np.abs(coefs)
objective_weights = magnitude_weights / np.sum(magnitude_weights)
weights_df = pd.DataFrame({
    'Metric': X.columns, # Use original feature names from DataFrame X
    'Raw Logit Coefficient (Beta)': coefs,
    'Objective Weight (Magnitude)': objective_weights
}).sort_values(by='Objective Weight (Magnitude)', ascending=False)

weights_df['Signed Objective Weight'] = np.where(
    # Condition: If the raw coefficient is negative
    weights_df['Raw Logit Coefficient (Beta)'] > 0,
    
    # Value if True: Multiply the magnitude by -1
    weights_df['Objective Weight (Magnitude)'] *-1,
    
    # Value if False: Keep the positive magnitude
    weights_df['Objective Weight (Magnitude)']
)
weights_df = weights_df[['Metric','Objective Weight (Magnitude)','Signed Objective Weight']]
defensive_metrics_df = weights_df[
    weights_df['Metric'].str.contains('defense', case=False, na=False)
].copy()

# Option 2: Get a list of the metric names (if you need it for external use)
def_col_names = defensive_metrics_df['Metric'].tolist()

merged_df[def_col_names + ['defteam']]
ranking_df = merged_df[def_col_names + ['defteam']].copy()
defense_performance = ranking_df.groupby('defteam')[def_col_names].mean().reset_index()

weight_map = defensive_metrics_df.set_index('Metric')['Signed Objective Weight'].to_dict()

X_agg = defense_performance[def_col_names]
X_agg_scaled = scaler.fit_transform(X_agg)
scaled_df = pd.DataFrame(
    X_agg_scaled, 
    columns=[f'{col}_ZScore' for col in def_col_names], 
    index=defense_performance.index
)
# Re-attach the defteam column
scaled_df['defteam'] = defense_performance['defteam']

scaled_df['Composite_Score'] = 0

# Apply the weighted sum using the 'Signed Objective Weight'
for metric, weight in weight_map.items():
    # Find the corresponding Z-score column name
    z_score_col = f'{metric}_ZScore'
    
    # Add the weighted score contribution
    # (Z-Score * Signed Objective Weight)
    scaled_df['Composite_Score'] += scaled_df[z_score_col] * weight

# --- STEP 4: Final Ranking ---

# Sort the teams by the composite score (Highest score is best defense)
final_ranking = scaled_df[['defteam', 'Composite_Score']].sort_values(
    by='Composite_Score', 
    ascending=False
).reset_index(drop=True)

final_ranking.to_csv(r'Composite_Score.csv', index=False)

sns.barplot(data = final_ranking, x='defteam', y='Composite_Score', palette='rocket')
plt.title("Defenses' Composite Score Ranking")
plt.xlabel('Team')
plt.xticks(rotation=45)
plt.ylabel('Composite Score')
plt.savefig(r'Images/Composite Score.png')

K = 3

X_scores = final_ranking[['Composite_Score']].values
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
final_ranking['Cluster'] = kmeans.fit_predict(X_scores)

# Map cluster numbers to sequential tier names
centroid_map = final_ranking.groupby('Cluster')['Composite_Score'].mean().sort_values(ascending=False)
tier_mapping = {cluster_id: f'Tier {i+1}' for i, cluster_id in enumerate(centroid_map.index)}
final_ranking['KMeans_Tier'] = final_ranking['Cluster'].map(tier_mapping)
final_ranking['Rank'] = final_ranking.index + 1

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=final_ranking,
    x='Rank',
    y='Composite_Score',
    hue='KMeans_Tier',
    palette='rocket',
    s=150,
    style='KMeans_Tier',
    legend='full'
)

for index, row in final_ranking.iterrows():
    # Annotate the top 5 teams and every 4th team to prevent visual clutter
    if row['Rank'] <= 5 or (row['Rank'] - 1) % 4 == 0:
        plt.annotate(
            row['defteam'],
            (row['Rank'] + 0.3, row['Composite_Score']), # Offset slightly to the right of the point
            fontsize=9,
            alpha=0.8,
            fontweight='bold' if row['Rank'] <= 5 else 'normal'
        )

# Set up the plot title and labels
plt.title(f'Defensive Ranking: Composite Score and K={K} Tiers', fontsize=16, fontweight='bold')
plt.xlabel('Defensive Rank (1 = Best)', fontsize=12)
plt.ylabel('Composite Score (Weighted Z-Score)', fontsize=12)
plt.xticks(range(1, final_ranking['Rank'].max() + 1, 3))
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(r'Images/Composite Score Tiered.png')

import joblib
# Save the tuned LGBM model for rushing yards
#joblib.dump(rf_tuned_model, 'rf_passing_model.joblib')

# Save the tuned Random Forest model for rushing yards
joblib.dump(rf_tuned_model, 'rf_passing_model.joblib')
X_train.to_csv(r'model_trained_on_data.csv')

print("All optimal models have been saved to disk.")