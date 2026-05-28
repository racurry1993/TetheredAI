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
import joblib

os.chdir(r'C:\Users\rfo7799\Desktop\Git\TetheredAI\NFL\PROD')

pd.set_option('display.max_columns', None)

import sqlite3
import pandas as pd
DATABASE_NAME = 'odds.db'
#SOURCE_DB = r'C:\Users\rfo7799\Downloads\odds.db'

conn = sqlite3.connect(DATABASE_NAME)
table_name = 'player_props'
prop_df = pd.read_sql(f"""
                 SELECT * FROM {table_name} WHERE market_type = 'player_pass_tds'
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

columns_to_drop = [col for col in ewma_cols if col != 'pass_touchdown']
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
from dateutil import parser

# Define your timezone mapping
tz_mapping = {"EDT": -14400, "EST": -18000} # Offsets in seconds

# Apply the parser to each string in the column
prop_df['event_commence_time'] = prop_df['event_commence_time'].apply(
    lambda x: parser.parse(x, tzinfos=tz_mapping)
)
prop_df['event_commence_time'] = pd.to_datetime(
    prop_df['event_commence_time'], 
    utc=True
)
prop_df['gameday'] = pd.to_datetime(prop_df['event_commence_time']).dt.strftime('%Y-%m-%d')
# Merge the DataFrames on the newly created common key
merged_df = final_df.merge(
    prop_df.drop(columns=['player_name','outcome_type']).drop_duplicates(), # Drop the redundant name column from prop_df
    on=['passer_player_name','gameday'],
    how='left'
)

merged_df.dropna(subset='sport_key', inplace=True)

merged_df.drop(columns=['event_id','event_name','sport_key','event_commence_time','updated_dttm'], inplace=True)

# Identify categorical columns to be encoded
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from scipy.stats import poisson

# Identify categorical columns to be encoded
#categorical_cols = ['passer_player_name', 'posteam', 'defteam', 'roof', 'surface']
categorical_cols = ['roof', 'surface']

# 2. PERFORM ENCODING
final_df = pd.get_dummies(merged_df, columns=categorical_cols, drop_first=True)

# 3. DEFINE FEATURES (Drop EVERYTHING that identifies a specific player or team)
# We want the model to learn: "If a QB has an EPA of 0.2 and 30 attempts, they score X"
# NOT: "If a QB is named Mahomes, they score X"
cols_to_drop = [
    'pass_touchdown', 'week', 'season', 'point_value', 'Status', 'odds',
    'passer_player_name', 'posteam', 'defteam', 'gameday', 'market_type'
]

from scipy.stats import poisson
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
# Ensure we aren't keeping the dummy name columns
# This regex removes any column that starts with 'passer_player_name_' or 'posteam_'
X = final_df.drop(columns=[c for c in final_df.columns if any(x in c for x in cols_to_drop)])
X = X.loc[:, ~X.columns.str.startswith(('passer_player_name_', 'posteam_', 'defteam_'))]

y = final_df['pass_touchdown']

tscv = TimeSeriesSplit(n_splits=5)

mae_scores = []
deviance_scores = []
win_rates = []

print("--- Starting Time-Series Validation (Poisson) ---")

def run_tuned_edge_backtest(X, y, df, line=1.5):
    def get_implied_prob(odds):
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    # 1. Define the Parameter Grid for Poisson Regression
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5]
    }

    # 2. Setup Time-Series Split
    tscv = TimeSeriesSplit(n_splits=5)
    all_results = []
    
    # We will track the best params for each fold to see if they stay consistent
    best_params_history = []

    print("Starting Backtest with Hyper-parameter Tuning...")

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_odds = df.iloc[test_index]['odds']
        
        # 3. Initialize the Poisson Model
        base_model = XGBRegressor(objective='count:poisson', random_state=42)

        # 4. Run Randomized Search INSIDE the fold
        # Note: cv=TimeSeriesSplit ensures the tuning itself doesn't leak data
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=15, 
            scoring='neg_mean_poisson_deviance', # Optimization target for Poisson
            cv=TimeSeriesSplit(n_splits=3),
            verbose=0,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params_history.append(search.best_params_)

        # 5. Predict with the Optimized Model
        lambdas = best_model.predict(X_test)
        prob_over = 1 - poisson.cdf(line, lambdas)
        
        fold_df = pd.DataFrame({
            'actual_tds': y_test,
            'model_prob': prob_over,
            'vegas_odds': test_odds
        })
        
        fold_df['implied_prob'] = fold_df['vegas_odds'].apply(get_implied_prob)
        fold_df['edge'] = fold_df['model_prob'] - fold_df['implied_prob']
        fold_df['won_bet'] = ((fold_df['model_prob'] > 0.5) & (fold_df['actual_tds'] > line)) | \
                             ((fold_df['model_prob'] < 0.5) & (fold_df['actual_tds'] <= line))
        
        all_results.append(fold_df)
        print(f"Fold complete. Best Params: {search.best_params_}")

    # Combine results
    full_backtest = pd.concat(all_results)
    
    # Performance Metrics
    high_edge_plays = full_backtest[full_backtest['edge'].abs() > 0.10]
    avg_win_rate = full_backtest['won_bet'].mean()
    high_edge_win_rate = high_edge_plays['won_bet'].mean() if len(high_edge_plays) > 0 else 0

    print("\n--- TUNED BACKTEST RESULTS ---")
    print(f"Average Win Rate: {avg_win_rate:.2%}")
    print(f"High Confidence (>10% Edge) Win Rate: {high_edge_win_rate:.2%}")

    # 6. Final Fit: Re-tune on ALL data for production use
    final_search = RandomizedSearchCV(
        XGBRegressor(objective='count:poisson'),
        param_grid, n_iter=20, cv=tscv, scoring='neg_mean_poisson_deviance'
    )
    final_search.fit(X, y)
    
    # Save tuned model and threshold
    joblib.dump(final_search.best_estimator_, 'nfl_pass_td_tuned_model.joblib')
    
    return full_backtest, final_search.best_params_, final_search, X_train

results, final_params, model, train_X = run_tuned_edge_backtest(X, y, final_df)

# Assuming 'full_backtest' is the results DataFrame from your run_edge_backtest function
def save_optimal_threshold(full_backtest, target_win_rate=0.60):
    # Sort by edge to find the threshold
    sorted_edges = full_backtest.sort_values(by='edge', ascending=False)
    
    # Simple logic to find the edge value where the rolling win rate hits your target
    sorted_edges['rolling_win_rate'] = sorted_edges['won_bet'].expanding().mean()
    
    # Find the edge threshold that satisfies the target win rate
    # We look for the smallest edge that still gives us our 60% (or target) WR
    threshold_df = sorted_edges[sorted_edges['rolling_win_rate'] >= target_win_rate]
    
    if not threshold_df.empty:
        # Get the edge value at this point
        optimal_threshold = threshold_df['edge'].min()
    else:
        # Fallback to a safe default if data is thin
        optimal_threshold = 0.20
        
    print(f"Optimal Edge Threshold found: {optimal_threshold:.4f}")
    joblib.dump(optimal_threshold, 'edge_threshold.joblib')
    return optimal_threshold

# Call this after your backtest
save_optimal_threshold(results)

top_n = 18
min_entries = 3
target_season = 2025
season_df = passer_df[passer_df['season'] == target_season].copy()
qb_stats = season_df.groupby('passer_player_name').agg(
    Avg_Passing_TDs=('pass_touchdown', 'mean'), 
    Game_Count=('pass_touchdown', 'count') # Use a distinct name for the count column
).reset_index() # Promote passer_player_name from index to column
qb_stats
sns.barplot(data=qb_stats.sort_values(by='Avg_Passing_TDs', ascending=False)[:top_n].dropna(), x='passer_player_name', y='Avg_Passing_TDs')
plt.xticks(rotation=45)
plt.xlabel('QB Name')
plt.ylabel('Average Passing TDs')
plt.title(f'Top {top_n} QBs by Avg Passing TDs - 2025 Season')
plt.savefig(r'Images/Avg PassTDs by QB.png')

min_entries = passer_df[passer_df['season'] == 2025]['week'].max()-2
target_season = 2025
season_df = passer_df[passer_df['season'] == target_season][['pass_touchdown','passer_player_name']].copy().dropna(subset='pass_touchdown')
min_df = passer_df[passer_df['season'] == target_season][['pass_touchdown','passer_player_name']].copy().dropna(subset='pass_touchdown').groupby('passer_player_name')['pass_touchdown'].count().reset_index()
min_df.rename({'pass_touchdown':'count'}, axis=1, inplace=True)
season_df = season_df.merge(min_df, how='inner', on='passer_player_name')
sns.boxplot(data=season_df[season_df['count'] >= min_entries].sort_values(by='pass_touchdown', ascending=False), x='passer_player_name', y='pass_touchdown')
plt.xticks(rotation=45)
plt.xlabel('QB Name')
plt.ylabel('Passing TDs')
plt.title(f'Top QBs by Passing TDs - 2025 Season')
plt.savefig(r'Images/BoxPlot PassTDs.png')


analysis_df = merged_df.copy()
analysis_df['Status'] = (analysis_df['pass_touchdown'] > analysis_df['point_value']).astype(int)
cols = analysis_df.select_dtypes(exclude='object')
cols = cols.drop(['week','season','home_flag_defense'], axis=1)
cols = cols.columns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

print("--- Filtered Defensive Weights DataFrame ---")
print(defensive_metrics_df)
print("\nList of Filtered Metric Names:")
print(def_col_names)

merged_df[def_col_names + ['defteam']]
ranking_df = merged_df[def_col_names + ['defteam']].copy()
defense_performance = ranking_df.groupby('defteam')[def_col_names].mean().reset_index()

weight_map = defensive_metrics_df.set_index('Metric')['Signed Objective Weight'].to_dict()

scaler = StandardScaler()
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

final_ranking.to_csv(r'Composite_Score_PassTDs.csv', index=False)

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
plt.savefig(r'Images/Composite Score Tiered PassTDs.png')


# Save the tuned Random Forest model for rushing yards
train_X.to_csv(r'model_trained_on_tds_data.csv')

print("All optimal models have been saved to disk.")