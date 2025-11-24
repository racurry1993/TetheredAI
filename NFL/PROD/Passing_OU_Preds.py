import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nfl_data_py as nfl
import os
os.chdir(r'C:\Users\rfo7799\Desktop\Git\TetheredAI\NFL\PROD')
schedule_data = pd.read_csv(r'2025_schedule.csv')
# Select only the relevant columns
columns = [
'season','game_type','week','gameday','weekday','gametime','away_team','home_team','home_coach','away_coach',
'away_qb_id','home_qb_id','away_qb_name','home_qb_name','surface','away_rest','home_rest'
]

schedule_df_final = schedule_data[columns]

def preprocessing_data():
	columns = ['passer_player_id','passer_player_name', 'posteam', 'defteam', 'season', 'week', 'home_team', 'away_team', 'play_type', 'air_yards', 
			   'yards_after_catch', 'epa', 'complete_pass', 'incomplete_pass', 'interception', 'qb_hit', 'sack', 'pass_touchdown',
			   'passing_yards', 'cpoe', 'roof', 'surface','drive','game_seconds_remaining', 'game_id']

	# Loading in the NFL pbp data
	data = nfl.import_pbp_data(range(2010,2026), downcast=True)
	data = data[columns]

	# nfl-data-py still loads other columns, so we again need to set our data equal to only the columns we want
	data = data[columns]

	# Drop all rows that are not a pass
	data = data[data['play_type'] == 'pass']

	# Drop the play type column
	passer_data = data.drop(columns=['play_type'])

	# Group the data together by passer, week, season and aggregate
	passer_df = passer_data.groupby(['passer_player_name', 'game_id', 'passer_player_id', 'week', 'season'], as_index=False).agg(
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

	# Drop the home_team and away_team columns
	passer_df = passer_df.drop(columns=['home_team', 'away_team'])

	# Reorder the columns
	#passer_df = passer_df[['passer_player_name','passer_player_id', 'posteam', 'defteam', 'season', 'week', 'passing_yards', 'home_flag', 'completion_percentage', 'pass_attempts',
	#                       'air_yards',  'yards_after_catch', 'epa', 'interception', 'qb_hit', 'sack', 'pass_touchdown', 
	#                        'cpoe', 'roof', 'surface']]

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
	passer_df_merged.drop('game_id', axis=1, inplace=True)

	ewma_cols = ['completion_percentage', 'pass_attempts', 'air_yards', 'yards_after_catch', 'epa', 'interception', 
				 'qb_hit', 'sack', 'pass_touchdown', 'passing_yards', 'cpoe','total_top_seconds']

	passer_df = passer_df_merged.sort_values(by=['passer_player_name', 'season', 'week']).reset_index(drop=True)

	for col in ewma_cols:
		new_col_name = f'{col}_ewma'
		
		# Group by player, apply EWM (which is calculated cumulatively), and then shift
		passer_df[new_col_name] = passer_df.groupby('passer_player_name')[col]\
			.transform(lambda x: x.ewm(min_periods=1, span=10).mean().shift(1))


	# 3. Drop the non-ewma columns
	passer_df = passer_df.drop(columns=ewma_cols)

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

	# Calculate the exponentially weighted moving average for each feature
	defense_df['completion_percentage_ewma'] = defense_df.groupby('defteam')['completion_percentage']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	defense_df['pass_attempts_ewma'] = defense_df.groupby('defteam')['pass_attempts']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	defense_df['air_yards_ewma'] = defense_df.groupby('defteam')['air_yards']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	defense_df['yards_after_catch_ewma'] = defense_df.groupby('defteam')['yards_after_catch']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	defense_df['epa_ewma'] = defense_df.groupby('defteam')['epa']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	defense_df['interception_ewma'] = defense_df.groupby('defteam')['interception']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	defense_df['qb_hit_ewma'] = defense_df.groupby('defteam')['qb_hit']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	defense_df['sack_ewma'] = defense_df.groupby('defteam')['sack']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	defense_df['pass_touchdown_ewma'] = defense_df.groupby('defteam')['pass_touchdown']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	defense_df['passing_yards_ewma'] = defense_df.groupby('defteam')['passing_yards']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	defense_df['cpoe_ewma'] = defense_df.groupby('defteam')['cpoe']\
		.transform(lambda x: x.ewm(min_periods=1, span=10).mean())

	# Drop the non-ewma columns
	defense_df = defense_df.drop(columns=['passing_yards','completion_percentage', 'pass_attempts', 'air_yards', 'yards_after_catch', 'epa', 
										'interception', 'qb_hit', 'sack', 'pass_touchdown', 'cpoe'])

	# Merge the defense and passer dataframes together
	df = passer_df.merge(defense_df, how='inner', on=['defteam', 'season', 'week', 'roof', 'surface'], suffixes=('_passer', '_defense'))
	df = df[df['pass_attempts_ewma_passer'] > 5]

	final_df = df.dropna()
	return final_df
	
def query_props():
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
	return prop_df
	
def name_match(main_df, props_df):
	from rapidfuzz import process, fuzz
	import pandas as pd
	import re

	# 1. Prepare Name Lists
	final_names = main_df['passer_player_name'].unique().tolist()
	prop_names = props_df['player_name'].unique().tolist()
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
	print("--- Sample Name Matches (props_df -> final_df) ---")
	print(match_df[['prop_name', 'final_name_match', 'similarity_score']].sort_values(by='similarity_score', ascending=False))
	print("-" * 50)
	print("\n--- Unmatched Players (Score < {}) ---".format(SCORE_THRESHOLD))
	print(match_df[match_df['final_name_match'].isna()])


	# 5. Perform the Merge
	# Create the map from prop_name (full name) to final_name (abbreviated name)
	valid_matches = match_df.dropna(subset=['final_name_match'])
	name_map = valid_matches.set_index('prop_name')['final_name_match'].to_dict()

	# Create the common key column in props_df using the map
	props_df['passer_player_name'] = props_df['player_name'].map(name_map)
	props_df['gameday'] = pd.to_datetime(props_df['event_commence_time']).dt.strftime('%Y-%m-%d')
	return props_df


def prediction_algorithm(schd_df, main_df):
	import joblib
	import numpy as np

	week_num = 12

	# Step 1: Load the saved model from the .joblib file.
	try:
		model = joblib.load('rf_passing_model.joblib')
		print("Model loaded successfully.")
	except FileNotFoundError:
		print("Error: The .joblib file was not found. Please check the file path and name.")
		#exit()


	# Step 3: Prepare the data for prediction.
	# We need to filter for Week 1 games and merge them with the most recent player and defensive stats.

	# Filter for Week 1 games from the schedule data.
	week_1_games = schd_df[schd_df['week'] == week_num].copy()

	# A helper function to find the most recent stats for a player or team
	def get_most_recent_stats(df, name, name_col, team_col, team):
		"""
		Finds the most recent stats for a given player or defensive team.
		"""
		# Filter for the specific player/team
		filtered_df = df[(df[name_col] == name) & (df[team_col] == team)].copy()
		
		# Sort by season and week to find the most recent game
		if not filtered_df.empty:
			most_recent_game = filtered_df.sort_values(by=['season', 'week'], ascending=False).iloc[0]
			return most_recent_game
		return None

	features_from_training = model.feature_names_in_.tolist()

	# Step 3b: Create a list of dictionaries for each QB in Week 1 games, populated with stats.
	prediction_rows = []
	for index, row in week_1_games.iterrows():
		# --- Process Home QB
		home_qb_name = row['home_qb_id']
		home_team = row['home_team']
		home_qb_stats = get_most_recent_stats(main_df, home_qb_name, 'passer_player_id', 'posteam', home_team)
		
		if home_qb_stats is not None:
			home_qb_row = home_qb_stats.to_dict()
			# Add schedule-specific info that the model might need
			home_qb_row['home_flag_passer'] = True # Home QB is playing at home
			home_qb_row['home_team'] = row['home_team']
			home_qb_row['away_team'] = row['away_team']
			home_qb_row['season'] = row['season']
			home_qb_row['week'] = row['week']
			
			# Add defensive stats for the opposing team (the 'away' team in this game)
			away_def_stats = get_most_recent_stats(main_df, row['away_team'], 'defteam', 'defteam', row['away_team'])
			if away_def_stats is not None:
				for col in away_def_stats.keys():
					if 'defense' in col:
						home_qb_row[col] = away_def_stats[col]

			prediction_rows.append(home_qb_row)

		# --- Process Away QB
		away_qb_name = row['away_qb_id']
		away_team = row['away_team']
		away_qb_stats = get_most_recent_stats(main_df, away_qb_name, 'passer_player_id', 'posteam', away_team)
		
		if away_qb_stats is not None:
			away_qb_row = away_qb_stats.to_dict()
			away_qb_row['home_flag_passer'] = False # Away QB is not playing at home
			away_qb_row['home_team'] = row['home_team']
			away_qb_row['away_team'] = row['away_team']
			away_qb_row['season'] = row['season']
			away_qb_row['week'] = row['week']

			# Add defensive stats for the opposing team (the 'home' team in this game)
			home_def_stats = get_most_recent_stats(main_df, row['home_team'], 'defteam', 'defteam', row['home_team'])
			if home_def_stats is not None:
				for col in home_def_stats.keys():
					if 'defense' in col:
						away_qb_row[col] = home_def_stats[col]

			prediction_rows.append(away_qb_row)

	# Create a DataFrame from the combined data
	if not prediction_rows:
		print("Warning: No Week 1 games with available stats found. Cannot make predictions.")
		#exit()

	week_qbs_df = pd.DataFrame(prediction_rows)
	print("\nPrepared Week 1 data with combined player and defensive stats.")

	# Store the original player and team names before one-hot encoding
	prediction_info = week_qbs_df[['passer_player_name', 'home_team', 'away_team', 'season', 'week']].copy()

	categorical_features = ['passer_player_name', 'posteam', 'defteam', 'home_team', 'away_team']
	week_dummies = pd.get_dummies(
		week_qbs_df.drop(['passer_player_id'], axis=1, errors='ignore'), 
		columns=categorical_features, 
		# CRITICAL FIX: Ensure drop_first=False here. Let reindex handle the alignment.
		drop_first=False, 
		dtype=int
	)
	# Step 3d: Align the columns to ensure they match the training data perfectly.
	# Any column in the training data not present in the prediction data will be filled with 0.
	X_predict = week_dummies.reindex(columns=features_from_training, fill_value=0)
	# Step 4: Use the loaded model to predict the passing yards.
	predictions = model.predict(X_predict) # <--- **FIXED LINE**

	# Step 5: Add the predictions to your DataFrame.
	probabilities = model.predict_proba(X_predict)
	prob_of_over = probabilities[:, 1]
	X_predict['probability_over'] = prob_of_over
	X_predict['status'] = predictions
	X_predict['status'] = X_predict['status'].apply(lambda x: 'Over' if x == 1 else 'Under')

	# Step 6: Join the original player/team names back to the predictions DataFrame.
	X_predict = pd.concat([prediction_info, X_predict], axis=1)

	# Step 6: Join the original player/team names back to the predictions DataFrame.
	X_predict = pd.concat([prediction_info, X_predict], axis=1)
	return X_predict
	
def main():
	final_df = preprocessing_data()
	prop_df = query_props()
	prop_df = name_match(final_df,prop_df)
	X_predict = prediction_algorithm(schedule_df_final,final_df)
	import datetime
	today = datetime.date.today()
	today = today.strftime("%Y-%m-%d")
	X_predict[['passer_player_name','week','home_team','away_team','status','probability_over']].sort_values(by='probability_over', ascending=False)[:10].to_csv(fr'Preds/10_over_passing_yds_{today}.csv')
	X_predict[['passer_player_name','week','home_team','away_team','status','probability_over']].sort_values(by='probability_over', ascending=True)[:10].to_csv(fr'Preds/10_under_passing_yds_{today}.csv')
	
if __name__ == "__main__":
    main()