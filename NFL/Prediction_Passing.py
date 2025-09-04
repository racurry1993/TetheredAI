import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nfl_data_py as nfl


def make_prediction(week):
    schedule_data = pd.read_csv(r'2025_schedule.csv')

    columns = [
        'season','game_type','week','gameday','weekday','gametime','away_team','home_team','home_coach','away_coach',
        'away_qb_id','home_qb_id','away_qb_name','home_qb_name','surface','away_rest','home_rest'
    ]

    schedule_df_final = schedule_data[columns]

    # Select only the relevant columns
    columns = ['passer_player_id','passer_player_name', 'posteam', 'defteam', 'season', 'week', 'home_team', 'away_team', 'play_type', 'air_yards', 
            'yards_after_catch', 'epa', 'complete_pass', 'incomplete_pass', 'interception', 'qb_hit', 'sack', 'pass_touchdown',
            'passing_yards', 'cpoe', 'roof', 'surface']

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
    passer_df = passer_data.groupby(['passer_player_name', 'passer_player_id', 'week', 'season'], as_index=False).agg(
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
    passer_df = passer_df[['passer_player_name','passer_player_id', 'posteam', 'defteam', 'season', 'week', 'passing_yards', 'home_flag', 'completion_percentage', 'pass_attempts',
                        'air_yards',  'yards_after_catch', 'epa', 'interception', 'qb_hit', 'sack', 'pass_touchdown', 
                            'cpoe', 'roof', 'surface']]

    # Calculate the exponentially weighted moving average for each feature
    passer_df['completion_percentage_ewma'] = passer_df.groupby('passer_player_name')['completion_percentage']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    passer_df['pass_attempts_ewma'] = passer_df.groupby('passer_player_name')['pass_attempts']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    passer_df['air_yards_ewma'] = passer_df.groupby('passer_player_name')['air_yards']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    passer_df['yards_after_catch_ewma'] = passer_df.groupby('passer_player_name')['yards_after_catch']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    passer_df['epa_ewma'] = passer_df.groupby('passer_player_name')['epa']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    passer_df['interception_ewma'] = passer_df.groupby('passer_player_name')['interception']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    passer_df['qb_hit_ewma'] = passer_df.groupby('passer_player_name')['qb_hit']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    passer_df['sack_ewma'] = passer_df.groupby('passer_player_name')['sack']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    passer_df['pass_touchdown_ewma'] = passer_df.groupby('passer_player_name')['pass_touchdown']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    passer_df['passing_yards_ewma'] = passer_df.groupby('passer_player_name')['passing_yards']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    passer_df['cpoe_ewma'] = passer_df.groupby('passer_player_name')['cpoe']\
        .transform(lambda x: x.ewm(min_periods=1, span=10).mean())

    # Drop the non-ewma columns
    passer_df = passer_df.drop(columns=['completion_percentage', 'pass_attempts', 'air_yards', 'yards_after_catch', 'epa', 
                                        'interception', 'qb_hit', 'sack', 'pass_touchdown', 'cpoe'])

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

    import joblib

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
    week_1_games = schedule_df_final[schedule_df_final['week'] == week].copy()

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

    # Step 3a: Extract the full feature list from the training data.
    # This list is crucial to ensure the prediction data has the same columns.
    features_from_training = pd.read_csv(r'model_trained_on_data.csv').columns.tolist()
    if 'passing_yards' in features_from_training:
        features_from_training.remove('passing_yards')

    # Step 3b: Create a list of dictionaries for each QB in Week 1 games, populated with stats.
    prediction_rows = []
    for index, row in week_1_games.iterrows():
        # --- Process Home QB
        home_qb_name = row['home_qb_id']
        home_team = row['home_team']
        home_qb_stats = get_most_recent_stats(final_df, home_qb_name, 'passer_player_id', 'posteam', home_team)
        
        if home_qb_stats is not None:
            home_qb_row = home_qb_stats.to_dict()
            # Add schedule-specific info that the model might need
            home_qb_row['home_flag_passer'] = True # Home QB is playing at home
            home_qb_row['home_team'] = row['home_team']
            home_qb_row['away_team'] = row['away_team']
            home_qb_row['season'] = row['season']
            home_qb_row['week'] = row['week']
            
            # Add defensive stats for the opposing team (the 'away' team in this game)
            away_def_stats = get_most_recent_stats(final_df, row['away_team'], 'defteam', 'defteam', row['away_team'])
            if away_def_stats is not None:
                for col in away_def_stats.keys():
                    if 'defense' in col:
                        home_qb_row[col] = away_def_stats[col]

            prediction_rows.append(home_qb_row)

        # --- Process Away QB
        away_qb_name = row['away_qb_id']
        away_team = row['away_team']
        away_qb_stats = get_most_recent_stats(final_df, away_qb_name, 'passer_player_id', 'posteam', away_team)
        
        if away_qb_stats is not None:
            away_qb_row = away_qb_stats.to_dict()
            away_qb_row['home_flag_passer'] = False # Away QB is not playing at home
            away_qb_row['home_team'] = row['home_team']
            away_qb_row['away_team'] = row['away_team']
            away_qb_row['season'] = row['season']
            away_qb_row['week'] = row['week']

            # Add defensive stats for the opposing team (the 'home' team in this game)
            home_def_stats = get_most_recent_stats(final_df, row['home_team'], 'defteam', 'defteam', row['home_team'])
            if home_def_stats is not None:
                for col in home_def_stats.keys():
                    if 'defense' in col:
                        away_qb_row[col] = home_def_stats[col]

            prediction_rows.append(away_qb_row)

    # Create a DataFrame from the combined data
    if not prediction_rows:
        print("Warning: No Week 1 games with available stats found. Cannot make predictions.")
        #exit()

    week_1_qbs_df = pd.DataFrame(prediction_rows)
    print("\nPrepared Week 1 data with combined player and defensive stats.")

    # Store the original player and team names before one-hot encoding
    prediction_info = week_1_qbs_df[['passer_player_name', 'home_team', 'away_team']].copy()

    # Step 3c: One-hot encode the categorical features to match the training data.
    categorical_features = ['passer_player_name', 'posteam', 'defteam', 'home_team', 'away_team']
    week_1_dummies = pd.get_dummies(week_1_qbs_df.drop(['passer_player_id'], axis=1), columns=categorical_features, drop_first=True, dtype=int)

    # Step 3d: Align the columns to ensure they match the training data perfectly.
    # Any column in the training data not present in the prediction data will be filled with 0.
    X_predict = week_1_dummies.reindex(columns=features_from_training, fill_value=0)

    # Step 4: Use the loaded model to predict the passing yards.
    predictions = model.predict(X_predict.drop(['season','week'], axis=1))

    # Step 5: Add the predictions to your DataFrame.
    X_predict['predicted_passing_yards'] = predictions

    # Step 6: Join the original player/team names back to the predictions DataFrame.
    X_predict = pd.concat([prediction_info, X_predict], axis=1)
    # Step 6: Display the results.
    print("\nWeek 1 Predictions with Passer Names:")
    print(X_predict)
    X_predict.to_csv(fr'Week - {week} Passing Predictions.csv', index=False)

    return