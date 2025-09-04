import pandas as pd
import nfl_data_py as nfl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

def create_model(end_year):
    # Select only the relevant columns
    columns = ['passer_player_name', 'posteam', 'defteam', 'season', 'week', 'home_team', 'away_team', 'play_type', 'air_yards', 
            'yards_after_catch', 'epa', 'complete_pass', 'incomplete_pass', 'interception', 'qb_hit', 'sack', 'pass_touchdown',
            'passing_yards', 'cpoe', 'roof', 'surface']

    # Loading in the NFL pbp data
    data = nfl.import_pbp_data(range(2010,end_year), downcast=True)
    data = data[columns]

    # nfl-data-py still loads other columns, so we again need to set our data equal to only the columns we want
    data = data[columns]

    # Drop all rows that are not a pass
    data = data[data['play_type'] == 'pass']

    # Drop the play type column
    passer_data = data.drop(columns=['play_type'])

    # Group the data together by passer, week, season and aggregate
    passer_df = passer_data.groupby(['passer_player_name', 'week', 'season'], as_index=False).agg(
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
    passer_df = passer_df[['passer_player_name', 'posteam', 'defteam', 'season', 'week', 'passing_yards', 'home_flag', 'completion_percentage', 'pass_attempts',
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

    # Identify categorical columns to be encoded
    categorical_cols = ['passer_player_name', 'posteam', 'defteam', 'roof', 'surface']

    # Perform one-hot encoding
    final_df = pd.get_dummies(final_df, columns=categorical_cols, drop_first=True)
    final_df.to_csv('model_trained_on_data.csv', index=False)
    # Define your features (X) and target (y)
    X = final_df.drop(['passing_yards','week','season'], axis=1)  # Features
    y = final_df['passing_yards']              # Target

    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    # Instantiate the models
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'SVR': SVR(),
    }

    # Regression models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results[name] = {'RMSE': rmse, 'R-squared': r2, 'MAE': mae}

    # Print the comparison results
    results_df = pd.DataFrame(results).T.sort_values(by='RMSE')
    print(results_df)

    # You would choose the best model based on the lowest RMSE
    best_model_name = results_df.index[0]
    best_model = models[best_model_name]
    print(f"\nThe best performing model is: {best_model_name}")

    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
    }

    # Instantiate the RandomForest model
    rf = RandomForestRegressor(random_state=42)

    # Perform Grid Search with cross-validation
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid,
                                cv=5, scoring='neg_root_mean_squared_error',
                                n_jobs=-1, verbose=1)

    grid_search_rf.fit(X_train, y_train)

    # Get the best model and its parameters
    tuned_rf = grid_search_rf.best_estimator_
    rf_best_params = grid_search_rf.best_params_
    print(f"Best parameters found for RandomForest: {grid_search_rf.best_params_}")


    from sklearn.metrics import mean_absolute_error

    rf_tuned_model = RandomForestRegressor(**rf_best_params, random_state=42)

    rf_tuned_model.fit(X_train, y_train)
    y_pred = rf_tuned_model.predict(X_test)

    # Calculate metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'R2 Score: {r2}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')

    import joblib
    # Save the tuned LGBM model for rushing yards
    joblib.dump(rf_tuned_model, 'rf_passing_model.joblib')

    return