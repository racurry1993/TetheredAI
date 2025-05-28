
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

#os.chdir('Desktop')

tournament_name = r'''the Memorial Tournament pres. by Workday'''

os.chdir(r'C:\Users\rfo7799\Desktop\Git\TetheredAI')

df = pd.read_csv(r'full_tournament_data.csv')
#df = pd.read_csv(r'data.csv')

df[['Position','Last T1 Finish','Last T2 Finish', 'Last T3 Finish']] = df[['Position','Last T1 Finish','Last T2 Finish', 'Last T3 Finish']].replace(-1, 100)

df['Last T2 Finish'] = df['Last T2 Finish'].fillna(df['Last T2 Finish'].median())
df['Last T3 Finish'] = df['Last T3 Finish'].fillna(df['Last T3 Finish'].median())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def convert_data_types(df):
    """Convert columns to appropriate data types"""
    df = df.copy()
    # Convert Position to numeric, replacing 'CUT' with -1
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(100)
    
    # Convert TOPAR to numeric
    df['TOPAR'] = pd.to_numeric(df['TOPAR'], errors='coerce')
    
    # Convert round scores and totals to numeric
    for col in ['R1', 'R2', 'R3', 'R4', 'Total']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Winnings to numeric, removing currency symbols and commas
    df['Winnings'] = df['Winnings'].replace('[\$,]', '', regex=True)
    df['Winnings'] = pd.to_numeric(df['Winnings'], errors='coerce')
    
    return df

def handle_missed_cuts(df):
    """Handle missing data for missed cuts"""
    df = df.copy()
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(100)
    missed_cut_mask = df['Position'] == 100
    
    # Fill missing values for missed cuts
    round_cols = ['R3', 'R4', 'Total', 'Winnings']
    for col in round_cols:
        df.loc[missed_cut_mask, col] = 0
    
    # Set TOPAR for missed cuts
    worst_topar = df[df['Position'] != 100]['TOPAR'].max()
    df.loc[missed_cut_mask, 'TOPAR'] = worst_topar + 5 if not pd.isna(worst_topar) else 10
    
    return df

def create_performance_features(df):
    """Create performance-based features"""
    df = df.copy()
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(100)
    df = df.sort_values(['Player', 'Final_Date'], ascending=[True,True])
    
    windows = [2, 3, 5]
    for window in windows:
        # Moving averages for positions
        df[f'Position_ma_{window}'] = (
            df.groupby('Player')['Position']
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(0, drop=True)
        )
        
        # Position volatility
        df[f'Position_std_{window}'] = (
            df.groupby('Player')['Position']
            .rolling(window=window, min_periods=1)
            .std()
            .shift(1)
            .reset_index(0, drop=True)
        )
        
        # TOPAR trends
        df[f'TOPAR_trend_{window}'] = (
            df.groupby('Player')['TOPAR']
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(0, drop=True)
        )
    
    # Momentum features
    #df['Position_momentum_5'] = df['Position_ma_5'] - df['Position_ma_10']
    df['Position_momentum_3'] = df['Position_ma_3'] - df['Position_ma_5']
    
    # Cut making features
    df['Made_Cut'] = (df['Position'] != 100).astype(float)  # Changed to float
    def preceding_cut_streak(scores):
        streak = 0
        streaks = [0]
        for score in scores[:-1]:
            if score == 100:
                streak = 0
            else:
                streak += 1
            streaks.append(streak)
            
        return streaks

    df['Cut_streak'] = df.groupby('Player')['Position'].transform(lambda x: preceding_cut_streak(x))
    
    return df

def create_golf_specific_features(df):
    """Create golf-specific performance metrics"""
    df = df.copy()
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(100)
    df = df.sort_values(['Player','Final_Date'], ascending=[True,True])
    # Round performance metrics
    df['Early_Rounds_Avg'] = df[['R1', 'R2']].shift(1).mean(axis=1)
    #df['Late_Rounds_Avg'] = df[['R3', 'R4']].shift(-1).mean(axis=1)
    df['Last_3_Early_Rounds_Avg'] = df.groupby('Player')['Early_Rounds_Avg'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
    #df['Last_3_Late_Rounds_Avg'] = df.groupby('Player')['Late_Rounds_Avg'].transform(lambda x: x.shift(-1).rolling(window=3, min_periods=1).mean())
    
    # Pull prior tournament averages for R1, R2, R3, and R4
    #df['Prior_R1_Avg'] = df.groupby('Player')['R1'].shift(-1)
    #df['Prior_R2_Avg'] = df.groupby('Player')['R2'].shift(-1)
    #df['Prior_R3_Avg'] = df.groupby('Player')['R3'].shift(-1)
    #df['Prior_R4_Avg'] = df.groupby('Player')['R4'].shift(-1)
    #df['Round_Progression'] = df['Last_3_Late_Rounds_Avg'] - df['Last_3_Early_Rounds_Avg']
    

    #df['Course_Par'] = df['Total'] - df['TOPAR']
    #df['Weekend_Performance'] = df[['R3', 'R4']].mean(axis=1) - df[['R1', 'R2']].mean(axis=1)
    
    # Recent form features
    #df = df.sort_values(['Player', 'Final_Date'], ascending=[True,True])
    
    # Position trend
    df['Position_Trend'] = df.groupby('Player')['Position'].transform(
        lambda x: np.polyfit(range(len(x[-5:])), x[-5:], 1)[0]
        if len(x) >= 5 else np.nan
    )
    
    # Performance ratios - Fixed to handle boolean operations
    for window in [5, 10, 20]:
        # Convert boolean to float before calculating mean
        df[f'Win_Ratio_{window}'] = df.groupby('Player')['Position'].transform(
            lambda x: (x == 1).astype(float).rolling(window=window, min_periods=1).mean().shift(1)
        )
        df[f'Top10_Ratio_{window}'] = df.groupby('Player')['Position'].transform(
            lambda x: (x <= 10).astype(float).rolling(window=window, min_periods=1).mean().shift(1)
        )
    
    # Tournament metrics
    #df['Avg_Tournament_Score'] = df.groupby('Tournament_ID')['TOPAR'].transform('mean')
    #df['Score_vs_Field'] = df['TOPAR'] - df['Avg_Tournament_Score']
    
    # Consistency metrics
    completed_rounds_mask = (df['Position'] != 100)
    #df['Score_Variance'] = np.nan
    df.loc[completed_rounds_mask, 'Score_Variance'] = df.loc[completed_rounds_mask, ['R1', 'R2', 'R3', 'R4']].std(axis=1)
    
    return df

# Convert data types and create base features
df = convert_data_types(df)
df['Tournament_ID'] = df['Tournament'] + '_' + df['Year'].astype(str)
#df = df.groupby('Tournament_ID').apply(normalize_positions)
df = handle_missed_cuts(df)

# Convert dates and create features
df['Final_Date'] = pd.to_datetime(df['Final_Date'])
df = create_performance_features(df)
df = create_golf_specific_features(df)

df.to_csv(r'Test.csv', index=False)

#df.isnull().sum()

df.drop(['Tournament_ID','TOPAR','R1','R2','R3','R4','Total','Winnings','Final_Date','State','Region','Score_Variance','Made_Cut'], axis=1, inplace=True)

df = df.dropna(subset=['Last T1 Finish'])

for col in df.columns.drop(['Tournament','Location','Player']):
    if col in ['Position_Trend', 'Score_vs_Field']:
        fill_value = 0
    elif col.startswith(('Win_Ratio', 'Top10_Ratio')):
        fill_value = 0
    else:
        fill_value = df[col].median()
        
    df[col] = df[col].fillna(fill_value)
    
#df.to_csv(r'Alt_Dataset.csv')

sns.heatmap(df.corr())
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


year = 2025
# Create prediction mask
prediction_mask = (df['Tournament'] == tournament_name) & (df['Year'] == year)

# Split data
train_data = df[~prediction_mask].copy()
predict_data = df[prediction_mask].copy()



# Assuming train_data is your DataFrame
X = train_data.drop(['Tournament', 'Location', 'Player', 'Position'], axis=1)
y = train_data['Position']


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_X = scaler.fit_transform(X)
pca = PCA(n_components = 0.95)
X_pca = pca.fit_transform(scaled_X)

cumulative_variance = pca.explained_variance_ratio_.cumsum()
print("Cumulative explained variance ratio: ", cumulative_variance)

# Convert the scaled data back to a DataFrame with original feature names and index
scaled_X_df = pd.DataFrame(scaled_X, columns=X.columns, index=X.index)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X_df, y, test_size=0.3, random_state=42)

###############################################################################
import statsmodels.api as sm

# Fit a model using statsmodels
X_train_with_intercept = sm.add_constant(X_train)  # Add constant for the intercept term

# Make sure the indices match for y_train and X_train_with_intercept
model_sm = sm.OLS(y_train, X_train_with_intercept).fit()

# Print model summary which includes p-values
print(model_sm.summary())

# Extract p-values from the model (excluding the intercept)
p_values = model_sm.pvalues[1:]  # Skip the constant term, we want p-values for features
print("P-values for features:")
print(p_values)

# List statistically significant features (p-value <= 0.06)
# Map p-values back to the actual feature names from the scaled DataFrame
significant_features = p_values[p_values <= 0.06].index.tolist()
print("Statistically significant features:")
print(significant_features)

###############################################################################

from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

results = []

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

elastic_net = ElasticNet()
ridge_model = Ridge()
lasso_model = Lasso()

###############################################################################

print('-------RIDGE----------')

# Instantiate GridSearchCV
grid_search_ridge = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model
grid_search_ridge.fit(X_train, y_train)
print("Best parameters:", grid_search_ridge.best_params_)
print("Best score:", grid_search_ridge.best_score_)
best_model = grid_search_ridge.best_estimator_
test_score = grid_search_ridge.score(X_test, y_test)
print("Test score:", test_score)
y_pred = grid_search_ridge.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

def R2(y_true,y_pred):    
     r2 = r2_score(y_true, y_pred)
     #print 'R2: %2.3f' % r2
     return r2
R2 = R2(y_test, y_pred)
print(f"R-Squared Value: {R2:.2f}")

results.append(['RIDGE',rmse,mae,R2,grid_search_ridge.best_params_])

###############################################################################

print('-------LASSO----------')

# Instantiate GridSearchCV
grid_search_lasso = GridSearchCV(lasso_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model
grid_search_lasso.fit(X_train, y_train)
print("Best parameters:", grid_search_lasso.best_params_)
print("Best score:", grid_search_lasso.best_score_)
best_model = grid_search_lasso.best_estimator_
test_score = grid_search_lasso.score(X_test, y_test)
print("Test score:", test_score)
y_pred = grid_search_lasso.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

def R2(y_true,y_pred):    
     r2 = r2_score(y_true, y_pred)
     #print 'R2: %2.3f' % r2
     return r2
R2 = R2(y_test, y_pred)
print(f"R-Squared Value: {R2:.2f}")

results.append(['LASSO',rmse,mae,R2,grid_search_lasso.best_params_])

###############################################################################

print('-------ELASTIC----------')

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'l1_ratio': np.arange(0, 1.01, 0.1)
}

elastic_net = ElasticNet()

# Instantiate GridSearchCV
grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test score:", test_score)
y_pred = grid_search.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

def R2(y_true,y_pred):    
     r2 = r2_score(y_true, y_pred)
     return r2
R2 = R2(y_test, y_pred)
print(f"R-Squared Value: {R2:.2f}")

results.append(['ELASTIC',rmse,mae,R2, grid_search.best_params_])

###############################################################################

results_df = pd.DataFrame(results, columns=['Model', 'RMSE','MAE','R2', 'PARAMS'])
print(results_df.iloc[:, 0:4])

###############################################################################

print('-------XGBoost----------')

#X = X[significant_features]
import xgboost as xgb
from sklearn.metrics import r2_score
model = xgb.XGBRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)

model.fit(X_train, y_train)
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
print('R2 Score: ', r2)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Root Mean Squared Error: {rmse}")

mae = mean_absolute_error(y_test, preds)
print("Mean Absolute Error:", mae)


###############################################################################

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# Define models
models = [
    ('RandomForest', RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)),
    ('XGBoost', xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)),
    ('LightGBM', lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42))
]

results = {
    'Model':[],
    'R2':[],
    'MAE':[],
    'RMSE':[]
    }

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models:
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
        X_val = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
        y_tr = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
        y_val = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        
    results['Model'].append(model_name)
    results['R2'].append(np.mean(r2_scores))
    results['MAE'].append(np.mean(mae_scores))
    results['RMSE'].append(np.mean(rmse_scores))
    
results_df = pd.DataFrame(results)

print("Cross-validation results: ")
print(results_df)



###############################################################################


print('-------LGBM-------')

model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)

model.fit(X_train, y_train)
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
print('R2 Score: ', r2)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Root Mean Squared Error: {rmse}")

mae = mean_absolute_error(y_test, preds)
print("Mean Absolute Error:", mae)

###############################################################################

predict_data.drop(['Position'], axis=1, inplace=True)
X = predict_data.drop(['Tournament', 'Location', 'Player'], axis=1)
#y = predict_data['Position']

#X = X[significant_features]

predict_data.isnull().sum()
#predict_data.to_csv('Predict_test.csv', index=False)

# Fit and transform the data
scaled_X = scaler.transform(X)
preds = grid_search_ridge.predict(scaled_X)

predict_data['Preds'] = preds.round(0)
predict_data[['Player','Preds']].sort_values('Preds', ascending=True)
from datetime import datetime
last_dw_update = datetime.today().strftime('%Y-%m-%d')
predict_data = predict_data.sort_values('Preds', ascending=True)
predict_data.to_csv(fr'C:\Users\rfo7799\Desktop\Git\TetheredAI\Predictions\LR_Preds_{last_dw_update}.csv')
print(predict_data[['Player','Preds']].sort_values('Preds', ascending=True))


#predict_data.drop(['Position'], axis=1, inplace=True)
X = predict_data.drop(['Tournament', 'Location', 'Player'], axis=1)
#y = predict_data['Position']

X.drop('Preds', axis=1, inplace=True)
preds = model.predict(X)
predict_data['Preds'] = preds
print(predict_data[['Player','Preds']].sort_values('Preds', ascending=True)[:10])
predict_data = predict_data.sort_values('Preds', ascending=True).reset_index()
predict_data.to_csv(fr'C:\Users\rfo7799\Desktop\Git\TetheredAI\Predictions\LGBM_Preds_{last_dw_update}.csv')

###############################################################################

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import classification_report, precision_recall_curve, auc
from xgboost import XGBClassifier

df['Top10_Position'] = df['Position'].apply(lambda x: 1 if x <= 10 else 0)

scale_ratio = round(len(df[df['Top10_Position'] == 0]) / len(df[df['Top10_Position'] == 1]),0)

# Split data
train_data = df[~prediction_mask].copy()
predict_data = df[prediction_mask].copy()

train_data.reset_index(inplace=True, drop=True)

# Assuming train_data is your DataFrame
X = train_data.drop(['Tournament', 'Location', 'Player', 'Position', 'Top10_Position'], axis=1)
y = train_data['Top10_Position']


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_X = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(scaled_X, y)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_param_grid = {'n_estimators':[100, 200],
                 'max_depth':[10,20, None],
                 'min_samples_split':[2,5],
                 'class_weight':['balanced']
                 }
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=kf, scoring='f1', n_jobs=-2)
rf_grid.fit(X_balanced, y_balanced)
print('Best Random Forest Params: ', rf_grid.best_params_)
rf_best = rf_grid.best_estimator_


pos_weight = (y == 0).sum() / (y==1).sum()
xgb_param_grid = {
    'n_estimators':[100,200],
    'max_depth':[3,6],
    'learning_rate':[.01, .1],
    'scale_pos_weight':[pos_weight]
    }

xgb = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv = kf, scoring='f1', n_jobs=-2)
xgb_grid.fit(X_balanced, y_balanced)
print('Best XGBoost Params: ', xgb_grid.best_params_)
xgb_best = xgb_grid.best_estimator_

rf_preds = np.zeros(len(y))
xgb_preds = np.zeros(len(y))
y_true = np.zeros(len(y))

for train_idx, val_idx in kf.split(scaled_X):
    X_train, X_val = scaled_X[train_idx], scaled_X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    rf_best.fit(X_train_bal, y_train_bal)
    xgb_best.fit(X_train_bal, y_train_bal)
    
    rf_preds[val_idx] = rf_best.predict_proba(X_val)[:, 1]
    xgb_preds[val_idx] = xgb_best.predict_proba(X_val)[:, 1]
    y_true[val_idx] = y_val
    
stacked_features = np.column_stack((rf_preds, xgb_preds))

meta_model = LogisticRegression()
meta_model.fit(stacked_features, y_true)
final_pred = meta_model.predict_proba(stacked_features)[:, 1]

precision, recall, _ = precision_recall_curve(y_true, final_pred)
pr_auc = auc(recall, precision)
print(f'Precision-Recall AUC: {pr_auc:.4f}')

threshold = .35
y_pred_binary = (final_pred >= threshold).astype(int)
print(classification_report(y_true, y_pred_binary))

winner_idx = np.argmax(final_pred)
print(f"Predicted winner: Player at index {winner_idx} with probability {final_pred[winner_idx]:.4f}")



predict_data = df[prediction_mask].copy()
X = predict_data.drop(['Tournament', 'Location', 'Player', 'Position', 'Top10_Position'], axis=1)
#y = predict_data['Position']

#X = X[significant_features]

predict_data.isnull().sum()
#predict_data.to_csv('Predict_test.csv', index=False)

# Fit and transform the data
scaled_X = scaler.transform(X)

rf_preds = rf_best.predict_proba(scaled_X)[:, 1]
xgb_preds = xgb_best.predict_proba(scaled_X)[:, 1]
stacked_features = np.column_stack((rf_preds, xgb_preds))
new_preds = meta_model.predict_proba(stacked_features)[: ,1]
predict_data['Preds'] = new_preds

#threshold = .3
#new_y_pred_binary = (new_preds >= threshold).astype(int)
#winner_idx = np.argmax(new_y_pred_binary)

predict_data[['Player','Preds']].sort_values('Preds', ascending=False)
from datetime import datetime
last_dw_update = datetime.today().strftime('%Y-%m-%d')
predict_data = predict_data.sort_values('Preds', ascending=False)
predict_data.to_csv(fr'C:\Users\rfo7799\Desktop\Git\TetheredAI\Predictions\LR_Preds_{last_dw_update}.csv')
print(predict_data[['Player','Preds']].sort_values('Preds', ascending=False))

###############################################################################