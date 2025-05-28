import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import os

os.chdir(r'C:\Users\rfo7799\Desktop\Git\TetheredAI')

#https://www.espn.com/golf/schedule

#Manual Input Needed for Future Tournament
tournament_name = '''the Memorial Tournament pres. by Workday'''
tournament_Id = '401703513'

class ESPNGolfScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def get_leaderboard(self, tournament_id, year):
        url = f'https://www.espn.com/golf/leaderboard?tournamentId={tournament_id}'
        #url = 'https://www.espn.com/golf/leaderboard?tournamentId=401703489'
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find leaderboard table
        leaderboard_data = []
        # Find all tables and target the one with the most rows (likely the main leaderboard)
        tables = soup.find_all('tbody', {'class': 'Table__TBODY'})
        leaderboard = max(tables, key=lambda tbl: len(tbl.find_all('tr')), default=None)

        if year == 2025:
            if leaderboard:
                for row in leaderboard.find_all('tr'):
                    cols = row.find_all('td')
                    if len(cols) >= 8:  # Ensure row has enough columns
                        player_data = {
                            'Position': cols[1].text.strip(),
                            'Player': cols[2].text.strip(),
                            'TOPAR': cols[3].text.strip(),
                            'R1': cols[4].text.strip(),
                            'R2': cols[5].text.strip(),
                            'R3': cols[6].text.strip(),
                            'R4': cols[7].text.strip(),
                            'Total': cols[8].text.strip(),
                            'Winnings': cols[9].text.strip() if len(cols) > 8 else '',
                            'FedEx': cols[10].text.strip() if len(cols) > 9 else ''
                        }
                        leaderboard_data.append(player_data)
        else:
            try:
                # Initialize leaderboard data
                leaderboard_data = []
                rows = soup.select('tbody.Table__TBODY tr')

                for row in rows:
                    if not row.select_one('td'):
                        continue
                    cells = row.select('td')
                    if len(cells) < 8:  # Basic validation for required columns
                        continue
                        
                    player_data = {
                        'position': cells[1].text.strip(),
                        'player': cells[2].text.strip(),
                        'to_par': cells[3].text.strip(),
                        'today': cells[4].text.strip(),
                        'thru': cells[5].text.strip(),
                        'round_1': cells[6].text.strip(),
                        'round_2': cells[7].text.strip(),
                        'round_3': cells[8].text.strip(),
                        'round_4': cells[9].text.strip() if len(cells) > 8 else '',
                        'total': cells[10].text.strip() if len(cells) > 9 else ''
                    }
                    leaderboard_data.append(player_data)
            except Exception as e:
                print(f"Error processing row: {e}")
        df = pd.DataFrame(leaderboard_data)
        
        return df
    
    def scrape_golf_schedule(self, url):
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all table containers
            table_containers = soup.find_all('div', {'class': 'Table__Scroller'})
            
            all_tournaments = []
            
            for table in table_containers:
                # Process each table separately
                tournament_rows = table.find_all('tr', {'class': 'Table__TR'})
                
                for row in tournament_rows:
                    tournament_data = self._extract_tournament_data(row)
                    if tournament_data:
                        all_tournaments.append(tournament_data)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_tournaments)
            print(f"Extracted {len(df)} tournaments")
            return df
        
    def _extract_tournament_data(self, row):
        """Extract tournament data from a single row"""
        try:
            # Find tournament link and ID
            tournament_link = row.select_one('a[href*="tournamentId"]')
            if not tournament_link:
                return None
                
            tournament_id_match = re.search(r'tournamentId=(\d+)', tournament_link['href'])
            if not tournament_id_match:
                return None
                
            tournament_id = tournament_id_match.group(1)
            
            # Find location and name
            location_div = row.select_one('.eventAndLocation__tournamentLocation')
            location_name = row.select_one('.eventAndLocation__tournamentLink')
            
            # Find date (checking both possible date column classes)
            date_cell = (row.select_one('td.dateAndTickets__col') or 
                        row.select_one('td.dateRange__col'))
            
            if not all([location_div, location_name, date_cell]):
                return None
                
            return {
                'tournament_id': tournament_id,
                'location': location_div.text.strip(),
                'Name': location_name.text.strip(),
                'Date': date_cell.text.strip()
            }
            
        except Exception as e:
            print(f"Error processing row: {e}")
            return None

scraper = ESPNGolfScraper() 
#url = 'https://www.espn.com/golf/schedule'
url = [(2025, 'https://www.espn.com/golf/schedule'), (2024, 'https://www.espn.com/golf/schedule/_/season/2024'), (2023, 'https://www.espn.com/golf/schedule/_/season/2023')]
full_tournament_data = pd.DataFrame()
for web in url:
    tournaments = scraper.scrape_golf_schedule(web[1])
    #tournaments = tournaments[tournaments['tournament_id'] != '401703492']
    for index, row in tournaments.iterrows():
        tournament_id = row['tournament_id']
        location = row['location']
        name = row['Name']
        date = row['Date']
        
        # Fetch leaderboard data
        tournament_data = scraper.get_leaderboard(tournament_id, web[0])
        tournament_data['Tournament'] = name
        tournament_data['Location'] = location
        tournament_data['Year'] = web[0]
        tournament_data['Date'] = date
        
        full_tournament_data = pd.concat([full_tournament_data, tournament_data], axis=0)
        
print("\nFinal Tournament Data:")
print(full_tournament_data.head())
#full_tournament_data[full_tournament_data['Year'] == 2025]['Tournament'].unique()
current_tournament_data = full_tournament_data.iloc[:, :13]
prior_tournament_data = pd.concat(
[full_tournament_data.iloc[:, :4], full_tournament_data.iloc[:, 14:]],
axis=1)
prior_tournament_data.rename({'position':'Position', 'player':'Player', 'to_par':'TOPAR',
                              'round_1':'R3', 'round_2':'R4', 'round_3':'Total', 'round_4':'Winnings',
                              'total':'FedEx', 'today':'R1', 'thru':'R2'}, axis=1, inplace=True)


full_tournament_data = pd.concat([current_tournament_data, prior_tournament_data], axis=0)
full_tournament_data = full_tournament_data.dropna(subset=['Player','R1'])
# Function to handle both formats
# Function to handle both formats
def parse_date(row):
    year = str(row['Year'])
    date_str = row['Date']

    # Case 1: Range format (e.g., "Feb 29 - Mar 3")
    if '-' in date_str and any(char.isdigit() for char in date_str.split('-')[-1]):
        parts = date_str.split('-')
        start_part = parts[0].strip()  # "Feb 29"
        end_part = parts[-1].strip()  # "Mar 3" or "5"
        
        if len(end_part.split()) == 2:  # If the end part includes both month and day (e.g., "Mar 3")
            month_day_end = end_part
        else:  # If the end part only includes the day (e.g., "5")
            month_day_end = start_part.split()[0] + ' ' + end_part  # Use the month from the start part

        final_date_str = f"{year} {month_day_end}"  # Combine into '2024 Mar 3'
        return pd.to_datetime(final_date_str, format='%Y %b %d', errors='coerce')
    
    # Case 2: Single date format (e.g., "17-Dec")
    elif '-' in date_str:
        final_date_str = date_str + '-' + year  # Example: '17-Dec-2024'
        return pd.to_datetime(final_date_str, format='%d-%b-%Y', errors='coerce')
    
    else:
        return pd.NaT

# Apply function to each row
full_tournament_data['Final_Date'] = full_tournament_data.apply(parse_date, axis=1)
full_tournament_data.drop(['FedEx', 'Date'], axis=1, inplace=True)
full_tournament_data.dropna(subset=['Final_Date', 'R1', 'Player'], inplace=True)
full_tournament_data = full_tournament_data[full_tournament_data['Tournament'] != 'Presidents Cup']
full_tournament_data = full_tournament_data[full_tournament_data['Player'] != '']
full_tournament_data['Tournament'] = full_tournament_data['Tournament'].apply(lambda x: 'The Sentry' if x == 'Sentry Tournament of Champions' else x)

full_tournament_data['Position'] = (full_tournament_data['Position'].str.replace('T', '', regex=True).replace('-', '-1').replace('', '-1').astype(float))

# Assuming 'full_tournament_data' is already sorted by 'Player' and 'Date'
full_tournament_data = full_tournament_data.sort_values(by=['Player', 'Final_Date'], ascending=[True, False])
# Ensure 'Final_Date' is in datetime format for sorting
full_tournament_data['Final_Date'] = pd.to_datetime(full_tournament_data['Final_Date'], errors='coerce')

# Drop any duplicates (if applicable)
full_tournament_data = full_tournament_data.drop_duplicates()

# Sort the data by 'Player' and 'Final_Date' to ensure chronological order
full_tournament_data = full_tournament_data.sort_values(by=['Player', 'Final_Date'], ascending=[True, True])
#full_tournament_data.to_csv(r'Test.csv', index=False)

url = f'https://www.espn.com/golf/leaderboard?tournamentId={tournament_Id}'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    page_content = response.content
else:
    print(f'Failed to retrieve the page. Status code: {response.status_code}')

soup = BeautifulSoup(page_content, 'html.parser')
leaderboard_data = []
tables = soup.find_all('tbody', {'class': 'Table__TBODY'})
course = soup.find('h1', {'class':'headline headline__h1 Leaderboard__Event__Title'})
course = course.text.strip()
date = soup.find('span', {'class':'Leaderboard__Event__Date n7'})
date = date.text.strip()
try:
    location = soup.find('div', {'class':'Leaderboard__Course__Location n8 clr-gray-05'})
    location = location.text.strip()
except:
    location = soup.find('div', {'class':'Leaderboard__Course__Location n8 clr-gray-04'})
    location = location.text.strip()

# Extract the last date and year
try:
    month_day_range, year = date.rsplit(", ", 1)  # Split into 'January 16 - 19' and '2025'
    month, day_range = month_day_range.split(" ", 1)  # Split into 'January' and '16 - 19'
    end_day = day_range.split(" - ")[-1]  # Extract the last day

    from datetime import datetime
    # Format the last date
    formatted_date = datetime.strptime(f"{month} {end_day}, {year}", "%B %d, %Y").strftime("%m/%d/%Y")
except:
    # Split the range
    parts = date.split(" - ")
    
    # Extract the final date part
    final_date_part = parts[1]
    
    # Ensure the year is included in the final part
    if "," not in final_date_part:
        final_date_part += ", " + parts[0].split(" ")[-1]  # Add the year from the first part
    
    # Convert to datetime object
    final_date = datetime.strptime(final_date_part, "%B %d, %Y")
    
    # Format as YYYY-MM-DD
    formatted_date = final_date.strftime("%Y-%m-%d")


leaderboard = max(tables, key=lambda tbl: len(tbl.find_all('tr')), default=None)
for row in leaderboard.find_all('tr'):
    cols = row.find_all('td')
    player_data = {
        'Tournament':course,
        'Location':location,
        'Year': int(year),
        'Player': cols[1].text.strip(),
        'Final_Date': formatted_date
        
        }
    leaderboard_data.append(player_data)

leaderboard_data = pd.DataFrame(leaderboard_data)
leaderboard_data['Final_Date'] = pd.to_datetime(leaderboard_data['Final_Date'], errors='coerce')
full_tournament_data = pd.concat([full_tournament_data, leaderboard_data], axis=0)


# Function to extract state from the Location string
def _extract_state(location):
    """Extract state from location string."""
    state_mapping = {
        'HI': 'Hawaii', 'CA': 'California', 'FL': 'Florida', 'TX': 'Texas', 'AZ':'Arizona',
        'CT':'Connecticut', 'TN':'Tennessee', 'MN':'Minnesota', 'MS':'Mississippi', 'NV':'Nevada',
        'NC': 'North Carolina', 'PR': 'Puerto Rico', 'MX': 'Mexico', 'Bermuda': 'Bermuda',
        'Scotland': 'Scotland', 'ON': 'Canada (ON)', 'Japan': 'International (Other Regions)', 
        'GA': 'Georgia', 'OH':'Ohio', 'KY': 'Kentucky', 'SC': 'South Carolina', 'CO': 'Colorado', 
        'MI': 'Michigan', 'IL': 'Illinois', 'NY': 'New York'
    }
    
    for state, region in state_mapping.items():
        if state in location:
            return state
    return None  # If no state is found

# Function to map state to region group
def _map_to_region(state):
    """Map state to region group."""
    region_group_mapping = {
        'West': ['CA', 'HI', 'WA', 'OR', 'NV', 'AZ', 'CO', 'ID', 'MT', 'NM', 'UT', 'AK'],
        'Midwest': ['IL', 'IN', 'IA', 'MI', 'MN', 'MO', 'NE', 'OH', 'WI'],
        'South': ['FL', 'GA', 'KY', 'NC', 'SC', 'TN', 'TX', 'AL', 'MS', 'LA', 'AR'],
        'Northeast': ['NY', 'NJ', 'PA', 'CT', 'RI', 'MA', 'VT', 'NH', 'ME'],
        'International': ['MX', 'Bermuda', 'Japan', 'Scotland', 'Canada', 'ON'],
        'Other': ['PR']  # Puerto Rico
    }
    
    if state in region_group_mapping['West']:
        return 'West'
    elif state in region_group_mapping['Midwest']:
        return 'Midwest'
    elif state in region_group_mapping['South']:
        return 'South'
    elif state in region_group_mapping['Northeast']:
        return 'Northeast'
    elif state in region_group_mapping['International']:
        return 'International'
    elif state in region_group_mapping['Other']:
        return 'Other'
    return 'Unknown'

full_tournament_data['State'] = full_tournament_data['Location'].apply(_extract_state)
full_tournament_data['Region'] = full_tournament_data['State'].apply(_map_to_region)

#full_tournament_data.to_csv(r'full_tournament_data.csv', index=False)

def get_last_3_places(player_data):
    # Sort the player data by Final_Date, then reset the index to ensure unique access
    player_data = player_data.sort_values(by='Final_Date', ascending=True)
    
    # Calculate the last 3 positions (using a rolling window)
    player_data['Last T1 Finish'] = player_data['Position'].shift(1)  # Most recent finish
    player_data['Last T2 Finish'] = player_data['Position'].shift(2)  # Second most recent finish
    player_data['Last T3 Finish'] = player_data['Position'].shift(3)  # Third most recent finish
    
    return player_data

# Apply the function to each player
full_tournament_data = full_tournament_data.groupby('Player').apply(get_last_3_places)

# Reset index if needed (to clean up after groupby)
full_tournament_data.reset_index(drop=True, inplace=True)



# Ensure Final_Date is in datetime format
full_tournament_data['Final_Date'] = pd.to_datetime(full_tournament_data['Final_Date'], errors='coerce')
full_tournament_data['Month'] = full_tournament_data['Final_Date'].dt.month

full_tournament_data = full_tournament_data.sort_values(by=['Player','Tournament','Year'], ascending=[True, True, True])
def get_previous_year_position(row, player_data):
    current_date = row['Final_Date']
    player = row['Player']
    current_year = row['Year']
    tournament = row['Tournament']
    
    previous_year_data = player_data[(player_data['Player'] == player) & (player_data['Year'] == current_year - 1) & (player_data['Tournament'] == tournament)]
    if previous_year_data.empty:
        return None  # If no tournaments in the previous year, return None
    return previous_year_data['Position'].values[0]

full_tournament_data['Previous_Year_Position'] = full_tournament_data.apply(lambda row: get_previous_year_position(row, full_tournament_data), axis=1)
#full_tournament_data = full_tournament_data[full_tournament_data['Year'] != 2023]
full_tournament_data['Previous_Year_Position'] = full_tournament_data['Previous_Year_Position'].fillna(100)
full_tournament_data['Previous_Year_Position'] = full_tournament_data['Previous_Year_Position'].apply(lambda x: 100 if x == -1 else x)
full_tournament_data = full_tournament_data.sort_values(by=['Player','Final_Date'], ascending=[True, True])
full_tournament_data['Days_Since_Last_Tournament'] = (
    full_tournament_data.groupby('Player')['Final_Date']
    .apply(lambda x: x - x.shift(1))
    .dt.days
)    

full_tournament_data.to_csv(r'full_tournament_data.csv', index=False)

########################################################################################3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import os

def convert_data_types(df):
    """Convert columns to appropriate data types"""
    df = df.copy()
    # Convert Position to numeric, replacing 'CUT' with -1
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(-1)
    
    # Convert TOPAR to numeric
    df['TOPAR'] = pd.to_numeric(df['TOPAR'], errors='coerce')
    
    # Convert round scores and totals to numeric
    for col in ['R1', 'R2', 'R3', 'R4', 'Total']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Winnings to numeric, removing currency symbols and commas
    df['Winnings'] = df['Winnings'].replace('[\$,]', '', regex=True)
    df['Winnings'] = pd.to_numeric(df['Winnings'], errors='coerce')
    
    return df

def normalize_positions(group):
    """Normalize positions within a tournament group"""
    group = group.copy()
    # Handle missed cuts
    group.loc[group['Position'] == -1, 'Normalized_Position'] = group['Position'].max() + 1
    # Handle regular positions
    group.loc[group['Position'] != -1, 'Normalized_Position'] = group.loc[group['Position'] != -1, 'Position']
    return group

def handle_missed_cuts(df):
    """Handle missing data for missed cuts"""
    df = df.copy()
    missed_cut_mask = df['Position'] == -1
    
    # Fill missing values for missed cuts
    round_cols = ['R3', 'R4', 'Total', 'Winnings']
    for col in round_cols:
        df.loc[missed_cut_mask, col] = 0
    
    # Set TOPAR for missed cuts
    worst_topar = df[df['Position'] != -1]['TOPAR'].max()
    df.loc[missed_cut_mask, 'TOPAR'] = worst_topar + 5 if not pd.isna(worst_topar) else 10
    
    return df

def create_performance_features(df):
    """Create performance-based features"""
    df = df.copy()
    df = df.sort_values(['Player', 'Final_Date'])
    
    windows = [2, 3, 5, 7, 10]
    for window in windows:
        # Moving averages for positions
        df[f'Position_ma_{window}'] = (
            df.groupby('Player')['Normalized_Position']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Position volatility
        df[f'Position_std_{window}'] = (
            df.groupby('Player')['Normalized_Position']
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        
        # TOPAR trends
        df[f'TOPAR_trend_{window}'] = (
            df.groupby('Player')['TOPAR']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    # Momentum features
    df['Position_momentum_5'] = df['Position_ma_5'] - df['Position_ma_10']
    df['Position_momentum_3'] = df['Position_ma_3'] - df['Position_ma_5']
    
    # Cut making features
    df['Made_Cut'] = (df['Position'] != -1).astype(float)  # Changed to float
    df['Cut_streak'] = (
        df.groupby('Player')['Made_Cut']
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(0, drop=True)
    )
    
    return df

def create_golf_specific_features(df):
    """Create golf-specific performance metrics"""
    df = df.copy()
    df = df.sort_values(['Player','Final_Date'], ascending=[True,False])
    # Round performance metrics
    df['Early_Rounds_Avg'] = df[['R1', 'R2']].mean(axis=1)
    df['Late_Rounds_Avg'] = df[['R3', 'R4']].mean(axis=1)
    df['Last_3_Early_Rounds_Avg'] = df.groupby('Player')['Early_Rounds_Avg'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['Last_3_Late_Rounds_Avg'] = df.groupby('Player')['Late_Rounds_Avg'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    
    # Pull prior tournament averages for R1, R2, R3, and R4
    df['Prior_R1_Avg'] = df.groupby('Player')['R1'].shift(-1)
    df['Prior_R2_Avg'] = df.groupby('Player')['R2'].shift(-1)
    df['Prior_R3_Avg'] = df.groupby('Player')['R3'].shift(-1)
    df['Prior_R4_Avg'] = df.groupby('Player')['R4'].shift(-1)
    df['Round_Progression'] = df['Last_3_Late_Rounds_Avg'] - df['Last_3_Early_Rounds_Avg']
    
    # Course performance
    df['Course_Par'] = df['Total'] - df['TOPAR']
    #df['Weekend_Performance'] = df[['R3', 'R4']].mean(axis=1) - df[['R1', 'R2']].mean(axis=1)
    
    # Recent form features
    df = df.sort_values(['Player', 'Final_Date'])
    
    # Position trend
    df['Position_Trend'] = df.groupby('Player')['Position'].transform(
        lambda x: np.polyfit(range(len(x[-5:])), x[-5:], 1)[0]
        if len(x) >= 5 else np.nan
    )
    
    # Performance ratios - Fixed to handle boolean operations
    for window in [5, 10, 20]:
        # Convert boolean to float before calculating mean
        df[f'Win_Ratio_{window}'] = df.groupby('Player')['Position'].transform(
            lambda x: (x == 1).astype(float).rolling(window=window, min_periods=1).mean()
        )
        df[f'Top10_Ratio_{window}'] = df.groupby('Player')['Position'].transform(
            lambda x: (x <= 10).astype(float).rolling(window=window, min_periods=1).mean()
        )
    
    # Tournament metrics
    df['Avg_Tournament_Score'] = df.groupby('Tournament_ID')['TOPAR'].transform('mean')
    #df['Score_vs_Field'] = df['TOPAR'] - df['Avg_Tournament_Score']
    
    # Consistency metrics
    completed_rounds_mask = (df['Position'] != -1)
    df['Score_Variance'] = np.nan
    df.loc[completed_rounds_mask, 'Score_Variance'] = df.loc[completed_rounds_mask, ['R1', 'R2', 'R3', 'R4']].std(axis=1)
    
    return df

def prepare_prediction_data(df, tournament_name, year):
    """Prepare data for prediction"""
    df = df.copy()
    
    # Convert data types and create base features
    df = convert_data_types(df)
    df['Tournament_ID'] = df['Tournament'] + '_' + df['Year'].astype(str)
    df = df.groupby('Tournament_ID').apply(normalize_positions)
    df = handle_missed_cuts(df)
    
    # Convert dates and create features
    df['Final_Date'] = pd.to_datetime(df['Final_Date'])
    df = create_performance_features(df)
    df = create_golf_specific_features(df)
    
    # Create prediction mask
    prediction_mask = (df['Tournament'] == tournament_name) & (df['Year'] == year)
    
    # Split data
    train_data = df[~prediction_mask].copy()
    predict_data = df[prediction_mask].copy()
    
    # Define features
    feature_columns = [
        'Position_ma_2', 'Position_std_2', 'TOPAR_trend_2',
        'Position_ma_3', 'Position_std_3', 'TOPAR_trend_3',
        'Position_ma_5', 'Position_std_5', 'TOPAR_trend_5',
        'Position_momentum_5', 'Position_momentum_3',
        'Cut_streak','Prior_R1_Avg','Prior_R2_Avg','Prior_R3_Avg','Prior_R4_Avg',
        'Last_3_Early_Rounds_Avg','Last_3_Late_Rounds_Avg',
        'Score_Variance', 'Round_Progression','Position_Trend',
        'Win_Ratio_5', 'Win_Ratio_10', 'Win_Ratio_20',
        'Top10_Ratio_5', 'Top10_Ratio_10', 'Top10_Ratio_20'
    ]
    
    # Handle missing values
    for col in feature_columns:
        if col in ['Position_Trend', 'Score_vs_Field']:
            fill_value = 0
        elif col.startswith(('Win_Ratio', 'Top10_Ratio')):
            fill_value = 0
        else:
            fill_value = train_data[col].median()
            
        train_data[col] = train_data[col].fillna(fill_value)
        predict_data[col] = predict_data[col].fillna(fill_value)
    
    X_train = train_data[feature_columns]
    y_train = train_data['Normalized_Position']
    X_predict = predict_data[feature_columns]
    
    return X_train, y_train, X_predict, predict_data, feature_columns

def train_and_predict(df, tournament_name, year):
    """Train models and make predictions"""
    # Prepare data
    X_train, y_train, X_predict, predict_data, feature_columns = prepare_prediction_data(
        df, tournament_name, year
    )
    
    # Define models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    }
    
    # Train and evaluate models
    best_rmse = float('inf')
    best_model = None
    best_predictions = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, 
                                      scoring='neg_mean_squared_error', 
                                      cv=5).mean())
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_predictions = model.predict(X_predict)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Player': predict_data['Player'],
        'Predicted_Position': best_predictions
    })
    
    results = results.sort_values('Predicted_Position')
    
    # Save predictions
    last_update = datetime.today().strftime('%Y-%m-%d')
    folder_path = 'Predictions'
    os.makedirs(folder_path, exist_ok=True)
    results.to_csv(f'{folder_path}/predictions_{last_update}.csv', index=False)
    
    return results, best_model, feature_columns

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance of the model"""
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    return importances.sort_values('importance', ascending=False)

def print_metrics(metrics):
    """
    Print formatted metrics
    """
    print("\nModel Performance Metrics:")
    print("\nCross-Validation Metrics (5-fold):")
    print(f"RMSE: {metrics['cv_rmse_mean']:.2f} (±{metrics['cv_rmse_std']:.2f})")
    print(f"MAE: {metrics['cv_mae_mean']:.2f} (±{metrics['cv_mae_std']:.2f})")
    print(f"R²: {metrics['cv_r2_mean']:.3f} (±{metrics['cv_r2_std']:.3f})")
    
    print("\nTraining Metrics:")
    print(f"RMSE: {metrics['train_rmse']:.2f}")
    print(f"MAE: {metrics['train_mae']:.2f}")
    print(f"R²: {metrics['train_r2']:.3f}")
    
    if 'test_rmse' in metrics:
        print("\nTest Metrics:")
        print(f"RMSE: {metrics['test_rmse']:.2f}")
        print(f"MAE: {metrics['test_mae']:.2f}")
        print(f"R²: {metrics['test_r2']:.3f}")


# Make predictions
results, best_model, feature_cols = train_and_predict(full_tournament_data, tournament_name, 2025)

# Analyze feature importance
importance = analyze_feature_importance(best_model, feature_cols)

print("Predictions:")
print(results)
print("\nFeature Importance:")
print(importance)

min_val = min(results['Predicted_Position'])
import math
mapped_values = [math.ceil(1 + (val - min_val)) for val in results['Predicted_Position']]
results['Normalized_Predictions'] = mapped_values

from datetime import datetime
last_dw_update = datetime.today().strftime('%Y-%m-%d')
import os
folder_path = r'C:\Users\rfo7799\Desktop\Git\TetheredAI\Predictions'
os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
results.to_csv(f'{folder_path}\\All_Results_{last_dw_update}.csv', index=False)
print(results[:50])
