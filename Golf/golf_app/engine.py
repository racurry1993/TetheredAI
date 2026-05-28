# engine.py

class GolfEngine:
    def __init__(self, csv_path):
        # MOVE IMPORTS HERE to prevent "Out of Memory" crashes on startup
        try:
            import pandas as pd
            import numpy as np
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            print(f"Library Missing: {e}")
            raise

        try:
            self.df = pd.read_csv(csv_path)
            # Define canonical working names
            self.kpi_cols = ['Up & Down', 'GIR', 'FIR', 'Putts per Rd', 'Avg Drive Dist']
            
            # Flexible mapping: Handles CSV variations
            column_mapping = {
                'Up Downs': 'Up & Down',
                'U&D': 'Up & Down',
                'Putts': 'Putts per Rd',
                'Avg Drive': 'Avg Drive Dist',
                'Dist': 'Avg Drive Dist'
            }
            self.df = self.df.rename(columns=column_mapping)
            
            # Ensure 'Handicap' column is numeric
            self.df['Handicap'] = pd.to_numeric(self.df['Handicap'], errors='coerce')
            self.df = self.df.dropna(subset=['Handicap'])
            
            self.scaler = StandardScaler()
            self.model = Ridge(alpha=1.0)
            self._train()
        except Exception as e:
            print(f"CRITICAL: Engine initialization failed: {e}")
            raise e

    def _train(self):
        # We need to re-import pandas/numpy here if they aren't stored in self
        # But since we used them in __init__, they are loaded in memory for this instance now.
        X = self.df[self.kpi_cols]
        y = self.df['Handicap']
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)

    def get_benchmark_stats(self, target_hcp):
        # Logic to get the row closest to the target handicap
        # Since we are inside the class, we can use the self.df loaded in __init__
        try:
            import numpy as np # Re-import safety
            idx = (self.df['Handicap'] - target_hcp).abs().idxmin()
            row = self.df.loc[idx]
            return {col: row[col] for col in self.kpi_cols}
        except Exception as e:
            print(f"Error getting benchmarks: {e}")
            return {}

    def get_priorities(self, user_avgs, target_hcp):
        try:
            target_hcp = float(target_hcp)
            target_stats = self.get_benchmark_stats(target_hcp)
            
            results = []
            for col in self.kpi_cols:
                u_val = float(user_avgs.get(col, 0))
                t_val = float(target_stats.get(col, 0))
                
                if t_val == 0:
                    gap_pct = 0
                elif col == 'Putts per Rd':
                    gap_pct = ((u_val - t_val) / t_val) * 100
                else:
                    gap_pct = ((t_val - u_val) / t_val) * 100
                
                results.append({
                    'category': col,
                    'user': round(u_val, 1),
                    'target': round(t_val, 1),
                    'gap': round(max(0, gap_pct), 1)
                })
            
            # Use coefficients to weight the gap
            # Coefficients from the trained model
            coeffs = dict(zip(self.kpi_cols, self.model.coef_))
            
            for res in results:
                # Weight by importance (absolute value of coefficient)
                importance = abs(coeffs.get(res['category'], 0))
                res['score'] = res['gap'] * importance

            return sorted(results, key=lambda x: x['score'], reverse=True)
        except Exception as e:
            print(f"Error calculating priorities: {e}")
            return []