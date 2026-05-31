# TetheredAI MLB Production Package

Copy these files into `MLB/mlb_betting_app`.

This package excludes lineup features and focuses on leakage-safe MLB moneyline modeling:

- MLB schedule/results and probable pitchers
- boxscore-derived starter stats
- bullpen workload and performance
- team form and run quality
- team offensive boxscore features
- team-vs-starter-handedness proxies
- Elo pregame ratings
- odds consensus and no-vig market probabilities
- manual champion model export from notebook
- daily GitHub scoring workflow using the champion model
- Streamlit dashboard with MLB Head-2-Head active and Spread/Total placeholders

## Recommended flow

1. GitHub Actions: fetch/update data and build features.
2. Notebook: run `notebooks/04_mlb_tetheredai_model_lab_no_lineups.ipynb` to evaluate/tune models.
3. Notebook: export the selected champion model to `models/mlb_moneyline_champion.joblib`.
4. Commit the champion model and metadata.
5. Daily GitHub workflow scores upcoming games with the committed champion model.
6. Streamlit reads `data/predictions/mlb_moneyline_predictions.csv`.

## No-lineup policy

The feature engineering intentionally avoids projected/confirmed lineup data. Lineups can be added later as a separate feature block, but this package is designed to work without them.
