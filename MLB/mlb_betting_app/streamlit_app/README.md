# TetheredAI Streamlit App

Run from the project root:

```bash
cd MLB/mlb_betting_app
python -m streamlit run streamlit_app/app.py
```

The app expects moneyline predictions at:

```text
data/predictions/mlb_moneyline_predictions.csv
```

The app does not call The Odds API directly. GitHub Actions should refresh the data and commit the latest predictions.
