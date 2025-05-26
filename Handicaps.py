import streamlit as st
import pandas as pd
import plotly.express as px


st.title("Tethered AI Golf Data Analysis")
st.header("Amateur Golf Handicap Statistics")

csv_file = "Handicap Stats.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error("CSV File not found. Please ensure 'golfers.golfers.csv' is in the correct directory.")
    st.stop()

st.subheader("Select Handicap Index")
handicap_options = df["Handicap"]
selected_handicap = st.selectbox("Choose a Handicap Index:", handicap_options)
filtered_df = df[df["Handicap"] == selected_handicap]

st.write(f"Data for Handicap Index: {selected_handicap}")

st.dataframe(filtered_df)
st.subheader("Statistics by Handicap Index")
fig = px.bar(
    filtered_df,
    x=['Avg Score to Par','Average Score by Hole Type - 3','Average Score by Hole Type - 4','Average Score by Hole Type - 5'],
    y=filtered_df.columns[1:5],
    title=f"Statistics for Handicap Index: {selected_handicap}",
    labels={'value': 'Statistics', 'variable': 'Category'}
)

fig.update_layout(showlegend=True, legend_title_text='Statistics')
st.plotly_chart(fig)

