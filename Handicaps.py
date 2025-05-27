import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Tethered AI Golf Data Analysis", layout="wide")

# Title and Header
st.title("Tethered AI Golf Data Analysis")
st.header("Amateur Golf Handicap Statistics")

# Load data
csv_file = "Handicap Stats.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error("CSV File not found. Please ensure 'Handicap Stats.csv' is in the correct directory.")
    st.stop()

# Organize layout with tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Trends", "Comparisons"])

with tab1:
    st.subheader("Select Handicap Index")
    handicap_options = sorted(df["Handicap"].unique())  # Sort for better UX
    selected_handicap = st.selectbox("Choose a Handicap Index:", handicap_options, key="overview_select")
    filtered_df = df[df["Handicap"] == selected_handicap]

    # Display key metrics for all columns except 'Handicap'
    st.subheader("Key Metrics")
    columns = df.columns[1:]  # Exclude the first column ('Handicap')
    num_columns = len(columns)
    num_cols_per_row = 4  # Number of metrics to display per row
    for i in range(0, num_columns, num_cols_per_row):
        cols = st.columns(num_cols_per_row)
        for j, col_name in enumerate(columns[i:i + num_cols_per_row]):
            with cols[j]:
                st.metric(col_name, f"{filtered_df[col_name].iloc[0]:.2f}")

    # Bar chart (fixed to handle all columns except 'Handicap')
    st.subheader("Statistics by Handicap Index")
    # Reshape the data using pd.melt to create a long-format DataFrame
    melted_df = pd.melt(
        filtered_df,
        id_vars=['Handicap'],
        value_vars=columns,
        var_name='Category',
        value_name='Score'
    )
    fig_bar = px.bar(
        melted_df,
        x='Category',
        y='Score',
        title=f"Statistics for Handicap Index: {selected_handicap}",
        labels={'Score': 'Score', 'Category': 'Category'}
    )
    fig_bar.update_layout(showlegend=False)  # No legend needed since it's a single handicap
    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar chart for selected handicap
    st.subheader("Performance Across Metrics")
    categories = columns.tolist()
    values = filtered_df[columns].iloc[0].values
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=f"Handicap {selected_handicap}"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title=f"Performance Profile for Handicap {selected_handicap}"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.subheader("Trends Across Handicap Levels")
    # Line plot for trends across handicaps
    fig_trend = px.line(
        df,
        x='Handicap',
        y=df.columns[1:],
        title="Performance Trends Across Handicap Levels",
        labels={'value': 'Score', 'variable': 'Metric'}
    )
    fig_trend.update_layout(showlegend=True, legend_title_text='Metrics')
    st.plotly_chart(fig_trend, use_container_width=True)

    # Box plot for score distribution
    st.subheader("Score Distribution Across All Handicaps")
    fig_box = px.box(
        df,
        x='Handicap',
        y='Avg Score to Par',
        title="Distribution of Average Score to Par by Handicap",
        labels={'Avg Score to Par': 'Average Score to Par'}
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Between Metrics")
    corr_matrix = df[df.columns[1:]].corr()
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        title="Correlation Heatmap of Golf Metrics",
        labels=dict(x="Metric", y="Metric", color="Correlation")
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.subheader("Compare Multiple Handicaps")
    selected_handicaps = st.multiselect(
        "Select Handicaps to Compare:", 
        handicap_options, 
        default=[handicap_options[0], handicap_options[-1]],
        key="compare_select"
    )
    if selected_handicaps:
        compare_df = df[df["Handicap"].isin(selected_handicaps)]
        # Bar chart for comparison
        fig_compare = px.bar(
            compare_df,
            x='Handicap',
            y=df.columns[1:],
            barmode='group',
            title="Comparison of Metrics Across Selected Handicaps",
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        fig_compare.update_layout(showlegend=True, legend_title_text='Metrics')
        st.plotly_chart(fig_compare, use_container_width=True)

        # Statistical summary
        st.subheader("Statistical Summary")
        stats = compare_df[df.columns[1:]].describe()
        st.dataframe(stats)

        # Downloadable report
        st.subheader("Download Data")
        csv = compare_df.to_csv(index=False)
        st.download_button(
            label="Download Selected Data as CSV",
            data=csv,
            file_name="selected_handicap_data.csv",
            mime="text/csv"
        )

# Explanatory text
st.sidebar.markdown("""
### About This App
This app analyzes amateur golf handicap statistics, providing insights into performance across different handicap levels. 

**Metrics Explained:**
- **Avg Score to Par**: Average score relative to par across all holes.
- **Par 3/4/5 Avg Score**: Average score on par 3, 4, and 5 holes, respectively.

**Tabs:**
- **Overview**: Detailed stats for a single handicap.
- **Trends**: Trends and distributions across all handicaps.
- **Comparisons**: Compare metrics across multiple handicaps.
""")