import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr # Import pearsonr for correlation calculation

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

# --- Function to generate personalized handicap advice ---
def generate_handicap_advice(df_full, current_handicap_focus):
    """
    Generates personalized advice based on correlation analysis of the full dataset.

    Args:
        df_full (pd.DataFrame): The complete DataFrame containing all handicap data.
        current_handicap_focus (int): The current handicap for which to generate advice.

    Returns:
        str: A markdown-formatted string containing the personalized advice.
    """
    numerical_cols = df_full.select_dtypes(include=['number']).columns.tolist()
    if 'Handicap' in numerical_cols:
        numerical_cols.remove('Handicap') # Remove Handicap itself from features to correlate

    if not numerical_cols:
        return "No other numerical columns found for correlation analysis besides 'Handicap' to generate advice."

    correlations = {}
    for col in numerical_cols:
        # Calculate Pearson correlation between the feature and 'Handicap'
        # This is done on the full dataset to understand overall drivers
        corr, _ = pearsonr(df_full[col], df_full['Handicap'])
        correlations[col] = corr

    # Sort correlations to find the strongest drivers (negative correlations are better for handicap)
    sorted_correlations = sorted(correlations.items(), key=lambda item: item[1])

    advice_lines = []
    target_handicap = current_handicap_focus - 1
    advice_lines.append(f"### Personalized Advice for a {current_handicap_focus} Handicap Golfer")
    advice_lines.append(f"To move from a {current_handicap_focus} handicap towards a {target_handicap} handicap, focus on the following key areas based on your data:")

    advice_given = False
    # Recommend improving stats with strong negative correlation (increase these values)
    # A negative correlation means as this stat increases, handicap decreases (good)
    for feature, corr_val in sorted_correlations:
        if corr_val < -0.3: # Using a threshold of -0.3 for 'strong' negative correlation
            advice_lines.append(f"  - **Improve {feature}:** This metric has a strong negative correlation ({corr_val:.2f}) with handicap. Increasing your performance in {feature} (e.g., hitting the ball further, hitting more greens, etc.) is highly likely to reduce your handicap.")
            advice_given = True

    # Recommend improving stats with strong positive correlation (decrease these values)
    # A positive correlation means as this stat increases, handicap increases (bad)
    # Iterate in reverse to get strongest positive correlations first
    for feature, corr_val in reversed(sorted_correlations):
        if corr_val > 0.3: # Using a threshold of 0.3 for 'strong' positive correlation
            advice_lines.append(f"  - **Reduce {feature}:** This metric has a strong positive correlation ({corr_val:.2f}) with handicap. Decreasing your performance in {feature} (e.g., fewer putts, fewer penalty strokes, etc.) is highly likely to reduce your handicap.")
            advice_given = True

    if not advice_given:
        advice_lines.append("No strong correlations found to provide specific advice based on the current data and thresholds. Your golf game might have a balanced set of strengths and weaknesses, or more data might be needed.")

    return "\n".join(advice_lines)

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
                # Ensure the column exists in filtered_df before trying to access it
                if col_name in filtered_df.columns and not filtered_df.empty:
                    st.metric(col_name, f"{filtered_df[col_name].iloc[0]:.2f}")
                else:
                    st.metric(col_name, "N/A") # Handle cases where data might be missing for a specific handicap

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

    # --- Integrate Personalized Advice Here ---
    st.markdown("---") # Add a separator for better visual organization
    if selected_handicap: # Ensure a handicap is selected before generating advice
        advice_text = generate_handicap_advice(df, selected_handicap)
        st.markdown(advice_text)
    else:
        st.info("Select a handicap index above to receive personalized improvement advice.")


with tab2:
    st.subheader("Trends Across Handicap Levels")

    # Correlation plots sorted by importance
    st.markdown("### Metric Trends by Handicap (Sorted by Correlation Importance)")
    st.markdown("""
        These plots show how each golf metric changes across different handicap levels.
        The charts are ordered by the **absolute strength of their correlation with Handicap**,
        meaning metrics with the strongest impact (positive or negative) on handicap are shown first.
    """)

    numerical_cols_for_trends = df.select_dtypes(include=['number']).columns.tolist()
    if 'Handicap' in numerical_cols_for_trends:
        numerical_cols_for_trends.remove('Handicap')

    if numerical_cols_for_trends:
        # Calculate correlations for sorting
        correlations_for_sorting = {}
        for col in numerical_cols_for_trends:
            corr, _ = pearsonr(df[col], df['Handicap'])
            correlations_for_sorting[col] = abs(corr) # Use absolute correlation for importance

        # Sort metrics by absolute correlation in descending order
        sorted_metrics_by_importance = sorted(correlations_for_sorting.items(), key=lambda item: item[1], reverse=True)

        for metric, abs_corr in sorted_metrics_by_importance:
            fig_metric_trend = px.line(
                df,
                x='Handicap',
                y=metric,
                title=f"Trend: {metric} vs. Handicap (Abs. Correlation: {abs_corr:.2f})",
                labels={'Handicap': 'Handicap Index', metric: metric}
            )
            fig_metric_trend.update_traces(mode='lines+markers') # Show points as well
            fig_metric_trend.update_layout(hovermode="x unified") # Unified hover for better comparison
            st.plotly_chart(fig_metric_trend, use_container_width=True)
    else:
        st.info("No numerical metrics available to plot trends by handicap.")


    # Box plot for score distribution
    st.subheader("Score Distribution Across All Handicaps")
    # Check if 'Avg Score to Par' exists before plotting
    if 'Avg Score to Par' in df.columns:
        fig_box = px.box(
            df,
            x='Handicap',
            y='Avg Score to Par',
            title="Distribution of Average Score to Par by Handicap",
            labels={'Avg Score to Par': 'Average Score to Par'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("Column 'Avg Score to Par' not found for box plot.")


    # Correlation heatmap
    st.subheader("Overall Correlation Between Metrics")
    # Ensure there are enough numerical columns for correlation calculation
    if len(df.columns[1:]) > 1:
        corr_matrix = df[df.columns[1:]].corr()
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Correlation Heatmap of Golf Metrics",
            labels=dict(x="Metric", y="Metric", color="Correlation")
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Not enough numerical columns to generate a meaningful correlation heatmap.")


with tab3:
    st.subheader("Compare Multiple Handicaps")
    selected_handicaps = st.multiselect(
        "Select Handicaps to Compare:",
        handicap_options,
        default=[handicap_options[0], handicap_options[-1]] if handicap_options else [], # Handle empty options
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
    else:
        st.info("Select at least one handicap to compare.")

# Explanatory text
st.sidebar.markdown("""
### About This App
This app analyzes amateur golf handicap statistics, providing insights into performance across different handicap levels.

**Metrics Explained:**
- **Avg Score to Par**: Average score relative to par across all holes.
- **Par 3/4/5 Avg Score**: Average score on par 3, 4, and 5 holes, respectively.

**Tabs:**
- **Overview**: Detailed stats for a single handicap, including personalized advice.
- **Trends**: Trends and distributions across all handicaps, with metrics sorted by their correlation importance to handicap.
- **Comparisons**: Compare metrics across multiple handicaps.
""")
