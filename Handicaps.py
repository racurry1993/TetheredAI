import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr  # Import pearsonr for correlation calculation

# Page configuration
st.set_page_config(page_title="Tethered AI Golf Data Analysis", layout="wide")

# Function to load data
def load_data():
    csv_file = "Handicap Stats.csv"
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        st.error("CSV File not found. Please ensure 'Handicap Stats.csv' is in the correct directory.")
        st.stop()

# Function to generate personalized handicap advice
def generate_handicap_advice(df_full, current_handicap_focus):
    numerical_cols = df_full.select_dtypes(include=['number']).columns.tolist()
    if 'Handicap' in numerical_cols:
        numerical_cols.remove('Handicap')  # Remove Handicap itself from features to correlate

    if not numerical_cols:
        return "No other numerical columns found for correlation analysis besides 'Handicap' to generate advice."

    correlations = {}
    for col in numerical_cols:
        corr, _ = pearsonr(df_full[col], df_full['Handicap'])
        correlations[col] = corr

    sorted_correlations = sorted(correlations.items(), key=lambda item: item[1])

    advice_lines = []
    target_handicap = current_handicap_focus - 1
    advice_lines.append(f"### Personalized Advice for a {current_handicap_focus} Handicap Golfer")
    advice_lines.append(f"To move from a {current_handicap_focus} handicap towards a {target_handicap} handicap, focus on the following key areas based on your data:")

    advice_given = False
    for feature, corr_val in sorted_correlations:
        if corr_val < -0.3:
            advice_lines.append(f"  - **Improve {feature}:** This metric has a strong negative correlation ({corr_val:.2f}) with handicap. Increasing your performance in {feature} (e.g., hitting the ball further, hitting more greens, etc.) is highly likely to reduce your handicap.")
            advice_given = True

    for feature, corr_val in reversed(sorted_correlations):
        if corr_val > 0.3:
            advice_lines.append(f"  - **Reduce {feature}:** This metric has a strong positive correlation ({corr_val:.2f}) with handicap. Decreasing your performance in {feature} (e.g., fewer putts, fewer penalty strokes, etc.) is highly likely to reduce your handicap.")
            advice_given = True

    if not advice_given:
        advice_lines.append("No strong correlations found to provide specific advice based on the current data and thresholds. Your golf game might have a balanced set of strengths and weaknesses, or more data might be needed.")

    return "\n".join(advice_lines)

# Landing page
def landing_page():
    st.title("Welcome to Tethered AI")
    st.markdown("Select an analysis option below to explore golf statistics:")
    
    analysis_option = st.selectbox(
        "Choose Analysis Type:",
        ["Amateur Handicap Analysis", "Professional Golf Tournament Analysis"],
        key="landing_select"
    )
    
    return analysis_option

# Amateur Handicap Analysis page
def amateur_handicap_analysis():
    df = load_data()
    
    st.title("Tethered AI Golf Data Analysis")
    st.header("Amateur Golf Handicap Statistics")
    
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
                    if col_name in filtered_df.columns and not filtered_df.empty:
                        st.metric(col_name, f"{filtered_df[col_name].iloc[0]:.2f}")
                    else:
                        st.metric(col_name, "N/A")

        # Personalized Advice
        st.markdown("---")
        if selected_handicap:
            advice_text = generate_handicap_advice(df, selected_handicap)
            st.markdown(advice_text)
        else:
            st.info("Select a handicap index above to receive personalized improvement advice.")

    with tab2:
        st.subheader("Trends Across Handicap Levels")
        st.markdown("""
            These plots show how each golf metric changes across different handicap levels.
            The charts are ordered by the **absolute strength of their correlation with Handicap**,
            meaning metrics with the strongest impact (positive or negative) on handicap are shown first.
        """)

        numerical_cols_for_trends = df.select_dtypes(include=['number']).columns.tolist()
        if 'Handicap' in numerical_cols_for_trends:
            numerical_cols_for_trends.remove('Handicap')

        if numerical_cols_for_trends:
            correlations_for_sorting = {}
            for col in numerical_cols_for_trends:
                corr, _ = pearsonr(df[col], df['Handicap'])
                correlations_for_sorting[col] = abs(corr)

            sorted_metrics_by_importance = sorted(correlations_for_sorting.items(), key=lambda item: item[1], reverse=True)

            for metric, abs_corr in sorted_metrics_by_importance:
                fig_metric_trend = px.line(
                    df,
                    x='Handicap',
                    y=metric,
                    title=f"Trend: {metric} vs. Handicap (Abs. Correlation: {abs_corr:.2f})",
                    labels={'Handicap': 'Handicap Index', metric: metric}
                )
                fig_metric_trend.update_traces(mode='lines+markers')
                fig_metric_trend.update_layout(hovermode="x unified")
                st.plotly_chart(fig_metric_trend, use_container_width=True)
        else:
            st.info("No numerical metrics available to plot trends by handicap.")

        # Box plot for score distribution
        st.subheader("Score Distribution Across All Handicaps")
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
            default=[handicap_options[0], handicap_options[-1]] if handicap_options else [],
            key="compare_select"
        )
        if selected_handicaps:
            compare_df = df[df["Handicap"].isin(selected_handicaps)]
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

            st.subheader("Statistical Summary")
            stats = compare_df[df.columns[1:]].describe()
            st.dataframe(stats)

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

# Professional Golf Tournament Analysis page (placeholder)
def professional_golf_analysis():
    st.title("Professional Golf Tournament Analysis")
    st.markdown("This section is under development. Check back soon for professional golf tournament insights!")

# Main app logic
def main():
    analysis_option = landing_page()
    
    if analysis_option == "Amateur Handicap Analysis":
        amateur_handicap_analysis()
    elif analysis_option == "Professional Golf Tournament Analysis":
        professional_golf_analysis()

# Sidebar explanatory text
st.sidebar.markdown("""
### About This App
This app provides insights into golf performance data. Choose an analysis type from the dropdown menu on the landing page:

- **Amateur Handicap Analysis**: Explore detailed statistics, trends, and comparisons for amateur golfers across different handicap levels.
- **Professional Golf Tournament Analysis**: Coming soon! This section will provide insights into professional golf tournament data.
""")

if __name__ == "__main__":
    main()