import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr  # Import pearsonr for correlation calculation

# Page configuration
st.set_page_config(page_title="Tethered AI Golf Data Analysis", layout="wide")

# Function to load data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure the CSV is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
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
        # Ensure there are no NaN values for correlation calculation
        temp_df = df_full[[col, 'Handicap']].dropna()
        if not temp_df.empty:
            corr, _ = pearsonr(temp_df[col], temp_df['Handicap'])
            correlations[col] = corr
        else:
            correlations[col] = 0 # Assign 0 if no valid data for correlation

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
    df = load_data("Handicap Stats.csv") # Assuming Handicap Stats.csv is in the root
    
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
                temp_df_corr = df[[col, 'Handicap']].dropna()
                if not temp_df_corr.empty:
                    corr, _ = pearsonr(temp_df_corr[col], temp_df_corr['Handicap'])
                    correlations_for_sorting[col] = abs(corr)
                else:
                    correlations_for_sorting[col] = 0

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
        numerical_df_for_heatmap = df.select_dtypes(include=['number'])
        if len(numerical_df_for_heatmap.columns) > 1:
            corr_matrix = numerical_df_for_heatmap.corr()
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
            
            # Melt the DataFrame for grouped bar chart
            melted_compare_df = compare_df.melt(id_vars=['Handicap'], var_name='Metric', value_name='Value')

            # Filter out 'Handicap' column from the metrics if it somehow got in
            metrics_to_plot = [col for col in df.columns[1:] if col != 'Handicap']
            filtered_melted_df = melted_compare_df[melted_compare_df['Metric'].isin(metrics_to_plot)]


            fig_compare = px.bar(
                filtered_melted_df,
                x='Handicap',
                y='Value',
                color='Metric',
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

# Professional Golf Tournament Analysis page
def professional_golf_analysis():
    st.title("Professional Golf Tournament Analysis")
    st.markdown("Insights into professional golf tournament data.")

    preds_file_path = "Predictions/LR_Preds_2025-05-28.csv"
    preds_df = load_data(preds_file_path)

    if preds_df is not None and not preds_df.empty:
        # Display Tournament name at the top
        if 'Tournament' in preds_df.columns and not preds_df['Tournament'].empty:
            tournament_name = preds_df['Tournament'].iloc[0]
            st.header(f"Tournament: {tournament_name}")

        st.subheader("Top 10 Player Predictions")
        # Ensure 'Player' column exists
        if 'Player' in preds_df.columns:
            top_10_players = preds_df.head(10)[['Player']]
            st.dataframe(top_10_players, hide_index=True)
        else:
            st.warning("The 'Player' column was not found in the predictions file.")

        st.subheader("Highest Probability of Winning")
        # Define the columns for the table, including the newly requested ones
        display_columns = [
            'Last T1 Finish',
            'Last T2 Finish',
            'Last T3 Finish',
            'Previous_Year_Position',
            'Early_Rounds_Avg',
            'Last_3_Early_Rounds_Avg',
            'Days_Since_Last_Tournament'
        ]
        
        # Check if all display columns exist in the DataFrame
        if all(col in preds_df.columns for col in display_columns):
            # Select the relevant columns and set 'Player' as index
            winning_prob_df = preds_df.set_index('Player')[display_columns]

            # Replace 100 with 'DNF' in the specified columns
            # Ensure the columns are numeric before replacing if they are not already
            for col in ['Last T1 Finish', 'Last T2 Finish', 'Last T3 Finish', 'Previous_Year_Position']:
                if col in winning_prob_df.columns:
                    winning_prob_df[col] = pd.to_numeric(winning_prob_df[col], errors='coerce')
                    winning_prob_df[col] = winning_prob_df[col].replace(100, 'DNF')

            st.dataframe(winning_prob_df)
        else:
            missing_cols = [col for col in display_columns if col not in preds_df.columns]
            st.warning(f"Missing one or more required columns for 'Highest Probability of Winning' table: {', '.join(missing_cols)}")
    else:
        st.info("No professional golf predictions available at this time.")


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