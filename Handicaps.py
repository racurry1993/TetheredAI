import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr  # Import pearsonr for correlation calculation

# Page configuration
st.set_page_config(page_title="Tethered AI Golf Data Analysis", layout="wide", initial_sidebar_state="expanded") # Added initial_sidebar_state

# Function to load data
@st.cache_data # Cache data to improve performance
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
    target_handicap = current_handicap_focus - 1 if current_handicap_focus > 0 else 0
    advice_lines.append(f"### Personalized Advice for a {current_handicap_focus} Handicap Golfer")
    advice_lines.append(f"To move from a **{current_handicap_focus} handicap** towards a **{target_handicap} handicap**, focus on the following key areas based on your data:")

    advice_given = False
    for feature, corr_val in sorted_correlations:
        if corr_val < -0.3: # Strong negative correlation
            advice_lines.append(f"  - <span style='color:green'>**Improve {feature}:**</span> This metric has a strong negative correlation ($${corr_val:.2f}$$) with handicap. Increasing your performance in {feature} (e.g., hitting the ball further, hitting more greens, etc.) is highly likely to reduce your handicap.")
            advice_given = True

    for feature, corr_val in reversed(sorted_correlations):
        if corr_val > 0.3: # Strong positive correlation
            advice_lines.append(f"  - <span style='color:red'>**Reduce {feature}:**</span> This metric has a strong positive correlation ($${corr_val:.2f}$$) with handicap. Decreasing your performance in {feature} (e.g., fewer putts, fewer penalty strokes, etc.) is highly likely to reduce your handicap.")
            advice_given = True

    if not advice_given:
        advice_lines.append("No strong correlations found to provide specific advice based on the current data and thresholds. Your golf game might have a balanced set of strengths and weaknesses, or more data might be needed.")

    return "\n".join(advice_lines)

# Landing page
def landing_page():
    st.title("🏌️‍♂️ Welcome to Tethered AI Golf Analysis")
    st.markdown("---")
    st.markdown("Select an analysis option below to explore fascinating golf statistics:")
    
    analysis_option = st.selectbox(
        "Choose Analysis Type:",
        ["Amateur Handicap Analysis", "Professional Golf Tournament Analysis"],
        key="landing_select"
    )
    
    st.markdown("---") # Add a separator for visual clarity
    return analysis_option

# Amateur Handicap Analysis page
def amateur_handicap_analysis():
    df = load_data("Handicap Stats.csv") # Assuming Handicap Stats.csv is in the root
    
    st.title("📊 Amateur Golf Handicap Statistics")
    st.markdown("Dive deep into your amateur golf performance and uncover insights to lower your handicap.")
    st.markdown("---")
    
    # Organize layout with tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Trends & Correlations", "Comparisons"])
    
    with tab1:
        # ... (tab1 code remains the same) ...
        pass # Placeholder, your tab1 code is here

    with tab2:
        # ... (tab2 code remains the same) ...
        pass # Placeholder, your tab2 code is here

    with tab3:
        st.header("⚖️ Compare Multiple Handicaps")
        st.markdown("Select two or more handicaps to compare their average performance across all metrics.")
        selected_handicaps = st.multiselect(
            "Select Handicaps to Compare:",
            handicap_options, # handicap_options comes from df in tab1, ensure it's available
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
                labels={'value': 'Average Value', 'variable': 'Metric'},
                template="plotly_white"
            )
            fig_compare.update_layout(showlegend=True, legend_title_text='Metrics',
                                      title_font_size=20,
                                      xaxis_title_font_size=16,
                                      yaxis_title_font_size=16)
            st.plotly_chart(fig_compare, use_container_width=True)

            st.markdown("---")
            st.subheader("Detailed Statistical Summary")

            # --- FIX STARTS HERE ---
            # Select only numerical columns from compare_df for describe() and styling
            numerical_cols_for_stats = compare_df.select_dtypes(include=['number']).columns.tolist()
            # Ensure 'Handicap' is excluded if it's treated as a category for stats,
            # though describe() usually handles it by providing stats if it's numeric.
            # If you want stats for 'Handicap' as well, remove this line.
            if 'Handicap' in numerical_cols_for_stats:
                numerical_cols_for_stats.remove('Handicap')

            if numerical_cols_for_stats:
                stats_df = compare_df[numerical_cols_for_stats].describe().T
                st.dataframe(stats_df.style.background_gradient(cmap='Blues'))
            else:
                st.info("No numerical data available to generate a statistical summary for selected handicaps.")
            # --- FIX ENDS HERE ---

            st.markdown("---")
            st.subheader("📥 Download Compared Data")
            csv = compare_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Selected Data as CSV",
                data=csv,
                file_name="selected_handicap_data.csv",
                mime="text/csv"
            )
        else:
            st.info("Select at least one handicap to compare.")

    with tab2:
        st.header("📈 Trends & Correlations Across Handicaps")
        st.markdown("""
            These plots illustrate how various golf metrics trend across different handicap levels.
            Charts are ordered by the **absolute strength of their correlation with Handicap**,
            highlighting metrics with the most significant impact on your score.
        """)
        st.markdown("---")

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
                    title=f"Trend: **{metric}** vs. Handicap (Abs. Correlation: $${abs_corr:.2f}$$)", # Bold title and LaTeX for correlation
                    labels={'Handicap': 'Handicap Index', metric: metric},
                    template="plotly_white" # Use a clean template
                )
                fig_metric_trend.update_traces(mode='lines+markers', marker=dict(size=8, symbol='circle', line=dict(width=2, color='DarkSlateGrey')),
                                            line=dict(width=3)) # Thicker lines, better markers
                fig_metric_trend.update_layout(hovermode="x unified",
                                              title_font_size=20,
                                              xaxis_title_font_size=16,
                                              yaxis_title_font_size=16) # Adjust font sizes
                st.plotly_chart(fig_metric_trend, use_container_width=True)
        else:
            st.info("No numerical metrics available to plot trends by handicap.")

        st.markdown("---")
        # Box plot for score distribution
        st.subheader("⛳ Score Distribution Across All Handicaps")
        if 'Avg Score to Par' in df.columns:
            fig_box = px.box(
                df,
                x='Handicap',
                y='Avg Score to Par',
                title="Distribution of **Average Score to Par** by Handicap",
                labels={'Avg Score to Par': 'Average Score to Par'},
                template="plotly_white"
            )
            fig_box.update_layout(title_font_size=20,
                                  xaxis_title_font_size=16,
                                  yaxis_title_font_size=16)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Column 'Avg Score to Par' not found for box plot.")

        st.markdown("---")
        # Correlation heatmap
        st.subheader("🔥 Overall Correlation Between Metrics")
        st.markdown("Understand how different golf metrics relate to each other.")
        numerical_df_for_heatmap = df.select_dtypes(include=['number'])
        if len(numerical_df_for_heatmap.columns) > 1:
            corr_matrix = numerical_df_for_heatmap.corr()
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=True,
                title="Correlation Heatmap of Golf Metrics",
                labels=dict(x="Metric", y="Metric", color="Correlation"),
                color_continuous_scale=px.colors.sequential.Viridis # Better color scale
            )
            fig_heatmap.update_layout(title_font_size=20)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Not enough numerical columns to generate a meaningful correlation heatmap.")

    with tab3:
        st.header("⚖️ Compare Multiple Handicaps")
        st.markdown("Select two or more handicaps to compare their average performance across all metrics.")
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
                labels={'value': 'Average Value', 'variable': 'Metric'},
                template="plotly_white"
            )
            fig_compare.update_layout(showlegend=True, legend_title_text='Metrics',
                                      title_font_size=20,
                                      xaxis_title_font_size=16,
                                      yaxis_title_font_size=16)
            st.plotly_chart(fig_compare, use_container_width=True)

            st.markdown("---")
            st.subheader("Detailed Statistical Summary")
            st.dataframe(compare_df[df.columns[1:]].describe().T.style.background_gradient(cmap='Blues')) # Styled dataframe

            st.markdown("---")
            st.subheader("📥 Download Compared Data")
            csv = compare_df.to_csv(index=False).encode('utf-8') # Encode to utf-8 for download
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
    st.title("🏆 Professional Golf Tournament Analysis")
    st.markdown("Harnessing the power of AI and Machine Learning to predict top performers in professional golf tournaments.")
    st.markdown("---")

    preds_file_path = "Predictions/LR_Preds_2025-05-28.csv"
    preds_df = load_data(preds_file_path)

    if preds_df is not None and not preds_df.empty:
        # Display Tournament name at the top
        if 'Tournament' in preds_df.columns and not preds_df['Tournament'].empty:
            tournament_name = preds_df['Tournament'].iloc[0]
            st.header(f"Tournament: **{tournament_name}**")
            st.markdown("---")

        st.subheader("Top 10 Player Predictions for the Tournament")
        st.success("These players are predicted to have the highest probability of performing well.")
        # Ensure 'Player' column exists
        if 'Player' in preds_df.columns:
            top_10_players = preds_df.head(10)[['Player']]
            st.dataframe(top_10_players.style.set_properties(**{'background-color': '#e6f3ff', 'color': 'black'}), hide_index=True) # Subtle blue background
        else:
            st.warning("The 'Player' column was not found in the predictions file.")

        st.markdown("---")
        st.subheader("Key Performance Indicators for Predicted Players")
        st.info("Below are the historical performance indicators for the players with the highest winning probabilities. 'DNF' indicates Did Not Finish.")
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
            for col in ['Last T1 Finish', 'Last T2 Finish', 'Last T3 Finish', 'Previous_Year_Position']:
                if col in winning_prob_df.columns:
                    winning_prob_df[col] = pd.to_numeric(winning_prob_df[col], errors='coerce')
                    winning_prob_df[col] = winning_prob_df[col].replace(100, 'DNF')
            
            st.dataframe(winning_prob_df.style.highlight_max(axis=0, subset=['Early_Rounds_Avg', 'Last_3_Early_Rounds_Avg'], color='lightgreen')
                                           .highlight_min(axis=0, subset=['Early_Rounds_Avg', 'Last_3_Early_Rounds_Avg'], color='pink')
                                           .set_properties(**{'text-align': 'center'})
                                           .format({'Early_Rounds_Avg': "{:.2f}", 'Last_3_Early_Rounds_Avg': "{:.2f}"})
                        )
        else:
            missing_cols = [col for col in display_columns if col not in preds_df.columns]
            st.warning(f"Missing one or more required columns for 'Highest Probability of Winning' table: {', '.join(missing_cols)}")
    else:
        st.info("No professional golf predictions available at this time. Please check back later!")


# Main app logic
def main():
    analysis_option = landing_page()
    
    if analysis_option == "Amateur Handicap Analysis":
        amateur_handicap_analysis()
    elif analysis_option == "Professional Golf Tournament Analysis":
        professional_golf_analysis()

# Sidebar explanatory text
st.sidebar.markdown("""
### ℹ️ About This App
This application, **Tethered AI Golf Analysis**, is designed to provide comprehensive insights into golf performance data using artificial intelligence and machine learning techniques.
""")

st.sidebar.markdown("""
---
**Choose an analysis type from the dropdown menu on the main page:**

- **Amateur Handicap Analysis**: 
    - Explore detailed statistics for amateur golfers across various handicap levels.
    - Receive personalized advice based on strong correlations in the data to help reduce your handicap.
    - Visualize trends and compare performance metrics between different handicap groups.

- **Professional Golf Tournament Analysis**: 
    - This section features AI-powered predictions and data breakdowns for professional golf tournaments, offering insights into top performers and predicted winners.
""")

st.sidebar.markdown("---")
st.sidebar.info("Developed by Tethered AI")


if __name__ == "__main__":
    main()