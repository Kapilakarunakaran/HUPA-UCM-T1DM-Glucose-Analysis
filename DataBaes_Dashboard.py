# ══════════════════════════════════════════════════════════════════════════════
# HUPA-UC T1DM Clinical Insights Dashboard
# ══════════════════════════════════════════════════════════════════════════════
# 
# PURPOSE:
# This Streamlit dashboard provides comprehensive analysis of Type 1 Diabetes 
# Mellitus (T1DM) patients using Continuous Glucose Monitoring (CGM) data from 
# the HUPA-UC dataset.
#
# DASHBOARD STRUCTURE:
# The dashboard is organized into 4 main tabs:
#
# TAB 1 - Patient Risk Overview:
#   - Population-level KPIs (TIR, variability, extreme values)
#   - Glucose zone distribution (hypo/normal/hyper)
#   - Time-of-day glucose patterns
#   - Demographic analysis (glucose spikes by race)
#
# TAB 2 - Insulin Management:
#   - Insulin delivery method distribution (pump vs manual)
#   - Bolus frequency analysis (patient engagement)
#   - Dose-response relationship (insulin dose vs glucose outcome)
#
# TAB 3 - Meals and Activity:
#   - Pre-meal and post-meal glucose trajectories
#   - Glucose spikes by meal size
#   - Physical activity patterns during high glucose
#   - Sleep quality impact on glucose control
#
# TAB 4 - Patient Risk Deep Dive:
#   - Age vs hypoglycemia risk
#   - Bolus frequency vs time in range
#   - Composite risk scoring (multi-factor patient ranking)
#   - Gap analysis (high-risk vs well-managed patients)
#
# KEY CLINICAL METRICS:
# - TIR (Time in Range): % of readings between 70-180 mg/dL (target ≥70%)
# - CV (Coefficient of Variation): Glucose variability (target ≤36%)
# - Hypoglycemia: Glucose < 70 mg/dL (dangerous low)
# - Hyperglycemia: Glucose > 180 mg/dL (too high)
# - Severe hyperglycemia: Glucose > 250 mg/dL (very high risk)
# - Severe hypoglycemia: Glucose < 54 mg/dL (critical low)
#
# DATA REQUIREMENTS:
# - Input file: clean_data/cgm_cleaned.csv
# - Required columns: Patient_ID, Date, Time, Glucose, Basal_Rate, Bolus_Volume,
#   Carbs, Steps, Sleep_Quality, Sleep_Disturbances, Age, Race
#
# TECHNICAL NOTES:
# - Uses Streamlit caching (@st.cache_data) for performance
# - CGM readings assumed at 5-minute intervals (12 per hour)
# - Time shifts: 6 readings = 30min, 12 = 60min, 24 = 120min
# - Custom CSS styling for clinical appearance
# - Matplotlib for static charts, consistent color scheme throughout
#
# ══════════════════════════════════════════════════════════════════════════════

# Import required libraries
import streamlit as st          # Web app framework for creating interactive dashboards
import pandas as pd             # Data manipulation and analysis
import numpy as np              # Numerical computing and array operations
import matplotlib.pyplot as plt # Static plotting library
import seaborn as sns           # Statistical data visualization (built on matplotlib)
import plotly.express as px     # Interactive plotting library
from scipy import stats         # Statistical functions for correlation analysis

# Configure the Streamlit page layout and title
# layout="wide" uses the full browser width for better visualization
st.set_page_config(layout="wide", page_title="HUPA-UC T1DM Clinical Insights Dashboard")

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ══════════════════════════════════════════════════════════════════════════════
# This section defines custom CSS to style the dashboard components for a
# professional, clinical appearance with clear visual hierarchy.
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Page Background ── */
/* Sets a light gray background for the entire dashboard */
[data-testid="stAppViewContainer"] > .main {
    background-color: #F8FAFC;
}

/* ── KPI Hero Card ── */
/* Special styling for the Time in Range (TIR) card - the most important metric */
.kpi-hero {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 22px 16px 18px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(15,23,42,0.07);
}

/* ── KPI Standard Cards ── */
/* Styling for secondary KPI cards (max glucose, min glucose, variability) */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(15,23,42,0.07);
}

/* ── Tab Navigation Styling ── */
/* Creates a pill-style tab interface with navy blue active state */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    width: 100%;
    background-color: #EEF2FF;  /* Light blue background for tab container */
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    flex: 1;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    padding: 12px 4px;
    border-radius: 8px;
    color: #475569;  /* Gray text for inactive tabs */
}
.stTabs [aria-selected="true"] {
    background-color: #1B2A4A !important;  /* Navy blue for active tab */
    color: #FFFFFF !important;              /* White text for active tab */
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
    height: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
# Load the cleaned CGM dataset with caching to improve performance.
# @st.cache_data decorator ensures data is loaded only once and reused across
# user sessions, significantly improving dashboard responsiveness.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """
    Load the cleaned CGM dataset from CSV file.
    
    Returns:
        pd.DataFrame: Cleaned CGM data containing patient glucose readings,
                      insulin data, meals, activity, and demographics
    """
    df = pd.read_csv('Team11_DataBaes_cleaned_data.csv')
    return df

# Load data and create a working copy
df = load_data()
clean = df.copy()  # Create a copy to preserve original data

# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB GLOBAL STYLING
# ══════════════════════════════════════════════════════════════════════════════
# Configure matplotlib's default appearance to match the dashboard's design system.
# These settings apply to all matplotlib charts created in the dashboard.
# ══════════════════════════════════════════════════════════════════════════════

# Set smaller font sizes for axis labels to fit more information
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

# Use subtle gray colors for chart elements (professional clinical appearance)
plt.rcParams['axes.edgecolor']  = '#CBD5E1'  # Light gray for axis borders
plt.rcParams['axes.labelcolor'] = '#475569'  # Medium gray for axis labels
plt.rcParams['xtick.color']     = '#64748B'  # Gray for x-axis tick marks
plt.rcParams['ytick.color']     = '#64748B'  # Gray for y-axis tick marks
plt.rcParams['text.color']      = '#1B2A4A'  # Navy blue for text

# Set white backgrounds for clean, clinical appearance
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'

# ══════════════════════════════════════════════════════════════════════════════
# TIME-SHIFTED GLUCOSE CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════
# Create lagged and leading glucose values to analyze pre-meal and post-meal
# glucose patterns. This is essential for understanding how meals and insulin
# affect glucose levels over time.
#
# CGM readings are typically taken every 5 minutes:
# - 6 readings = 30 minutes
# - 12 readings = 60 minutes  
# - 24 readings = 120 minutes
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def compute_shifted(_df_clean):
    """
    Compute time-shifted glucose values for temporal analysis.
    
    This function creates new columns with glucose values from different time points:
    - glucose_30min_before: Glucose level 30 minutes before current reading
    - glucose_60min_after: Glucose level 60 minutes after current reading
    - glucose_120min_after: Glucose level 120 minutes after current reading
    
    Args:
        _df_clean: DataFrame with CGM data
        
    Returns:
        pd.DataFrame: Original data with added time-shifted glucose columns
    """
    # Sort by patient, date, and time to ensure chronological order
    ds = _df_clean.sort_values(['Patient_ID', 'Date', 'Time']).copy()
    
    # Create shifted columns (grouped by patient to avoid mixing patient data)
    # shift(6) = 6 readings back = 30 minutes before
    ds['glucose_30min_before'] = ds.groupby('Patient_ID')['Glucose'].shift(6)
    
    # shift(-12) = 12 readings forward = 60 minutes after
    ds['glucose_60min_after']  = ds.groupby('Patient_ID')['Glucose'].shift(-12)
    
    # shift(-24) = 24 readings forward = 120 minutes after
    ds['glucose_120min_after'] = ds.groupby('Patient_ID')['Glucose'].shift(-24)
    
    return ds

# Compute shifted glucose values for the entire dataset
df_sorted = compute_shifted(clean)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD HEADER
# ══════════════════════════════════════════════════════════════════════════════

# Display centered dashboard title with custom styling
st.markdown(
    "<h2 style='text-align:center; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; color:#1B2A4A; font-weight:700; letter-spacing:-0.5px;'>HUPA-UCM T1DM Clinical Insights Dashboard</h2>",
    unsafe_allow_html=True
)
st.divider()  # Add horizontal line separator

# ══════════════════════════════════════════════════════════════════════════════
# TAB NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
# Create four main analysis tabs to organize different aspects of the analysis:
# 1. Patient Risk Overview - Population-level metrics and glucose distribution
# 2. Insulin Management - Insulin delivery methods and dosing patterns
# 3. Meals and Activity - Impact of food and exercise on glucose
# 4. Patient Risk Deep Dive - Individual patient risk scoring and comparison
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Patient Risk Overview",
    "💉 Insulin Management",
    "🍽️ Meals and Activity",
    "🔍 Patient Risk Deep Dive"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PATIENT RISK OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
# This tab provides a high-level overview of glucose control across the entire
# patient population, including:
# - Time in Range (TIR) - the primary metric for diabetes management
# - Glucose variability and stability metrics
# - Time-of-day patterns
# - Demographic differences in glucose control
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

    # ──────────────────────────────────────────────────────────────────────────
    # KPI CALCULATIONS
    # ──────────────────────────────────────────────────────────────────────────
    # Calculate key performance indicators for the entire patient population
    # ──────────────────────────────────────────────────────────────────────────
    
    # Time in Range (TIR): Percentage of readings between 70-180 mg/dL
    # This is the gold standard metric for diabetes management (target ≥70%)
    population_tir = ((clean['Glucose'] >= 70) & (clean['Glucose'] <= 180)).sum() / len(clean) * 100
    
    # Coefficient of Variation (CV) per patient: measures glucose variability
    # CV = (standard deviation / mean) × 100
    # High CV (>36%) indicates unstable blood sugar control
    per_pt_cv = clean.groupby('Patient_ID')['Glucose'].agg(lambda x: x.std() / x.mean() * 100)
    high_variability_count = (per_pt_cv > 36).sum()
    
    # Extreme glucose values - important for identifying dangerous episodes
    max_glucose = clean['Glucose'].max()  # Highest recorded glucose (hyperglycemia peak)
    min_glucose = clean['Glucose'].min()  # Lowest recorded glucose (hypoglycemia floor)

    # ──────────────────────────────────────────────────────────────────────────
    # KPI DISPLAY CARDS
    # ──────────────────────────────────────────────────────────────────────────
    # Display four key metrics in card format at the top of the dashboard
    # The TIR card gets special "hero" styling as it's the most important metric
    # ──────────────────────────────────────────────────────────────────────────
    
    k1, k2, k3, k4 = st.columns(4)  # Create 4 equal-width columns
    
    # Card 1: Time in Range (TIR) - Primary metric with hero styling
    with k1:
        st.markdown(f"""
        <div class="kpi-hero" style="height:148px; overflow:hidden;">
        <div style="font-size:15px; color:#64748B; font-weight:700;  letter-spacing:0.07em;">🟢 Time in Range (TIR)</div>
        <div style="font-size:44px; font-weight:800; color:#27AE60; margin-top:4px; letter-spacing:-1.5px; line-height:1;">{population_tir:.1f}%</div>
        <div style="font-size:11px; color:#94A3B8; margin-top:6px;">Population-level · target ≥ 70%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Card 2: High Variability Patients - Identifies unstable glucose control
    with k2:
        st.markdown(f"""
        <div class="kpi-card" style="height:148px; overflow:hidden;">
        <div style="font-size:15px; color:#64748B; font-weight:700;letter-spacing:0.07em;">🔴 Unstable Blood Sugar</div>
        <div style="font-size:44px; font-weight:700; color:#C0392B; margin-top:4px;letter-spacing:-1.5px; line-height:1;"> {high_variability_count} <span style="font-size:16px; font-weight:500; color:#94A3B8;">/ 25 patients</span></div>
        <div style="font-size:11px; color:#94A3B8; margin-top:6px;">CV &gt; 36%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Card 3: Maximum Glucose - Hyperglycemia peak
    with k3:
        st.markdown(f"""
        <div class="kpi-card" style="height:148px; overflow:hidden;">
        <div style="font-size:15px; color:#64748B; font-weight:700;letter-spacing:0.07em;">📈 Highest Recorded</div>
        <div style="font-size:44px; font-weight:700; color:#E67E22; margin-top:4px;letter-spacing:-1.5px; line-height:1;">{max_glucose:.0f} <span style="font-size:15px; font-weight:500; color:#94A3B8;">mg/dL</span></div>
        <div style="font-size:11px; color:#94A3B8; margin-top:6px;">Hyperglycemia peak</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Card 4: Minimum Glucose - Hypoglycemia floor (dangerous low)
    with k4:
        st.markdown(f"""
        <div class="kpi-card" style="height:148px; overflow:hidden;">
        <div style="font-size:15px; color:#64748B; font-weight:700;letter-spacing:0.07em;">📉 Lowest Recorded</div>
        <div style="font-size:44px; font-weight:700; color:#C0392B; margin-top:4px;letter-spacing:-1.5px; line-height:1;">{min_glucose:.0f} <span style="font-size:15px; font-weight:500; color:#94A3B8;">mg/dL</span></div>
        <div style="font-size:11px; color:#94A3B8; margin-top:6px;">Hypoglycemia floor</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # CHART ROW 1: Glucose Distribution & Time-of-Day Patterns
    # ──────────────────────────────────────────────────────────────────────────
    # Left: Donut chart showing glucose zone distribution (hypo/normal/hyper)
    # Right: Bar chart showing average glucose by time of day
    # ──────────────────────────────────────────────────────────────────────────
    
    col_a, col_b = st.columns(2)  # Create two equal-width columns

    # ── LEFT CHART: Glucose Stability & Time in Range Analysis ───────────────
    with col_a:
        # Define function to classify glucose readings into clinical zones
        def classify_glucose(g):
            """
            Classify glucose reading into clinical zones.
            
            Clinical thresholds:
            - Hypoglycemia: < 70 mg/dL (dangerously low)
            - Normal/TIR: 70-180 mg/dL (target range)
            - Hyperglycemia: > 180 mg/dL (too high)
            """
            if g < 70:     return 'Hypoglycemia\n(<70)'
            elif g <= 180: return 'Normal / TIR\n(70–180)'
            else:          return 'Hyperglycemia\n(>180)'

        # Define order for consistent display
        zone_order = ['Hypoglycemia\n(<70)', 'Normal / TIR\n(70–180)', 'Hyperglycemia\n(>180)']
        
        # Classify all glucose readings and count occurrences
        clean_z = clean.copy()
        clean_z['glucose_zone'] = clean_z['Glucose'].apply(classify_glucose)
        cohort_counts = clean_z['glucose_zone'].value_counts().reindex(zone_order)
        cohort_pct    = (cohort_counts / len(clean_z) * 100).round(1)

        # Color scheme: Red (hypo), Green (normal), Orange (hyper)
        COLORS = ['#C0392B', '#27AE60', '#E67E22']
        LABELS = ['Hypoglycemia\n<70 mg/dL', 'Normal / TIR\n70–180 mg/dL', 'Hyperglycemia\n>180 mg/dL']

        # Create donut chart (pie chart with center cutout)
        fig, ax = plt.subplots(figsize=(3.5, 3))
        wedges, texts, autotexts = ax.pie(
            cohort_pct, labels=LABELS, autopct='%1.1f%%', colors=COLORS,
            startangle=90,  # Start from top
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),  # width<1 creates donut
            pctdistance=0.75,  # Position percentage labels
            textprops={'fontsize': 6}
        )
        
        # Style percentage labels (white text, bold)
        for at, c in zip(autotexts, COLORS):
            at.set_color('white')
            at.set_fontweight('bold')
            at.set_fontsize(7)
        
        # Add TIR value in center of donut
        tir_val = cohort_pct['Normal / TIR\n(70–180)']
        ax.text(0,  0.1,  'TIR',         ha='center', fontsize=8, fontweight='bold', color='#27AE60')
        ax.text(0, -0.15, f'{tir_val}%', ha='center', fontsize=10, fontweight='bold', color='#27AE60')
        
        ax.set_title('Glucose Stability & Time in Range Analysis', fontsize=8, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        
        # Clinical interpretation
        st.write("Over 70% of glucose readings remained within the healthy range, indicating overall stable blood sugar control with limited hypo- and hyperglycemic events.")

    # ── RIGHT CHART: Average Glucose Trends by Time of Day ───────────────────
    with col_b:
        # Check if Time column exists in the dataset
        if 'Time' in clean.columns:
            tod = clean.copy()
            
            # Extract hour from time string (format: HH:MM:SS)
            tod['Hour'] = pd.to_datetime(tod['Time'], format='%H:%M:%S', errors='coerce').dt.hour
            tod = tod.dropna(subset=['Hour'])  # Remove any rows where time parsing failed

            # Define function to categorize hours into time-of-day periods
            def get_tod(h):
                """
                Categorize hour into time-of-day period.
                
                Periods align with typical meal and activity patterns:
                - Morning: 6am-12pm (breakfast, morning activity)
                - Afternoon: 12pm-6pm (lunch, afternoon activity)
                - Evening: 6pm-10pm (dinner, post-meal period)
                - Night: 10pm-6am (sleep, fasting period)
                """
                if 6 <= h < 12:    return 'Morning\n(6am–12pm)'
                elif 12 <= h < 18: return 'Afternoon\n(12pm–6pm)'
                elif 18 <= h < 22: return 'Evening\n(6pm–10pm)'
                else:              return 'Night\n(10pm–6am)'

            # Apply time-of-day categorization
            tod['Time_of_Day'] = tod['Hour'].apply(get_tod)
            
            # Calculate average glucose for each time period
            tod_order = ['Morning\n(6am–12pm)', 'Afternoon\n(12pm–6pm)', 'Evening\n(6pm–10pm)', 'Night\n(10pm–6am)']
            tod_mean  = tod.groupby('Time_of_Day')['Glucose'].mean().reindex(tod_order)
            
            # Highlight the highest period in red, others in gray
            bar_colors = ['#C0392B' if v == tod_mean.max() else '#94A3B8' for v in tod_mean]

            # Create bar chart
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(tod_order, tod_mean.values, color=bar_colors)
            
            # Add value labels on top of each bar
            for bar, val in zip(bars, tod_mean.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.0f}', ha='center', fontsize=8, fontweight='bold')
            
            ax.set_ylabel('Average Glucose (mg/dL)')
            ax.set_title('Average Glucose Trends by Time', fontsize=9, fontweight='bold')
            
            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            
            # Clinical interpretation
            st.write("Evening hours showed the highest average glucose levels, indicating stronger post-meal glucose spikes and increased monitoring needs during that period.")
        else:
            st.warning("Time column not found — cannot generate time of day chart.")

    # ──────────────────────────────────────────────────────────────────────────
    # CHART ROW 2: Demographic Analysis - Glucose Spike Patterns by Race
    # ──────────────────────────────────────────────────────────────────────────
    # Dual-axis chart showing both spike rate (%) and average glucose by race
    # This helps identify demographic groups that may need targeted interventions
    # ──────────────────────────────────────────────────────────────────────────
    
    col_c, col_d = st.columns(2)

    with col_c:
        # Filter to rows with valid race and glucose data
        race_df = clean.dropna(subset=['Race', 'Glucose']).copy()
        
        # Create binary indicator: 1 if glucose >= 180 (spike), 0 otherwise
        race_df['Glucose_Spike'] = np.where(race_df['Glucose'] >= 180, 1, 0)

        # Calculate spike rate (percentage of readings with spikes) by race
        spike_by_race = (race_df.groupby('Race')['Glucose_Spike'].mean() * 100).sort_values(ascending=False)
        
        # Calculate average glucose level by race
        avg_glucose_race = race_df.groupby('Race')['Glucose'].mean()

        # Combine both metrics into a single dataframe
        result = pd.DataFrame({
            'Spike_Rate': spike_by_race,
            'Average_Glucose': avg_glucose_race
        }).sort_values('Spike_Rate', ascending=False)

        # Define distinct colors for each racial group
        race_colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#6A0572', '#1A936F', '#3A86FF']

        # Create dual-axis chart (bar + line)
        fig, ax1 = plt.subplots(figsize=(4, 3.5))
        
        # Primary axis: Bar chart for spike rate
        bars = ax1.bar(result.index, result['Spike_Rate'],
                       color=race_colors[:len(result)], alpha=0.85)
        
        # Add value labels on bars
        for bar in bars:
            y = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, y + 0.5,
                     f'{y:.1f}%', ha='center', fontsize=7, fontweight='bold')

        ax1.set_xlabel('Race', fontsize=8, fontweight='bold')
        ax1.set_ylabel('Glucose Spike Rate (%)', fontsize=8, fontweight='bold')
        ax1.tick_params(axis='x', rotation=15, labelsize=7)
        ax1.spines['top'].set_visible(False)
        
        # Set y-axis limits for primary axis (spike rate)
        ax1.set_ylim(0, result['Spike_Rate'].max() * 1.2)

        # Secondary axis: Line chart for average glucose
        ax2 = ax1.twinx()
        ax2.plot(result.index, result['Average_Glucose'],
                 color='navy', marker='o', linewidth=2, markersize=6, label='Avg Glucose')
        ax2.set_ylabel('Average Glucose (mg/dL)', fontsize=8, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=7)
        ax2.legend(loc='upper right', fontsize=8)
        
        # Set y-axis limits for secondary axis (glucose) to align with primary
        # Calculate proportional limits to synchronize the axes
        spike_range = result['Spike_Rate'].max() * 1.2
        glucose_min = result['Average_Glucose'].min() * 0.95
        glucose_max = result['Average_Glucose'].max() * 1.05
        glucose_range = glucose_max - glucose_min
        
        # Adjust secondary axis to align with primary axis scale
        ax2.set_ylim(glucose_min, glucose_min + (glucose_range * (spike_range / result['Spike_Rate'].max())))

        ax1.set_title('Glucose Spike Risk Across Demographic Groups', fontsize=8, fontweight='bold')
        ax1.grid(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        
        # Clinical interpretation
        st.write("Glucose spike patterns varied significantly across demographic groups, highlighting the importance of personalized diabetes prevention and treatment strategies.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INSULIN MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
# This tab analyzes insulin delivery methods and dosing patterns:
# - Insulin delivery method distribution (pump vs manual injection)
# - Bolus frequency patterns (patient engagement in management)
# - Relationship between insulin dose size and glucose outcomes
# ══════════════════════════════════════════════════════════════════════════════

with tab2:

    # ──────────────────────────────────────────────────────────────────────────
    # DATA PREPARATION: Calculate bolus events and monitoring days per patient
    # ──────────────────────────────────────────────────────────────────────────
    
    # Count bolus events (insulin doses) per patient
    # Only count rows where Bolus_Volume > 0 (actual insulin delivery)
    bolus_events_df = df[df['Bolus_Volume'] > 0].groupby('Patient_ID').size().reset_index(name='bolus_events')
    
    # Count unique monitoring days per patient (to calculate daily averages)
    monitoring_days = df.groupby('Patient_ID')['Date'].nunique().reset_index(name='days')

    # ── Row 1: Chart 1 + Chart 2 ──────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        basal_summary = df.groupby('Patient_ID')['Basal_Rate'].max().reset_index()
        basal_summary['zero_basal'] = basal_summary['Basal_Rate'] == 0
        zero_count    = int(basal_summary['zero_basal'].sum())
        nonzero_count = len(basal_summary) - zero_count

        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.pie(
            [zero_count, nonzero_count],
            labels=[f'Manual Injection\n({zero_count} patients)', f'Insulin Pump\n({nonzero_count} patients)'],
            colors=['#E74C3C', '#2ECC71'],
            autopct='%1.0f%%', startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2),
            textprops={'fontsize': 9}
        )
        ax.set_title('Insulin Delivery Method Distribution Among Patients', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        st.write("Most patients use an insulin pump, while only a small group uses manual injection. "
                 "This shows insulin delivery methods are different, so manual injection patients may need separate monitoring.")

    with col_b:
        bolus_freq = bolus_events_df.merge(monitoring_days, on='Patient_ID', how='right').fillna(0)
        bolus_freq['events_per_day'] = bolus_freq['bolus_events'] / bolus_freq['days']
        bolus_freq = bolus_freq.sort_values('events_per_day', ascending=False)
        mean_freq  = bolus_freq['events_per_day'].mean()

        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.bar(bolus_freq['Patient_ID'], bolus_freq['events_per_day'], color='#27AE60')
        ax.axhline(mean_freq, color='#1B2A4A', linestyle='--', linewidth=1.5,
                   label=f'Mean: {mean_freq:.1f} events/day')
        ax.set_xlabel('Patient ID')
        ax.set_ylabel('Average Bolus Events per Day')
        ax.set_title('Patient Engagement in Blood Sugar Management', fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        st.write("Some patients actively manage their blood sugar more often than others, with an average of about 3.8 bolus events per day. "
                 "Patients with very low activity may need follow-up for better insulin management.")

    # ── Row 2: Chart 3 ────────────────────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        bolus_scatter = df_sorted[df_sorted['Bolus_Volume'] > 0].dropna(subset=['glucose_60min_after'])
        if len(bolus_scatter) > 0:
            corr_b = bolus_scatter['Bolus_Volume'].corr(bolus_scatter['glucose_60min_after'])
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.scatter(bolus_scatter['Bolus_Volume'], bolus_scatter['glucose_60min_after'],
                       color='#94A3B8', alpha=0.35, s=10)
            z = np.polyfit(bolus_scatter['Bolus_Volume'], bolus_scatter['glucose_60min_after'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(bolus_scatter['Bolus_Volume'].min(), bolus_scatter['Bolus_Volume'].max(), 100)
            ax.plot(x_line, p(x_line), color='#1B2A4A', linewidth=2, label=f'Trend (r={corr_b:.2f})')
            ax.axhline(180, color='#E67E22', linestyle='--', linewidth=1.5, label='Hyperglycemia threshold (180 mg/dL)')
            ax.set_xlabel('Insulin Dose Size (units)')
            ax.set_ylabel('Blood Sugar 60 Minutes Later (mg/dL)')
            ax.set_title('Impact of Insulin Dose Size on Later Blood Sugar Levels', fontsize=9, fontweight='bold')
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.write("Higher insulin doses are linked with lower blood sugar after 60 minutes, but many readings still remain high. "
                     "This suggests some patients may need dose review, diet planning, or closer glucose monitoring.")
        else:
            st.warning("Not enough data to generate bolus scatter chart.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MEALS AND ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════
# This tab analyzes the impact of meals and physical activity on glucose control:
# - Pre-meal and post-meal glucose trajectories (30min before to 120min after)
# - Glucose spike magnitude by meal size (small/medium/large)
# - Physical activity patterns during high glucose episodes
# - Sleep quality correlation with glucose stability (TIR and hyperglycemia)
#
# KEY INSIGHTS:
# - Post-meal glucose response indicates insulin effectiveness
# - Meal size impact on glucose spikes (surprisingly, small meals spike more)
# - Sedentary behavior during hyperglycemia (missed opportunity for correction)
# - Sleep disturbances strongly correlate with poor glucose control
# ══════════════════════════════════════════════════════════════════════════════

with tab3:

    # ──────────────────────────────────────────────────────────────────────────
    # CHART ROW 1: Meal Impact Analysis
    # ──────────────────────────────────────────────────────────────────────────
    # Left: Line chart showing glucose trajectory around meals
    # Right: Bar chart comparing glucose spikes by meal size
    # ──────────────────────────────────────────────────────────────────────────
    
    col_a, col_b = st.columns(2)

    # ── LEFT CHART: Pre-Meal and Post-Meal Glucose Monitoring ────────────────
    with col_a:
        # Filter to meal events (Carbs > 0) with complete time-shifted data
        meals = df_sorted[df_sorted['Carbs'] > 0].dropna(
            subset=['glucose_30min_before', 'glucose_60min_after', 'glucose_120min_after']
        )
        
        if len(meals) > 0:
            # Calculate average glucose at each time point relative to meal
            time_points = {
                '30min Before': meals['glucose_30min_before'].mean(),  # Pre-meal baseline
                'At Meal':      meals['Glucose'].mean(),                # Meal time
                '60min After':  meals['glucose_60min_after'].mean(),   # Peak response
                '120min After': meals['glucose_120min_after'].mean()   # Recovery phase
            }
            labels = list(time_points.keys())
            values = list(time_points.values())

            # Create line chart showing glucose trajectory
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.plot(labels, values, color='#1B2A4A', marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Time Relative to Meal')
            ax.set_ylabel('Mean Glucose (mg/dL)')
            ax.set_title('Pre-Meal and Post-Meal Glucose Monitoring', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            
            # Clinical interpretation
            st.write("This pattern indicates slow glucose recovery after meals, suggesting that insulin may not be "
                     "reducing glucose levels efficiently. It could also be a sign of impaired glucose regulation "
                     "or reduced insulin sensitivity.")
        else:
            st.warning("Not enough meal data to generate post-meal glucose chart.")

    # ── RIGHT CHART: Comparison of Glucose Spikes by Meal Size ───────────────
    with col_b:
        # Filter to meals with post-meal glucose data
        meals_spike = df_sorted[df_sorted['Carbs'] > 0].dropna(subset=['glucose_60min_after']).copy()
        
        if len(meals_spike) > 0:
            # Calculate glucose spike: difference between 60min after and at meal
            meals_spike['spike'] = meals_spike['glucose_60min_after'] - meals_spike['Glucose']
            
            # Categorize meals into size groups based on carb content
            # qcut divides into equal-sized bins (tertiles: 33%, 33%, 33%)
            meals_spike['meal_size'] = pd.qcut(meals_spike['Carbs'], q=3, labels=['Small', 'Medium', 'Large'])
            
            # Calculate average spike for each meal size
            spike_by_size = meals_spike.groupby('meal_size', observed=True)['spike'].mean()

            # Create bar chart
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.bar(spike_by_size.index, spike_by_size.values, color='#E67E22')
            
            # Add baseline reference line at zero (no spike)
            ax.axhline(0, color='#1B2A4A', linestyle='--', linewidth=1.5, label='No spike baseline')
            
            # Add value labels on bars
            for i, val in enumerate(spike_by_size.values):
                ax.text(i, val + (0.5 if val >= 0 else -1.5), f'{val:.1f}',
                        ha='center', fontsize=8, fontweight='bold')
            
            ax.set_xlabel('Meal Size')
            ax.set_ylabel('Mean Blood Sugar Spike (mg/dL)')
            ax.set_title('Comparison of Glucose Spikes by Meal Size', fontsize=9, fontweight='bold')
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            
            # Clinical interpretation (counterintuitive finding)
            st.write("Surprisingly, smaller meals are associated with larger glucose spikes compared to larger meals. "
                     "Normally, larger meals are expected to raise blood sugar more significantly, but this chart "
                     "suggests the opposite pattern. Patients should monitor glucose levels after all meals, including "
                     "small snacks, and ensure proper insulin coverage for smaller carbohydrate intake.")
        else:
            st.warning("Not enough data to generate meal size spike chart.")

    # ──────────────────────────────────────────────────────────────────────────
    # CHART ROW 2: Physical Activity Analysis
    # ──────────────────────────────────────────────────────────────────────────
    # Pie chart showing activity levels during high glucose episodes
    # ──────────────────────────────────────────────────────────────────────────
    
    col_c, col_d = st.columns(2)

    # ── LEFT CHART: Patient Movement Analysis During High Glucose ────────────
    with col_c:
        # Count readings with activity (Steps > 0) vs sedentary (Steps = 0)
        active_count    = (clean['Steps'] > 0).sum()
        sedentary_count = (clean['Steps'] == 0).sum()

        # Create pie chart
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.pie(
            [active_count, sedentary_count],
            labels=[f'Active\n(Steps > 0)', f'Sedentary\n(Steps = 0)'],
            colors=['#27AE60', '#CBD5E1'],  # Green for active, gray for sedentary
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2),
            textprops={'fontsize': 9}
        )
        ax.set_title('Patient Movement Analysis During High Glucose Conditions', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        
        # Clinical interpretation
        st.write("The chart indicates that most patients remained inactive even when their blood sugar levels were high. "
                 "This suggests a strong association between sedentary behavior and elevated glucose levels. "
                 "Physical activity helps muscles use glucose for energy, which can naturally reduce blood levels.")

    # ──────────────────────────────────────────────────────────────────────────
    # CHART ROW 3: Sleep Quality Impact on Glucose Control (Full Width)
    # ──────────────────────────────────────────────────────────────────────────
    # Two side-by-side charts showing correlation between sleep disturbances and:
    # - Left: Time in Range (TIR) - should decrease with poor sleep
    # - Right: Hyperglycemia percentage - should increase with poor sleep
    # Includes Pearson correlation coefficients to quantify relationships
    # ──────────────────────────────────────────────────────────────────────────
    
    # Calculate glucose metrics per patient
    glucose_stats_t3 = df.groupby('Patient_ID')['Glucose'].agg(
        TIR       = lambda g: ((g >= 70) & (g <= 180)).mean() * 100,  # Time in Range
        hyper_pct = lambda g: (g > 180).mean() * 100,                  # Hyperglycemia %
    ).round(2).reset_index()
    
    # Get sleep disturbance data per patient
    sleep_stats_t3 = df.groupby('Patient_ID').agg(
        sleep_disturbances=('Sleep_Disturbances', 'first')  # Assume constant per patient
    ).reset_index()
    
    # Merge glucose and sleep data
    patient_df_t3 = glucose_stats_t3.merge(sleep_stats_t3, on='Patient_ID')
    
    # Bin patients into sleep disturbance groups (low/medium/high)
    patient_df_t3['disturbance_bin'] = pd.qcut(
        patient_df_t3['sleep_disturbances'], q=3,
        labels=['Low\nDisturbances', 'Medium\nDisturbances', 'High\nDisturbances']
    )
    
    # Calculate average metrics for each sleep disturbance group
    bin_stats_t3 = patient_df_t3.groupby('disturbance_bin', observed=True).agg(
        avg_TIR      =('TIR',       'mean'),
        avg_hyper    =('hyper_pct', 'mean'),
        patient_count=('Patient_ID','count')
    ).round(1).reset_index()
    
    # Calculate Pearson correlation coefficients
    # r = correlation coefficient (-1 to 1), p = statistical significance
    r_tir_t3,   p_tir_t3   = stats.pearsonr(patient_df_t3['sleep_disturbances'], patient_df_t3['TIR'])
    r_hyper_t3, p_hyper_t3 = stats.pearsonr(patient_df_t3['sleep_disturbances'], patient_df_t3['hyper_pct'])

    # Prepare chart data
    x_t3  = np.arange(len(bin_stats_t3))
    w_t3  = 0.5  # Bar width
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))  # Two side-by-side charts

    # ── LEFT CHART: Time-in-Range by Sleep Disturbance Level ─────────────────
    # Color scheme: Green (good) → Orange → Red (poor)
    COLORS_TIR = ['#27AE60', '#E67E22', '#C0392B']
    bars1 = axes[0].bar(x_t3, bin_stats_t3['avg_TIR'], width=w_t3, color=COLORS_TIR, edgecolor='none')
    
    # Add ADA (American Diabetes Association) target line at 70%
    axes[0].axhline(70, color='#1B2A4A', linestyle='--', linewidth=1.8, label='ADA 70% TIR target')
    
    # Add value labels on bars
    for bar, val, n in zip(bars1, bin_stats_t3['avg_TIR'], bin_stats_t3['patient_count']):
        # TIR percentage on top
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
        # Patient count inside bar
        axes[0].text(bar.get_x() + bar.get_width()/2, 2,
                     f'n={n}', ha='center', fontsize=9, color='white', fontweight='bold')
    
    axes[0].set_xticks(x_t3)
    axes[0].set_xticklabels(bin_stats_t3['disturbance_bin'], fontsize=8)
    axes[0].set_ylabel('Time in Range (%)')
    axes[0].set_ylim(0, 100)
    axes[0].set_title('Time-in-Range by Sleep Disturbance Level', fontsize=8, fontweight='bold')
    axes[0].legend(fontsize=9)
    
    # Add correlation statistics box
    axes[0].text(0.98, 0.92, f'r={r_tir_t3:.2f}, p={p_tir_t3:.3f}', transform=axes[0].transAxes,
                 ha='right', fontsize=9, bbox=dict(boxstyle='round', facecolor='#F0F3F4', alpha=0.8))
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # ── RIGHT CHART: Hyperglycemia % by Sleep Disturbance Level ──────────────
    # Color scheme: Light blue (good) → Orange → Red (poor)
    COLORS_HYPER = ['#BEE3F8', '#E67E22', '#C0392B']
    bars2 = axes[1].bar(x_t3, bin_stats_t3['avg_hyper'], width=w_t3, color=COLORS_HYPER, edgecolor='none')
    
    # Add value labels on bars
    for bar, val, n in zip(bars2, bin_stats_t3['avg_hyper'], bin_stats_t3['patient_count']):
        # Hyperglycemia percentage on top
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
        # Patient count inside bar
        axes[1].text(bar.get_x() + bar.get_width()/2, 0.5,
                     f'n={n}', ha='center', fontsize=9, color='white', fontweight='bold')
    
    axes[1].set_xticks(x_t3)
    axes[1].set_xticklabels(bin_stats_t3['disturbance_bin'], fontsize=8)
    axes[1].set_ylabel('Time in Hyperglycemia (>180 mg/dL) %')
    axes[1].set_ylim(0, bin_stats_t3['avg_hyper'].max() + 10)
    axes[1].set_title('Hyperglycemia % by Sleep Disturbance Level', fontsize=8, fontweight='bold')
    
    # Add correlation statistics box
    axes[1].text(0.98, 0.92, f'r={r_hyper_t3:.2f}, p={p_hyper_t3:.3f}', transform=axes[1].transAxes,
                 ha='right', fontsize=9, bbox=dict(boxstyle='round', facecolor='#F0F3F4', alpha=0.8))
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle('Effect of Poor Sleep on Glucose Stability', fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    
    # Clinical interpretation
    st.write("The left chart shows the percentage of time patients' glucose levels remained within the healthy target "
             "range. The right chart shows the percentage of time patients experienced hyperglycemia (glucose >180 mg/dL). "
             "The chart strongly suggests that poor sleep quality negatively impacts blood sugar regulation.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PATIENT RISK DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
# This tab provides individual patient-level risk analysis:
# - Age vs hypoglycemia risk (younger patients at higher risk)
# - Bolus frequency vs time in range (engagement correlation)
# - Composite risk scoring using multiple factors
# - Gap analysis comparing high-risk vs well-managed patients
#
# COMPOSITE RISK SCORE METHODOLOGY:
# The risk score (0-1 scale) combines 6 weighted factors:
# - Average glucose (25%): Higher glucose = higher risk
# - Hyperglycemia % (25%): More time >180 mg/dL = higher risk
# - Time out of range (20%): Lower TIR = higher risk
# - Glucose variability (15%): Higher std dev = higher risk
# - Severe hyperglycemia (10%): Time >250 mg/dL = higher risk
# - Severe hypoglycemia (5%): Time <54 mg/dL = higher risk
#
# PATIENT TIERS:
# - Highest Risk: Top 25% of patients (need immediate intervention)
# - Average: Middle 50% of patients (standard monitoring)
# - Well Managed: Bottom 25% of patients (best practices to share)
# ══════════════════════════════════════════════════════════════════════════════

with tab4:

    # ──────────────────────────────────────────────────────────────────────────
    # DATA PREPARATION: Recalculate bolus events for this tab
    # ──────────────────────────────────────────────────────────────────────────
    
    bolus_events_df2 = df[df['Bolus_Volume'] > 0].groupby('Patient_ID').size().reset_index(name='bolus_events')
    monitoring_days2 = df.groupby('Patient_ID')['Date'].nunique().reset_index(name='days')

    # ── Row 1: Chart 1 + Chart 2 ──────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        hypo_rate = (
            clean.groupby('Patient_ID')
            .apply(lambda x: (x['Glucose'] < 70).sum() / len(x) * 100)
            .reset_index(name='hypo_pct')
        )
        age_data = clean.groupby('Patient_ID')['Age'].first().reset_index()
        hypo_age = hypo_rate.merge(age_data, on='Patient_ID')

        hypo_age['Age_Group'] = pd.cut(
            hypo_age['Age'],
            bins=[19, 29, 39, 49, 59, 69, 79],
            labels=['20s', '30s', '40s', '50s', '60s', '70s']
        )
        group_hypo   = hypo_age.groupby('Age_Group', observed=True)['hypo_pct'].mean().reset_index()
        overall_mean = hypo_age['hypo_pct'].mean()

        n_bars   = len(group_hypo)
        colors_h = [plt.cm.RdYlGn(i / (n_bars - 1)) for i in range(n_bars)]

        fig, ax = plt.subplots(figsize=(4, 3.5))
        bars = ax.bar(group_hypo['Age_Group'], group_hypo['hypo_pct'],
                      color=colors_h, width=0.55, edgecolor='white', linewidth=1.2)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f'{h:.1f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333333')
        ax.axhline(overall_mean, color='#555555', linestyle='--', linewidth=1.8,
                   label=f'Overall mean: {overall_mean:.1f}%')
        ax.set_title('Patient Age vs Hypoglycemia Rate', fontsize=9, fontweight='bold')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Average % of Readings Below 70 mg/dL')
        ax.legend(fontsize=9)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(0, group_hypo['hypo_pct'].max() * 1.25)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        st.write("Younger patients tend to experience more episodes of dangerously low blood sugar. "
                 "The patient with the highest hypoglycemia rate is 49 years old, while several older patients (60+) "
                 "have near-zero hypoglycemia rates.")

    with col_b:
        tir = (
            clean.groupby('Patient_ID')
            .apply(lambda x: ((x['Glucose'] >= 70) & (x['Glucose'] <= 180)).sum() / len(x) * 100)
            .reset_index(name='tir_pct')
        )
        bolus_freq2 = bolus_events_df2.merge(monitoring_days2, on='Patient_ID', how='right').fillna(0)
        bolus_freq2['events_per_day'] = bolus_freq2['bolus_events'] / bolus_freq2['days']
        combined = tir.merge(bolus_freq2[['Patient_ID', 'events_per_day']], on='Patient_ID')

        combined['Dosing_Group'] = pd.cut(
            combined['events_per_day'],
            bins=[-0.01, 1, 3, 5, float('inf')],
            labels=['Rare\n(0–1/day)', 'Low\n(1–3/day)', 'Moderate\n(3–5/day)', 'Frequent\n(5+/day)']
        )
        group_tir        = combined.groupby('Dosing_Group', observed=True)['tir_pct'].mean().reset_index()
        overall_mean_tir = combined['tir_pct'].mean()

        n_bars   = len(group_tir)
        colors_t = [plt.cm.RdYlGn(i / (n_bars - 1)) for i in range(n_bars)]

        fig, ax = plt.subplots(figsize=(4, 3.5))
        bars = ax.bar(group_tir['Dosing_Group'], group_tir['tir_pct'],
                      color=colors_t, width=0.5, edgecolor='white', linewidth=1.2)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f'{h:.1f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333333')
        ax.axhline(overall_mean_tir, color='#555555', linestyle='--', linewidth=1.8,
                   label=f'Overall mean TIR: {overall_mean_tir:.1f}%')
        ax.set_title('Bolus Frequency vs Time in Range per Patient', fontsize=9, fontweight='bold')
        ax.set_xlabel('Dosing Frequency Group')
        ax.set_ylabel('Average Time in Range (%)')
        ax.legend(fontsize=9)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(0, group_tir['tir_pct'].max() * 1.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        st.write("Patients who take mealtime insulin more frequently tend to have better overall blood sugar control. "
                 "The best controlled patient takes insulin 2.5 times per day and spends 88% of the time in the healthy range.")

    # ──────────────────────────────────────────────────────────────────────────
    # CHART ROW 3: Gap Analysis - High Risk vs Well Managed (Full Width)
    # ──────────────────────────────────────────────────────────────────────────
    # Comparative analysis showing key metric differences between patient groups
    # ──────────────────────────────────────────────────────────────────────────
    
    st.divider()
    
    # ── DATA PREPARATION: Calculate patient-level metrics ────────────────────
    
    # Calculate glucose management metrics
    patient_summary = df.groupby('Patient_ID').agg({
        'Glucose': ['mean', 'std'],
        'Basal_Rate': 'mean',
        'Sleep_Quality': 'first',
        'Sleep_Disturbances': 'first'
    })
    
    # Flatten column names
    patient_summary.columns = ['Avg_Glucose', 'Std_Dev', 'Basal_Rate', 'Sleep_Quality', 'Sleep_Disturbances']
    patient_summary = patient_summary.reset_index()
    
    # Calculate TIR, Hyper%, Hypo% per patient
    patient_summary['TIR'] = df.groupby('Patient_ID')['Glucose'].apply(
        lambda g: ((g >= 70) & (g <= 180)).mean() * 100).values
    patient_summary['Hyper_Pct'] = df.groupby('Patient_ID')['Glucose'].apply(
        lambda g: (g > 180).mean() * 100).values
    patient_summary['Hypo_Pct'] = df.groupby('Patient_ID')['Glucose'].apply(
        lambda g: (g < 70).mean() * 100).values
    
    # Calculate average daily steps
    daily_steps = df.groupby(['Patient_ID', pd.to_datetime(df['Date']).dt.date])['Steps'].sum().reset_index()
    daily_steps.columns = ['Patient_ID', 'Date', 'Daily_Steps']
    avg_daily_steps = daily_steps.groupby('Patient_ID')['Daily_Steps'].mean().reset_index()
    patient_summary = patient_summary.merge(avg_daily_steps, on='Patient_ID', how='left')
    
    # ── RISK SCORE CALCULATION ───────────────────────────────────────────────
    # Simple composite risk score based on key metrics
    from sklearn.preprocessing import MinMaxScaler
    
    risk_features = ['Avg_Glucose', 'Hyper_Pct', 'Hypo_Pct', 'Std_Dev', 'Sleep_Disturbances']
    good_features = ['TIR', 'Sleep_Quality', 'Daily_Steps']
    
    scaler = MinMaxScaler()
    risk_scaled = scaler.fit_transform(patient_summary[risk_features])
    good_scaled = scaler.fit_transform(patient_summary[good_features])
    
    # Composite risk score (higher = more risk)
    patient_summary['Risk_Score'] = risk_scaled.mean(axis=1) - good_scaled.mean(axis=1)
    patient_summary['Risk_Score'] = MinMaxScaler().fit_transform(patient_summary[['Risk_Score']])
    
    # Sort by risk score
    patient_summary = patient_summary.sort_values(by='Risk_Score', ascending=False)
    
    # ── IDENTIFY TOP AND BOTTOM QUARTILES ────────────────────────────────────
    top_n = max(1, int(len(patient_summary) * 0.25))
    highest_risk = patient_summary.head(top_n)
    well_managed = patient_summary.tail(top_n)
    
    # ── GAP ANALYSIS METRICS ──────────────────────────────────────────────────
    comparison_metrics = [
        'Avg_Glucose', 'TIR', 'Hyper_Pct', 'Hypo_Pct',
        'Std_Dev', 'Basal_Rate', 'Sleep_Quality',
        'Sleep_Disturbances', 'Daily_Steps'
    ]
    
    risk_means = highest_risk[comparison_metrics].mean()
    well_means = well_managed[comparison_metrics].mean()
    
    # ── NORMALIZE FOR VISUALIZATION ──────────────────────────────────────────
    risk_norm = []
    well_norm = []
    
    for metric in comparison_metrics:
        max_val = max(risk_means[metric], well_means[metric])
        if max_val == 0:
            risk_norm.append(0)
            well_norm.append(0)
        else:
            risk_norm.append(risk_means[metric] / max_val)
            well_norm.append(well_means[metric] / max_val)
    
    # ── CREATE GAP ANALYSIS CHART ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(comparison_metrics))
    width = 0.35
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, risk_norm, width, label='Highest Risk', color='#e74c3c')
    bars2 = ax.bar(x + width/2, well_norm, width, label='Well Managed', color='#2ecc71')
    
    # Add value annotations (raw values)
    for i, metric in enumerate(comparison_metrics):
        ax.text(i - width/2, risk_norm[i] + 0.03, f"{risk_means[metric]:.1f}",
                ha='center', fontsize=9, color='#e74c3c', fontweight='bold')
        ax.text(i + width/2, well_norm[i] + 0.03, f"{well_means[metric]:.1f}",
                ha='center', fontsize=9, color='#2ecc71', fontweight='bold')
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_metrics, rotation=30, ha='right', fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel('Normalized Score (0-1)', fontsize=11)
    ax.set_title('What Separates Well Managed from Highest Risk?\n(raw values annotated | bars normalized per metric)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    
    # Clinical interpretation
    st.write("This gap analysis reveals the key differences between highest-risk and well-managed patients. "
             "Well-managed patients show better Time in Range (TIR), lower glucose variability (Std Dev), "
             "better sleep quality, and higher physical activity levels. These metrics can guide targeted interventions "
             "for high-risk patients.")

# ══════════════════════════════════════════════════════════════════════════════
# END OF DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
