import streamlit as st
import pandas as pd
import altair as alt
import processor

st.set_page_config(page_title="Fitness Aggregator", layout="wide")
st.title("Cutting & Metabolic Response Dashboard")

# Single Sidebar Uploader
st.sidebar.header("Data Ingestion")
mf_upload = st.sidebar.file_uploader("MacroFactor Export (XLSX)", type="xlsx")

if mf_upload:
    try:
        df = processor.process_mf_export(mf_upload)
        
        # Top Level KPIs
        latest = df.dropna().iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Trend Weight", f"{latest['Trend Weight (kg)']:.2f} kg", f"{latest['Weight Velocity (kg/week)']:.2f} kg/wk")
        col2.metric("Latest Expenditure (MA7)", f"{latest['Expenditure_MA7']:.0f} kcal")
        col3.metric("Latest Intake", f"{latest['Calories (kcal)']:.0f} kcal", f"Deficit: {latest['Actual Deficit']:.0f} kcal")
        col4.metric("Latest Steps (MA7)", f"{latest['Steps_MA7']:.0f}", f"{latest['Steps_MA7'] - 12000:.0f} from goal")

        st.markdown("---")

        # Chart 1: Energy Balance
        st.subheader("Energy Balance (Intake vs TDEE - 7-Day MA)")
        base = alt.Chart(df).encode(x='Date:T')
        line_exp = base.mark_line(color='red', strokeWidth=2).encode(y=alt.Y('Expenditure_MA7:Q', title='kcal (MA7)'))
        line_intake = base.mark_line(color='blue', strokeWidth=1, opacity=0.5).encode(y='Calories (kcal):Q')
        st.altair_chart(line_exp + line_intake, use_container_width=True)

        # Chart 2: Activity Volume
        st.subheader("Activity Volume (Steps - 7-Day MA)")
        line_steps = base.mark_line(color='orange', point=True).encode(y=alt.Y('Steps_MA7:Q', title='Steps (MA7)'))
        rule_steps = alt.Chart(pd.DataFrame({'y': [12000]})).mark_rule(color='green', strokeDash=[5,5]).encode(y='y:Q')
        st.altair_chart(line_steps + rule_steps, use_container_width=True)

        # Chart 3: Correlation
        st.subheader("Metabolic Response to Activity (Steps vs TDEE)")
        scatter = alt.Chart(df.dropna(subset=['Steps_MA7', 'Expenditure_MA7'])).mark_circle(size=60).encode(
            x=alt.X('Steps_MA7:Q', title='Steps (7-Day MA)', scale=alt.Scale(zero=False)),
            y=alt.Y('Expenditure_MA7:Q', title='Expenditure (7-Day MA)', scale=alt.Scale(zero=False)),
            tooltip=['Date', 'Steps_MA7', 'Expenditure_MA7']
        )
        trendline = scatter.transform_regression('Steps_MA7', 'Expenditure_MA7').mark_line(color='red')
        st.altair_chart(scatter + trendline, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Ensure the XLSX contains sheets: 'Calories & Macros', 'Weight Trend', 'Expenditure', 'Steps'")

else:
    st.info("Upload MacroFactor export file to begin.")