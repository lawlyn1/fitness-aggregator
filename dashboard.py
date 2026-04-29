import streamlit as st
import pandas as pd
import altair as alt
import processor

st.set_page_config(page_title="Fitness Aggregator", layout="wide")
st.title("Cutting & Metabolic Response Dashboard")

st.sidebar.header("Data Ingestion")
mf_upload = st.sidebar.file_uploader("MacroFactor Export (XLSX)", type="xlsx")
manual_upload = st.sidebar.file_uploader("Manual Logs (CSV) - Optional", type="csv")

if mf_upload:
    try:
        df = processor.process_data(mf_upload, manual_upload)
        
        latest = df.dropna(subset=['Expenditure_MA7', 'Steps_MA7']).iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trend Weight", f"{latest['Trend Weight (kg)']:.2f} kg", f"{latest['Weight Velocity (kg/week)']:.2f} kg/wk")
        col2.metric("Expenditure (MA7)", f"{latest['Expenditure_MA7']:.0f} kcal")
        col3.metric("Latest Intake", f"{latest['Calories (kcal)']:.0f} kcal", f"Deficit vs Target (1550): {latest['Actual Deficit'] - latest['Target Deficit']:.0f} kcal")
        col4.metric("Steps (MA7)", f"{latest['Steps_MA7']:.0f}", f"{latest['Steps_MA7'] - 12000:.0f} from goal")

        st.markdown("---")

        colA, colB = st.columns(2)
        base = alt.Chart(df).encode(x='Date:T')
        
        with colA:
            st.subheader("Energy Balance")
            line_exp = base.mark_line(color='red', strokeWidth=2).encode(y=alt.Y('Expenditure_MA7:Q', title='kcal (MA7)'))
            line_intake = base.mark_line(color='blue', strokeWidth=1, opacity=0.5).encode(y='Calories (kcal):Q')
            st.altair_chart(line_exp + line_intake, width='stretch')

        with colB:
            st.subheader("Activity Volume")
            line_steps = base.mark_line(color='orange', point=True).encode(y=alt.Y('Steps_MA7:Q', title='Steps (MA7)'))
            rule_steps = alt.Chart(pd.DataFrame({'y': [12000]})).mark_rule(color='green', strokeDash=[5,5]).encode(y='y:Q')
            st.altair_chart(line_steps + rule_steps, width='stretch')

        st.markdown("---")

        colC, colD = st.columns(2)

        with colC:
            st.subheader("Metabolic Response")
            scatter