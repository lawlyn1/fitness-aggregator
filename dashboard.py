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
            scatter = alt.Chart(df.dropna(subset=['Steps_MA7', 'Expenditure_MA7'])).mark_circle(size=60).encode(
                x=alt.X('Steps_MA7:Q', title='Steps (7-Day MA)', scale=alt.Scale(zero=False)),
                y=alt.Y('Expenditure_MA7:Q', title='Expenditure (7-Day MA)', scale=alt.Scale(zero=False)),
                tooltip=['Date', 'Steps_MA7', 'Expenditure_MA7']
            )
            trendline = scatter.transform_regression('Steps_MA7', 'Expenditure_MA7').mark_line(color='red')
            st.altair_chart(scatter + trendline, width='stretch')

        with colD:
            st.subheader("Lag Analysis: Day T Input vs T+1 Scale Weight Delta")
            
            lag_options = [col for col in ['Sodium (mg)', 'Fiber (g)', 'Carbs (g)', 'Calories (kcal)', 'Water (g)'] if col in df.columns]
            
            if 'Defecation' in df.columns and not df['Defecation'].isna().all():
                lag_options.append('Defecation')
            
            if lag_options:
                selected_var = st.selectbox("Independent Variable (Day T)", lag_options)
                
                if selected_var == 'Defecation':
                    chart_lag = alt.Chart(df.dropna(subset=[selected_var, 'Daily_Scale_Weight_Delta'])).mark_boxplot().encode(
                        x=alt.X(f'{selected_var}:O', title='Defecation (1=Yes, 0=No)'),
                        y=alt.Y('Daily_Scale_Weight_Delta:Q', title='Scale Weight Delta kg (T+1 - T)')
                    )
                else:
                    chart_lag = alt.Chart(df.dropna(subset=[selected_var, 'Daily_Scale_Weight_Delta'])).mark_circle(size=60).encode(
                        x=alt.X(f'{selected_var}:Q', title=f'{selected_var} (Day T)', scale=alt.Scale(zero=False)),
                        y=alt.Y('Daily_Scale_Weight_Delta:Q', title='Scale Weight Delta kg (T+1 - T)'),
                        tooltip=['Date', selected_var, 'Daily_Scale_Weight_Delta']
                    )
                    chart_lag += chart_lag.transform_regression(selected_var, 'Daily_Scale_Weight_Delta').mark_line(color='red')
                
                st.altair_chart(chart_lag, width='stretch')
            else:
                st.warning("No lag variables detected. Ensure Micros sheet is present.")
        
    except Exception as e:
        st.error(f"Pipeline failure: {e}")
        st.write("Please ensure the uploaded XLSX is a full MacroFactor export.")
else:
    st.info("Awaiting MacroFactor payload.")