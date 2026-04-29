import streamlit as st
import pandas as pd
import altair as alt
import processor

st.set_page_config(page_title="Fitness Aggregator", layout="wide")
st.title("Cutting & Metabolic Response Dashboard")

st.sidebar.header("Global Parameters")
global_target_intake = st.sidebar.number_input("Target Intake (kcal)", value=1550, step=50)
st.sidebar.header("Data Ingestion")
mf_upload = st.sidebar.file_uploader("MacroFactor Export (XLSX)", type="xlsx")
manual_upload = st.sidebar.file_uploader("Manual Logs (CSV) - Optional", type="csv")

tab_overview, tab_model, tab_projection = st.tabs(["Daily Overview", "Metabolic Model", "Forecasting & Goals"])

if mf_upload:
    try:
        df = processor.process_data(mf_upload, manual_upload)
        multivariate = processor.calculate_multivariate_tdee(mf_upload)
        baseline_metrics = processor.calculate_baseline_metrics(mf_upload)
        current_weight = baseline_metrics['current_weight']
        current_tdee = baseline_metrics['current_tdee']
        current_daily_steps = baseline_metrics['current_daily_steps']
        current_weekly_sets = baseline_metrics['current_weekly_sets']
        target_sets = df['Daily_Lifting_Sets'].dropna().tail(7).mean()
        BMR_KG_MULT = 14.0
        STEP_MULT = 0.45
        SET_MULT = 5.0
        current_step_cost = (current_daily_steps / 1000) * (current_weight * STEP_MULT)
        current_set_cost = (current_weekly_sets / 7) * SET_MULT
        sedentary_base = current_tdee - current_step_cost - current_set_cost
        forecasted_tdee_12k = (
            sedentary_base +
            ((current_weight - current_weight) * BMR_KG_MULT) +
            ((12000 / 1000) * (current_weight * STEP_MULT)) +
            ((target_sets * 7 / 7) * SET_MULT)
        )
        implied_deficit = forecasted_tdee_12k - global_target_intake
        
        latest = df.dropna(subset=['Expenditure_MA7', 'Steps_MA7']).iloc[-1]
        with tab_overview:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Trend Weight", f"{latest['Trend Weight (kg)']:.2f} kg", f"{latest['Weight Velocity (kg/week)']:.2f} kg/wk")
            col2.metric("Expenditure (MA7)", f"{latest['Expenditure_MA7']:.0f} kcal")
            col3.metric(
                "Latest Intake",
                f"{latest['Calories (kcal)']:.0f} kcal",
                f"Deficit vs Target ({global_target_intake:.0f}): {latest['Expenditure'] - latest['Calories (kcal)'] - (latest['Expenditure'] - global_target_intake):.0f} kcal"
            )
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

        with tab_model:
            st.markdown("### Metabolic Adaptation")
            metabolic_df = df.dropna(subset=['Weight (kg)', 'Relative_TDEE', 'Date']).copy()
            metabolic_scatter = alt.Chart(metabolic_df).mark_circle(size=70).encode(
                x=alt.X('Weight (kg):Q', title='Weight (kg)', scale=alt.Scale(zero=False)),
                y=alt.Y('Relative_TDEE:Q', title='Relative TDEE (kcal/kg)', scale=alt.Scale(zero=False)),
                color=alt.Color('Date:T', scale=alt.Scale(scheme='viridis'), title='Date'),
                tooltip=['Date:T', 'Weight (kg):Q', 'Relative_TDEE:Q']
            )
            metabolic_regression = metabolic_scatter.transform_regression('Weight (kg)', 'Relative_TDEE').mark_line(color='black')
            st.altair_chart(metabolic_scatter + metabolic_regression, width='stretch')

            st.markdown("### Multivariate Context")
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            fcol1.metric("Step Cost @ Current BW (1k)", f"{current_weight * STEP_MULT:.1f}")
            fcol2.metric("Forecasted TDEE @ 12k", f"{forecasted_tdee_12k:.0f}")
            fcol3.metric(f"Implied Deficit @ {global_target_intake:.0f} kcal", f"{implied_deficit:.0f}")
            fcol4.metric("Multivariate R²", f"{multivariate['r2']:.2f}")

        model_table_weight = st.session_state.get("sim_weight_input", float(current_weight))
        step_value_per_1k = model_table_weight * STEP_MULT

        with tab_model:
            st.markdown(
                f"""| Variable | Energy Value | Units | Metabolic Implication |
|---|---:|---|---|
| Body Mass | {BMR_KG_MULT:.2f} | kcal / kg | Change in BMR per kg. |
| Step Volume | {step_value_per_1k:.2f} | kcal / 1k steps | Scales dynamically with user weight input. |
| Lifting (1 Set) | {SET_MULT:.2f} | kcal / set | Caloric cost per set performed. |
| Sedentary Baseline | {sedentary_base:.2f} | kcal | Derived BMR + NEAT (excluding steps/lifting). |"""
            )

        with tab_projection:
            st.markdown("### Current State Anchor")
            anchor_col1, anchor_col2, anchor_col3, anchor_col4 = st.columns(4)
            anchor_col1.metric("Current Weight", f"{current_weight:.1f} kg")
            anchor_col2.metric("MacroFactor TDEE", f"{current_tdee:.0f} kcal")
            anchor_col3.metric("Avg Daily Steps", f"{current_daily_steps:.0f}")
            anchor_col4.metric("Avg Weekly Sets", f"{current_weekly_sets:.0f}")
            st.divider()

            st.markdown("### Interactive TDEE Simulator")
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                sim_weight = st.number_input("Target Weight (kg)", value=float(current_weight), step=0.1, key="sim_weight_input")
            with sim_col2:
                sim_steps = st.number_input("Sim_Steps", value=12000, step=500)
            with sim_col3:
                sim_weekly_sets = st.number_input("Simulated_Weekly_Sets", value=float(target_sets * 7), step=1.0)

            sim_bmr_adjustment = (sim_weight - current_weight) * BMR_KG_MULT
            sim_step_cost = (sim_steps / 1000) * (sim_weight * STEP_MULT)
            sim_set_cost = (sim_weekly_sets / 7) * SET_MULT
            simulated_tdee = sedentary_base + sim_bmr_adjustment + sim_step_cost + sim_set_cost
            simulated_deficit = simulated_tdee - global_target_intake

            sim_out1, sim_out2 = st.columns(2)
            sim_out1.metric("Simulated TDEE", f"{simulated_tdee:.0f} kcal")
            sim_out2.metric("Simulated Deficit", f"{simulated_deficit:.0f} kcal")

            st.markdown("### Goal Projection Engine")
            today = pd.Timestamp.today().date()
            gcol1, gcol2, gcol3, gcol4 = st.columns(4)
            with gcol1:
                goal_weight = st.number_input("Goal_Weight", value=float(current_weight - 2), step=0.1)
            with gcol2:
                target_date = st.date_input("Target_Date", value=(pd.Timestamp.today() + pd.Timedelta(days=30)).date())
            with gcol3:
                planned_steps = st.number_input("Planned_Steps", value=12000, step=500)
            with gcol4:
                planned_weekly_sets = st.number_input("Planned_Weekly_Sets", value=float(target_sets * 7), step=1.0)

            days_remaining = (pd.to_datetime(target_date).date() - today).days
            kg_to_lose = current_weight - goal_weight
            total_deficit_needed = kg_to_lose * 7700
            required_daily_deficit = total_deficit_needed / days_remaining if days_remaining > 0 else 0
            avg_projected_weight = (current_weight + goal_weight) / 2
            proj_bmr_adj = (avg_projected_weight - current_weight) * BMR_KG_MULT
            proj_step_cost = (planned_steps / 1000) * (avg_projected_weight * STEP_MULT)
            proj_set_cost = (planned_weekly_sets / 7) * SET_MULT
            projected_average_tdee = sedentary_base + proj_bmr_adj + proj_step_cost + proj_set_cost
            target_daily_intake = projected_average_tdee - required_daily_deficit

            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Required Daily Deficit", f"{required_daily_deficit:.0f} kcal")
            pcol2.metric("Projected Average TDEE", f"{projected_average_tdee:.0f} kcal")
            pcol3.metric("Target Daily Intake", f"{target_daily_intake:.0f} kcal")
        
    except Exception as e:
        st.error(f"Pipeline failure: {e}")
        st.write("Please ensure the uploaded XLSX is a full MacroFactor export.")
else:
    st.info("Awaiting MacroFactor payload.")