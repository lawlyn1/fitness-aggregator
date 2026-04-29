import streamlit as st
import pandas as pd
import altair as alt
import processor

st.set_page_config(page_title="Fitness Aggregator", layout="wide")
st.title("Cutting & Metabolic Response Dashboard")

BMR_KG_MULT = 14.0
STEP_MULT = 0.45
SET_MULT = 5.0


def estimate_tdee(weight_kg, height_cm, age_years, sex, activity_multiplier, body_fat_pct=None):
    if body_fat_pct is not None:
        lean_body_mass = weight_kg * (1 - (body_fat_pct / 100))
        bmr = 370 + (21.6 * lean_body_mass)  # Katch-McArdle
        formula_used = "Katch-McArdle"
    else:
        sex_offset = 5 if sex == "Male" else -161
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) + sex_offset  # Mifflin-St Jeor
        formula_used = "Mifflin-St Jeor"
    return bmr * activity_multiplier, formula_used


def build_manual_dataframe(weight_kg, tdee_kcal, daily_steps, daily_intake):
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30, freq='D')
    df = pd.DataFrame(
        {
            'Date': dates,
            'Trend Weight (kg)': float(weight_kg),
            'Weight (kg)': float(weight_kg),
            'Weight Velocity (kg/week)': 0.0,
            'Expenditure': float(tdee_kcal),
            'Expenditure_MA7': float(tdee_kcal),
            'Calories (kcal)': float(daily_intake),
            'Total Steps': float(daily_steps),
            'Steps_MA7': float(daily_steps),
            'Daily_Lifting_Sets': 0.0,
            'Sets_MA7': 0.0,
            'Relative_TDEE': float(tdee_kcal) / float(weight_kg),
        }
    )
    return df


def calculate_weekly_intake(df):
    latest_date = pd.to_datetime(df['Date']).max()
    last_7_days = pd.date_range(end=latest_date, periods=7, freq='D')
    intake_series = df.set_index('Date')['Calories (kcal)'].reindex(last_7_days, fill_value=0)
    return float(intake_series.mean())


if "dashboard_loaded" not in st.session_state:
    st.session_state.dashboard_loaded = False
if "mode" not in st.session_state:
    st.session_state.mode = "MacroFactor CSV"

st.sidebar.header("Global Parameters")
st.sidebar.caption("Choose your daily calorie target here. This drives deficit calculations across all modes.")
global_target_intake = st.sidebar.number_input("Target Intake (kcal)", value=1550, step=50)
st.sidebar.info("You can change Target Intake at any time.")

if not st.session_state.dashboard_loaded:
    st.subheader("Start Session")
    st.write("Choose data mode before loading the analytics dashboard.")
    st.info("Set your Target Intake in the left sidebar before loading, or adjust it later during analysis.")
    mode = st.radio(
        "Data Source",
        ["MacroFactor CSV", "Manual Mode (No Export)"],
        index=0 if st.session_state.mode == "MacroFactor CSV" else 1
    )
    st.session_state.mode = mode

    if mode == "MacroFactor CSV":
        mf_upload = st.file_uploader("MacroFactor Export (XLSX)", type="xlsx")
        manual_upload = st.file_uploader("Manual Logs (CSV) - Optional", type="csv")
        if st.button("Load Dashboard", type="primary"):
            if mf_upload is None:
                st.warning("Please upload a MacroFactor XLSX export.")
            else:
                st.session_state.dashboard_loaded = True
                st.session_state.mf_upload = mf_upload
                st.session_state.manual_upload = manual_upload
                st.rerun()
    else:
        st.markdown("### Manual TDEE Estimator")
        st.caption("Body fat percentage is optional. If provided, Katch-McArdle is used. Otherwise Mifflin-St Jeor is used.")
        est_col1, est_col2, est_col3 = st.columns(3)
        with est_col1:
            manual_age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
            manual_sex = st.selectbox("Sex", ["Male", "Female"])
        with est_col2:
            manual_height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=175.0, step=1.0)
            manual_activity = st.selectbox(
                "Activity Level",
                [
                    ("Sedentary", 1.2),
                    ("Light Exercise", 1.375),
                    ("Moderate Exercise", 1.55),
                    ("Heavy Exercise", 1.725),
                    ("Athlete", 1.9),
                ],
                format_func=lambda x: x[0],
            )
        with est_col3:
            has_body_fat = st.checkbox("I know my body fat percentage (optional)", value=False)
            body_fat_pct = st.number_input("Body Fat (%)", min_value=3.0, max_value=60.0, value=20.0, step=0.5, disabled=not has_body_fat)

        manual_weight = st.number_input("Current Weight (kg)", value=75.0, step=0.1)
        calc_tdee, formula_used = estimate_tdee(
            weight_kg=float(manual_weight),
            height_cm=float(manual_height_cm),
            age_years=float(manual_age),
            sex=manual_sex,
            activity_multiplier=float(manual_activity[1]),
            body_fat_pct=float(body_fat_pct) if has_body_fat else None,
        )
        st.caption(f"Estimated with {formula_used}: {calc_tdee:.0f} kcal")
        manual_tdee = st.number_input("Estimated TDEE (kcal)", value=float(round(calc_tdee)), step=50.0)
        if st.button("Load Dashboard", type="primary"):
            st.session_state.dashboard_loaded = True
            st.session_state.manual_weight = manual_weight
            st.session_state.manual_tdee = manual_tdee
            st.rerun()
else:
    try:
        if st.session_state.mode == "MacroFactor CSV":
            df = processor.process_data(st.session_state.mf_upload, st.session_state.get("manual_upload"))
            multivariate = processor.calculate_multivariate_tdee(st.session_state.mf_upload)
            baseline_metrics = processor.calculate_baseline_metrics(st.session_state.mf_upload)
        else:
            baseline_metrics = processor.initialize_manual_baseline(
                current_weight=st.session_state.manual_weight,
                current_tdee=st.session_state.manual_tdee,
                target_intake=global_target_intake,
                baseline_steps=12000,
            )
            df = build_manual_dataframe(
                weight_kg=baseline_metrics['current_weight'],
                tdee_kcal=baseline_metrics['current_tdee'],
                daily_steps=baseline_metrics['current_daily_steps'],
                daily_intake=global_target_intake,
            )
            multivariate = {'r2': 0.0}

        current_weight = baseline_metrics['current_weight']
        current_tdee = baseline_metrics['current_tdee']
        current_daily_steps = baseline_metrics['current_daily_steps']
        current_weekly_sets = baseline_metrics['current_weekly_sets']

        avg_daily_steps_30d = float(current_daily_steps)
        avg_weekly_sets_30d = float(current_weekly_sets)
        avg_daily_sets_30d = avg_weekly_sets_30d / 7 if avg_weekly_sets_30d else 0.0

        current_step_cost = (current_daily_steps / 1000) * (current_weight * STEP_MULT)
        current_set_cost = (current_weekly_sets / 7) * SET_MULT
        sedentary_base = current_tdee - current_step_cost - current_set_cost
        forecasted_tdee_12k = sedentary_base + ((12000 / 1000) * (current_weight * STEP_MULT)) + ((avg_weekly_sets_30d / 7) * SET_MULT)
        implied_deficit = forecasted_tdee_12k - global_target_intake
        avg_weekly_intake = calculate_weekly_intake(df)

        latest = df.dropna(subset=['Expenditure_MA7', 'Steps_MA7']).iloc[-1]

        if st.session_state.mode == "Manual Mode (No Export)":
            tab_projection = st.tabs(["Forecasting & Goals"])[0]
        else:
            tab_overview, tab_model, tab_projection = st.tabs(["Daily Overview", "Metabolic Model", "Forecasting & Goals"])

            with tab_overview:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Trend Weight", f"{latest['Trend Weight (kg)']:.2f} kg", f"{latest['Weight Velocity (kg/week)']:.2f} kg/week")
                col2.metric("Expenditure (MA7)", f"{latest['Expenditure_MA7']:.0f} kcal")
                col3.metric(
                    "Average Weekly Intake",
                    f"{avg_weekly_intake:.0f} kcal",
                    f"vs target {global_target_intake:.0f} kcal: {avg_weekly_intake - global_target_intake:.0f}"
                )
                col4.metric("Steps (MA7)", f"{latest['Steps_MA7']:.0f} steps", f"{latest['Steps_MA7'] - 12000:.0f} from goal")

                st.markdown("---")
                colA, colB = st.columns(2)
                base = alt.Chart(df).encode(x=alt.X('Date:T', title='Date (calendar day)'))

                with colA:
                    st.subheader("Energy Balance")
                    line_exp = base.mark_line(color='red', strokeWidth=2).encode(y=alt.Y('Expenditure_MA7:Q', title='Expenditure (kcal, MA7)'))
                    line_intake = base.mark_line(color='blue', strokeWidth=1, opacity=0.7).encode(y=alt.Y('Calories (kcal):Q', title='Intake (kcal)'))
                    st.altair_chart(line_exp + line_intake, width='stretch')

                with colB:
                    st.subheader("Activity Volume")
                    line_steps = base.mark_line(color='orange', point=True).encode(y=alt.Y('Steps_MA7:Q', title='Steps (steps, MA7)'))
                    rule_steps = alt.Chart(pd.DataFrame({'y': [12000]})).mark_rule(color='green', strokeDash=[5, 5]).encode(y=alt.Y('y:Q', title='Steps (steps)'))
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
                fcol1.metric("Step Cost @ Current BW (1k)", f"{current_weight * STEP_MULT:.1f} kcal")
                fcol2.metric("Forecasted TDEE @ 12k", f"{forecasted_tdee_12k:.0f} kcal")
                fcol3.metric(f"Implied Deficit @ {global_target_intake:.0f} kcal", f"{implied_deficit:.0f} kcal")
                fcol4.metric("Multivariate R²", f"{multivariate['r2']:.2f}")

                model_table_weight = st.session_state.get("sim_weight_input", float(current_weight))
                step_value_per_1k = model_table_weight * STEP_MULT
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
            if st.session_state.mode == "Manual Mode (No Export)":
                anchor_col1, anchor_col2 = st.columns(2)
                anchor_col1.metric("Current Weight", f"{current_weight:.1f} kg")
                anchor_col2.metric("Current TDEE", f"{current_tdee:.0f} kcal")
            else:
                anchor_col1, anchor_col2, anchor_col3, anchor_col4 = st.columns(4)
                anchor_col1.metric("Current Weight", f"{current_weight:.1f} kg")
                anchor_col2.metric("Current TDEE", f"{current_tdee:.0f} kcal")
                anchor_col3.metric("Average Daily Steps (30-day)", f"{avg_daily_steps_30d:.0f} steps")
                anchor_col4.metric("Avg Weekly Sets (30-day)", f"{avg_weekly_sets_30d:.0f} sets")
            st.divider()

            st.markdown("### Interactive TDEE Simulator")
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                sim_weight = st.number_input("Target Weight (kg)", value=float(current_weight), step=0.1, key="sim_weight_input")
            with sim_col2:
                sim_steps = st.number_input("Simulated Steps", value=int(avg_daily_steps_30d), step=500)
            with sim_col3:
                sim_weekly_sets = st.number_input("Simulated Weekly Sets", value=float(avg_weekly_sets_30d), step=1.0)

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
                goal_weight = st.number_input("Goal Weight", value=float(current_weight - 2), step=0.1)
            with gcol2:
                target_date = st.date_input("Target Date", value=(pd.Timestamp.today() + pd.Timedelta(days=30)).date())
            with gcol3:
                planned_steps = st.number_input("Planned Steps", value=int(avg_daily_steps_30d), step=500)
            with gcol4:
                planned_weekly_sets = st.number_input("Planned Weekly Sets", value=float(avg_weekly_sets_30d), step=1.0)

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

        if st.sidebar.button("Reset Session"):
            st.session_state.dashboard_loaded = False
            st.rerun()

    except Exception as e:
        st.error(f"Pipeline failure: {e}")
        st.write("Please ensure the uploaded source data is valid for the selected mode.")