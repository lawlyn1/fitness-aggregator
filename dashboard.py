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


def build_manual_editor_seed(weight_kg, baseline_steps, baseline_intake):
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30, freq='D')
    return pd.DataFrame(
        {
            "Date": dates,
            "Weight (kg)": float(weight_kg),
            "Calories (kcal)": float(baseline_intake),
            "Total Steps": float(baseline_steps),
            "Daily_Lifting_Sets": 0.0,
            "Protein (g)": pd.NA,
            "Carbs (g)": pd.NA,
            "Fat (g)": pd.NA,
            "Sodium (mg)": pd.NA,
            "Defecation": 0.0,
        }
    )


def build_manual_dataframe_from_editor(editor_df, estimated_tdee, target_intake):
    df = editor_df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    df["Weight (kg)"] = pd.to_numeric(df["Weight (kg)"], errors="coerce").ffill().bfill()
    df["Calories (kcal)"] = pd.to_numeric(df["Calories (kcal)"], errors="coerce").fillna(float(target_intake))
    df["Total Steps"] = pd.to_numeric(df["Total Steps"], errors="coerce").fillna(12000.0)
    df["Daily_Lifting_Sets"] = pd.to_numeric(df["Daily_Lifting_Sets"], errors="coerce").fillna(0.0)
    df["Protein (g)"] = pd.to_numeric(df.get("Protein (g)"), errors="coerce")
    df["Carbs (g)"] = pd.to_numeric(df.get("Carbs (g)"), errors="coerce")
    df["Fat (g)"] = pd.to_numeric(df.get("Fat (g)"), errors="coerce")
    df["Sodium (mg)"] = pd.to_numeric(df["Sodium (mg)"], errors="coerce")
    df["Defecation"] = pd.to_numeric(df["Defecation"], errors="coerce").fillna(0.0)

    df["Expenditure"] = float(estimated_tdee)
    df["Trend Weight (kg)"] = df["Weight (kg)"].ewm(span=14, adjust=False, min_periods=1).mean()
    df["Relative_TDEE"] = df["Expenditure"] / df["Weight (kg)"]
    df["Expenditure_MA7"] = df["Expenditure"].ewm(span=14, adjust=False, min_periods=1).mean()
    df["Steps_MA7"] = df["Total Steps"].rolling(window=7, min_periods=1).mean()
    df["Sets_MA7"] = df["Daily_Lifting_Sets"].rolling(window=7, min_periods=1).mean()
    df["Weight Velocity (kg/week)"] = df["Trend Weight (kg)"].diff(7)
    df["Target Deficit"] = df["Expenditure"] - float(target_intake)
    df["Actual Deficit"] = df["Expenditure"] - df["Calories (kcal)"]
    df["Next_Day_Scale_Weight"] = df["Weight (kg)"].shift(-1)
    df["Daily_Scale_Weight_Delta"] = df["Next_Day_Scale_Weight"] - df["Weight (kg)"]
    return df


def calculate_weekly_review(df):
    weekly_df = df.copy()
    weekly_df["Week Start"] = pd.to_datetime(weekly_df["Date"]).dt.to_period("W-MON").dt.start_time
    grouped = weekly_df.groupby("Week Start", as_index=False).agg(
        avg_weekly_weight=("Weight (kg)", "mean"),
        avg_daily_energy_balance=("Actual Deficit", "mean"),
        total_weekly_steps=("Total Steps", "sum"),
    )
    # Positive Actual Deficit means cutting; invert so negative = deficit, positive = surplus.
    grouped["avg_daily_energy_balance"] = -grouped["avg_daily_energy_balance"]
    grouped["Weekly Phase"] = grouped["avg_daily_energy_balance"].apply(
        lambda x: "Surplus (Bulking)" if x > 0 else "Deficit (Cutting)"
    )
    return grouped.sort_values("Week Start")


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
            st.session_state.manual_editor_seed = build_manual_editor_seed(
                weight_kg=manual_weight,
                baseline_steps=12000,
                baseline_intake=global_target_intake,
            )
            st.rerun()
else:
    try:
        if st.session_state.mode == "MacroFactor CSV":
            df = processor.process_data(st.session_state.mf_upload, st.session_state.get("manual_upload"))
            multivariate = processor.calculate_multivariate_tdee(st.session_state.mf_upload)
            baseline_metrics = processor.calculate_baseline_metrics(st.session_state.mf_upload)
        else:
            st.markdown("### Manual Data Entry")
            st.caption("Paste or type daily records to unlock the full dashboard without an export.")
            seed_df = st.session_state.get("manual_editor_seed", build_manual_editor_seed(st.session_state.manual_weight, 12000, global_target_intake))
            editable_df = st.data_editor(
                seed_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date"),
                    "Weight (kg)": st.column_config.NumberColumn("Weight (kg)", format="%.2f"),
                    "Calories (kcal)": st.column_config.NumberColumn("Calories (kcal)", format="%.0f"),
                    "Total Steps": st.column_config.NumberColumn("Total Steps", format="%.0f"),
                    "Daily_Lifting_Sets": st.column_config.NumberColumn("Daily Lifting Sets", format="%.1f"),
                    "Protein (g)": st.column_config.NumberColumn("Protein (g)", format="%.0f"),
                    "Carbs (g)": st.column_config.NumberColumn("Carbs (g)", format="%.0f"),
                    "Fat (g)": st.column_config.NumberColumn("Fat (g)", format="%.0f"),
                    "Sodium (mg)": st.column_config.NumberColumn("Sodium (mg)", format="%.0f"),
                    "Defecation": st.column_config.NumberColumn("Defecation", format="%.0f"),
                },
                key="manual_editor_table",
            )
            st.download_button(
                "Download Data (CSV)",
                data=editable_df.to_csv(index=False),
                file_name="manual_fitness_data.csv",
                mime="text/csv",
            )
            st.session_state.manual_editor_seed = editable_df
            df = build_manual_dataframe_from_editor(editable_df, st.session_state.manual_tdee, global_target_intake)
            baseline_metrics = {
                "current_tdee": float(st.session_state.manual_tdee),
                "current_daily_steps": float(df["Total Steps"].tail(30).mean()),
                "current_weekly_sets": float(df["Daily_Lifting_Sets"].tail(30).mean() * 7),
                "current_weight": float(df["Weight (kg)"].dropna().iloc[-1]),
            }
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

        tab_overview, tab_model, tab_reviews, tab_macros, tab_projection = st.tabs(
            ["Daily Overview", "Metabolic Model", "Reviews", "Macro Composition", "Forecasting & Goals"]
        )

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
                energy_chart_df = df[['Date', 'Expenditure_MA7', 'Calories (kcal)']].rename(
                    columns={'Expenditure_MA7': 'Expenditure (EMA14)', 'Calories (kcal)': 'Intake'}
                )
                energy_long = energy_chart_df.melt(
                    id_vars=['Date'],
                    value_vars=['Expenditure (EMA14)', 'Intake'],
                    var_name='Series',
                    value_name='kcal'
                )
                energy_line = alt.Chart(energy_long).mark_line(strokeWidth=2).encode(
                    x=alt.X('Date:T', title='Date (calendar day)'),
                    y=alt.Y('kcal:Q', title='Energy (kcal)'),
                    color=alt.Color(
                        'Series:N',
                        scale=alt.Scale(domain=['Expenditure (EMA14)', 'Intake'], range=['red', 'blue']),
                        legend=alt.Legend(title='Series')
                    )
                )
                st.altair_chart(energy_line, width='stretch')

            with colB:
                st.subheader("Activity Volume")
                steps_chart_df = df[['Date', 'Steps_MA7']].copy()
                steps_chart_df['Step Goal'] = 12000.0
                steps_chart_df = steps_chart_df.rename(columns={'Steps_MA7': 'Steps (MA7)'})
                steps_long = steps_chart_df.melt(
                    id_vars=['Date'],
                    value_vars=['Steps (MA7)', 'Step Goal'],
                    var_name='Series',
                    value_name='Steps'
                )
                steps_line = alt.Chart(steps_long).mark_line(point=True).encode(
                    x=alt.X('Date:T', title='Date (calendar day)'),
                    y=alt.Y('Steps:Q', title='Steps (steps)'),
                    color=alt.Color(
                        'Series:N',
                        scale=alt.Scale(domain=['Steps (MA7)', 'Step Goal'], range=['orange', 'green']),
                        legend=alt.Legend(title='Series')
                    ),
                    strokeDash=alt.StrokeDash('Series:N', scale=alt.Scale(domain=['Steps (MA7)', 'Step Goal'], range=[[1, 0], [5, 5]]), legend=None)
                )
                st.altair_chart(steps_line, width='stretch')

            st.markdown("---")
            st.subheader("Expected vs Actual Deficit")
            deficit_df = df.dropna(subset=['Date', 'Actual Deficit', 'Weight Velocity (kg/week)']).copy()
            deficit_df['Deficit From Weight Trend'] = (-deficit_df['Weight Velocity (kg/week)'] * 7700) / 7
            deficit_long = deficit_df.melt(
                id_vars=['Date'],
                value_vars=['Actual Deficit', 'Deficit From Weight Trend'],
                var_name='Series',
                value_name='Deficit (kcal)'
            )
            deficit_long['Series'] = deficit_long['Series'].map({
                'Actual Deficit': 'Logged Deficit',
                'Deficit From Weight Trend': 'Weight-Trend Equivalent'
            })
            deficit_chart = alt.Chart(deficit_long).mark_line().encode(
                x=alt.X('Date:T', title='Date (calendar day)'),
                y=alt.Y('Deficit (kcal):Q', title='Daily Deficit (kcal)'),
                color=alt.Color(
                    'Series:N',
                    scale=alt.Scale(domain=['Logged Deficit', 'Weight-Trend Equivalent'], range=['purple', 'brown']),
                    legend=alt.Legend(title='Series')
                )
            )
            st.altair_chart(deficit_chart, width='stretch')

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
            fcol4.metric("Multivariate R²", f"{multivariate['r2']:.2f}" if st.session_state.mode == "MacroFactor CSV" else "N/A")

            with st.expander("View Energy Value Reference Table"):
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

        with tab_reviews:
            review_mode = st.radio(
                "Select Review Type",
                ["Lag Analysis", "Weekly Review"],
                label_visibility="collapsed"
            )

            if review_mode == "Lag Analysis":
                st.markdown("### Daily Fluctuations and Lag Analysis")
                lag_df = df.dropna(subset=['Daily_Scale_Weight_Delta']).copy()
                lag_col1, lag_col2 = st.columns(2)
                with lag_col1:
                    sodium_df = lag_df.dropna(subset=['Sodium (mg)'])
                    sodium_scatter = alt.Chart(sodium_df).mark_circle(size=65).encode(
                        x=alt.X('Sodium (mg):Q', title='Sodium (mg)'),
                        y=alt.Y('Daily_Scale_Weight_Delta:Q', title='Next-Day Scale Weight Delta (kg)'),
                        color=alt.value('teal'),
                        tooltip=['Date:T', 'Sodium (mg):Q', 'Daily_Scale_Weight_Delta:Q']
                    )
                    st.altair_chart(sodium_scatter, width='stretch')
                with lag_col2:
                    defecation_df = lag_df.dropna(subset=['Defecation'])
                    defecation_scatter = alt.Chart(defecation_df).mark_circle(size=65).encode(
                        x=alt.X('Defecation:Q', title='Defecation (count)'),
                        y=alt.Y('Daily_Scale_Weight_Delta:Q', title='Next-Day Scale Weight Delta (kg)'),
                        color=alt.value('darkorange'),
                        tooltip=['Date:T', 'Defecation:Q', 'Daily_Scale_Weight_Delta:Q']
                    )
                    st.altair_chart(defecation_scatter, width='stretch')
            else:
                st.markdown("### Week-over-Week Review")
                weekly_review = calculate_weekly_review(df.dropna(subset=["Date"]))
                wk1, wk2, wk3 = st.columns(3)
                wk1.metric("Latest Weekly Avg Weight", f"{weekly_review['avg_weekly_weight'].iloc[-1]:.2f} kg")
                wk2.metric("Latest Avg Daily Balance", f"{weekly_review['avg_daily_energy_balance'].iloc[-1]:.0f} kcal")
                wk3.metric("Latest Weekly Steps", f"{weekly_review['total_weekly_steps'].iloc[-1]:.0f} steps")

                with st.expander("30-Day Adherence Score"):
                    adherence_window = df.dropna(subset=["Date"]).sort_values("Date").tail(30).copy()
                    intake_ratio = (global_target_intake / adherence_window["Calories (kcal)"]).clip(upper=1.0)
                    steps_ratio = (adherence_window["Total Steps"] / float(avg_daily_steps_30d)).clip(upper=1.0)
                    adherence_score = ((intake_ratio.fillna(0) + steps_ratio.fillna(0)) / 2).mean() * 100
                    st.metric("30-Day Adherence Score", f"{adherence_score:.1f}%")
                    st.caption("Measures how well you adhered to your calorie and step goals over the last 30 days. Calculated as the average of (target/actual intake) and (actual/average steps), both capped at 100%. Higher is better.")

                deficit_bar = alt.Chart(weekly_review).mark_bar().encode(
                    x=alt.X("Week Start:T", title="Week Start (date)"),
                    y=alt.Y("avg_daily_energy_balance:Q", title="Average Daily Energy Balance (kcal)"),
                    color=alt.Color(
                        "Weekly Phase:N",
                        scale=alt.Scale(
                            domain=["Deficit (Cutting)", "Surplus (Bulking)"],
                            range=["#B22222", "#2E8B57"]
                        ),
                        legend=alt.Legend(title="Phase")
                    ),
                    tooltip=["Week Start:T", "avg_weekly_weight:Q", "avg_daily_energy_balance:Q", "total_weekly_steps:Q", "Weekly Phase:N"]
                )
                st.altair_chart(deficit_bar, width='stretch')

        with tab_macros:
            st.markdown("### Macronutrient Composition Over Time")
            macro_df = df.dropna(subset=["Date"]).copy()
            macro_cols = ["Protein (g)", "Carbs (g)", "Fat (g)"]
            for m_col in macro_cols:
                if m_col not in macro_df.columns:
                    macro_df[m_col] = 0
            melted_macros = macro_df.melt(
                id_vars=["Date"],
                value_vars=macro_cols,
                var_name="Macronutrient",
                value_name="Grams",
            )
            melted_macros["Grams"] = pd.to_numeric(melted_macros["Grams"], errors="coerce").fillna(0)
            macro_area = alt.Chart(melted_macros).mark_area(opacity=0.7).encode(
                x=alt.X("Date:T", title="Date (calendar day)"),
                y=alt.Y("Grams:Q", title="Macronutrients (g)"),
                color=alt.Color("Macronutrient:N", title="Macronutrient"),
                tooltip=["Date:T", "Macronutrient:N", "Grams:Q"]
            )
            st.altair_chart(macro_area, width="stretch")

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

            with st.expander("Interactive TDEE Simulator"):
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

            with st.expander("Goal Projection Engine"):
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