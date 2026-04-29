import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def clean_and_normalize(df):
    df.columns = df.columns.str.strip()
    date_col = next((c for c in df.columns if c.lower() in ['date', 'startdate']), None)
    if not date_col: 
        raise KeyError(f"Date column missing in {df.columns.tolist()}")
    df = df.rename(columns={date_col: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', utc=True).dt.tz_convert(None).dt.normalize()
    return df

def calculate_step_tdee_regression(mf_file):
    df_expenditure = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Expenditure"))
    df_steps = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Steps"))
    merged_regression = df_steps[['Date', 'Steps']].merge(
        df_expenditure[['Date', 'Expenditure']],
        on='Date',
        how='inner'
    ).sort_values('Date')
    merged_regression['Steps_MA7'] = merged_regression['Steps'].rolling(window=7).mean()
    merged_regression = merged_regression.dropna(subset=['Steps_MA7', 'Expenditure'])
    slope, intercept, r_value, _, _ = linregress(merged_regression['Steps_MA7'], merged_regression['Expenditure'])
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value
    }

def calculate_multivariate_tdee(mf_file):
    df_expenditure = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Expenditure"))
    df_steps = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Steps"))
    df_weight = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Scale Weight"))[['Date', 'Weight (kg)']]
    df_sets = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Exercises - Total Sets"))
    df_sets['Daily_Lifting_Sets'] = df_sets.select_dtypes(include='number').sum(axis=1)
    df_sets = df_sets[['Date', 'Daily_Lifting_Sets']].copy()

    model_df = df_expenditure[['Date', 'Expenditure']].merge(
        df_steps[['Date', 'Steps']],
        on='Date',
        how='inner'
    ).merge(
        df_weight,
        on='Date',
        how='inner'
    ).merge(
        df_sets[['Date', 'Daily_Lifting_Sets']],
        on='Date',
        how='inner'
    ).sort_values('Date')

    model_df['Steps_MA7'] = model_df['Steps'].rolling(7).mean()
    model_df['Weight_MA7'] = model_df['Weight (kg)'].rolling(7).mean()
    model_df['Sets_MA7'] = model_df['Daily_Lifting_Sets'].rolling(7).mean()
    model_df[['Steps_MA7', 'Weight_MA7', 'Sets_MA7']] = model_df[['Steps_MA7', 'Weight_MA7', 'Sets_MA7']].ffill()
    model_df = model_df.dropna(subset=['Steps_MA7', 'Weight_MA7', 'Sets_MA7', 'Expenditure'])

    X = model_df[['Steps_MA7', 'Weight_MA7', 'Sets_MA7']]
    y = model_df['Expenditure']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    multivariate_r2 = r2_score(y, y_pred)

    return {
        'step_coef': model.coef_[0],
        'weight_coef': model.coef_[1],
        'set_coef': model.coef_[2],
        'intercept': model.intercept_,
        'r2': multivariate_r2
    }

def calculate_baseline_metrics(mf_file):
    df_expenditure = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Expenditure"))[['Date', 'Expenditure']]
    df_steps = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Steps"))[['Date', 'Steps']]
    df_weight = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Scale Weight"))[['Date', 'Weight (kg)']]
    df_sets = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Exercises - Total Sets"))
    df_sets['Daily_Lifting_Sets'] = df_sets.select_dtypes(include='number').sum(axis=1)
    df_sets = df_sets[['Date', 'Daily_Lifting_Sets']].copy()

    df_expenditure = df_expenditure.sort_values('Date')
    df_steps = df_steps.sort_values('Date')
    df_weight = df_weight.sort_values('Date')
    df_sets = df_sets.sort_values('Date')

    current_tdee = df_expenditure['Expenditure'].dropna().iloc[-1]
    current_weight = df_weight['Weight (kg)'].dropna().iloc[-1]
    current_daily_steps = df_steps['Steps'].dropna().tail(30).mean()
    latest_date = df_sets['Date'].max()
    last_30_days = pd.date_range(end=latest_date, periods=30, freq='D')
    sets_calendar = df_sets.set_index('Date')['Daily_Lifting_Sets'].reindex(last_30_days, fill_value=0)
    current_weekly_sets = sets_calendar.mean() * 7

    return {
        'current_tdee': current_tdee,
        'current_daily_steps': current_daily_steps,
        'current_weekly_sets': current_weekly_sets,
        'current_weight': current_weight
    }

def process_data(mf_file, manual_file=None):
    # Core sheets
    df_nutrition = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Calories & Macros"))
    df_expenditure = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Expenditure"))
    df_steps = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Steps")).rename(columns={'Steps': 'Total Steps'})
    df_weight_trend = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Weight Trend"))
    
    # New detailed sheets
    df_micros = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Micronutrients"))
    df_scale_weight = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Scale Weight"))
    df_sets = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Exercises - Total Sets"))
    df_weight = df_scale_weight[['Date', 'Weight (kg)']].copy()
    df_sets['Daily_Lifting_Sets'] = df_sets.select_dtypes(include='number').sum(axis=1)
    df_sets = df_sets[['Date', 'Daily_Lifting_Sets']].copy()
    print(df_sets['Daily_Lifting_Sets'].tail())
    
    # Select specific micros to avoid bloat, but ensure we have the main culprits for water retention/digestion
    micros_subset = df_micros[['Date', 'Sodium (mg)', 'Fiber (g)', 'Water (g)']].copy()
    
    # Merge pipeline
    merged_df = df_nutrition.merge(df_weight_trend, on='Date', how='outer') \
                            .merge(df_weight, on='Date', how='outer') \
                            .merge(df_expenditure, on='Date', how='outer') \
                            .merge(df_steps, on='Date', how='outer') \
                            .merge(df_sets[['Date', 'Daily_Lifting_Sets']], on='Date', how='outer') \
                            .merge(micros_subset, on='Date', how='outer')
                            
    # Ingest Manual Logs (Defecation) if provided
    if manual_file:
        df_manual = clean_and_normalize(pd.read_csv(manual_file))
        merged_df = merged_df.merge(df_manual, on='Date', how='left')
        if 'Defecation' in merged_df.columns:
            merged_df['Defecation'] = merged_df['Defecation'].fillna(0)
    else:
        # Initialize column so dashboard code doesn't break before you start logging
        merged_df['Defecation'] = pd.NA 

    # Clean up and filter
    merged_df = merged_df.sort_values('Date')
    merged_df['Weight (kg)'] = merged_df['Weight (kg)'].ffill()
    merged_df = merged_df.dropna(subset=['Expenditure'])
    merged_df = merged_df[merged_df['Calories (kcal)'].notna() & (merged_df['Calories (kcal)'] > 0)].copy()
    merged_df['Relative_TDEE'] = merged_df['Expenditure'] / merged_df['Weight (kg)']
    merged_df['Date_Ordinal'] = pd.to_datetime(merged_df['Date']).map(pd.Timestamp.toordinal)
    
    # Moving Averages
    merged_df['Steps_MA7'] = merged_df['Total Steps'].rolling(window=7, min_periods=3).mean()
    merged_df['Expenditure_MA7'] = merged_df['Expenditure'].rolling(window=7, min_periods=3).mean()
    merged_df['Weight_MA7'] = merged_df['Weight (kg)'].rolling(window=7, min_periods=3).mean()
    merged_df['Sets_MA7'] = merged_df['Daily_Lifting_Sets'].rolling(window=7, min_periods=3).mean()
    
    # Engineering metrics based on your goals
    merged_df['Weight Velocity (kg/week)'] = merged_df['Trend Weight (kg)'].diff(7)
    merged_df['Target Deficit'] = merged_df['Expenditure'] - 1550 # Based on your 1550 kcal target
    merged_df['Actual Deficit'] = merged_df['Expenditure'] - merged_df['Calories (kcal)']
    
    # Phase 2: Lag Analysis Metrics
    # Using Scale Weight (Weight (kg)) for daily bloat analysis, not Trend Weight
    merged_df['Next_Day_Scale_Weight'] = merged_df['Weight (kg)'].shift(-1)
    merged_df['Daily_Scale_Weight_Delta'] = merged_df['Next_Day_Scale_Weight'] - merged_df['Weight (kg)']
    
    return merged_df