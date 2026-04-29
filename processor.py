import pandas as pd

def clean_and_normalize(df):
    df.columns = df.columns.str.strip()
    date_col = next((c for c in df.columns if c.lower() in ['date', 'startdate']), None)
    if not date_col: 
        raise KeyError(f"Date column missing in {df.columns.tolist()}")
    df = df.rename(columns={date_col: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', utc=True).dt.tz_convert(None).dt.normalize()
    return df

def process_data(mf_file, manual_file=None):
    # Core sheets
    df_nutrition = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Calories & Macros"))
    df_expenditure = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Expenditure"))
    df_steps = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Steps")).rename(columns={'Steps': 'Total Steps'})
    df_weight_trend = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Weight Trend"))
    
    # New detailed sheets
    df_micros = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Micronutrients"))
    df_scale_weight = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Scale Weight"))
    
    # Select specific micros to avoid bloat, but ensure we have the main culprits for water retention/digestion
    micros_subset = df_micros[['Date', 'Sodium (mg)', 'Fiber (g)', 'Water (g)']].copy()
    
    # Merge pipeline
    merged_df = df_nutrition.merge(df_weight_trend, on='Date', how='outer') \
                            .merge(df_scale_weight[['Date', 'Weight (kg)']], on='Date', how='outer') \
                            .merge(df_expenditure, on='Date', how='outer') \
                            .merge(df_steps, on='Date', how='outer') \
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
    merged_df = merged_df.sort_values('Date').dropna(subset=['Expenditure'])
    merged_df = merged_df[merged_df['Calories (kcal)'].notna() & (merged_df['Calories (kcal)'] > 0)].copy()
    
    # Moving Averages
    merged_df['Steps_MA7'] = merged_df['Total Steps'].rolling(window=7, min_periods=3).mean()
    merged_df['Expenditure_MA7'] = merged_df['Expenditure'].rolling(window=7, min_periods=3).mean()
    
    # Engineering metrics based on your goals
    merged_df['Weight Velocity (kg/week)'] = merged_df['Trend Weight (kg)'].diff(7)
    merged_df['Target Deficit'] = merged_df['Expenditure'] - 1550 # Based on your 1550 kcal target
    merged_df['Actual Deficit'] = merged_df['Expenditure'] - merged_df['Calories (kcal)']
    
    # Phase 2: Lag Analysis Metrics
    # Using Scale Weight (Weight (kg)) for daily bloat analysis, not Trend Weight
    merged_df['Next_Day_Scale_Weight'] = merged_df['Weight (kg)'].shift(-1)
    merged_df['Daily_Scale_Weight_Delta'] = merged_df['Next_Day_Scale_Weight'] - merged_df['Weight (kg)']
    
    return merged_df