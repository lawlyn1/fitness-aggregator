import pandas as pd

def clean_and_normalize(df):
    df.columns = df.columns.str.strip()
    date_col = next((c for c in df.columns if c.lower() in ['date', 'startdate']), None)
    if not date_col: 
        raise KeyError(f"Date column missing in {df.columns.tolist()}")
    df = df.rename(columns={date_col: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', utc=True).dt.tz_convert(None).dt.normalize()
    return df

def process_mf_export(mf_file):
    # MacroFactor names its sheets exactly as exported
    df_nutrition = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Calories & Macros"))
    df_weight = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Weight Trend"))
    df_expenditure = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Expenditure"))
    df_steps = clean_and_normalize(pd.read_excel(mf_file, sheet_name="Steps"))
    
    # Rename for consistency with existing dashboard code
    df_steps = df_steps.rename(columns={'Steps': 'Total Steps'})
    
    # Merge sequentially
    merged_df = df_nutrition.merge(df_weight, on='Date', how='outer') \
                            .merge(df_expenditure, on='Date', how='outer') \
                            .merge(df_steps, on='Date', how='outer')
    
    # Transformations & Filtering
    merged_df = merged_df.sort_values('Date').dropna(subset=['Expenditure'])
    merged_df = merged_df[merged_df['Calories (kcal)'].notna() & (merged_df['Calories (kcal)'] > 0)].copy()
    
    # Moving Averages
    merged_df['Steps_MA7'] = merged_df['Total Steps'].rolling(window=7, min_periods=3).mean()
    merged_df['Expenditure_MA7'] = merged_df['Expenditure'].rolling(window=7, min_periods=3).mean()
    
    # Engineering metrics
    merged_df['Weight Velocity (kg/week)'] = merged_df['Trend Weight (kg)'].diff(7)
    merged_df['Actual Deficit'] = merged_df['Expenditure'] - merged_df['Calories (kcal)']
    merged_df['Step Target Variance'] = merged_df['Total Steps'] - 12000
    
    return merged_df