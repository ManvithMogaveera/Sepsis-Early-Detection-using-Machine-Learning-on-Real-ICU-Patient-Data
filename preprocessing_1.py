import pandas as pd
import numpy as np

VITALS = ['HR', 'Temp', 'SBP', 'MAP', 'Resp', 'O2Sat']

LAB_COLS = [
    'WBC', 'Creatinine', 'Platelets', 'Hgb',
    'Lactate', 'BUN', 'Glucose', 'Calcium',
    'Potassium', 'Magnesium', 'Phosphate'
]

# Missingness here carries clinical signal
#   (doctor ordered Lactate → suspicion of sepsis/shock)
CLINICALLY_IMPORTANT_MISSING = ['Lactate', 'Creatinine', 'Hct']
IMPORTANT_MISSING_FLAGS = [
    'FiO2_missing', 'pH_missing', 'PaCO2_missing',
    'Hct_missing', 'Lactate_missing', 'Creatinine_missing'
]

IMPORTANT_LABS_2 = ['FiO2', 'pH', 'PaCO2', 'Hct']

HIGH_MISSING_THRESHOLD = 0.95


def load_and_sort(path: str = "Dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(by=['Patient_ID', 'Hour']).reset_index(drop=True)
    print(f"[load]  shape={df.shape}")
    return df


def impute_vitals(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in VITALS if c in df.columns]
    df[cols] = (
        df.groupby('Patient_ID')[cols]
          .transform(lambda x: x.interpolate(limit=2))
    )
    for col in cols:
        df[col] = df[col].fillna(df[col].median())
    return df



def impute_labs_round1(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in LAB_COLS if c in df.columns]
    df[cols] = df.groupby('Patient_ID')[cols].ffill()
    df[cols] = df[cols].fillna(df[cols].median())
    return df


def add_clinical_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    for col in CLINICALLY_IMPORTANT_MISSING:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
    return df


def impute_labs_round2(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ['Lactate', 'Creatinine', 'WBC'] if c in df.columns]
    df[cols] = df.groupby('Patient_ID')[cols].ffill()
    df[cols] = df[cols].fillna(df[cols].median())
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    # Temporal diffs (per-patient) — captures rate-of-change
    # Early sepsis: HR and Resp rise, MAP falls rapidly
    for col in ['HR', 'Resp', 'MAP']:
        if col in df.columns:
            df[f'{col}_diff'] = (
                df.groupby('Patient_ID')[col]
                  .diff()
                  .fillna(0)
            )

    # Rolling statistics — 3-hour window (min 1 sample)
    if 'HR' in df.columns:
        df['HR_roll_mean'] = (
            df.groupby('Patient_ID')['HR']
              .transform(lambda x: x.rolling(3, min_periods=1).mean())
              .fillna(0)
        )
    if 'MAP' in df.columns:
        df['MAP_roll_std'] = (
            df.groupby('Patient_ID')['MAP']
              .transform(lambda x: x.rolling(3, min_periods=1).std())
              .fillna(0)
        )

    # HR × Temp interaction
    # Fever + tachycardia together are a stronger sepsis signal than either alone
    if 'HR' in df.columns and 'Temp' in df.columns:
        df['HR_Temp'] = df['HR'] * df['Temp']

    # Shock Index = HR / SBP
    # Rises in septic shock as HR↑ and SBP↓; +1e-6 avoids division by zero
    if 'HR' in df.columns and 'SBP' in df.columns:
        df['Shock_Index'] = df['HR'] / (df['SBP'] + 1e-6)

    return df


def drop_high_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > HIGH_MISSING_THRESHOLD].index.tolist()
    if drop_cols:
        print(f"[drop_high_missing]  dropping {len(drop_cols)} cols: {drop_cols}")
    df = df.drop(columns=drop_cols, errors='ignore')
    return df



def impute_admin(df: pd.DataFrame) -> pd.DataFrame:
    if 'HospAdmTime' in df.columns:
        df['HospAdmTime'] = df['HospAdmTime'].fillna(df['HospAdmTime'].median())
    if 'Unit1' in df.columns:
        df['Unit1'] = df['Unit1'].fillna(0)
    if 'Unit2' in df.columns:
        df['Unit2'] = df['Unit2'].fillna(0)
    return df



def impute_dbp_and_pulse_pressure(df: pd.DataFrame) -> pd.DataFrame:
    if 'DBP' not in df.columns:
        return df

    df['DBP'] = (
        df.groupby('Patient_ID')['DBP']
          .transform(lambda x: x.interpolate(limit=2))
    )
    df['DBP'] = df.groupby('Patient_ID')['DBP'].ffill()
    df['DBP'] = df['DBP'].fillna(df['DBP'].median())

    if 'SBP' in df.columns:
        df['Pulse_Pressure'] = df['SBP'] - df['DBP']

    return df



def drop_base_excess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['BaseExcess'], errors='ignore')
    return df


def impute_respiratory_labs(df: pd.DataFrame) -> pd.DataFrame:
    flag_cols = ['FiO2', 'pH', 'PaCO2']
    for col in flag_cols:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    cols = [c for c in IMPORTANT_LABS_2 if c in df.columns]
    df[cols] = df.groupby('Patient_ID')[cols].ffill()
    for col in cols:
        df[col] = df[col].fillna(df[col].median())

    return df



def drop_uninformative_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col for col in df.columns
        if col.endswith('_missing') and col not in IMPORTANT_MISSING_FLAGS
    ]
    if drop_cols:
        print(f"[drop_missing_flags]  dropping {len(drop_cols)} cols: {drop_cols}")
    df = df.drop(columns=drop_cols, errors='ignore')
    return df


def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols   = df.select_dtypes(include=['int64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols]   = df[int_cols].astype('int32')
    return df


def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df




def run_pipeline(path: str = "Dataset.csv") -> pd.DataFrame:
    df = load_and_sort(path)
    df = impute_vitals(df)
    df = impute_labs_round1(df)
    df = add_clinical_missing_flags(df)    
    df = impute_labs_round2(df)
    df = engineer_features(df)
    df = drop_high_missing(df)
    df = impute_admin(df)
    df = impute_dbp_and_pulse_pressure(df)
    df = drop_base_excess(df)
    df = impute_respiratory_labs(df)
    df = drop_uninformative_missing_flags(df)
    df = optimise_dtypes(df)
    # df = drop_patient_id(df)
    df = final_cleanup(df)

    
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0].sort_values(ascending=False)
    if remaining.empty:
        print("[null_audit] ✓  Zero nulls remaining")
    else:
        print("[null_audit] Remaining nulls:\n", remaining)

    print(f"[done]  final shape={df.shape}")
    return df



if __name__ == "__main__":
    df_clean = run_pipeline("Dataset.csv")
    df_clean.to_csv("Dataset_processed.csv", index=False)
    print("Saved → Dataset_processed.csv")