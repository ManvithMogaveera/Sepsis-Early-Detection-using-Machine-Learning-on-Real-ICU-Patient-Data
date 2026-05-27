import pandas as pd
import numpy as np

df = pd.read_csv("Dataset.csv")
df = df.sort_values(
    by=['Patient_ID', 'Hour']
)
vitals = ['HR','Temp','SBP','MAP','Resp','O2Sat']
df[vitals] = (
    df.groupby('Patient_ID')[vitals]
      .transform(
          lambda x: x.interpolate(limit=2)
      )
)
df['HR']= df['HR'].fillna(df['HR'].median())
df['Temp']= df['Temp'].fillna(df['Temp'].median())
df['SBP']= df['SBP'].fillna(df['SBP'].median())
df['MAP']= df['MAP'].fillna(df['MAP'].median())
df['Resp']= df['Resp'].fillna(df['Resp'].median())
df['O2Sat']= df['O2Sat'].fillna(df['O2Sat'].median())
lab_cols = [
    'WBC','Creatinine','Platelets','Hgb',
    'Lactate','BUN','Glucose','Calcium',
    'Potassium','Magnesium','Phosphate'
]

df[lab_cols] = (
    df.groupby('Patient_ID')[lab_cols]
      .ffill()
)
df[lab_cols] = df[lab_cols].fillna(df[lab_cols].median())
lab_cols = ['Lactate','Creatinine','WBC']
important_lab_missing = [
    'Lactate',
    'Creatinine',
    'Hct'
]

for col in important_lab_missing:
    df[col + '_missing'] = (
        df[col].isnull().astype(int)
    )
df[lab_cols] = (
    df.groupby('Patient_ID')[lab_cols]
      .ffill()
)

df[lab_cols] = df[lab_cols].fillna(
    df[lab_cols].median()
)
# df['HR_diff'] = df['HR'].diff()
# df[vitals] = df[vitals].diff()
df['HR_diff'] = df.groupby('Patient_ID')['HR'].diff().fillna(0)
df['Resp_diff'] = df.groupby('Patient_ID')['Resp'].diff().fillna(0)
df['MAP_diff'] = df.groupby('Patient_ID')['MAP'].diff().fillna(0)
df['HR_roll_mean'] = (
    df.groupby('Patient_ID')['HR']
      .transform(
          lambda x: x.rolling(3, min_periods=1).mean()
      )
      .fillna(0)
)

df['MAP_roll_std'] = (
    df.groupby('Patient_ID')['MAP']
      .transform(
          lambda x: x.rolling(3, min_periods=1).std()
      )
      .fillna(0)
)
df['HR_Temp'] = df['HR'] * df['Temp']
df['Shock_Index'] = (
    df['HR'] /
    (df['SBP'] + 1e-6)
)
threshold = 0.95

missing_ratio = df.isnull().mean()

drop_cols = missing_ratio[
    missing_ratio > threshold
].index

df = df.drop(columns=drop_cols)
df['HospAdmTime'] = df['HospAdmTime'].fillna(
    df['HospAdmTime'].median()
)
df['Unit1'] = df['Unit1'].fillna(0)
df['Unit2'] = df['Unit2'].fillna(0)
df['DBP'] = (
    df.groupby('Patient_ID')['DBP']
      .transform(lambda x: x.interpolate(limit=2))
)

df['DBP'] = (
    df.groupby('Patient_ID')['DBP']
      .ffill()
)

df['DBP'] = df['DBP'].fillna(df['DBP'].median())
df['Pulse_Pressure'] = df['SBP'] - df['DBP']
df = df.drop('BaseExcess',axis=1)
important_missing = ['FiO2', 'pH', 'PaCO2']

for col in important_missing:
    df[col + '_missing'] = df[col].isnull().astype(int)


important_labs = ['FiO2', 'pH', 'PaCO2', 'Hct']

df[important_labs] = (
    df.groupby('Patient_ID')[important_labs]
      .ffill()
)


for col in important_labs:
    df[col] = df[col].fillna(df[col].median())

remaining_nulls = df.isnull().sum()

print(
    remaining_nulls[remaining_nulls > 0]
        .sort_values(ascending=False)
)


print(df.shape)
#
important_missing = [
    'FiO2_missing',
    'pH_missing',
    'PaCO2_missing',
    'Hct_missing',
    'Lactate_missing',
    'Creatinine_missing'
]
drop_missing = [
    col for col in df.columns
    if (
        col.endswith('_missing')
        and col not in important_missing
    )
]

df = df.drop(columns=drop_missing)

float_cols = df.select_dtypes(include=['float64']).columns
int_cols = df.select_dtypes(include=['int64']).columns

df[float_cols] = df[float_cols].astype('float32')
df[int_cols] = df[int_cols].astype('int32')

df = df.drop(columns=['Patient_ID'])

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.fillna(df.median(numeric_only=True), inplace=True)