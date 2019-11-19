import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

base_df = pd.read_csv(r"check.csv")
#  objects -> datettimes
#  im just going to drop these features, so do i even need to do this
for column in ["DOB", "DOD", "DOD_HOSP", "DOD_SSN", 
"ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME", "EDOUTTIME"]:
    base_df[column] = pd.to_datetime(base_df[column])

#  binary encoding of gender
base_df = base_df.replace(["F", "M"], [0, 1])

#  list of categorical features to be encoded
to_encode = ['ADMISSION_TYPE', 'ADMISSION_LOCATION', 
'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 
'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']

#  dummy encoding
base_df = pd.get_dummies(base_df, columns = to_encode)

#  import first_day values into dataframes for processing
labs_df = pd.read_csv(r'first_day_labs.csv')
vitals_df = pd.read_csv(r'vitals_first_day.csv')
height_df = pd.read_csv(r'height_first_day.csv')
weight_df = pd.read_csv(r'weight_first_day.csv')

#  replace nulls with 0 for informative missingness
labs_df = labs_df.fillna(0)

#  drop all the height and weight columns besides height, weight, icustay_id
height_df = height_df.drop(["height_chart", "height_echo"], axis=1)
weight_df = weight_df.drop(["weight_admit", "weight_daily"], axis=1)

#  convert height icustay_id to int so it can be merged with all other dataframes
height_df = height_df.astype({'icustay_id': 'int64'})
vitals_df = vitals_df.rename(columns={'glucose_mean': 'glucose_vital_mean',
                                        'glucose_min': 'glucose_vital_min',
                                        'glucose_max': 'glucose_vital_max'})

#  merge everything together
height_weight_df = pd.merge(height_df, weight_df, left_on='icustay_id', right_on='icustay_id')
vitals_df = pd.merge(height_weight_df, vitals_df, left_on='icustay_id', right_on='icustay_id')
first_days_df = pd.merge(vitals_df, labs_df, left_on=['subject_id','hadm_id','icustay_id'], right_on=['subject_id','hadm_id','icustay_id'])

#  final df should have 67 columns - good.

#  merge base and first days based on hadm and subj id
final_df = pd.merge(base_df, first_days_df, how="left", left_on=["SUBJECT_ID", "HADM_ID"], right_on=["subject_id", "hadm_id"])
final_df = final_df.drop(["subject_id", "hadm_id"], axis=1)

'''drop duplicates of hadm_id - duplicates appear because of icustay_ids which we
are not concerned with because we are addressing hospital readmission
and hence interested in first-day of hospital admission vitals, etc.'''
final_df = final_df.drop_duplicates(['HADM_ID'], keep='first')

#  fill in vitals with medians - MCAR
MCAR_values = ['height', 'weight', 'heartrate_min', 'heartrate_max', 'heartrate_mean', 'sysbp_min', 'sysbp_max', 'sysbp_mean', 
    'diasbp_min', 'diasbp_max', 'diasbp_mean', 'meanbp_min', 'meanbp_max', 'meanbp_mean', 'resprate_min', 'resprate_max', 'resprate_mean', 
    'tempc_min', 'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 'spo2_mean', 'glucose_vital_min', 'glucose_vital_max', 'glucose_vital_mean']

for i in range(len(MCAR_values)):
    final_df[MCAR_values[i]] = final_df[MCAR_values[i]].fillna(final_df[MCAR_values[i]].median())

IM_values = ['aniongap_min', 'aniongap_max', 'albumin_min', 'albumin_max', 'bands_min', 'bands_max', 'bicarbonate_min', 
    'bicarbonate_max', 'bilirubin_min', 'bilirubin_max', 'creatinine_min', 'creatinine_max', 'chloride_min', 'chloride_max', 'glucose_min', 
    'glucose_max', 'hematocrit_min', 'hematocrit_max', 'hemoglobin_min', 'hemoglobin_max', 'lactate_min', 'lactate_max', 'platelet_min', 
    'platelet_max', 'potassium_min', 'potassium_max', 'ptt_min', 'ptt_max', 'inr_min', 'inr_max', 'pt_min', 'pt_max', 'sodium_min', 
    'sodium_max', 'bun_min', 'bun_max', 'wbc_min', 'wbc_max']

for i in range(len(IM_values)):
    final_df[IM_values[i]] = final_df[IM_values[i]].fillna(0)

#  split into features and label dataframes
x = final_df.drop(["Unnamed: 0", "SUBJECT_ID", "DIAGNOSIS", "LABEL",
    "DOB", "DOD", "DOD_HOSP", "DOD_SSN", "icustay_id",
    "ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME", "EDOUTTIME"], axis=1)
y = final_df["LABEL"]

#  scale non-binary data
to_scale = []

for i in range(len(list(x))):
    if x.dtypes[i] == float:
        to_scale.append((list(x))[i])

scaler = preprocessing.MinMaxScaler()
pd.DataFrame(scaler.fit_transform(x[to_scale]), columns=to_scale)

#  pickle format lets us preserve the data structure without any added indeces, etc.
x.to_pickle("x.pkl")
y.to_pickle("y.pkl")
