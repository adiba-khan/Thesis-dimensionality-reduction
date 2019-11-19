import pandas as pd
import numpy as np

patients_dataframe = pd.read_csv(r"PATIENTS.csv")
admissions_dataframe = pd.read_csv(r"ADMISSIONS.csv")

for column in ["DOB", "DOD", "DOD_HOSP", "DOD_SSN"]:
    patients_dataframe[column] = pd.to_datetime(patients_dataframe[column])

for column in ["ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME", "EDOUTTIME"]:
    admissions_dataframe[column] = pd.to_datetime(admissions_dataframe[column])

patients_dataframe_subj_dob = patients_dataframe.filter(["SUBJECT_ID", "DOB"], axis=1)

life_span_column = []
gender_column = []

#  calculate lifespan (days)
for row_id, data in patients_dataframe_subj_dob.iterrows(): #for each patient
    this_subject_id = data["SUBJECT_ID"] #patient hospital id
    this_dob = data["DOB"] # (1) #patient dateofbirth

    #  take all admissions instanes of the patient
    his_rows_in_admissions = admissions_dataframe.loc[admissions_dataframe["SUBJECT_ID"] == this_subject_id]

    #  get the index in his_rows_in_admission and find earliest admission date (min)
    index_of_the_min = his_rows_in_admissions["ADMITTIME"].idxmin(axis=0)
    this_first_admittime = his_rows_in_admissions.loc[index_of_the_min]["ADMITTIME"] # (2)

    #  CALCULATE life span by taking the difference between earliest admissions and dob
    life_span = (this_first_admittime - this_dob).total_seconds()
    life_span_column.append(life_span)
    #print(this_first_admittime - this_dob)

#  add new column to patients dataframe representing life span
patients_dataframe["LIFE_SPAN"] = np.array(life_span_column)

pt_ad_df = pd.merge(patients_dataframe, admissions_dataframe, left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])

#  dataframe must be sorted by patient and then chronologically in order for our for loop to work
pt_ad_df = pt_ad_df.sort_values(['SUBJECT_ID','ADMITTIME'])
rows = pt_ad_df.shape[0]

#  binary column - are they readmitted or not, by default 0
pt_ad_df['READMIT_FLAG'] = np.zeros(rows)

#  column for time between admission(i+1)-admission(i) in days, by default 0
pt_ad_df['READMIT_DT'] = np.zeros(rows)

#  create label column, default 0; possible values 0, 1, 2
pt_ad_df['LABEL'] = np.zeros(rows, dtype=np.int)

for idx in np.arange(0,rows-1): #for each hospital admission
    if pt_ad_df.SUBJECT_ID[idx] == pt_ad_df.SUBJECT_ID[idx + 1]: #check if the patient appears in the next row
        pt_ad_df.at[idx, {"DOD", "DOD_HOSP", "DOD_SSN", "DEATHTIME"}] = None #remove death-events for all previous admissions so death info is only associated with last visit
        pt_ad_df.at[idx, "EXPIRE_FLAG"] = 0 #set expire flag to 1
        discharge = pt_ad_df.DISCHTIME[idx] #create initial time point with discharge time of the current hospital visit
        next_adm = pt_ad_df.ADMITTIME[idx+1] #create next time point with admission time of the next hospital visit
        time = (next_adm - discharge).days #calculate time between admission(i+1)-admission(i) in days
    
        pt_ad_df.set_value(idx,'READMIT_FLAG', 1) #will column with 1 if we have entered the for loop since the patient revisits hospital
        pt_ad_df.set_value(idx,'READMIT_DT',time) #fill column with time between visits in days    
    
        if (pt_ad_df.READMIT_FLAG[idx] and pt_ad_df.READMIT_DT[idx] <= 30): #check for patients readmitted within 30 days of discharge
            pt_ad_df.set_value(idx, 'LABEL', 1) #assign label 1 for 30-day readmissions

    elif (pt_ad_df.EXPIRE_FLAG[idx] == 1 and 
    	(pt_ad_df.DOD[idx]-pt_ad_df.DISCHTIME[idx]).days <= 30): #check for patients who die witin 30 days of discharge
            pt_ad_df.set_value(idx, 'LABEL', 2) #assign label 2 for 30-day deaths

patient_admissions_dataframe = pt_ad_df.drop(['ROW_ID_x', 'ROW_ID_y', 'READMIT_FLAG', 'READMIT_DT'], axis=1)
print(patient_admissions_dataframe)
patient_admissions_dataframe.to_csv(r'check.csv')

