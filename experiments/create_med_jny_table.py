import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

import csv

import itertools
import collections
import time
import random

# Reading Data
pat_table = pd.read_csv('./saved_data/patients.csv')
adm_table = pd.read_csv('./saved_data/admissions.csv')
icu_table = pd.read_csv('./saved_data/icustays.csv')

pat_fields = ['uniquepid','patientunitstayid','patienthealthsystemstayid','hospitaladmitoffset','unitvisitnumber','unitdischargestatus']

# creating medical journey table of patients
pd_merged = pd.merge(adm_table, pat_table, on='subject_id', how='outer')
med_jny_og = pd.merge(icu_table, pd_merged, on='hadm_id', how='outer')
med_jny_og = med_jny_og.sort_values(by=['hadm_id'])

# adding visits as a column
med_jny_og['visits'] = med_jny_og.groupby("subject_id_x")["subject_id_x"].transform('count')

# converting required columns to datetime64
med_jny_og['intime'] = pd.to_datetime(med_jny_og['intime'])
med_jny_og['admittime'] = pd.to_datetime(med_jny_og['admittime'])

# Creating medical journeys final version
med_jny = pd.DataFrame(columns=pat_fields)
for i in range(len(med_jny_og)):
    curr_row = len(med_jny.index)
    med_jny.loc[curr_row, "uniquepid"] = med_jny_og.loc[i, "subject_id_x"]
    med_jny.loc[curr_row, "patientunitstayid"] = med_jny_og.loc[i, "icustay_id"]
    med_jny.loc[curr_row, "patienthealthsystemstayid"] = med_jny_og.loc[i, "hadm_id"]
    med_jny.loc[curr_row, "unitvisitnumber"] = med_jny_og.loc[i, "visits"]
med_jny['hospitaladmitoffset'] = (med_jny_og['admittime'] - med_jny_og['intime']).dt.total_seconds()/60
med_jny['unitdischargestatus'] = np.where(med_jny_og['hospital_expire_flag']==1, 'Expired', np.where(med_jny_og['hospital_expire_flag']==0, 'Alive', None))

# Tysetting to the correct datatype
med_jny['uniquepid'] = med_jny['uniquepid'].astype("string")
med_jny['patientunitstayid'] = med_jny['patientunitstayid'].astype("int")
med_jny['patienthealthsystemstayid'] = med_jny['patienthealthsystemstayid'].astype("int")
med_jny['unitvisitnumber'] = med_jny['unitvisitnumber'].astype("int")
med_jny['unitdischargestatus'] = med_jny['unitdischargestatus'].astype("string")
med_jny['hospitaladmitoffset'] = round(med_jny['hospitaladmitoffset'], 0)
med_jny['hospitaladmitoffset'] = med_jny['hospitaladmitoffset'].astype("int")

# Writing to csv file
med_jny.to_csv('./saved_data/patient.csv', index=False)

# pnt = pd.read_csv("../saved_data/patient.csv")
