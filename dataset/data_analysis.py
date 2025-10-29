import os
import pandas as pd 
from source.data_processing.data_model import ClinicalText, Diagnosis, Patient
from source.data_processing.data_reader import clinical_texts, diagnoses, patients 
from collections import Counter

with open('source/annotation_processing/dataset/train_i.txt') as f:
    train = f.read().splitlines()

with open('source/annotation_processing/dataset/test_i.txt') as f:
    test = f.read().splitlines()

full = [float(i) for i in train+test]
print(full)

df = clinical_texts

df = df[df[ClinicalText.NOTE_ID.value].isin(full)]
df['word_count'] = df[ClinicalText.TEXT.value].apply(lambda note: len(note.strip().split()) if note.strip() else 0)

word_count = df.value_counts('word_count')
word_count = pd.DataFrame(word_count).to_csv('/output/word_count.csv')

patient_ids = list(df[ClinicalText.PATIENT_ID.value])
num_patients = print(len(set(patient_ids)))

diagnosis_df = diagnoses[diagnoses[Diagnosis.PATIENT_ID.value].isin(patient_ids)]
diagnosis_df= diagnosis_df.drop_duplicates(subset=[Diagnosis.DSM5_CODE.value, Diagnosis.TYPE.value, Diagnosis.PATIENT_ID.value])
diagnosis_count = diagnosis_df.value_counts([Diagnosis.DSM5_CODE.value, Diagnosis.TYPE.value])
diagnosis_count = pd.DataFrame(diagnosis_count).to_csv('/output/diagnosis_count.csv')

num_traj = len(set(list(df[ClinicalText.TRAJECTORY_ID.value])))
print(num_traj)

patien_df = patients[patients[Patient.ID.value].isin(patient_ids)]
patient_count = patien_df.value_counts([Patient.GENDER.value, Patient.AGE.value])
patient_count = pd.DataFrame(patient_count).to_csv('/output/patient_count.csv')
