"""
Dataset format: NoteID,final_spans,final_labels,PseudoID,Creation_datetime_tekst,Mutation_datetime_tekst,text,Bron_tekst,BehandelaarID,TrajectID,word_count,Geslacht,Geboortejaar,Postcode,age,admission_status,word_count_bin,strata,date_time
"""

import os,ast
import pandas as pd 
from source.data_processing.data_model import Trajectory, Patient, Admission, ClinicalText
from source.data_processing.data_reader import admissions, patients, clinical_texts
from sklearn.model_selection import train_test_split
from collections import Counter

def create_dataset(sample_list: list[str]):
    print("Creating train-test split ...\n")
    
    dir = 'data/annotation_files'

    ## combine all annotations into 1 dataset 
    annotation_dfs = [] 

    for sample in os.listdir(dir):
        if not sample in sample_list: continue
        print(f"processing {sample}")
        sample_path = os.path.join(dir, sample)
        if not os.path.isdir(sample_path): 
            print("\tnot a directory, continuing ...")
            continue
        file_path = os.path.join(sample_path, f'final_annotations_{sample}.csv')
        try:
            df = pd.read_csv(file_path, index_col=0)
        except:
            print("\tno annotation file found, continuing ...")
            continue

        df[ClinicalText.NOTE_ID.value] = df.apply(get_noteID, axis = 1) 
        print(f"\tBefore filtering out notes without 1-1 mapping to unique ID: {len(df)}") 
        df = df[df[ClinicalText.NOTE_ID.value].notnull()] # remove annotation for notes that do not have a unique ID
        print(f"\tAfter filtering: {len(df)}\n")

        annotation_dfs.append(df)

    annotated_dataset = pd.concat(annotation_dfs)
    annotated_dataset = annotated_dataset.set_index(ClinicalText.NOTE_ID.value)

    #merge the extened feature df into the annotated note df 
    print(f"len all: {len(all_df)}, len annotated: {len(annotated_dataset)}", end=", ")
    annotated_df = annotated_dataset.merge(all_df, left_index= True, right_index= True, how = 'left')
    print(f"len combined: {len(annotated_df)}")

    print(annotated_df)

    for c in range(12):
        annotated_df[f'cat_{c}'] = annotated_df.apply(lambda row: split_labels(row, c), axis = 1)

    return annotated_df

def get_noteID(row):
    try: 
        traj_date, noteID =row.name.split('#')
    except:
        traj_date = row.name; noteID = ""

    if noteID: #convert date-time note_id to the unique NoteID
        return float(noteID)
    else: 
        traj, date = traj_date.split('@'); date = date.replace('T', ' ')
        corresponding_data = clinical_texts[(clinical_texts[ClinicalText.DATE.value].astype(str) == date) & (clinical_texts[ClinicalText.TRAJECTORY_ID.value].astype(str) == traj)]
        if len(corresponding_data)>1:
            print('\tnon unique date-time found:', row.name)
            return None 

        return corresponding_data[ClinicalText.NOTE_ID.value].item() 

## stratified sampling for train, test 
def check_if_during_admission(note_row, admissions_df):
    patient_admissions = admissions_df[admissions_df[Patient.ID.value] == note_row[Patient.ID.value]]
    for _, adm in patient_admissions.iterrows():
        if adm[Admission.START.value] <= note_row[ClinicalText.DATE.value] <= adm[Admission.STOP.value]:
            return "during admission"
    return "outside admission"

def bin_column(series, bins=3, labels=['low', 'medium', 'high']):
    return pd.qcut(series, q=bins, labels=labels, duplicates='drop')

def extend_features(df: pd.DataFrame):
    """"
    extends features, filters on word count and decursus, changes index to NoteID
    """
    df['word_count'] = df[ClinicalText.TEXT.value].apply(lambda note: len(note.strip().split()) if note.strip() else 0)

    df = pd.merge(df, patients, on=Patient.ID.value, how='left')
    # Apply the function to each row
    df['admission_status'] = df.apply(
        lambda row: check_if_during_admission(row, admissions), axis=1
    )
    df = df[(df['word_count'] > 5) & (df[ClinicalText.SOURCE.value] == 'decursus')]
    df['word_count_bin'] = bin_column(df['word_count'])
    #df['medication_count_bin'] = bin_column(df['medication_count'])
    # Create a stratification label by combining bins
    #df[Patient.GENDER.value].astype(str) + "_" +
    df['strata'] = (
                    df['admission_status'].values.astype(str) + "_" +
                df['word_count_bin'].astype(str))
    
    #make the unique NoteID the index
    df = df.set_index(ClinicalText.NOTE_ID.value) #? redundant?

    return df

def stratified_train_test_split(df: pd.DataFrame, train_size=0.8):
    # removes items that are the only when a specific value combination used for stratified sampling
    strata_counts = df['strata'].value_counts()
    print(f"{len(df)} before unique stratified removal.. ")
    df = df[df['strata'].isin(strata_counts[strata_counts > 1].index)].copy()
    print(f"{len(df)} after unique stratified removal.. ")

    train, test = train_test_split(
        df,
        train_size=train_size,
        stratify=df['strata'],
        random_state=35 + 22
    )

    for split in [train, test]:
        split = split.reset_index(drop=True)

    return train, test

def split_labels(row, c):
    label = ast.literal_eval(row['final_labels'])[str(c)]
    if len(label) > 1:
        raise ValueError("there are more than 1 labels present")
    if len(label) == 0:
        return None
    else:
        return int(label[0])
    
def data_analysis(dataset, name):

    counts = {}
    
    for c in range(12):
        counts[c] = dataset.value_counts(f'cat_{c}', dropna=False)
    
    counts = pd.DataFrame(counts)
    counts.to_csv(f'source/annotation_processing/dataset/{name}_distribution.csv')


if __name__ == "__main__": 
    #extend the features of the full (unannotated) dataset for stratified sampling and select only word count>5 and decursus 
    print('before extend and select', len(clinical_texts))
    all_df = extend_features(clinical_texts)
    all_df['date_time'] = all_df.apply(lambda row: str(row[ClinicalText.TRAJECTORY_ID.value]) + '@' + str(row[ClinicalText.DATE.value]).replace(' ', 'T'), axis = 1)
    print('after extend and select', len(all_df), '\n')

    ## trajectory dataset
    traj_dataset = create_dataset(['S111826687_1']) 
    traj_dataset.to_csv('/data/dataset/traj_dataset.csv')

    ## original dataset -> S0 & S1
    original_dataset = create_dataset(['S0', 'S1']) 
    original_train, original_test = stratified_train_test_split(original_dataset)

    data_analysis(original_train, 'llm_validation')

    with open('source/annotation_processing/dataset/train_i.txt', 'w') as f:
        f.writelines([str(i)+'\n' for i in original_train.index])
    with open('source/annotation_processing/dataset/llm_val_i.txt', 'w') as f:
        f.writelines([str(i)+'\n' for i in original_train.index])
    with open('source/annotation_processing/dataset/test_i.txt', 'w') as f:
        f.writelines([str(i)+'\n' for i in original_test.index])

    ## new sample -> S2m3m 
    new_dataset = create_dataset(['S2m3m']) 
    new_train, new_test = stratified_train_test_split(new_dataset)

    ## Final datasets 
    final_dataset = pd.concat([original_dataset, new_dataset])
    final_train = pd.concat([original_train, new_train])
    final_test = pd.concat([original_test, new_test])
    assert set(list(final_dataset.index)) == set(list(final_train.index) + list(final_test.index))

    data_analysis(final_dataset, 'full')
    data_analysis(final_train, 'train')
    data_analysis(final_test, 'test')

    with open('source/annotation_processing/dataset/train_i.txt', 'w') as f:
        f.writelines([str(i)+'\n' for i in final_train.index])
    with open('source/annotation_processing/dataset/test_i.txt', 'w') as f:
        f.writelines([str(i)+'\n' for i in final_test.index])

    final_train.to_csv('/data/dataset/train_set.csv')
    final_test.to_csv('/data/dataset/test_set.csv')


