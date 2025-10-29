"""
Run to check if any annotation has a mismatched "selected words" and "selected spans". 

Depends on:
- directory {dir} with .json files in correct annotation format 
"""

import json, os
from source.data_processing.data_model import ClinicalText
from source.data_processing.data_reader import clinical_texts

## setup 
dir = 'data/annotation_files'
annotation_files = [] 

# get dataframe of all clinical notes 
all_df = clinical_texts
all_df['date_time'] = clinical_texts.apply(lambda row: str(row[ClinicalText.TRAJECTORY_ID.value]) + '@' + str(row[ClinicalText.DATE.value]).replace(' ', 'T'), axis = 1)

## check words and indices
#for each annotation sample 
for sample in os.listdir(dir):
    sample_path = os.path.join(dir, sample)
    if not os.path.isdir(sample_path):
        continue
    for filename in os.listdir(sample_path):
        print(filename)
        if os.path.splitext(filename)[1] != '.json': #this is not an annotation file 
            continue

        file_path = os.path.join(sample_path, filename)
        with open(file_path) as f:
            file = json.load(f)

        for annotation in file:
            if 'NoteID' in annotation: #some annotation files contain the NoteID
                note_id = annotation['NoteID']
                note = all_df[all_df[ClinicalText.NOTE_ID.value].astype(str) == note_id]['text'].values
            else: #when it does not contain NoteID, use the date_time as id 
                note_id = annotation['TrajectID'] + '@' + annotation['Creation_datetime_tekst']
                note = all_df[all_df['date_time'] == note_id]['text'].values
            
            if len(note) > 1: #when using data_time, more than 1 note can match the ID 
                continue

            note = note[0]

            indices = [tuple(span.values()) for span in annotation['selected_positions']]

            correct_indices = []; correct_words = [] 
            for i, w in zip(indices, annotation['selected_words']): # check corresponding indices and words 
                if note[i[0]:i[1]] == w:
                    correct_indices.append({"start": i[0], "end": i[1]})
                    correct_words.append(w)
                if note[i[0]:i[1]] != w: #print which instances are a mismatch 
                    print(note_id)
                    print(f'not equal: {note[i[0]:i[1]]} | {w}' )
                    print(f'from file {filename} remove {i} - {w}\n')
            annotation['selected_positions'] = correct_indices
            annotation['selected_words'] = correct_words

        with open(file_path, 'w') as f:
            json.dump(file, f, indent=4)
        