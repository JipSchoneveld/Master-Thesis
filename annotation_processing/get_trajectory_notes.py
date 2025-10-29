from source.data_processing.data_reader import clinical_texts 
from source.data_processing.data_model import ClinicalText
import pandas as pd
import json
from collections import defaultdict

trajectory = "111826687_1"

def get_notes():
    notes = []     
    trajectory_notes = clinical_texts[clinical_texts[ClinicalText.TRAJECTORY_ID.value] == trajectory]
    for _, data in trajectory_notes.iterrows():
        date = str(data[ClinicalText.DATE.value])
        traj_id = str(data[ClinicalText.TRAJECTORY_ID.value])
        id = str(data[ClinicalText.NOTE_ID.value])
        note_entry = {ClinicalText.DATE.value: date, ClinicalText.TRAJECTORY_ID.value: traj_id, ClinicalText.NOTE_ID.value: id}
        notes.append(note_entry)
    df = pd.DataFrame(notes)
    df.to_csv(f"data/annotation_files/S{trajectory}/S{trajectory}.csv")

if __name__ == "__main__":
    get_notes()