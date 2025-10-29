from source.annotation.data_reader import clinical_texts, ClinicalText
import pandas as pd
import json
from collections import defaultdict

batch_number = 2

def get_batches(batch: str):
    res = {}
    batch_ids = pd.read_csv(f'./data/annotation_batches/sample_{batch}.csv', header = 0)
    for date, traj in zip(batch_ids['Creation_datetime_tekst'], batch_ids['TrajectID']):
        corresponding_data = clinical_texts[(clinical_texts[ClinicalText.DATE.value].astype(str) == date) & (clinical_texts[ClinicalText.TRAJECTORY_ID.value].astype(str) == traj)]
        for id, text in zip(corresponding_data[ClinicalText.NOTE_ID.value], corresponding_data[ClinicalText.TEXT.value]):
            res[id] = text 

    res = pd.Series(res)

    return res 

def get_none_notes(path):
    res = defaultdict(set)
    with open(path) as f:
        predictions = json.load(f)
    for cat, prompts in predictions.items():
        for id, label in prompts['T:q|R:c|OG:d|PI:-'].items():
            res[id].add(label)
    print(res)

    mentioned_notes = []     
    for id, label_set in res.items():
        if label_set == {100}:
            corresponding_note = clinical_texts[clinical_texts[ClinicalText.NOTE_ID.value].astype(str) == id]
            date = str(corresponding_note[ClinicalText.DATE.value].item())
            traj_id = str(corresponding_note[ClinicalText.TRAJECTORY_ID.value].item())
            note_entry = {ClinicalText.DATE.value: date, ClinicalText.TRAJECTORY_ID.value: traj_id, ClinicalText.NOTE_ID.value: id}
            mentioned_notes.append(note_entry)
    df = pd.DataFrame(mentioned_notes)
    df.to_csv(f"/output/none_notes_batch{batch_number}.csv")

if __name__ == "__main__":
    get_none_notes(f'/output/batch{batch_number}_preselection/1s@qwen2.5:14b.json')