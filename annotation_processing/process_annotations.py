import os, json, csv, sys
from collections import defaultdict
from source.annotation_processing.annotation_tasks import Symptom_progression, Treatments
import pandas as pd 

eval_output_dir = '/output' 

DIR = 'data/annotation_files' 
SEP = '@'
DEBUG = False

def main(batches: list):
    """"
    batches: list of batch folders in DIR that will be processed 
    """
    print(f"Processing {len(batches)} batch(es): {batches}")

    batches_symptom_annotation = []; batches_treatment_annotations = []

    for batch in batches:
        print(f"{'-'*10}\nprocessing batch: {batch}")
        path = os.path.join(DIR, batch)

        # setting up en checking directories  
        check_batch_existence(path)
        output = set_output_dir(str(batch)) #each batch will have its own output folder 

        batch_csv = pd.read_csv(os.path.join(path, f"{batch}.csv")) #this file contains all notes in the batch 

        symptom_annotations, treatment_annotations = process_annotations(path, batch_csv) #process the annotations in the batch 
        batches_symptom_annotation.append(symptom_annotations); batches_treatment_annotations.append(treatment_annotations)

        for annotation in [symptom_annotations, treatment_annotations]:
            annotation.compute_IAA(output, batch)
            annotation.get_merged_labels()
            annotation.write_files(output) 
            annotation.get_conflicts(output)
            annotation.print_ratio_no_mention()

        ## collect all notes with a conflict 
        conflict_notes = {}; notes = []
        for annotation_task in ['conf_symp_annotations', 'conf_treat_annotations']:
            conflict_df = pd.read_csv(os.path.join(output,f'{annotation_task}.csv'), index_col = 0) 
            notes += [note for note in conflict_df.index]
        notes = sorted(list(set(notes)))

        conflict_notes = defaultdict(list)
        for id in notes:
            date_timeID, noteID = id.split('#')
            trajID, date  = date_timeID.split('@')
            conflict_notes['TrajectID'].append(trajID)
            conflict_notes['Creation_datetime_tekst'].append(date)
            conflict_notes['NoteID'].append(noteID)

        conflict_notes = pd.DataFrame(conflict_notes)
        conflict_notes.to_csv(os.path.join(output, f'conflicts_{batch}.csv'))

        ## process tie_breaker annotations 
        conflicts_file = f"conf_{batch}.json"
        try:
            with open(os.path.join(path, conflicts_file)) as f:
                resolvers = json.load(f)

            process_conflicts(resolvers, treatment_annotations, symptom_annotations)
        except:
            print("Get final annotations without tie-breaker annotations? (y/n)")
            answer = input()
            if answer == 'n':
                continue #process conflicts when there is a conflicts file 
            if answer == 'y':
               print("Getting final annotations based on the main annotators.") 
            else: 
                raise ValueError("Not a valid answers")
        
        treatment_annotations.get_final_labels(); symptom_annotations.get_final_labels()
        treatment_annotations.write_files(output); symptom_annotations.write_files(output)
        treatment_annotations.print_ratio_no_mention(final = True); symptom_annotations.print_ratio_no_mention(final = True)

        combined_path = os.path.join(path, f"final_annotations_{batch}.csv") #final files are written to the dir with the annotation files 
        treatment_df = treatment_annotations.get_df(); symptom_df = symptom_annotations.get_df()
        assert sorted(treatment_df.index) == sorted(symptom_df.index)
        final_treatment = treatment_df['final_spans'] 
        final_symptom = symptom_df['final_labels']
        combined_df = pd.concat([final_treatment, final_symptom], axis=1, join = 'inner')
        assert sorted(treatment_df.index) == sorted(combined_df.index)
        combined_df.to_csv(combined_path)

def check_batch_existence(path):
    if not os.path.exists(path):
            raise NameError(f"The supplied batch directory {path} does not exist.")
    if not os.path.isdir(path):
        raise TypeError(f"Found a non-directory: {path}. The {DIR} folder should only contain folders of annotation batches.")

def set_output_dir(folder):
    global eval_output_dir

    output = os.path.join(eval_output_dir, folder)
    if not os.path.exists(output):
        os.mkdir(output)
    else:
        while True: 
            print(f"This output dir already exists, are you sure you want to possibly overwrite the files in {output}? (y/n)", end = " ")
            answer = input() 
            if answer == 'n':
                print("Exiting the current process because of already existing output directory")
                quit()
            elif answer == 'y':
                break
            print("Invalid answer: write y/n")
    
    return output

def get_files(dir):
    """
    dir: a directory with annotation files with filenames: [annotator]_[...].json

    Return -> list of tuples -> (annotator name, loaded annotation json file) 
    """
    annotation_files = [] 
    for filename in os.listdir(dir):
        if os.path.splitext(filename)[1] != '.json': 
            if DEBUG: print(f"non json file found: {filename}")
            continue
        if filename.startswith("conf"):
            continue
        if DEBUG: print(f"file found: {filename}")
        annotator = filename.split('_')[0]
        with open(os.path.join(dir, filename)) as f:
            annotation_files.append((annotator, json.load(f))) 

    return annotation_files

def process_conflicts(resolvers, treatment_annotations: Treatments, symptom_annotations: Symptom_progression):
    for item in resolvers:
        date_time = item['TrajectID'] + SEP + item['Creation_datetime_tekst']
        if 'NoteID' in item:
                noteID = item['NoteID']
        else: 
            noteID = ''
        note = date_time + '#' + noteID
        if item['symptom_category'] == "PSYCHOTHERAPY":
            treatment_annotations.update(item, note, 'tie_breaker')
        else:
            symptom_annotations.update(item, note, 'tie_breaker')

def process_annotations(dir, batch_items: pd.DataFrame):
    """
    dir: path to directory with annotation json files 
    batch_items: df with columns TrajectID and Creation_datetime_tekst 

    return -> 2 pandas dataframes of all annotations for symptoms and treatments 
    """
    annotation_files = get_files(dir) #collect the annotation files
    annotators = [a for a, _ in annotation_files] 
    print(f"Annotators in this batch: {', '.join(annotators)}")

    batch_notes = [
        (   #create the unique note ID 
            row['TrajectID'] + SEP + row['Creation_datetime_tekst'].replace(' ', 'T'),
            row['NoteID'] if 'NoteID' in row else ''
        ) 
        for _, row in batch_items.iterrows()
    ]
    batch_notes = [date_time + '#' + str(noteID) for date_time, noteID in batch_notes]

    symptom_annotations = Symptom_progression(annotators, batch_notes)
    treatment_annotations = Treatments(annotators, batch_notes)

    for annotator, annotations in annotation_files:
        if DEBUG: print(f'\n...processing annotations from {annotator}...')
        for item in annotations:
            date_time = item['TrajectID'] + SEP + item['Creation_datetime_tekst']
            if 'NoteID' in item:
                noteID = item['NoteID']
            else: 
                noteID = ''
            note = date_time + '#' + noteID
            if item['symptom_category'] == "PSYCHOTHERAPY":
                treatment_annotations.update(item, note, annotator)
            else:
                symptom_annotations.update(item, note, annotator)
    
    if DEBUG:
        print(f"treatment annotations:\n{symptom_annotations.preview()}")
        print(f"symptom annotations:\n{treatment_annotations.preview()}")

    return symptom_annotations, treatment_annotations

if __name__ == "__main__":
    if len(sys.argv) > 1:
        batches = ['S'+ number for number in sys.argv[1:]]
        main(batches)
    else:
        print("\n\nWhich batch(es) should be processed? Write the batch ID(s) (0, 1, 2m3m, etc.) separated with a space.")
        batches = input().strip().split(' ')
        batches = ['S'+ number for number in batches]
        main(batches)