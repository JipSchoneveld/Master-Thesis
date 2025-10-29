Orgnize the annotation files as follows (repeat the pattern for more batches): 
.
├── annotation_files
│   ├── S0
│   │   ├── <a1>_annotations_S0.json
│   │   ├── conf_S0.json.         #annotations of the tie breaker
│   │   ├── <a2>_annotations_S0.json
│   │   └── S0.csv                  # csv from data/annotation_batches that indicates all notes in the batch 
│   └── S1
│       ├── <a1>_annotations_S1.json
│       ├── conf_S1.json
│       ├── <a2>_annotations_S1.json
│       └── S1.csv
├── output

To fully process the annotation, do the following: (note: the paths from the code need to be changed to reflect the file structure)
1. Manually remove any mistakes 
2. Run check_span_annotations.py (has to be run somehwere with an access to the data) and remove the mistakes highlighted by this 
3. Run preprocess_annotation_files.py
4. Run process_annotations.py

Aftwerwards manually copy the final_annotations_S*.csv files from the output/S* folder to the annotation_files/S* folder. 
If, afterwards, you want to make any chaged to the annotations, do this in the final_annotations files in the annotation_files folder. 