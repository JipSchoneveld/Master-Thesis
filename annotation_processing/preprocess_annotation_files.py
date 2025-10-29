"""
Run to remove any trailing interpunction and spaces. 

Depends on:
- directory {dir} with .json files in correct annotation format 
"""

import os, json, re

## setup 
dir = 'data/annotation_files' #!local path 

## clean up annotations 
#for each annotation sample 
for sample in os.listdir(dir):
    sample_path = os.path.join(dir, sample)
    if not os.path.isdir(sample_path):
        continue
    for file in os.listdir(sample_path):
        if os.path.splitext(file)[1] != '.json':  #this is not an annotation file 
            continue
        file_path = os.path.join(sample_path, file)

        with open(file_path, 'r') as f:
            annotations = json.load(f)

        #for each annotation 
        for item in annotations:
            if item['selected_words']:
                # for each string of selected words, check for trailing interpunction and whitespace 
                for i in range(len(item['selected_words'])):
                    if re.search(r'\W$', item['selected_words'][i]): #trim end
                        if item['selected_words'][i][-1] == ')': continue #brackets can stay
                        print(item['selected_words'][i], tuple(item['selected_positions'][i].values()))
                        item['selected_positions'][i]['end'] = item['selected_positions'][i]['end'] - 1 #update index 
                        item['selected_words'][i] = item['selected_words'][i][:-1] #update string
                        print(item['selected_words'][i], tuple(item['selected_positions'][i].values()))
                    if re.search(r'^\W', item['selected_words'][i]): #trim start
                        if item['selected_words'][i][0] == '(': continue
                        print(item['selected_words'][i], tuple(item['selected_positions'][i].values()))
                        item['selected_positions'][i]['start'] = item['selected_positions'][i]['start'] + 1 #update index 
                        item['selected_words'][i] = item['selected_words'][i][1:] #update string
                        print(item['selected_words'][i], tuple(item['selected_positions'][i].values()))

        with open(file_path, 'w') as f:
            json.dump(annotations, f, indent=4)
