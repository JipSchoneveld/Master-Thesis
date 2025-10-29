## Creating a dataset from the annotations
Docker dataset_service runs: 
1. create_data.set.py -> writes 2 csv files (train and test) to the specified data location ('srv/data/dataset) + writes 2 txt files to source/annotation listing the note ids in train and test (for version control purposes)
2. data_analysis.py ->  writes a txt files to source/annotation listing data distributions of the train set
