"""
Main file for training and testing for symptom status prediction

depends on: 
- dataset csv files in specified directory: /data/dataset/train_set.csv
- list of available LLMs in {llm_models} with format: {number of prompting steps (1 or 2)}@{model name}

outputs folder in specified {output_dir} with:
- raw LLM output 
- json files of all models that were run 
- json file with the gold labels 
"""

import pandas as pd 
import os, sys, json 
import numpy as np 
from tqdm import tqdm
from source.models.llm_models.llm_models import make_llm_prediction
from source.models.llm_models.symptom_baselines import majority_baseline, keyword_baseline
from source.models.llm_models.symptom_BERT import BERT_classification
from collections import defaultdict 

output_dir = 'test_set_13-10' + '/'
llm_models = ['1s@qwen3:8b', '1s@gpt-oss:20b']

def create_dataset(data, cat):
    """
    Selects the relevant labels for the category given the dataset. Splits into x and y while keeping the original indices. 

    Parameters: 
    data (pd.DataFrame): dataset with x values and y values 
    cat (int): category number 

    Returns:
    x (pd.Series): input text
    y (pd.Series): gold output labels 
    """
    filtered_data = data[~data[f'cat_{cat}'].isnull()] #remove rows with a None label 
    x = filtered_data['text']; y = filtered_data[f'cat_{cat}']

    return x, y

def run_baselines(x_train, y_train, cat, X_test = None):
    """
    Run the baseline model(s) on the dataset.

    Parameters: 
    x_train (pd.Series): input text from train set with original indices 
    y_train (pd.Series): gold output labels with original indices 
    cat (int): the category for which predictions are made 

    Returns:
    res (dict[str, dict[int, dict[int, int]: res['majority'][cat] = a dictionary from note IDs to predictions 
    """
    res = {
        'majority': {}, 
        'keyword': {}}

    mb = majority_baseline(x_train, y_train, X_test)
    res['majority'][cat] = mb

    kw = keyword_baseline(x_train, cat, X_test)
    res['keyword'][cat] = kw

    return res 

def run_BERT(x_train, y_train, cat, X_test = None):
    """
    Run the BERT-based model(s) on the dataset.

    Parameters: 
    x_train (pd.Series): input text from train set with original indices 
    y_train (pd.Series): gold output labels with original indices 
    cat (int): the category for which predictions are made 

    Returns:
    res (dict[str, dict[int, dict[int, int]: res['BERT'][cat] = a dictionary from note IDs to predictions 
    """
    best_bert = {
        0: ('trunc', 'mean', 2, 22),
        1: ('chunk', 'mean', 1, 45),
        2: ('trunc', 'mean', 2, 29),
        3: ('trunc', 'mean', 2, 28),
        4: ('trunc', 'first', 2, 15),
        5: ('trunc', 'first', 2, 21),
        6: ('trunc', 'mean', 2, 25),
        7: ('trunc', 'mean', 2, 25),
        8: ('trunc', 'first', 2, 9),
        9: ('trunc', 'mean', 2, 31),
        10: ('trunc', 'mean', 2, 26), 
        11: ('trunc', 'mean', 2, 29)
    }

    res = {
        'bestBERT': {}
    }

    text, token, layers, e = best_bert[cat]

    bert = BERT_classification(x_train, y_train, cat, text_mode = text, token_mode = token, layers = layers, epochs = e, X_test = X_test)
    res['bestBERT'][cat] = bert

    ## hyperparameter tuning
    # res = {
    #     'firstBERT1': {},
    #     'firstBERT2': {}, 
    #     'meanBERT1': {},
    #     'meanBERT2':{}
    # }

    # bert = BERT_classification(x_train, y_train, cat, text_mode = 'chunk', token_mode = 'first', layers = 1)
    # res['firstBERT1'][cat] = bert

    # bert = BERT_classification(x_train, y_train, cat, text_mode = 'chunk', token_mode = 'first', layers = 2)
    # res['firstBERT2'][cat] = bert

    # bert = BERT_classification(x_train, y_train, cat, text_mode = 'chunk', token_mode = 'mean', layers = 1)
    # res['meanBERT1'][cat] = bert

    # bert = BERT_classification(x_train, y_train, cat, text_mode = 'chunk', token_mode = 'mean', layers = 2)
    # res['meanBERT2'][cat] = bert

    return res 

def run_llms(x_train, y, cat, models: list, X_test = None):
    """
    Run the all LLMs listed on the dataset on the train data or test data if given. 

    Parameters: 
    x_train (pd.Series): input text from train set with original indices 
    y_train (pd.Series): gold output labels with original indices 
    cat (int): the category for which predictions are made 
    models (list[str]): models to run in format {number of prompting steps (1 or 2)}@{model name}
    X_test (pd.Series): input text from test set 

    Returns:
    res (dict[str, dict[int, dict[int, int]: res[{model}][cat] = a dictionary from note IDs to predictions made by {model}
    """
    res = {m: {} for m in models}
    for model in tqdm(models, desc = "model", position = 1, file = sys.stdout, leave = False, dynamic_ncols=True):
        mode, llm = model.split('@')
        if mode == '2s':
            res[model][cat] = make_llm_prediction(x_train, y, cat, llm, output_dir, steps = 2, X_test = X_test)
        else: 
            res[model][cat] = make_llm_prediction(x_train, y, cat, llm, output_dir, steps = 1, X_test = X_test)
    return res 

def main():
    """
    Given the train data at /data/dataset/train_set.csv:
    -> train and run various models and save their output at /output/{output_dir}

    """

    #setup dictionary for collecting results 
    models = defaultdict(dict)

    # setup output folder
    os.makedirs(f'/output/{output_dir}', exist_ok=True)
    
    # set train set
    train = pd.read_csv('/data/dataset/train_set.csv', index_col=0)
    #train = train.sample(frac = 1) #DEBUG 
    print(f"Train set:\n{train}")

    test = pd.read_csv('/data/dataset/test_set.csv', index_col=0)
    print(f"Test set:\n{test}")

    # run everything for each category
    for cat in tqdm(range(12), desc = "symptom categories", position = 0, file = sys.stdout, dynamic_ncols=True):

        x_train, y_train = create_dataset(data = train, cat = cat) 
        if test is not None:
            x_test, y_test = create_dataset(data = test, cat = cat) 
        else: 
            x_test, y_test = None, None

        # ## save gold labels
        models['gold'][cat] = {id:label for id, label in y_test.items()}

        ## baselines
        for model, results in run_baselines(x_train, y_train, cat, x_test).items():
            models[model].update(results)

        ## BERT 
        for model, results in run_BERT(x_train, y_train, cat, x_test).items():
            models[model].update(results)

        ## LLMs
        for model, results in run_llms(x_train, y_train, cat, llm_models, x_test).items():
            models[model].update(results)


    #save the output 
    for model, results in models.items():
        with open(f'/output/{output_dir}{model}.json', 'w') as f:
            json.dump(results, f)

if __name__ == "__main__":
    main()

