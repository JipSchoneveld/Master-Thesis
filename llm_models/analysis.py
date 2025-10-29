"""
Main file for analysing the predicted outputs for symptom status prediction

depends on: 
- prediction json files in specified directory {DIR}
- list of possibly used LLMs in {llm_models} with format: {number of prompting steps (1 or 2)}@{model name}
- list of all possible labels {LABELS}

outputs a 'results' folder in specified {DIR} with for each model:
- standard scores: recall, precision, accuracy, recall_binary, precision_binary, accuracy_binary, recall_(-1,0,1), precision_(-1,0,1)
- F-scores: f1_macro, f1_micro, f1_macro_(-1,0,1), f1_({label}), f1_binary
- ONLY FOR LLMs: none (= number of instances with a non-conforming output)
"""

import os, json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np 
import pandas as pd
from collections import defaultdict

DIR = 'output/symptom_prediction/keyword_7-10'
llm_models = [
    '1s@qwen2.5:7b', '2s@qwen2.5:7b',
    '1s@qwen2.5:14b', '2s@qwen2.5:14b', 
    '1s@qwen3:8b', '2s@qwen3:8b',
    '1s@llama3:8b', 
    '1s@llama3.1:8b',
    '1s@deepseek-r1:8b', 
    '1s@gpt-oss:20b', '2s@gpt-oss:20b'
]
LABELS = [-1,0,1,100]

def analyse_output():
    """
    Computes evaluation scores for all available predictions and writes them to csv files
    """

    #setup results folder in output dir
    os.makedirs(os.path.join(DIR, 'results'), exist_ok=True)

    ## get gold labels 
    with open(os.path.join(DIR, 'gold.json'), 'r') as f:
        gold = json.load(f)

    ## get predictions 
    for file_name in os.listdir(DIR):
        model_name, ext = os.path.splitext(file_name)
        print(file_name)
        if file_name == 'gold.json' or ext != '.json': #only use prediction json files 
            continue 
        with open(os.path.join(DIR, file_name), 'r') as f:
            predictions = json.load(f)

        model_name, _ = os.path.splitext(file_name) #llm predictions are handled different because of the various prompts 
        if model_name in llm_models:
            llm_metrics = evaluate_llm(gold, predictions)
            for metric, scores in llm_metrics.items(): #seperate file for each metric 
                scores.to_csv(os.path.join(DIR, 'results', f'{model_name}_{metric}.csv'))
        else:
            non_llm_metrics = evaluate_non_llm(gold, predictions)
            non_llm_metrics.to_csv(os.path.join(DIR, 'results', f'{model_name}.csv')) #only file for all metrics  baseline
            

def evaluate_non_llm(gold, predictions):
    """
    Collects the metrics for all non-llm (BERT, majority baseline). 

    Parameters: 
    gold (dict): the gold labels as a dictionary of: category -> noteID -> label 
    predictions (dict): predicted labels as a dictionary of: category -> noteID -> label 

    Returns: 
    non_llm_metrics (dict[int, dict[str, float]]): al metrics for all categories as a dictionary of: category -> metric -> score 
    """

    ##setup dictionary
    metrics = {}

    ## calculate metrics for all categories 
    for cat in [str(i) for i in range(12)]:
        y_true = gold[cat]; y_pred = predictions[cat]

        # make sure the predictions are in the same order as the gold labels 
        y_pred_sorted = {}
        for k in y_true.keys():
            y_pred_sorted[k] = y_pred[k]

        # check that the noteIDs are the same 
        assert list(y_true.keys()) == list(y_pred_sorted.keys()), f"y_true: {y_true.keys()}\ny_pred: {y_pred_sorted.keys()} \
            \nmissing: {set(y_true.keys()) - set(y_pred_sorted.keys())}\nextra: {set(y_pred_sorted.keys()) - set(y_true.keys())}"
        
        # get only the labels  
        y_true = list(y_true.values()); y_pred_sorted = list(y_pred_sorted.values())

        # collect all metrics 
        metrics[cat] = {}
        metrics[cat].update(compute_standard_scores(y_true, y_pred_sorted))
        metrics[cat].update(compute_f_scores(y_true, y_pred_sorted))
    
    non_llm_metrics = pd.DataFrame(metrics)
    
    return non_llm_metrics

def evaluate_llm(gold: dict[str, dict[str, int]], predictions: dict[str, dict[str, dict[str, int]]]) -> dict[str, pd.DataFrame]:
    """
    Collects the metrics for all LLMs. 

    Parameters: 
    gold (dict): the gold labels as a dictionary of: category -> noteID -> label 
    predictions (dict): predicted labels as a dictionary of: category -> promptID -> noteID -> label 

    Returns: 
    llm_metrics (dict[str, DataFrame[prompt_id, category]]): al metrics for all categories as a dictionary of: metric -> DataFrame[cat, promptID] = label  
    """

    ## setup dictionary 
    metrics = defaultdict(lambda: defaultdict(dict))

    ## compute metrics for all categories 
    for cat in [str(i) for i in range(12)]:
        all_y_pred = predictions[cat]
        for prompt_id, y_pred in all_y_pred.items():
            y_true = gold[cat]

            # make sure the predictions are in the same order as the gold labels 
            y_pred_sorted = {}
            for k in y_true.keys():
                y_pred_sorted[k] = y_pred[k]
            y_pred = y_pred_sorted
            
            # check that the noteIDs are the same
            assert list(y_true.keys()) == list(y_pred.keys())

            # get only the labels  
            y_true = list(y_true.values()); y_pred = list(y_pred.values())

            #LLM can output a None prediction -> needs to be removeds
            none_i = [i for i, elem in enumerate(y_pred) if elem == 13] #collect indices of None items
            metrics['none'][cat][prompt_id] = len(none_i) #number of none items 
            y_true = [elem for i, elem in enumerate(y_true) if not i in none_i]
            y_pred = [elem for i, elem in enumerate(y_pred) if not i in none_i]

            #collect all metrics and save per metric for this cat and prompt_id 
            for metric, score in compute_standard_scores(y_true, y_pred).items():
                metrics[metric][cat][prompt_id] = score 
            for metric, score in compute_f_scores(y_true, y_pred).items():
                metrics[metric][cat][prompt_id] = score 

    llm_metrics = {k: pd.DataFrame(v) for k, v in metrics.items()}

    return llm_metrics  

def compute_standard_scores(y_true, y_pred):
    """
    Compute all standard evluation metrics (recall, precision, accuracy) for normal, binary, and mentioned only settings. 

    Parameters: 
    y_true: list of all gold labels in corresponding order to y_pred
    y_pred: list of all gold labels in corresponding order to y_true

    Returns: 
    res (dict[str, float]): all metrics as a dictionary of: metric -> score 
    """

    ##setup results dictionary
    res = {}

    ## normal metrics 
    res['recall_macro'] = recall_score(y_true=y_true, y_pred=y_pred, average = 'macro', labels=LABELS, zero_division=0) 
    res['precision_macro'] = precision_score(y_true=y_true, y_pred=y_pred, average = 'macro', labels=LABELS, zero_division=0) 
    res['recall_micro'] = recall_score(y_true=y_true, y_pred=y_pred, average = 'micro', labels=LABELS, zero_division=0) 
    res['precision_micro'] = precision_score(y_true=y_true, y_pred=y_pred, average = 'micro', labels=LABELS, zero_division=0) 
    res['accuracy'] = accuracy_score(y_true=y_true, y_pred=y_pred) #gives mean of empty slice error 

    ## binary metrics (treat 1, 0, and -1 all as 'mention' labels)
    #convert to binary labels 
    y_true_binary = ['no mention' if a == 100 else 'mention' for a in y_true]
    y_pred_binary = ['no mention' if a == 100 else 'mention' for a in y_pred]

    # calculate binary metrics 
    res['recall_binary'] = recall_score(y_true=y_true_binary, y_pred=y_pred_binary, pos_label='mention', labels=LABELS, zero_division=0)
    res['precision_binary'] = precision_score(y_true=y_true_binary, y_pred=y_pred_binary, pos_label='mention', labels=LABELS, zero_division=0)
    res['accuracy_binary'] = accuracy_score(y_true=y_true_binary, y_pred=y_pred_binary) #gives mean of empty slice error 

    ## mention metrics (ignore the no-mention label (100))
    res['recall_macro_(-1,0,1)'] = recall_score(y_true=y_true, y_pred=y_pred, average = 'macro', labels = [-1,0,1], zero_division=0) 
    res['precision_macro_(-1,0,1)'] = precision_score(y_true=y_true, y_pred=y_pred, average = 'macro', labels = [-1,0,1], zero_division=0) 

    return res

def compute_f_scores(y_true, y_pred):
    """
    Compute all f-scores (macro, micro, macro-mention, per-class, binary).

    Parameters: 
    y_true: list of all gold labels in corresponding order to y_pred
    y_pred: list of all gold labels in corresponding order to y_true

    Returns: 
    res (dict[str, float]): all metrics as a dictionary of: metric -> score 
    """

    ##setup dictionary 
    res = {}

    ## combined macro and micro 
    res['f1_macro'] = f1_score(y_true= y_true, y_pred= y_pred, average = 'macro', labels = LABELS, zero_division=0) # TODO zero division = 0 
    res['f1_micro'] = f1_score(y_true= y_true, y_pred= y_pred, average = 'micro', labels = LABELS, zero_division=0) 

    ## mentioned macro-f1 (ignore no-mention label)
    res['f1_macro_(-1,0,1)'] = f1_score(y_true= y_true, y_pred= y_pred, average = 'macro', labels = [-1,0,1], zero_division=0) 

    ## per class f1 (treat as binary per class)
    class_f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None, labels=LABELS, zero_division=0) 
    for i, label in enumerate(LABELS):
        res[f'f1_({label})'] = class_f1[i] 

    ## binary f1 (treat 1, 0, and -1 all as 'mention' labels)
    #convert to binary labels 
    y_true_binary = ['no mention' if a == 100 else 'mention' for a in y_true]
    y_pred_binary = ['no mention' if a == 100 else 'mention' for a in y_pred]

    res['f1_binary'] = f1_score(y_true=y_true_binary, y_pred=y_pred_binary, pos_label='mention', zero_division=0) 
    
    return res

if __name__ == "__main__":
    analyse_output()



