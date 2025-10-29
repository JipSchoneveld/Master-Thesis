"""
All functions for running LLMs for symptom status prediction

depends on: 
- availability ollama api at port 11434
- list of selected prompts as promptIDs (if empty, all prompts are run)
"""

import requests, sys, json, re
from source.models.llm_models.prompts import get_1step_prompts, get_2step_prompts
import pandas as pd 
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import os


URL = "http://ollama:11434/api/generate"  # LLMs are running in port 11434 in the VM.
batch_size = 50                     # notes per batch (tune for your setup)
output_file = "predictions.pkl"    # file to save intermediate results

category_prompts = {
    "gpt-oss:20b" :
        {0: ['T:q|R:c|OG:d|PI:d'], 1: ['T:i|R:c|OG:-|PI:d'], 2: ['T:i|R:-|OG:d|PI:-'], 3: ['T:i|R:-|OG:-|PI:-'], 4: ['T:q|R:-|OG:-|PI:d'], 5: ['T:q|R:-|OG:d|PI:s'], 6: ['T:i|R:c|OG:-|PI:d'], 7: ['T:q|R:-|OG:d|PI:d'], 8: ['T:i|R:-|OG:d|PI:-'], 9: ['T:i|R:-|OG:d|PI:s'], 10: ['T:i|R:c|OG:-|PI:-'], 11: ['T:i|R:-|OG:i|PI:-']},
    "qwen3:8b":
        {0: ['T:q|R:-|OG:d|PI:-'], 1: ['T:i|R:c|OG:d|PI:s'], 2: ['T:q|R:c|OG:d|PI:s'], 3: ['T:i|R:-|OG:-|PI:d'], 4: ['T:q|R:-|OG:d|PI:s'], 5: ['T:q|R:c|OG:d|PI:d'], 6: ['T:i|R:c|OG:-|PI:d'], 7: ['T:i|R:c|OG:d|PI:s'], 8: ['T:i|R:c|OG:d|PI:d'], 9: ['T:i|R:-|OG:d|PI:d'], 10: ['T:i|R:-|OG:i|PI:s'], 11: ['T:q|R:c|OG:d|PI:d']}
}

def _convert_answer(s):
    """
    Converts a raw output string to the desired label format 

    Parameters: 
    s (str): raw output string 

    Returns: 
    (int): corresponding integer 
    """
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL) #remove think block 

    # conversion from label to int 
    char2i = {
        'A': 0,
        'B': -1,
        'C': 1,
        'D': 100
    }
    s = re.findall(r'\b[ABCD]\b', s) #find all standalone A, B, C, or D 
    s = list(set(s)) # the answer can contain the output token twice 
    if len(s) == 0 or len(s)>1: #if no ABCD or coflicting, output none
        return  13 # unqiue interger for none output 
    else:
        return char2i[s[0]] # return unique integer for found letter 
    
def _convert_mention_answer(s):
    """
    Converts a raw output string for mention/no-mention to mention or no-mention letter  

    Parameters: 
    s (str): raw output string for mention/no-mention

    Returns: 
    (int): corresponding mention (Y) or no-mention (D)    
    """
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL) #remove think block 
    s = s.lower() #capitalization does not matter
    s = re.findall(r'\b(no|yes)\b', s)
    s = list(set(s)) # the answer can contain the output token twice 
    if len(s) == 0 or len(s)>1: #if no yes or no or conflicting, output none
       return '' # output not according to format so return empty string
    if 'yes' in s:
        return 'Y' #it was mentioned
    if 'no' in s:
        return 'D' #label for no mention
    else:
        return ''


def prompt_LLM(note_id, prompt, note, params, second_prompt):
    ## First prompt 
    note_id, generated_text = send_note(note_id, prompt, note, params)

    ## if present, perform second step   
    if second_prompt: 
        answer = _convert_mention_answer(generated_text) #see if answer was 'mention' or 'no-mention' 
        if answer == 'Y': #symptom was mentioned, so now we determine the progression label 
            note_id, generated_text = send_note(note_id, second_prompt, note, params)
        else: 
            generated_text = answer #symptom was not mentioned so we use that as the final label 

    return note_id, generated_text

# ---- Function to send a single note to Ollama ----
def send_note(note_id, prompt, note, params):
    prompt_text = prompt.format(note)
    data = params.copy()
    data['prompt'] = prompt_text
    response = requests.post(URL, json=data)
    response.raise_for_status()
    generated_text = response.json()['response']
    try:
        reasoning = response.json()['thinking']
        reasoning = "<think>\n" + reasoning + "\n</think>\n"
        generated_text = reasoning + generated_text
    except:
        pass 
    if not generated_text: #TODO write to DEBUG file 
        print(f"No output. Response:\n{response.json()}\n") 
    return note_id, generated_text


def make_llm_prediction(X_train: pd.Series, y: pd.Series, cat, model, output_dir, steps = 1, X_test = None):
    """
    Makes a prediction for all X_train or X_test using the specified model and steps. By default, predictions are made on X_train, but when given X_test, predictions are only made on this set. 

    Parameters:
    X_train (pd.Series): train text from notes with NoteID as index
    y_train (pd.Series): train labels with NoteID as index
    cat (int): the catgeory to make predictions for 
    model (str): the specific model to use
    output_dir (str): path to save raw predictions 
    steps (int): number of steps in which final prediction is made 
    X_test (pd.Series / None): test text from notes with NoteID as index
    
    Returns: 
    predicted_labels (dict[str, dict[str, int]]): all predicted labels as a dictionary of: prompt_id -> NoteID -> predicted label 
    """

    ## setup 
    # paramaters 
    params = {
        "model": f"{model}", 
        "prompt": "{}",
        "stream": False, 
        "options":
            {"temperature":0, 
             "seed": 42}
    }

    # predictions dictionary
    predictions = {} 

    ## settings 
    # number of steps determines the prompts 
    if steps == 1: 
        get_prompts = get_1step_prompts 
    elif steps == 2:
        get_prompts = get_2step_prompts 
    else: 
        raise ValueError("incorrect number of steps given, should be 1 or 2")

    # if test set is given, use this dataset 
    if X_test is not None:
        X = X_test
        dataset = 'test' 
    else:
        X = X_train
        dataset = 'train'

    selected_prompts = category_prompts[model][cat]

    model_name = model.replace(":", "-")
    # pickle_path = f'/output/{output_dir}predictions_{str(steps)}s@{model_name}#{dataset}_cat{cat}.pkl'

    ## run all prompts 
    for prompt, prompt_id in tqdm(get_prompts(cat), desc="prompts", position=2, file=sys.stdout, leave = False, dynamic_ncols=True): 

        # run only selected prompts, or all if [] 
        if not prompt_id in selected_prompts and selected_prompts != []: 
            continue 
        
        # for two steps, the prompt is 2 elements 
        if steps == 2:
            prompt, second_prompt = prompt 
        else: second_prompt = None
        
        predictions[prompt_id] = {}

        ## make a prediction for each note in X 
        for note_id, note in tqdm(X.items(), total= len(X), desc=f"Processing {prompt_id}", position=3, file=sys.stdout, leave = False, dynamic_ncols=True): 
            _, result = prompt_LLM(note_id, prompt, note, params, second_prompt)
            predictions[prompt_id][note_id] = result

            # intermediate save after each batch of completed notes
            # if len(predictions[prompt_id]) % batch_size == 0:
            #     with open(pickle_path, "wb") as f:
            #         pickle.dump(predictions, f)

    # ---- Final save ----
    # with open(pickle_path, "wb") as f:
    #     pickle.dump(predictions, f)
          
    predictions = pd.DataFrame(predictions)
    predictions = predictions.fillna("")
    predictions.to_csv(f'/output/{output_dir}raw_output_{str(steps)}s@{model_name}#{dataset}_cat{cat}.csv') #write raw output 

    # convert raw output to desired label and output format 
    predicted_labels = predictions.map(_convert_answer)
    predicted_labels = predicted_labels.to_dict()

    return predicted_labels