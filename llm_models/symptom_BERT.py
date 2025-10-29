"""
All functions for running BERT for symptom status prediction

depends on: 
- list of all possible labels {LABELS}
- available {device}
"""

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch, math, sys, re, copy
import more_itertools
from sklearn.model_selection import StratifiedKFold
import numpy as np 
from collections import defaultdict

LABELS = [-1,0,1,100]

# torch.manual_seed(42)

## settiings 
batch_size = 16 
device = 'cuda'
epochs = 50
skf = StratifiedKFold(n_splits=3) #3-fold cross validation
input_mode = None
patience = 5

## setup tokenizer and BERT 
tokenizer = AutoTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")

## setup label and index mapping 
l2i = {label : i for i, label in enumerate(LABELS)}
i2l = {v : k for k,v in l2i.items()}

## regression classifier 
class Regression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
     def forward(self, x):
        outputs = self.linear(x)
        return outputs
          
## regression classifier with BERT model integrated           
class BERTRegression(torch.nn.Module):
    def __init__(self, token_mode, layers):
        super().__init__()

        ## BERT model
        bert_nl = AutoModel.from_pretrained("CLTL/MedRoBERTa.nl", add_pooling_layer=False)

        # freeze the first layers 
        for name, param in bert_nl.named_parameters(): 
            if name.startswith('embedding'):
                param.requires_grad = False
            elif name.startswith('encoder.layer'):
                number = int(re.findall('encoder.layer.([0-9]*).', name)[0])
                if number <= 11 - layers: #only keeps the last ... layers
                    param.requires_grad = False #freeze

        self.bert_model = bert_nl
        self.token_mode = token_mode #what kind of embedding to use as input 

        ## regression model 
        model_embdim = self.bert_model.config.hidden_size
        logreg_model = Regression(model_embdim, len(LABELS))
        self.regression_classifier = logreg_model

    def forward(self, inputs):
        if 'overflow_to_sample_mapping' in inputs: #each item represents the encoding of a chunk 
            mapping = inputs['overflow_to_sample_mapping'] #save mapping 
            del inputs['overflow_to_sample_mapping']

            chunk_bert_output = self.bert_model(**inputs, output_hidden_states = True) #get BERT embeddings for all chunks
            chunk_embeddings = self._pool_output(inputs, chunk_bert_output)

            # Save all embeddings per input text 
            embeddings_dict = defaultdict(list)
            for s, e in zip(mapping, chunk_embeddings):
                embeddings_dict[s.item()].append(e)

            #calculate mean 
            embeddings = []
            for s, l in embeddings_dict.items():
                mean = torch.mean(torch.stack(l), dim=0)
                embeddings.append(mean)

            embeddings = torch.stack(embeddings)
        else:
            bert_output = self.bert_model(**inputs, output_hidden_states = True) #get BERT embeddings 
            embeddings = self._pool_output(inputs, bert_output)

        class_output = self.regression_classifier(embeddings) 
        return class_output

    def _pool_output(self, inputs, bert_output):
        if self.token_mode == 'first':
            embedding = bert_output.last_hidden_state[:,0] #get first token embedding 
        elif self.token_mode == 'mean':
            mask = inputs['attention_mask'].unsqueeze(-1)
            masked_embeddings = bert_output.last_hidden_state * mask
            summed_embeddings = torch.sum(masked_embeddings, 1)
            n_masks = mask.sum(1)
            embedding = summed_embeddings / n_masks
        else: 
            raise ValueError(f"{self.token_mode} is not a valid sequence embedding mode.")
        
        return embedding
    
def BERT_classification(X_train, y_train, cat, text_mode, token_mode, layers, epochs, X_test = None):
    """
    Trains and runs the BERT-based classifier for all X_train using k-fold cross validations. 

    Parameters:
    X_train (pd.Series): train text from notes with NoteID as index
    y_train (pd.Series): train labels with NoteID as index
    
    Returns: 
    predicted_labels (dict[str, int]]): all predicted labels collected from the test/validation folds as a dictionary of: NoteID -> predicted label 
    """

    # tqdm.write(f"\n===== {cat} =====")

    global input_mode
    input_mode = text_mode

    ## setup dictionary 
    predicted_labels = {}

    if X_test is None: #CV mode
        ## Run training and evaluation for all train and dev splits 
        for train_index, dev_index in skf.split(X_train, y_train):

            # collect train and dev data 
            train_index = list(train_index); dev_index = list(dev_index)
            X_train_folds = X_train.iloc[train_index]
            y_train_folds = y_train.iloc[train_index]
            X_dev = X_train.iloc[dev_index]
            y_dev = y_train.iloc[dev_index]

            #from labels to indices
            y_train_folds = [torch.tensor(l2i[label]) for label in y_train_folds] 
            y_dev = [torch.tensor(l2i[label]) for label in y_dev]

            bert_logreg_model = BERTRegression(token_mode, layers).to(device) 

            _, predicted_labels = run_model(bert_logreg_model, epochs, X_train_folds, y_train_folds, predicted_labels, X_dev = X_dev, y_dev = y_dev)
    
    else: #train on full and get labels for test set 
        y_train = [torch.tensor(l2i[label]) for label in y_train] 

        bert_logreg_model = BERTRegression(token_mode, layers).to(device) 

        model, predicted_labels = run_model(bert_logreg_model, epochs, X_train, y_train, predicted_labels, X_test = X_test)
        # try: torch.save(model.state_dict(), f"/output/cat_{cat}.pt")
        # except: pass 

    return predicted_labels

def run_model(bert_logreg_model, epochs, X_train_folds, y_train_folds, predicted_labels, X_dev = None, y_dev = None, X_test = None, model_name = None):
    optimiser = torch.optim.Adam(bert_logreg_model.parameters(), lr=1e-5) #??
    loss_function = torch.nn.CrossEntropyLoss()

    # number of batches 
    batches_total = math.ceil(len(X_train_folds) / batch_size)

    epochs_no_improve = 0 
    best_loss = 100
    best_model_state = None; best_epoch = 0 

    # for each epoch 
    for epoch in tqdm(range(epochs), desc='Training Epochs', position = 1, file = sys.stdout, leave = False, dynamic_ncols=True):

        # TRAIN: for each batch, train the model 
        for X, y in tqdm(get_batches(X_train_folds, y_train_folds), total=batches_total, desc='Batches', position = 2, file = sys.stdout, leave = False, dynamic_ncols=True):
            bert_logreg_model.train()
            optimiser.zero_grad()
            outputs = bert_logreg_model(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimiser.step()

        if X_dev is not None and y_dev is not None: 
            # EVALUATE PROGRESS: after the full epoch, evaluate the progress on the dev fold data 
            with torch.no_grad():
                bert_logreg_model.eval()
                X4eval, y4eval = get_correct_format(X_dev, y_dev) #get required input format 
                pred_dev = bert_logreg_model(X4eval) #predictions 
                acc_dev = (y4eval == pred_dev.argmax(-1)).float().mean().item()
                loss_dev =  loss_function(pred_dev, y4eval).item()

                # tqdm.write(f'\n[Epoch {epoch}] Acc: train {acc_dev} / Loss: train {loss_dev}')

                if loss_dev < best_loss:
                    epochs_no_improve = 0
                    best_loss = loss_dev
                    best_model_state = copy.deepcopy(bert_logreg_model.state_dict())
                    best_epoch = epoch
                else:
                    epochs_no_improve += 1 
                    # print(epochs_no_improve)

                if epochs_no_improve >= patience or epoch == epochs - 1: #5 epochs no improvement or last epoch
                    tqdm.write(f"best model at epoch: {best_epoch}")
                    bert_logreg_model.load_state_dict(best_model_state)
                    best_pred_dev = bert_logreg_model(X4eval) #predictions 
                    for noteID, prediction in zip(X_dev.index, best_pred_dev): #same return format as the other models (llm, baseline, etc.)
                        predicted_labels[noteID] = int(i2l[prediction.argmax().item()])
                    break
        
        elif X_test is not None and epoch == epochs - 1: 
            with torch.no_grad():
                bert_logreg_model.eval()
                X4eval, _ = get_correct_format(X_test) #get required input format 
                pred_test = bert_logreg_model(X4eval) #predictions             

            for noteID, prediction in zip(X_test.index, pred_test): #same return format as the other models (llm, baseline, etc.)
                predicted_labels[noteID] = int(i2l[prediction.argmax().item()])

        if X_dev is None and X_test is NotImplementedError:
            raise ValueError("No evaluation data provided")
    
    return bert_logreg_model, predicted_labels


def get_batches(input, target, batch_size=batch_size):
    """
    Turns the data into batches.

    Parameters: 
    input: input data for the model 
    target: target output labels 
    batch_size (int): size of each batch 

    Yields:
    x: input per batch  
    y: target per batch 
    """
    for batch in more_itertools.chunked(zip(input, target), batch_size):
        x, y = zip(*batch)
        x, y = get_correct_format(x, y)
        yield x, y

def get_correct_format(x, y = None):
    """
    Gets the required format for the model and puts it on the correct device. 

    Parameters: 
    x: input data for the model 
    y: output data for the model 

    Returns: 
    x: tokenized version of x on device 
    y: torch version y on device 
    """
    if input_mode == "trunc": #truncate
        x = tokenizer(list(x), max_length= 512, padding=True, truncation=True, return_tensors='pt').to(device)
        if y: y = torch.stack(y).to(device)
        else: y = None

        return x, y
    elif input_mode == "chunk":
        x = tokenizer(
            list(x),
            max_length= 512, 
            padding=True, 
            return_tensors='pt',
            truncation=True,
            stride=256, # 256 by default will be overlapping text among chunks.
            return_overflowing_tokens=True
        ).to(device)

        if y: y = torch.stack(y).to(device)
        else: y = None

        return x, y
    else:
        raise ValueError(f"{input_mode} is not a valid sequence tokenization mode.")

