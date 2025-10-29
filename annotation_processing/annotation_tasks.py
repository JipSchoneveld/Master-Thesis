import pandas as pd 
import csv, os, copy, math
from nltk.metrics.agreement import AnnotationTask, binary_distance
from nltk.metrics.confusionmatrix import ConfusionMatrix
from nltk import edit_distance
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np 
from abc import ABC, abstractmethod
from fuzzywuzzy import fuzz
import re
from sklearn.metrics import f1_score

DEBUG = False

class MyAnnotationTask(ABC):
    annotation_dict: dict

    def get_df(self):
        annotation_dict = copy.deepcopy(self.annotation_dict)
        return pd.DataFrame(annotation_dict).T

    def preview(self):
        print(self.get_df().head())

    def write_files(self, dir):  
        (self.get_df()).to_csv(os.path.join(dir, f"{self.to_string()}.csv"))

    @staticmethod
    @abstractmethod
    def to_string() -> str:
        pass
    
class Symptom_progression(MyAnnotationTask):
    annotation_dict: dict[str, dict]

    with open('data/symptoms.csv') as f: #!local path
        reader = csv.DictReader(f)
        SYMP2I = {r['symptom']: r['score'] for r in reader}
    I2SYMP = {value: key for key, value in SYMP2I.items()}
        
    def __init__(self, annotators: list[str], note_ids: list[str]):
        """
        sets up the default annotation value for all <notes> and all <annotators>
        """

        self.symptom_cats = [s for s in Symptom_progression.SYMP2I.values()]
        self.annotation_dict = {
            note: {s: 
                {str(a):100 for a in annotators + ['tie_breaker']} for s in self.symptom_cats} #default score is 100
            for note in note_ids
        }
        self.annotators = annotators

    def update(self, item, note, annotator): 
        """
        add annotation <item> for <note> by <annotator> to the annotations task 
        """
        symptom_cat = Symptom_progression.SYMP2I[item['symptom_category']]
        symptom_score = item['annotation_score']
        if DEBUG: print(f"annotation by {annotator} found for {note}: {symptom_cat} = {symptom_score}")

        self.annotation_dict[note][symptom_cat][annotator] = symptom_score
    
    @staticmethod
    def to_string() -> str:
        return "symp_annotations"

    def compute_IAA(self, dir, batch):
        """
        dir: output location for the IAA-score files 
        computes some IAA scores for the annotations in the annotation_dict 
        """
        annotation_dict = copy.deepcopy(self.annotation_dict)
        data_arrays = Symptom_progression._get_data_arrays(annotation_dict)

        metrics = ['avg_Ao', 'Ae_kappa', 'kappa', 'weighted_kappa', 'alpha']
        all_IAA = {m: {} for m in metrics}
        try:
            open(os.path.join(dir, 'confusion_matrix.txt'), 'w').close() #empty file if already exists 
        except:
            pass 

        f_scores = {}; labels = [0,1,-1,100]

        for cat, data_array in data_arrays.items():
            nltk_annotation = AnnotationTask(data=data_array)
            for metric in metrics:
                if metric == 'weighted_kappa': nltk_annotation.distance = Symptom_progression._label_distance #only use custom distance function for weighted kappa 
                else: nltk_annotation.distance = binary_distance
                try:
                    score = getattr(nltk_annotation, metric)()
                except ZeroDivisionError: 
                        score = 1.0 #? is this the correct way to handle this -> only 0 when expeted agreement is 1, so all the same label 
                except TypeError as or_E:
                    try:
                        score = getattr(nltk_annotation, metric)(*self.annotators)
                    except:
                        raise or_E 
                all_IAA[metric][cat] = score 

            # confusion matrix per category 
            a1, a2 = self.annotators
            ref = [note_annotations[cat][a1] for note_annotations in annotation_dict.values()] 
            print(f'{cat}, ref\n{Counter(ref)}')
            pred = [note_annotations[cat][a2] for note_annotations in annotation_dict.values()] 
            print(f'{cat}, pred\n{Counter(pred)}')

            ref_mention = ['no mention' if a == 100 else 'mention' for a in ref]
            pred_mention = ['no mention' if a == 100 else 'mention' for a in pred]


            f_scores[cat] = {
                'binary': f1_score(ref_mention, pred_mention, pos_label='mention', zero_division=0),
                'micro': f1_score(ref, pred, labels=labels, average='micro', zero_division=0),
                'macro': f1_score(ref, pred, labels = labels, average='macro', zero_division=0),
                'macro_mention': f1_score(ref, pred, labels = [-1,0,1], average='macro', zero_division=0),
                'micro_mention': f1_score(ref, pred, labels = [-1,0,1], average='micro', zero_division=0)
            }
            f_scores[cat].update({
                l: f1 for l, f1 in zip(labels, f1_score(ref, pred, labels = labels, average= None, zero_division=0))}) # type: ignore

            with open(os.path.join(dir, 'confusion_matrix.txt'), 'a') as f:
                symp = Symptom_progression.I2SYMP[cat]
                f.write(f"====={cat}.{symp}=====\n")
                f.write('\n')
                f.write(str(ConfusionMatrix(ref, pred)) + "\n")

        #get overall confusion matrix 
        # print([(k, v) for note_annotation in annotation_dict.values() for k, v in note_annotation.items()])
        a1, a2 = self.annotators
        ref = [annotations[a1] for note_annotations in annotation_dict.values() for cat, annotations in note_annotations.items() if int(cat) in range(12)] 
        pred = [annotations[a2] for note_annotations in annotation_dict.values() for cat, annotations in note_annotations.items() if int(cat) in range(12)] 

        ref_mention = ['no mention' if a == 100 else 'mention' for a in ref]
        pred_mention = ['no mention' if a == 100 else 'mention' for a in pred]

        f_scores['combined'] = {
                'binary': f1_score(ref_mention, pred_mention, pos_label='mention', zero_division=0),
                'micro': f1_score(ref, pred, labels=labels, average='micro', zero_division=0),
                'macro': f1_score(ref, pred, labels = labels, average='macro', zero_division=0),
                'macro_mention': f1_score(ref, pred, labels = [-1,0,1], average='macro', zero_division=0),
                'micro_mention': f1_score(ref, pred, labels = [-1,0,1], average='micro', zero_division=0)
            }
        f_scores['combined'].update({l: f1 for l, f1 in zip(labels, f1_score(ref, pred, labels = labels, average= None, zero_division=0))}) # type: ignore

        with open(os.path.join(dir, 'confusion_matrix.txt'), 'a') as f:
            f.write(f"=====COMBINED=====\n")
            f.write('\n')
            f.write(str(ConfusionMatrix(ref, pred)) + "\n")

        f_scores = pd.DataFrame(f_scores)
        f_scores.to_markdown(os.path.join(dir, f'{batch}_symp_f_scores.md'))
        f_scores.to_csv(os.path.join(dir, f'{batch}_symp_f_scores.csv'))
        all_IAA = pd.DataFrame(all_IAA)
        all_IAA.to_markdown(os.path.join(dir, f'{batch}_symp_IAA.md'))
        all_IAA.to_csv(os.path.join(dir, f'{batch}_symp_IAA.csv'))
        if DEBUG: print(all_IAA)

    @staticmethod
    def _label_distance(a, b):
        """
        returns a value between 0.0 and 1.0 to represent the distance between label a and b
        To be used as a distance funtion for the NLTK AnnotationTask object 
        """
        if a == b: return 0.0

        combination = set([a, b])

        if combination in [set([-1, 0]), set([1, 0]), set([1,2])]:
            return 1/5
        if combination in [set([0, 2]), set([0,100])]:
            return 2/5
        if combination in [set([-1, 100]), set([1, 100]), set([2, 100])]:
            return 3/5
        if combination in [set([-1, 1])]:
            return 4/5
        if combination in [set([-1, 2])]:
            return 5/5
        
        raise NameError(f"undefined combination:, {combination}")
    


    @staticmethod
    def _get_data_arrays(annotation_dict: dict[str, dict]) -> dict[str, list[tuple[str, str, str]]]:
        """
        annotation_dict: annotation dictionary {note: {symptom: {annotator: label}}} for all notes, symptoms, and annotators 
        reformats all annotations in the annotation_dict in nltk tuple format 

        return -> dict with keys= symptom cat, values= list with tuples (coder, item, label) formatted for nltk
        """
        data_arrays = defaultdict(list)
        for note, annotations in annotation_dict.items():
            for cat, cat_annotations in annotations.items():
                array = [] 
                for annotator, score in cat_annotations.items():
                    if annotator == 'tie_breaker':
                        continue
                    array.append((annotator, note, score))
                data_arrays[cat] += array

        return data_arrays
    
    def get_conflicts(self, dir):
        """
        dir: output directory
        collects all annotations for notes that have any conflicting annotations 
        """
        annotation_df = self.get_df()
        conflicts = annotation_df[annotation_df['conflict'] == True]
        conflicts.to_csv(os.path.join(dir, f"conf_{self.to_string()}.csv"))

    @staticmethod
    def _get_majority(labels: list[str]):
        """"
        labels: iterable containing labels 
        determines the majority labels, defined as the label with at least 2 votes 
        
        returns -> list of labels with at least 2 votes, or [] if none
        """
        return [label for label, count in Counter(labels).most_common() if count >= 2] #a label needs at least 2 votes 

    def get_merged_labels(self):
        """
        collects the merged labels as merged_labels and adds this to the annotation_dict. Also records if the labels could not be merged due to conflict 
        """
        for note in self.annotation_dict.keys():
            conflict = False #conflict is recorded per note  
            merged_labels = {}
            for symptom_cat in self.symptom_cats:
                a1, a2 = self.annotators
                labels = [self.annotation_dict[note][symptom_cat][a1], self.annotation_dict[note][symptom_cat][a2]]
                majority = Symptom_progression._get_majority(labels)
                if not majority:
                    conflict = True
                merged_labels[symptom_cat] = majority
            
            self.annotation_dict[note]['merged_labels'] = merged_labels
            self.annotation_dict[note]['conflict'] = conflict

    def get_final_labels(self):
        """
        collects the final labels as final_labels and adds this to the annotation_dict. This takes into account the tie_breaker annotations and changes the values in the conflict column. 
        """
        for note in self.annotation_dict.keys():
            if self.annotation_dict[note]['conflict']: #there was a conlfict before which will be resolved 
                conflict = False #conflict is recorded per note  
                final_labels = {}
                for symptom_cat in self.symptom_cats:
                    labels = self.annotation_dict[note][symptom_cat].values()
                    # print(note, symptom_cat, labels)
                    majority = Symptom_progression._get_majority(labels)
                    if not majority:
                        conflict = True
                    final_labels[symptom_cat] = majority
                
                self.annotation_dict[note]['final_labels'] = final_labels
                self.annotation_dict[note]['conflict'] = conflict
            else:
                self.annotation_dict[note]['final_labels'] = copy.copy(self.annotation_dict[note]['merged_labels'])

    def print_ratio_no_mention(self, final = False):
        if final: 
            print('final')
            label_type = 'final_labels'
        else:
            print('merged')
            label_type = 'merged_labels'
        no_mention = defaultdict(int)
        annotation_dict = copy.deepcopy(self.annotation_dict)
        total = len(annotation_dict)
        for note, annotations in annotation_dict.items():
            for cat, label in annotations[label_type].items():
                if label == [100]:
                    no_mention[cat] += 1
        no_mention = {k: v/total for k,v in no_mention.items()}
        print('no mention:\n',  no_mention)

            
    
class Treatments(MyAnnotationTask):
    annotation_dict: dict[str, dict]

    def __init__(self, annotators: list[str], note_ids: list[str]):
        """
        sets up the default annotation value for all <notes> and all <annotators>
        """  

        self.annotation_dict = {
            note: 
                {a:[] for a in annotators + ['tie_breaker']} #default is empty list 
            for note in note_ids
        }
        self.annotators = annotators
        self.word_spans = {a:set() for a in annotators + ['tie_breaker']}

    def update(self, item, note, annotator):
        """
        add annotation <item> for <note> by <annotator> to the annotations task 
        """
        spans = item['selected_positions']
        if DEBUG: print(f"Note: {note}")
        if DEBUG: print(f"Spans: {spans}")
        if not spans: return #if there is not actually a span selected

        selected_words = [word_span.lower() for word_span in item['selected_words']]

        self.word_spans[annotator].update(selected_words)

        spans = [tuple(span.values()) for span in spans] # from 'start' and 'stop' dict to span tuples 

        current_spans = copy.copy(self.annotation_dict[note][annotator])
        updated_spans = Treatments._get_merged_spans([current_spans + spans]) #merge overlap 

        if DEBUG: print(f"Merged spans: {updated_spans}")
        
        self.annotation_dict[note][annotator] = updated_spans

    @staticmethod
    def get_all_spans(annotations):
        """
        annnotations: dict with key= annotator, value=list of annotated spans
        """
        all_spans = [span for span in annotations.values()]
        all_spans = list(set(all_spans)) #remove duplicates 

        return all_spans
    
    def compute_IAA(self, dir, batch):
        """
        dir: output location for the IAA-score files 
        computes some IAA scores for the annotations in the annotation_dict 
        """
        IAA = {}; differences={}
        annotation_dict = copy.deepcopy(self.annotation_dict)
        word_spans = copy.deepcopy(self.word_spans)

        for mode in ['overlapping', 'exact']:
            IAA_metrics, diff = Treatments._get_IAA_metrics(annotation_dict, self.annotators, word_spans, mode)
            IAA[mode] = IAA_metrics
            differences[mode] = diff
        
        IAA = pd.DataFrame(IAA)
        IAA.to_markdown(os.path.join(dir, f'{batch}_treat_IAA.md'))
        IAA.to_csv(os.path.join(dir, f'{batch}_treat_IAA.csv'))
        differences = pd.DataFrame(differences['exact']) #only use exact for this 
        differences.to_markdown(os.path.join(dir, f'{batch}_differences.md'))
        if DEBUG: print(IAA)

    @staticmethod    
    def _get_IAA_metrics(annotations, annotators, word_spans, mode):
        # print('-----', mode, '-----')
        f1 = Treatments.spans_f1(annotations, annotators, mode)
        overlap = Treatments.spans_overlap(annotations, annotators, mode)
        similarity, differences = Treatments.jaccard_similarity(word_spans, annotators, mode)
        fuzzy_similarity, _ = Treatments.jaccard_similarity(word_spans, annotators, mode, fuzzy = True)
        overlap_coefficient, _ = Treatments.jaccard_similarity(word_spans, annotators, mode, overlap_coefficient = True)
        fuzzy_overlap_coefficient, _ = Treatments.jaccard_similarity(word_spans, annotators, mode, fuzzy = True, overlap_coefficient=True)
        return {
            'f1' : f1, 
            'overlap': overlap, 
            'jaccard': similarity, 
            'fuzzy_jaccard': fuzzy_similarity, 
            'overlap_coefficient': overlap_coefficient,
            'fuzzy_overlap_coefficient': fuzzy_overlap_coefficient
            }, differences

    @staticmethod
    def spans_overlap(annotations, annotators, mode):
        """
        annotations: dict[annotator(str)] = spans (list((start(int), end(int)))
        mode: determines if annotaions are first converted to their overlapping counterpart if this exists  
        return: overlap in selected indices 
        """
        overlap = []
        for note, annotations in annotations.items():
            a1, a2 = annotators
            a1_spans = annotations[a1]; a2_spans = annotations[a2]
            if mode == "overlapping": 
                merged_spans = Treatments._get_merged_spans([a1_spans, a2_spans])  
                a1_spans = Treatments._exact2overlap(a1_spans, merged_spans) #covert exact individual annotions to their matching overlapping span 
                a2_spans = Treatments._exact2overlap(a2_spans, merged_spans)

            
            a1_indices = Treatments._spans2i(a1_spans) 
            a2_indices = Treatments._spans2i(a2_spans)   
            intersection = a1_indices.intersection(a2_indices)
            union = a1_indices.union(a2_indices)
            if union: 
                overlap.append(len(intersection)/len(union))
            else: 
                overlap.append(1.0) #2 empty spans are 100% similar 
        
        return np.mean(overlap)

    @staticmethod
    def jaccard_similarity(words_dict: dict[str, set[str]], annotators, mode, fuzzy = False, overlap_coefficient = False):
        words_dict = copy.deepcopy(words_dict)

        for annotator, words in words_dict.items():
            words_dict[annotator] = set([w.replace(' & ', ' en ').replace('&', ' en ').replace('+', ' en ').replace(' + ', ' en ') 
                     for w in words])

        if mode == "overlapping":
            for a, word_spans in words_dict.items():
                words = set()
                for word_span in word_spans:
                    word_list = [re.sub(r'^\W(?=\w)|(?<=\w)\W$', '', w) for w in word_span.split(' ')]
                    words.update(word_list)
                words_dict[a] = words

        a1, a2 = annotators
        a1_words = words_dict[a1]; a2_words = words_dict[a2]
        intersection = a1_words.intersection(sorted(list(a2_words)))
        union = a1_words.union(a2_words)

        diffa1 = '\n'.join(sorted(list(a1_words - a2_words)))
        diffa2 = '\n'.join(sorted(list(a2_words - a1_words)))

        if overlap_coefficient: 
            union = min(a1_words, a2_words)

        if fuzzy:
            intersection_c = Treatments._fuzzy_count(intersection)
            union_c = Treatments._fuzzy_count(union)
        else:
            intersection_c = len(intersection)
            union_c = len(union)

        try:
            res = intersection_c/union_c
        except ZeroDivisionError:
            res = None
        
        return res, {a1: [diffa1], a2: [diffa2]}

    @staticmethod
    def _fuzzy_count(words):
        count = 0
        words = list(words)
        for i in range(len(words)):
            if not Treatments._has_fuzzy_equal(words[i], words[:i]): #if there is a fuzzy equal in the previous words
                count += 1
        return count 

    @staticmethod
    def _has_fuzzy_equal(word, list):
        for w in list:
            if fuzz.ratio(word, w) > 87:
                # print(word, '|', w)
                return True
        return False

    @staticmethod 
    def spans_f1(annotations, annotators, mode):
        """
        annotations: dict[annotator(str)] = spans (list((start(int), end(int)))
        mode: determines if overlap is treated as matching annotation 
        return: F1-score 
        """
        TP, FP, FN = [], [], []
        for _, annotations in annotations.items():
            a1, a2 = annotators
            true = annotations[a1]; pred = annotations[a2]
            if mode == "overlapping": 
                merged_spans = Treatments._get_merged_spans([true, pred])  
                true = Treatments._exact2overlap(true, merged_spans) #covert exact individual annotions to their matching overlapping span 
                pred = Treatments._exact2overlap(pred, merged_spans)
                
            true = set(true); pred = set(pred)   
            TP += list(true.intersection(pred))
            FP += list(pred - true)
            FN += list(true - pred)
        
        TP = len(TP); FP = len(FP); FN = len(FN)
        if TP>0:
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            f1 = (2*precision*recall)/(precision+recall)
        else:
            f1= 0

        return f1
    
    @staticmethod
    def _exact2overlap(annotations: list[tuple[int, int]], merged_spans: list[tuple[int, int]]) -> list:
        """
        Converts the exact spans of 2 annotators, to the spans of the overlap between the annotators (when there is overlap)
        """ 
        res = copy.copy(annotations)
        for annotation in annotations:
            for merged_span in merged_spans:
                annotation_i = set(range(*annotation))
                merged_span_i = set(range(*merged_span))
                if annotation_i.intersection(merged_span_i): #if there is overlap replace with the merged span
                    if annotation in res: res.remove(annotation) #could have been already removed bc of other overlap
                    res.append(merged_span)

        if len(res) != len(set(res)): #check if no duplicates
            raise AssertionError(f"After converting to overlap, the list contains duplicates. \nResulting list:{res}") 
        return res

    @staticmethod
    def to_string():
        return "treat_annotations"
    
    def get_conflicts(self, dir, overlapping = False):
        """
        Get all conflicting annotions of this annotation task and output them to a CSV file 
        """
        annotation_df = self.get_df()
        conflicts = annotation_df[annotation_df['conflict'] == True]
        conflicts.to_csv(os.path.join(dir, f"conf_{self.to_string()}.csv"))

    def get_merged_labels(self):
        """
        collects the merged labels as final labels and adds this to the annotation_dict. Also records if the labels could not be merged due to conflict 
        """
        for note in self.annotation_dict.keys():
            a1, a2  = self.annotators
            conflict = set(self.annotation_dict[note][a1]) != set(self.annotation_dict[note][a2]) #not exact match is seen as conflict

            all_spans = [self.annotation_dict[note][a1], self.annotation_dict[note][a2]]
            merged_spans = Treatments._get_merged_spans(all_spans)
            
            self.annotation_dict[note]['merged_spans'] = merged_spans
            self.annotation_dict[note]['conflict'] = conflict

    def get_final_labels(self):
        """
        collects the merged labels as final labels and adds this to the annotation_dict. Also records if the labels could not be merged due to conflict 
        """
        for note in self.annotation_dict.keys():
            if self.annotation_dict[note]['conflict']: #there was a conlfict before which will be resolved 
                intersections = [] #we collect the intersection of all annotator pairs 
                for annotator_pair in combinations(self.annotators + ['tie_breaker'], 2):
                    a1, a2 = annotator_pair

                    both_spans = [self.annotation_dict[note][a1], self.annotation_dict[note][a2]]
                    intersection = Treatments._get_merged_spans(both_spans)
                    intersections.append(intersection)

                intersections = [Treatments._spans2i(intersection) for intersection in intersections]

                final_spans = set.union(*intersections) #if an index is in at least 1 intersection it got at least 2 votes 

                if final_spans: #if there are any indices left after merging 
                    self.annotation_dict[note]['final_spans'] = Treatments._i2spans(final_spans)
                else:
                    self.annotation_dict[note]['final_spans'] = []
                self.annotation_dict[note]['conflict'] = False
            else:
                self.annotation_dict[note]['final_spans'] = copy.copy(self.annotation_dict[note]['merged_spans'])
    
    def print_ratio_no_mention(self, final = False):
        if final: 
            print('final')
            label_type = 'final_spans'
        else:
            print('merged')
            label_type = 'merged_spans'
        no_mention = 0
        annotation_dict = copy.deepcopy(self.annotation_dict)
        total = len(annotation_dict)
        for note, annotations in annotation_dict.items():
            if annotations[label_type] == []:
                no_mention += 1
        no_mention = no_mention/total
        print('no mention:\n',  no_mention)

    @staticmethod
    def _get_merged_spans(all_spans): #for 3 sets -> union of all 3 intersections
        """
        all_spans: any iterable containing a list of spans as tuples (int, int)
        """
        all_i = [Treatments._spans2i(spans) for spans in all_spans] #combine the collection of spans into a list of char indices representng the spans
        merged_i = set.intersection(*all_i)
        if merged_i:
            merged_spans = Treatments._i2spans(merged_i)
        else:
            merged_spans = [] #empty list
        return  merged_spans
    
    @staticmethod
    def _spans2i(spans: list[tuple[int, int]]) -> set[int]:
        """
        convert a list of spans into the set of all indices contained in the spans 
        """
        res = set()
        for span in spans: 
            res.update(range(*span))
        return res
    
    @staticmethod
    def _i2spans(indices: set[int]) -> list[tuple[int, int]]:
        """
        converts a set of indices to a list of spans that cover the all indices 
        """
        sorted_indices = sorted(list(indices))
        spans = [] 

        start = sorted_indices[0]
        for i in range(1, len(sorted_indices)):
            if (sorted_indices[i] - sorted_indices[i-1] > 1):
                spans.append((start, sorted_indices[i-1] + 1)) #add 1 to make it a proper range 
                start = sorted_indices[i]
        spans.append((start, sorted_indices[-1] + 1)) #last value for start is the starts value of the final span 
        
        return spans 

