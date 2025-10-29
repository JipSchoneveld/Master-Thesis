"""
All functions for running baselines for symptom status prediction
"""

from sklearn.dummy import DummyClassifier
import clinlp, spacy 
from unidecode import unidecode
import re 

# majority baseline
def majority_baseline(X_train, y_train, X_test):
    """
    Trains and runs a majority baseline classifier.

    Parameters: 
    X_train (pd.Series): train text from notes with NoteID as index
    y_train (pd.Series): train labels with NoteID as index

    Returns: 
    predicted_labels (dict[str, int]]): all predicted labels as a dictionary of: NoteID -> predicted label 
    """

    ##setup dictionary 
    predicted_labels = {}

    ## train majority classifier 
    majority_cls = DummyClassifier()
    majority_cls.fit(X_train, y_train)

    ## make predictions 
    if X_test is not None: 
        y_pred = majority_cls.predict(X_test) 
        #save prediction per noteID
        for noteID, prediction in zip(X_test.index, y_pred):
            predicted_labels[noteID] = int(prediction)
    else:
        y_pred = majority_cls.predict(X_train) 
        #save prediction per noteID
        for noteID, prediction in zip(X_train.index, y_pred):
            predicted_labels[noteID] = int(prediction)

    return predicted_labels

def keyword_baseline(X_train, cat, X_test):

    if X_test is not None:
        X = X_test
    else:
        X = X_train

    nlp = spacy.blank('clinlp')
    nlp.add_pipe("clinlp_sentencizer")

    ##setup dictionary 
    predicted_labels = {}

    for noteID, note in X.items():
        doc = nlp(note)
        predicted_labels[noteID] = 100 #no match is the default  

        for item in doc.sents:
            sent = unidecode((item.text).lower())
            label = match_keyword(sent, cat)
            if label is not None: #we take the first match 
                predicted_labels[noteID] = label
                break

    return predicted_labels

def match_keyword(sent, cat):
    
    lookup_dict = {
        0: {
            "general": [
                "depressieve", "stemming", "somber", "moedeloos", "hopeloos", "non-verbale", "verdrietig", "voelt zich slecht", "neerslachtig", "malaise", "treurig", "melancholiek", "dysthymie"
            ],
            "worsened": [
                "depressiever", "slechter", "moedelozer", "hopelozer", "treuriger", "verdrietiger", "somberder"
            ],
            "worsened_": ["meer"],
            "improved": ["beter", "opgewekter", "positiever"],
            "improved_": ["minder"]
        },

        1: {
            "general": [
                "zelfdepreciatie", "(gebrek aan|lage|weinig|geen) zelfwaardering", "zelfverwijt", "(gebrek aan|weinig|geen) zelfrespect", "schuld", "straf", "faalangst", "(laag|weinig) zelfvertrouwen", "waardeloos", "inferieur", "onzeker"
            ],
            "worsened": [
                "schuldiger", "waardelozer", "(lager|minder|afname \w*) (zelfvertrouwen|zelfrespect|zelfwaardering)", "onzekerder"
            ],
            "worsened_": ["meer"],
            "improved": [
                "zelfverzekerder", "zekerder", "beter zelfbeeld", "(meer|hoger|toename \w*) (zelfvertrouwen|zelfrespect|zelfwaardering)"
            ],
            "improved_": ["minder"]
        },

        2: {
            "general": [
                "leven", "sterven", "zelfmoord", "suicid", "doodswens", "euthanasie", "TS"
            ],
            "worsened": ["suicidaler"],
            "worsened_": ["meer", "recent"],
            "improved": ["leefwens"],
            "improved_": ["minder"]
        },

        3: {
            "general": [
                "slecht slapen", "(niet|slecht) doorslapen", "wakker liggen", "insomni"
            ],
            "worsened": [
                "slechter slapen", "(minder|slechter) doorslapen"
            ],
            "worsened_": ["meer"],
            "improved": [
                "(meer|beter) doorslapen", "beter slapen"
            ],
            "improved_": ["minder"]
        },

        4: {
            "general": [
                "anhedonie", "(geen|gebrek aan|weinig) interesse", "(geen|gebrek aan|weinig) motivatie", "geen activiteit", "werkt niet", "weinig plezier", "gedemotiveerd"
            ],
            "worsened": [
                "minder plezier", "gedemotiveerder", "(minder|afname \w*) (interesse|motivatie|activiteit|actief)", "werkt minder"
            ],
            "worsened_": ["meer"],
            "improved": [
                "gemotiveerder", "actiever", "geïnteresseerder", "(toename \w*|meer) (interesse|motivatie)", "werkt meer"
            ],
            "improved_": ["minder"]
        },

        5: {
            "general": [
                "vertraging", "vertraagd", "langzaam", "vlak", "verminderd modulerend affect", "trage", "traag"
            ],
            "worsened": ["trager", "vlakker", "langzamer"],
            "worsened_": ["meer"],
            "improved": ["sneller", "levendiger"],
            "improved_": ["minder"]
        },

        6: {
            "general": [
                "opwinding", "agitatie", "lichamelijke onrust", "rusteloosheid", "(moeite met|niet) stilzitten", "hyperactiviteit", "gejaagd"
            ],
            "worsened": [
                "onrustiger", "gejaagder", "rustelozer", "lichamelijk onrustiger"
            ],
            "worsened_": ["meer"],
            "improved": ["rustiger", "kalmer"],
            "improved_": ["minder"]
        },

        7: {
            "general": [
                "angst", "bezorgdheid", "bang", "onveiligheid", "bedreiging", "paniek", "gespannenheid", "piekeren", "prikkelbaarheid", "vrees", "hypervigilantie", "nervositeit", "zenuwachtig"
            ],
            "worsened": [
                "banger", "paniekeriger", "prikkelbaarder", "angstiger", "onveiliger", "gespannener", "zenuwachtiger"
            ],
            "worsened_": ["meer"],
            "improved": ["kalmer", "meer ontspannen"],
            "improved_": ["minder"]
        },

        8: {
            "general": [
                "gastro-intestina", "(weinig|geen|gebrek aan) (eetlust|voedselinname|smaak|energie)", "verstopping", "krampen", "transpireren", "beven", "hyperventilatie", "droge mond", "pijn", "vermoeid", "uitputting", "bleke indruk", "magere indruk", "dyspepsie", "misselijk"
            ],
            "worsened": [
                "vermoeider", "(afname \w*|minder) (eetlust|voedselinname|smaak|energie)", "misselijker"
            ],
            "worsened_": ["meer"],
            "improved": [
                "(toename \w*|meer) (eetlust|voedselinname|smaak|energie)"
            ],
            "improved_": ["minder"]
        },

        9: {
            "general": [
                "hypochondrie", "hypochondrisch", "angst voor ziekte", "bezorgd om gezondheid", "lichamelijke gewaarwording", "lichamelijke symptomen", "ernstige ziekte", "waangedacht", "overbezorgd"
            ],
            "worsened": ["overbezorgder", "hypochondrischer"],
            "worsened_": ["meer"],
            "improved": [],
            "improved_": ["minder"]
        },

        10: {
            "general": [
                "(gebrek aan|verminderd|afname \w*) inzicht", "ontkenning", "ongerelateerde factoren", "inzichtloos"
            ],
            "worsened": ["minder inzicht"],
            "worsened_": ["meer"],
            "improved": ["meer inzicht"],
            "improved_": ["minder"]
        },

        11: {
            "general": ["gewichtsverlies"],
            "worsened": [
                "lichter", "afgenomen gewicht", "afgevallen", "weegt minder", "gewichtsverlies"
            ],
            "worsened_": ["meer"],
            "improved": ["aangekomen", "weegt meer"],
            "improved_": ["minder"]
        }
    }

    
    cat_dict = lookup_dict[cat]
    
    for p in cat_dict['improved']:
        match = re.search(p, sent)
        if match:
            return 1 

    for p in cat_dict['worsened']:
        match = re.search(p, sent)
        if match:
            return -1

    for p in cat_dict['general']:
        match = re.search(p, sent)
        if match:
            for p in cat_dict['improved_']:
                match = re.search(p, sent)
                if match:
                    return 1
            for p in cat_dict['worsened_']:
                match = re.search(p, sent)
                if match:
                    return -1 
            return 0 
        
    return None

