#main idea of this file is to extract thre entities of a given text and built a.json file with marked entities

from .abstract_extraction import *
import re
import itertools
import json
import spacy
from spacy.util import filter_spans
from difflib import SequenceMatcher
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher 
from spacy.tokens import Span 
import nltk
from nltk import sent_tokenize
from allennlp.predictors.predictor import Predictor

# download punkt from nltk if it is not downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# create coreference resolution predictor
model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
predictor = Predictor.from_path(model_url)

#function to calculate similarity between 2 strings
#Input: 2 strings
#Output: ratio between [0-1] representing how much similar both the strings are. Ratio of 1 means strings are exactly same.
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

#Function to find root word in the sentence
def get_relation(sent):
# relation between the 2 entities
    doc = nlp(sent)

  # Matcher class object 
    matcher = Matcher(nlp.vocab)

  #define the pattern 
    pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1
    if(k>=0):
        span = doc[matches[k][1]:matches[k][2]] 
        return(span.text)
    return ""

#remove E tags from sentence 
def remove_etag(text):
    CLEANR = re.compile('\[.*?\]') 
    cleantext = re.sub(CLEANR, '', text)
    return cleantext    

# Rule based entity extraction English
# rule: Extract subject and all objects along withh all modifiers, compound words and punctuations.
#Input: sentence
#Output: List of entities in the text
def get_entities(sent):
    entity_list = []
    sub_obj_list = []
    current_text = ""
    modified_entity_list = []
    tt = False
    punct = False
    sent = re.sub('-','',sent)
    doc = nlp(sent)
    
    for tok in doc:
        if(tok.dep_.find("subj") == True or tok.dep_ == "pobj"):
            sub_obj_list.append(tok.text)

    for tok in doc:
        if(tok.text in sub_obj_list):
            tt = False
            if(punct == True):
                current_text = current_text + tok.text
            else:
                current_text = current_text + " " + tok.text
            current_text = current_text.strip()
            current_text = re.sub(' +', ' ', current_text)
            if(len(nltk.word_tokenize(current_text))<=4):
                entity_list.append(current_text)
            current_text = ""
            punct = False
        elif(tok.dep_ == "compound" or tok.dep_.endswith("mod") == True or tt == True):
            tt = True
            if(tok.dep_ != "punct" and punct == False):
                current_text = current_text + " " + tok.text
            elif(tok.dep_ != "punct" and punct == True):
                current_text = current_text + tok.text
                # reset punct after a not punct token has been added
                # punct = False
            else:
                current_text = current_text + tok.text
                punct = True
    root = get_relation(sent)
    if(root !=""):
        token_root = nltk.word_tokenize(root)
        token_sent = nltk.word_tokenize(sent)
        index_root = token_sent.index(token_root[0])
        # remove empty entity
        while('' in entity_list):
            entity_list.remove('')
        for i in range(len(entity_list)):
            token_entity =  nltk.word_tokenize(entity_list[i])
            ind = token_sent.index(token_entity[0])
            if(ind >= index_root):
                # check for similar entities
                add_e1 = True
                add_e2 = True
                # dont add entity if its similarity ratio with atleast one already present entity is over 0.7
                for j in range(len(modified_entity_list)):
                    if(similar(modified_entity_list[j],entity_list[i-1])>0.7 or len(nltk.word_tokenize(entity_list[i-1]))>4 ):
                        add_e1 = False
                    if(similar(modified_entity_list[j],entity_list[i])>0.7 or len(nltk.word_tokenize(entity_list[i]))>4 ):
                        add_e2 = False
                if(add_e1 == True):
                    modified_entity_list.append(entity_list[i-1])
                if(add_e2 == True):
                    modified_entity_list.append(entity_list[i])
                break
    return modified_entity_list

# Rule based entity extraction German
#Input: sentence
#Output: List of entities in the text
def get_entities_de(sent):
    Entity_List=[]
    sent = re.sub('-','',sent)
    doc = nlp(sent)

  # Matcher class object 
    matcher = Matcher(nlp.vocab)

  #define the pattern 
    pattern = [ 
            {'POS':'ADJ','OP':"*"}, {'POS':'NOUN'}] 

    matcher.add("matching_1", [pattern]) 

    matches = matcher(doc,as_spans=True)
    matches = filter_spans(matches)
    k = len(matches)
    for i in range(k):
        if(len(matches[i])>2):
            span = doc[matches[i][1]:matches[i][2]]
            Entity_List.append(span.text)

    return(Entity_List)

# Function to create a dictionary with sentence as key and list of all the entites(rule based + Spacy model) as value
def create_entity_dict(path):
    text, lang = extract_abstract(path)
    #co-ref resolution
    prediction = predictor.predict(document=text)
    text = predictor.coref_resolved(text)
    text = re.sub('-','',text)
    text = re.sub(':',' ',text)
    # sent_tokenize
    text = sent_tokenize(text)
    Entity_rulebased = []
    Entity_spacy = []
    # rule based entity
    if(lang): #english text
        for sent in text:
            Entity_rulebased.append(get_entities(sent))
        #spacy based entity
        NER = spacy.load("en_core_web_trf")
    else:
        for sent in text:
            Entity_rulebased.append(get_entities_de(sent))
        NER = spacy.load("de_core_news_lg")
    for sent in text:
        current_list = []
        tt = NER(sent)
        for word in tt.ents:
            if word.label_ not in ["CARDINAL","DATE"]: #remove cardinal and date entity
                #check if its similar to previous entities. Do not add if ratio is over 0.7 with atleast one of the entity.
                add = True
                for i in range(len(current_list)):
                    if(similar(str(word),current_list[i]) > 0.7):
                        add = False
                if(add == True):
                    current_list.append(str(word))
        Entity_spacy.append(current_list)
    
    # merging entities from rule based and Spacy model
    final_entity = []
    for i in range(len(text)):
        current = Entity_spacy[i]
        for word in Entity_rulebased[i]:
            add_all = True
            for j in range(len(current)):
                if(similar(str(word),str(current[j]))>0.7):
                    add_all = False
            if(add_all == True):
                current.append(str(word))
        final_entity.append(current)

        
    # dont take sentences with 1 entity
    dt = {}
    for i in range(len(text)):
        if(len(final_entity[i]) > 1):
            dt[text[i]] = final_entity[i]
            
    return dt

# function to add [E] and [/E] tags around entities in an sentence
def add_entity_tokens(sentence, entities):
    if len(entities) == 0:
        return False
    idx1 = sentence.find(entities[0])
    idx2 = sentence.find(entities[1])

    first = [idx1, idx1 + len(entities[0])] if idx1 < idx2 else [idx2, idx2 + len(entities[1])]
    second = [idx1, idx1 + len(entities[0])] if idx1 > idx2 else [idx2, idx2 + len(entities[1])]

    sentence = (sentence[:first[0]] 
        + "[E1]" + sentence[first[0]:first[1]] + "[/E1]"
        + sentence[first[1]:second[0]] 
        + "[E2]" + sentence[second[0]:second[1]] + "[/E2]"
        + sentence[second[1]:])
    
    return sentence

# function to create a json file with sentences where entities are marked with tags
def create_ip_file(path): 
    dct = create_entity_dict(path)
    #print(dct)
    final_sent = []
    for key in dct:
        #print(dct[key])
        for comb in itertools.combinations(dct[key], 2):
            tagged_sent = add_entity_tokens(key, comb)
            final_sent.append(tagged_sent)

    sentence_dict = [{'text': l } for l in final_sent]

    with open(r'data/input.json', "w") as f:
        json.dump(sentence_dict, f)

    return "\n".join(final_sent)

# Function to find text between both entites. Used in rule based relationship extraction.
def get_text_between_entities(text):
    result = re.search('\[E1\](.*)\[/E2\]', text)
    new_text = result.group(1)
    new_text = re.sub('\[/E1\]','',new_text)
    new_text = re.sub('\[E2\]','',new_text)
    
    return new_text
