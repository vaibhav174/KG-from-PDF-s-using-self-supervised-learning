import nltk
from nltk import sent_tokenize,word_tokenize
from preprocessing.IP_data_generation import *
import numpy as np
import torch
import operator
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Function to create a sliding window of size n over a word tokenized sentence i.e. list of words
# Input: list of words,size of window
# Output: List of lists where each list is a window of size n from the text.
def create_window(lst,n):
    batch_list = []
    for i in range(len(lst)-n+1):
        batch = lst[i:i+n]
        batch_list.append(batch)
    return batch_list

# Function to find most occurring n-gram for a perticular cluster.
# Input: List of all the sentences that belong to same cluster, parameter n of n-gram
# output: Most occurring n-gram as list of words  
def n_gram_relation(all_sent,n):
    all_grams = {}
    for sent in all_sent:
        word_token = word_tokenize(sent)
        window = create_window(word_token,n)
        for lt in window:
            lt = str(lt)
            if lt in all_grams:
                all_grams[lt] = all_grams[lt] + 1
            else:
                all_grams[lt] = 1
    relationship = max(all_grams.items(), key=operator.itemgetter(1))[0]
    return relationship
#function to find relationships using most occurring n-gram for all the clusters
def get_relations(classes,text):
	relation_dict = {}
	classes = classes.tolist()
	unique_classes = list(set(classes))
	for c in unique_classes:
	    indices = [i for i, x in enumerate(classes) if x == c]
	    all_sent = []
	    for ind in indices:
	        all_sent.append(get_text_between_entities(text[ind]))
	    relation_dict[c] = n_gram_relation(all_sent,3)
	return relation_dict
