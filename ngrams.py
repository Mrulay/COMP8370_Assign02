import json
import os 
import pandas as pd 
from nltk.corpus import brown
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize.treebank import TreebankWordDetokenizer
import string
import csv

def trainNgram(n, trainWords):
    '''
    A simple function that returns a trained n-gram model. 
    
    Parameter Descriptions:    
        n: determines the context limit in the n-gram model
        trainWords: the text data that our n-gram model learns on
    '''
    train_data, padded_sents = padded_everygram_pipeline(n, trainWords)
    
    model = MLE(n)
    model.fit(train_data, padded_sents)
    
    return model

def testNgram(k, model, testSet, targetLabels): 
    '''
    Takes in 4 parametes and returns the number of words that were correctly predicted by the n-gram model.
    
    Parameter Descriptions:    
        k: number of top-k most frequent words that we are considering for evaluation
        model: the n-gram model that we select 
        testSet: the set of context (past n-1 words) from BirkBeck word corpus that we use for each iteration
        targetLabels: the correct words that we are trying to predict using Language Model 
    '''
    hits = 0
    for i in range(len(testSet)):
        input = testSet[i][:testSet[i].index("*")].strip()
        output = []
        while len(output) <= k:
            x = generate_sent(model, 20, [input])
            x = x.translate(str.maketrans('', '', string.punctuation))
            output.append(x.split(' ', 1)[0])
        if targetLabels[i] in output:
            hits += 1
    return hits 

def generate_sent(model, num_words, text): 
    '''
    Utility function for testNgram() that formats the generated word list into string and removes <s> (unknown words)
    '''
    detokenize = TreebankWordDetokenizer().detokenize
    content = []
    for token in model.generate(num_words, text_seed=text):
        if token == '<s>':
            continue
        if token == '</s>':
            break  
        content.append(token)
    return detokenize(content)

def savetoCSV(s_at_k): 
    '''
    Function that writes all the outputs (s@k) to the csv
    '''
    row = ",".join(s_at_k)
    with open('output.csv') as f:
        writer = csv.writer(f)
        writer.writerow(row)

if __name__ == '__main__':
    
    with open("APPLING1DAT.643") as f:
        bbkLines = f.readlines() # reading misspelled sentences from the APPLING1DAT.643 file
    
    bbkLines = [l for l in bbkLines if "$" not in l] # removing words with $ tags  
    bbkLines = [l.split(' ', 1)[1].strip() for l in bbkLines] # removing misspelled words  
    
    # Splitting correct words and the sentence and storing them in a dictionary
    correctWords = [l.split(' ', 1)[0].lower().strip() for l in bbkLines]
    testSents = [l.split(' ', 1)[1].lower().strip() for l in bbkLines]
    bbkPairs = dict(zip(testSents, correctWords))
    
    brWords = brown.sents() # Extracting all sentences from the brown corpus
    brWords = [[word.lower() for word in element] 
              for element in brWords] #lowering all words 
    
    modelList = [1,2,3,5,10] # defining the number of n for each n-gram model
    kList = [1,5,10] # defining k for evaluation
    
    for i in modelList:
        ngramModel = trainNgram(i, brWords)
        s_at_k = []
        for k in kList:
            hits_at_k = testNgram(k, ngramModel, testSents, correctWords)
            s_at_k.append(hits_at_k)
        savetoCSV(s_at_k)
     
    

    
