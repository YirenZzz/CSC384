import io
import os
import sys
import math
import random
import numpy as np
from collections import defaultdict
from pandas import *
import time
SOS = '<start>'
EOS = '<end>'
N = 1e6
LAMBDA = 0.95

def prob_trans(key, model):
    """Get the transition probability from the HMM model,
    described in Neubig's slide p.10.
    Args:
        key: <str> The key of transition dict,
             usually in the form of 'WORD_{i-1} WORD_i'.
        model: <dict> The transition part of HMM model.
    Returns:
        A corresponding transition probability.
    """
    return model[key]

def prob_emiss(key, model):
    """Get the emission probability from the HMM model,
    described in Neubig's slide p.10.
    Notice that we should smooth for unknown words.
    Args:
        key: <str> The key of emission dict,
             usually in the form of 'TAG_i WORD_i'.
        model: <dict> The emission part of HMM model.
    Returns:
        A corresponding smoothed emission probability.
    """
    return LAMBDA * model[key] + (1 - LAMBDA) * 1 / N
'''
training_list=["training1.txt"]
test_file="test1.txt"
output_file="autooutput.txt"
'''
training_list=["autotraining.txt","training7.txt"]
test_file="autotest.txt"
output_file="autooutput.txt"
count=0
train_sentence_array=[]
for files in training_list:
    with open(files, 'r') as f:
        train_word=[]
        
        train_sentence=''
        tags=[]
        for line in f:
            line=line.replace(" : ", "_")
            train_word.append(line.strip())
            
            train_sentence=train_sentence+line.strip()+':'
            if line.strip()== '._PUN':
                train_sentence_array.append(train_sentence.strip().split(':'))
                train_sentence=''
                
trainingChange=[]
for sent in train_sentence_array:
    s=''
    for word in sent:
        s=s+word+' '
    trainingChange.append(s)

#compute the emit and transition probability
emit = defaultdict(int)
transition = defaultdict(int)
context = defaultdict(int)
transitionArray=[]
emitArray=[]
trans_prob = {}
emiss_prob = {}    
for line in trainingChange:
    previous = SOS
    context[previous] += 1
    for wordtag in line.strip().split(' '):
        if len(wordtag.split('_')) !=2:
            continue
        word, tag = wordtag.split('_')
        transition['{} {}'.format(previous, tag)] += 1
        context[tag] += 1 
        emit['{} {}'.format(tag, word)] += 1
        previous = tag
        transition['{} {}'.format(previous, EOS)] += 1

for key, value in sorted(transition.items(),
                         key=lambda x: x[1], reverse=True):
    previous, word = key.split(' ')
    prob=value / context[previous]
    transitionArray.append('{} {}\n'.format(key, value / context[previous]))
    if previous not in trans_prob:
        trans_prob[previous] = {word: float(prob)}
    else:
        trans_prob[previous].update({word: float(prob)})    
        
for key, value in sorted(emit.items(),
                         key=lambda x: x[1], reverse=True):
    previous, tag = key.split(' ')
    prob=value / context[previous] 
    emitArray.append('{} {}\n'.format(key, value / context[previous]))
    if previous not in emiss_prob:
        emiss_prob[previous] = {tag: float(prob)}
    else:
        emiss_prob[previous].update({tag: float(prob)})         


transition = defaultdict(float)
emission = defaultdict(float)
possible_tags = defaultdict(float)
tag_matrix=defaultdict(dict)
emission_matrix=defaultdict(dict)
for line in transitionArray:
    context, word, prob = line.strip().split(' ')
    possible_tags[context] = 1
    transition[' '.join([context, word])] = float(prob)
    tag_matrix[context][word]=float(prob)
    
for line in emitArray:
    context, word, prob = line.strip().split(' ')
    possible_tags[context] = 1
    emission[' '.join([context, word])] = float(prob)
    emission_matrix[context][word]=float(prob)

#df_transition.drop(index='<end>', columns='<start>').dot(df_emission)
df_transition = DataFrame.from_dict(tag_matrix, orient='columns', dtype=None)
df_transition=df_transition.replace(np.nan, 0)
#df_transition=np.log(df_transition)
df_emission = DataFrame.from_dict(emission_matrix,orient='index', dtype=None)
df_emission=df_emission.replace(np.nan, 0)
#df_emission=np.log(df_emission)
#format the test file, combine the word into sentence 
with open(test_file, 'r') as f:
    test_word=[]
    sentence_array=[]
    sentence=''
    for line in f:
        test_word.append(line.strip())
        sentence=sentence+line.strip()+' '
        if line.strip() == '.':
            sentence_array.append(sentence.strip().split(' '))
            sentence=''         
testArray=[]        
for sen in sentence_array:
    s=''
    for word in sen:
        s=s+word+' '
    testArray.append(s)
#write result to output file    
tag_arr=[]
c=0
data = {'<start>':df_transition['<start>']}
df = DataFrame(data)
for line in testArray:
    #forward
    if SOS in possible_tags:
        possible_tags.pop(SOS)
        
    words = line.strip().split(' ')
    l = len(words)
    best_score = {}
    best_edge = []
    #best_score['{} {}'.format(0, SOS)] = 0
    #best_edge['{} {}'.format(0, SOS)] = None
    start=time.time()
    if words[0] not in df_emission:
        df_emission[words[0]]=0
    column=df_transition['<start>'].drop(index='<end>').multiply(df_emission[words[0]])
    data[words[0]]=column
    best_value=column.max()
    best_idx=column.idxmax()
    best_edge.append(best_idx)
    for i in range(1,l):
        print(words[i])
        c=c+1
        print(c)   
        print(best_idx)
        if words[i] not in df_emission:
            df_emission[words[i]]=0        
        column=best_value*df_transition[best_idx].drop(index='<end>').multiply(df_emission[words[i]])
        data[words[i]]=column
        best_value=column.max()
        best_idx=column.idxmax()
        best_edge.append(best_idx)
    '''
    column=best_value*df_transition[best_idx].drop(columns='<start>')
    best_value=column.max()
    best_idx=column.idxmax()
    best_edge.append(best_idx)
    '''
    '''
    for next in possible_tags.keys():
        prev=SOS
        prev_key = '{} {}'.format(0, prev)
        next_key = '{} {}'.format(1, next)
        trans_key = '{} {}'.format(prev, next)
        emiss_key = '{} {}'.format(next, words[0])
        if prev_key in best_score and trans_key in transition:
            score = best_score[prev_key] + \
                    -df_transition[prev][next] + \
                    -math.log2(prob_emiss(emiss_key, emission))
            if next_key not in best_score or best_score[next_key] > score:
                best_score[next_key] = score
                best_edge[next_key] = prev_key
                
    for i in range(1, l):
        c=c+1
        print(c)            
        
        for next in possible_tags.keys():
            for prev in possible_tags.keys():
                prev_key = '{} {}'.format(i, prev)
                next_key = '{} {}'.format(i + 1, next)
                trans_key = '{} {}'.format(prev, next)
                emiss_key = '{} {}'.format(next, words[i])
                if prev_key in best_score and trans_key in transition:
                    score = best_score[prev_key] + \
                                -df_transition[prev][next] + \
                                -math.log2(prob_emiss(emiss_key, emission))
                    if next_key not in best_score or best_score[next_key] > score:
                        best_score[next_key] = score
                        best_edge[next_key] = prev_key
    
    for prev in possible_tags.keys():
        prev_key = '{} {}'.format(l, prev)
        next_key = '{} {}'.format(l + 1, EOS)
        trans_key = '{} {}'.format(prev, EOS)
        emiss_key = '{} {}'.format(EOS, EOS)
        if prev_key in best_score and trans_key in transition:
            score = best_score[prev_key] + \
                    -df_transition[prev][next]
            if next_key not in best_score or best_score[next_key] > score:
                best_score[next_key] = score
                best_edge[next_key] = prev_key   
    '''
    end=time.time()
    deltatime=end-start
    print('deltatime',deltatime)
    #backward
    '''
    words = line.strip().split(' ')
    l = len(words)
    tags = []
    next_edge = best_edge['{} {}'.format(l+1, EOS)]
    while next_edge != '{} {}'.format(0, SOS):
        position, tag = next_edge.split(' ')
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    tag_arr=tag_arr+tags
    '''
    tag_arr=tag_arr+best_edge

with open(output_file, 'w') as f:
    for j in range(len(test_word)):
        f.write('{} : {}\n'.format(test_word[j], tag_arr[j]))