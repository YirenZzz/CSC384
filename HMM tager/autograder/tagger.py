'''
speed up methods:
1. segmenting the training and test test into sentences
2. use matrix to store transition and emission probability, use numpy and pandas to do multiplication (multiply by layers instead of use multiple nested loop)
3. use log value of probability, that make the multiplication between layers to be addition (add calculation always faster than multiple)
'''

# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import numpy as np
from collections import defaultdict
from pandas import *

SOS = '<start>'
EOS = '<end>'


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    #
    # YOUR IMPLEMENTATION GOES HERE
    #
    #combine the words into sentance
    train_sentence_array=[]
    for files in training_list:
        with open(files, 'r') as f:
            train_word=[]
            train_sentence=''
            for line in f:
                line=line.replace(" : ", "_")
                train_word.append(line.strip())
                
                train_sentence=train_sentence+line.strip()+' : '
                if line.strip()== '._PUN':
                    train_sentence_array.append(train_sentence.strip().split(' : '))
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
        transitionArray.append('{} {}'.format(key, float(prob)))
        if previous not in trans_prob:
            trans_prob[previous] = {word: float(prob)}
        else:
            trans_prob[previous].update({word: float(prob)})    
            
    for key, value in sorted(emit.items(),
                             key=lambda x: x[1], reverse=True):
        previous, tag = key.split(' ')
        prob=value / context[previous] 
        emitArray.append('{} {}'.format(key, float(prob)))
        if previous not in emiss_prob:
            emiss_prob[previous] = {tag: float(prob)}
        else:
            emiss_prob[previous].update({tag: float(prob)})         
    
    
    possible_tags = defaultdict(float)
    tag_matrix=defaultdict(dict)
    emission_matrix=defaultdict(dict)
    for line in transitionArray:
        context, word, prob = line.strip().split(' ')
        possible_tags[context] = 1
        tag_matrix[context][word]=float(prob)
        
    for line in emitArray:
        context, word, prob = line.strip().split(' ')
        possible_tags[context] = 1
        emission_matrix[context][word]=float(prob)
    #convert dict to matrix
    #df_transition.drop(index='<end>', columns='<start>').dot(df_emission)
    df_transition = DataFrame.from_dict(tag_matrix, orient='columns', dtype=None)
    df_transition=df_transition.replace(np.nan, 1e-11)
    df_transition=np.log2(df_transition)
    df_emission = DataFrame.from_dict(emission_matrix,orient='index', dtype=None)
    df_emission=df_emission.replace(np.nan, 1e-11)
    df_emission=np.log2(df_emission)
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
    data = {'<start>':df_transition['<start>']}
    df = DataFrame(data)
    for line in testArray:
            
        words = line.strip().split(' ')
        l = len(words)
        best_idx_arr = []
        #use Viterbi
        if words[0] not in df_emission:
            df_emission[words[0]]=0          
        column=df_transition['<start>'].drop(index='<end>').add(df_emission[words[0]])
        data[words[0]]=column
        best_value=column.max()
        best_idx=column.idxmax()
        best_idx_arr.append(best_idx)
        for i in range(1,l):
            if words[i] not in df_emission:
                df_emission[words[i]]=0              
            column=df_transition[best_idx].drop(index='<end>').add(df_emission[words[i]].add(best_value))
            data[words[i]]=column
            best_value=column.max()
            best_idx=column.idxmax()
            best_idx_arr.append(best_idx)
        
        tag_arr=tag_arr+best_idx_arr
    
    with open(output_file, 'w') as f:
        for j in range(len(test_word)):
            f.write('{} : {}\n'.format(test_word[j], tag_arr[j]))
    
if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    #os.system("python3 tagger.py -d autotraining.txt -t autotest.txt -o autooutput.txt")
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    #training_list=["autotraining.txt"]
    #test_file="autotest.txt"
    #output_file="autooutput.txt"
    #print("Training files: " + str(training_list))
    #print(training_list)
    #print("Test file: " + test_file)
    #print("Ouptut file: " + output_file)

    # Start the training and tagging operation.
    tag (training_list, test_file, output_file)
  

    