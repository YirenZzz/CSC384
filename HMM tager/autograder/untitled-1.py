import io
import os
import sys
import math
import random
import numpy as np
from collections import defaultdict
SOS = '<s>'
EOS = '</s>'
N = 1e6
LAMBDA = 0.95
def load_model(model_file):
    """Load the model file.
    Args:
        model_file: <str> The model file path.
    Returns:
        A transition probability dict and a emission probability dict.
    (Both are <dict of dicts>)
    """
    trans_prob = {}
    emiss_prob = {}
    with open(model_file, 'r') as f:
        for line in f:
            type, prev, next, prob = line.strip().split(' ')
            if type == 'T':  # Transition probability.
                if prev not in trans_prob:
                    trans_prob[prev] = {next: float(prob)}
                else:
                    trans_prob[prev].update({next: float(prob)})
            elif type == 'E':  # Emission probability.
                if prev not in emiss_prob:
                    emiss_prob[prev] = {next: float(prob)}
                else:
                    emiss_prob[prev].update({next: float(prob)})
    return trans_prob, emiss_prob
def forward(transition, emission, possible_tags, line):
    """The forward process of the Viterbi algorithm,
    described in Neubig's slides p.38-40.
    Notice: Maybe this version of `forward()` is more easy for understanding.

    Args:
        transition: <dict>
        emission: <dict>
        possible_tags: <dict>
        line: <str> A line of the file.

    Returns:
        The best edges <dict> derived from the forward process.
    """
    # Remove the SOS (default <s>) from the possible tags.
    if SOS in possible_tags:
        possible_tags.pop(SOS)
    words = line.strip().split(' ')
    l = len(words)
    best_score = {}
    best_edge = {}
    best_score['{} {}'.format(0, SOS)] = 0 # Start with SOS (default <s>).
    best_edge['{} {}'.format(0, SOS)] = None

    # Following three parts are corresponding to the Neubig's slides p.38-40.
    # I make them looks nearly the same in the forms to let you easy to compare.
    # Of course you can combine these parts together to make the code shorter!

    # First part, described in Neubig's slides p.38.
    for next in possible_tags.keys():
        for prev in [SOS]:
            prev_key = '{} {}'.format(0, prev)
            next_key = '{} {}'.format(1, next)
            trans_key = '{} {}'.format(prev, next)
            emiss_key = '{} {}'.format(next, words[0])
            if prev_key in best_score and trans_key in transition:
                score = best_score[prev_key] + \
                        -math.log2(prob_trans(trans_key, transition)) + \
                        -math.log2(prob_emiss(emiss_key, emission))
                if next_key not in best_score or best_score[next_key] > score:
                    best_score[next_key] = score
                    best_edge[next_key] = prev_key

    # Middle part, described in Neubig's slides p.39.
    for i in range(1, l):
        for next in possible_tags.keys():
            for prev in possible_tags.keys():
                prev_key = '{} {}'.format(i, prev)
                next_key = '{} {}'.format(i + 1, next)
                trans_key = '{} {}'.format(prev, next)
                emiss_key = '{} {}'.format(next, words[i])
                if prev_key in best_score and trans_key in transition:
                    score = best_score[prev_key] + \
                            -math.log2(prob_trans(trans_key, transition)) + \
                            -math.log2(prob_emiss(emiss_key, emission))
                    if next_key not in best_score or best_score[next_key] > score:
                        best_score[next_key] = score
                        best_edge[next_key] = prev_key

    # Final part, described in Neubig's slides p.40.
    for next in [EOS]:
        for prev in possible_tags.keys():
            prev_key = '{} {}'.format(l, prev)
            next_key = '{} {}'.format(l + 1, next)
            trans_key = '{} {}'.format(prev, next)
            emiss_key = '{} {}'.format(next, EOS)
            if prev_key in best_score and trans_key in transition:
                score = best_score[prev_key] + \
                        -math.log2(prob_trans(trans_key, transition))
                if next_key not in best_score or best_score[next_key] > score:
                    best_score[next_key] = score
                    best_edge[next_key] = prev_key

    return best_edge

def backward(best_edge, line):
    """The backward part of Viterbi algorithm.

    Args:
        best_edge: <list> Each component contains previous best edge.
        line: <str> A line of the file.

    Returns:
        The tag sequence.
    """
    words = line.strip().split(' ')
    l = len(words)
    tags = []
    next_edge = best_edge['{} {}'.format(l+1, EOS)]
    while next_edge != '{} {}'.format(0, SOS):
        position, tag = next_edge.split(' ')
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags
def norm(raw_list):
    """Normalize a list of float numbers.
    Args:
        raw_list: <list of float>
    Returns:
        A normalized list of float numbers.
    """
    return [i/sum(raw_list) for i in raw_list]

def random_sample(model_file):
    """Make a random sampling process in a HMM model.
    Initialize a random POS tag and generate a series of tags randomly
    according to the transition probability, as well as output a certain
    word randomly according to the emission probability.
    Args:
        model_file: <str> The model file path.
    Returns:
        A funny word sequence. :-)
    """
    trans_prob, emiss_prob = load_model(model_file)
    output_seq = []
    next_tag = random.sample(emiss_prob.keys(), 1)[0]  # Initialize.
    while next_tag != EOS:  # Until see the end of sentence mark.
        # Generate a output word.
        candidate_word = list(emiss_prob[next_tag].keys())
        candidate_word_prob = norm(emiss_prob[next_tag].values())
        output_word = np.random.choice(candidate_word, p=candidate_word_prob)
        output_seq.append(output_word)

        # Generate the next tag.
        candidate_tag = list(trans_prob[next_tag].keys())
        candidate_tag_prob = norm(trans_prob[next_tag].values())
        next_tag = np.random.choice(candidate_tag, p=candidate_tag_prob)
    #print(' '.join(output_seq))
    
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

training_list=["training7.txt"]
test_file="test7.txt"
output_file="autooutput.txt"
count=0
for files in training_list:
    with open(files, 'r') as f:
        train_word=[]
        train_sentence_array=[]
        train_sentence=''
        tags=[]
        for line in f:
            line=line.replace(" : ", "_")
            count+=1
            train_word.append(line.strip())
            
            train_sentence=train_sentence+line.strip()+':'
           
            if line.strip()== '._PUN':
                train_sentence_array.append(train_sentence.strip().split(':'))
                train_sentence=''

out = io.StringIO()
for sent in train_sentence_array:
    s=''
    for word in sent:
        s=s+word+' '
    out.write('{}\n'.format(s))
with open("trainingChange.txt", 'w') as f:
    f.write(out.getvalue().strip())
    

SOS = '<s>'
EOS = '</s>'
emit = defaultdict(int)
transition = defaultdict(int)
context = defaultdict(int)
training_file="trainingChange.txt"
model_file="my_model"
with open(training_file, 'r') as f:
    for line in f:
        previous = SOS  # Make the sentence start.
        context[previous] += 1
        for wordtag in line.strip().split(' '):
            if len(wordtag.split('_')) !=2:
                continue
            word, tag = wordtag.split('_')
            # Count the transition.
            transition['{} {}'.format(previous, tag)] += 1
            context[tag] += 1  # Count the context.
            # Count the emission.
            emit['{} {}'.format(tag, word)] += 1
            previous = tag
        # Make the sentence end.
        transition['{} {}'.format(previous, EOS)] += 1

# Save the info into a buffer temporarily.
out = io.StringIO()
for key, value in sorted(transition.items(),
                         key=lambda x: x[1], reverse=True):
    previous, word = key.split(' ')
    out.write('T {} {}\n'.format(key, value / context[previous]))
for key, value in sorted(emit.items(),
                         key=lambda x: x[1], reverse=True):
    previous, tag = key.split(' ')
    out.write('E {} {}\n'.format(key, value / context[previous]))

# Print on the screen or save in the file.
if model_file == 'stdout':
    print(out.getvalue().strip())
else:
    with open(model_file, 'w') as f:
        f.write(out.getvalue().strip())

random_sample(model_file)


transition = defaultdict(float)
emission = defaultdict(float)
possible_tags = defaultdict(float)
with open(model_file, 'r') as f:
    for line in f:
        type, context, word, prob = line.strip().split(' ')
        possible_tags[context] = 1
        if type == 'T':
            transition[' '.join([context, word])] = float(prob)
        else:
            emission[' '.join([context, word])] = float(prob)


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
    
    
out = io.StringIO()

with open("out5.txt", 'w') as f:
    for sen in sentence_array:
        s=''
        for word in sen:
            s=s+word+' '
        f.write('{}\n'.format(s))
       
test_file="out5.txt"

out = io.StringIO()

with open(test_file, 'r') as f:
    tag_arr=[]
    for line in f:
        best_edge = forward(transition, emission, possible_tags, line)
        tags = backward(best_edge, line)
        tag_arr=tag_arr+tags
        print(tags)
        out.write(' '.join(tags) + '\n')

# Print on the screen or save in the file.
if output_file == 'stdout':
    print(out.getvalue().strip())
else:
    with open(output_file, 'w') as f:
        for j in range(len(test_word)):
            f.write('{} : {}\n'.format(test_word[j], tag_arr[j]))