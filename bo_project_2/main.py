import pyximport
pyximport.install()

import fileparser
import os
import hmm

from LogProbability import LogProbability

import time

def increment_transitions(transition_map, tag, given):
    """update transition frequencies """
    if transition_map.has_key(given):
        if transition_map[given].has_key(tag):
            transition_map[given][tag] = transition_map[given][tag] + 1.0
        else:
            transition_map[given][tag] = 1.0
    else:
        transition_map[given] = { tag:1.0 }

def increment_emissions(emission_map, tag, word):
    """ update emission frequencies """
    if emission_map.has_key(tag):
        if emission_map[tag].has_key(word):
            emission_map[tag][word] = emission_map[tag][word] + 1.0
        else:
            emission_map[tag][word] = 1.0
    else:
        emission_map[tag] = { word:1.0 }

def train(transition_map, emission_map, sentence_list, vocabulary):
    """processes a list of sentences and updates the transition-, observation- and prior frequency"""
    for sentence in sentence_list:
        previous = fileparser.START
        for word, tags in sentence:
            word = fileparser.normalize_word(word)
            vocabulary.add(word)
            tag = tags.split('|')[0]
            increment_transitions(transition_map, tag, previous)
            increment_emissions(emission_map, tag, word)
            previous = tag
        increment_transitions(transition_map, fileparser.END, previous)

def main():
    aa = {}
    bb = {}
    vocabulary = set([])

    file_list = os.listdir(fileparser.resource_path)
    # training
    print('Training...')
    for file in file_list:
        if file.startswith(fileparser.training_prefix):
            training_file = open(fileparser.resource_path + file, 'r')
            sentence_list = fileparser.parse(training_file)
            train(aa, bb, sentence_list, vocabulary)
    print('DONE')
    
    # transform into a and b
    t_start = time.time()
    a = {}
    b = {}
    user_states = list(aa.iterkeys())
    states = list(aa.iterkeys()) + [hmm.START, hmm.END]
    for state in aa.iterkeys():
        sum_counts = sum([aa[state][next_state] for next_state in aa[state].iterkeys()])
        for next_state in states:
            if aa[state].has_key(next_state):
                a[(state, next_state)] = LogProbability(aa[state][next_state]) / sum_counts            
            else:
                a[(state, next_state)] = LogProbability(0.0)
    # Extract vocabulary
    vocab = {}
    for state in bb.iterkeys():
        for output in bb[state].iterkeys():
            vocab[output] = vocab.get(output, 0) + 1
    
    # Create matrix B and apply smoothing
    for state in bb.iterkeys():
        sum_emmited = (sum([bb[state][output] for output in bb[state].iterkeys()]) if bb.has_key(state) else 0)
        b[state] = {}
        for output in vocab.iterkeys():
            b[state][output] = LogProbability(bb.get(state, {}).get(output, 0.0) + 1.0) / (sum_emmited + len(vocab))

    # Calculate average of singleton words
    unknown_b = {}
    singletons = [word for word, count in vocab.iteritems() if count == 1]
    for s in bb.iterkeys():
        sm = LogProbability(0.0)
        for singleton in singletons:
            sm += b[s].get(singleton, LogProbability(0.0))
        b[s][hmm.UNKNOWN] = sm / len(singletons)
    
    print hmm.states(a, b)
    
    def unknown_b_mapper(s, word):
        print 'Could not find word %s in state %s' % (word, s)
        print 'State has: %s' % (list(b[s].iterkeys()))
        assert word not in vocab
        print '**UNKNOWN** %s' % word
        return unknown_b[s]
    
    # computing likelihood
    print('computing likelihood...')
    forward_file = open('forward.txt', 'w')
    for file in file_list:
        if file.startswith(fileparser.test_prefix):
            training_file = open(fileparser.resource_path + file, 'r')
            sentence_list = fileparser.parse(training_file)
            for sentence in sentence_list:
                words = [word for word, tag in sentence]
                words = [(word if word in vocab else hmm.UNKNOWN) for word in words]
                
                forward_table = {}
                backward_table = {}
                forward_p = hmm.forward_algorithm(words, a, b, forward=forward_table)
                backward_p = hmm.backward_algorithm(words, a, b, backward=backward_table)
                forward_file.write('%s\n %s\n %s\n\n' % (words, forward_p.logv, backward_p.logv))
    forward_file.close()
    print('likelihood computed.')
    print 'Took %ds' % (time.time() - t_start)
    
    # computing most likely tag sequencesanc accuracy
    print('computing most likely tag sequence and tagger accuracy...')
    match_count = 0.0
    total_count = 0.0
    for file in file_list:
        if file.startswith(fileparser.test_prefix):
            training_file = open(fileparser.resource_path + file, 'r')
            sentence_list = fileparser.parse(training_file)
            for sentence in sentence_list:
                words = [word for word, tag in sentence]
                words = [(word if word in vocab else hmm.UNKNOWN) for word in words]            
                
                tagger_sequence = hmm.viterbi(words, a, b)
                human_sequence = [tag for word, tag in sentence]
                
                #print tagger_sequence
                #print human_sequence
                #print '----'
                
                # update tagger accuracy information
                for i in range(min(len(human_sequence), len(tagger_sequence))): # because of underflow it is possible that the tag sequences are not equal in length...s
                    if tagger_sequence[i] == human_sequence[i]:
                        match_count = match_count + 1.0
                total_count = total_count + max(len(human_sequence), len(tagger_sequence))
                #print('%s\n%s\nProbability: %f\n' % (human_sequence, tagger_sequence, p))
    print('most likely tag sequence computed.')
    print('accuracy of tagger is: %f' % (match_count / total_count, ))

if __name__ == '__main__':    
    print 'Starting program..'
    main()
