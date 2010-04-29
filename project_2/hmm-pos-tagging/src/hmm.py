'''
Created on 20.04.2010
@author: Willy Lai
@contact: laiw@student.ethz.ch
'''

import os
from math import log, exp

resource_path = '../debug/'
test_prefix = ('test', )
training_prefix = ('train', )

line_delimiters = ('*', '\n', '=')
word_delimiters = ('[', ']')
sentence_separator = ('.')

START = '<s>'
END = '</s>'
EMPTY = (0,0)

a = {} # transition frequency
b = {} # observation frequency
c = {} # prior frequency
vocabulary = set([])

def filter_unused_lines(line):
	""" extracts all relevant lines """
	return not line.startswith(line_delimiters)

def filter_unused_strings(string):
	""" extracts all word/tag constructs """
	return not (string in word_delimiters)

def filter_start_end_states(state):
	return not (state in (START, END))

def map_extract_word(term):
	return term.split('/')[0]

def map_extract_tag(term):
	return term.split('/')[1].split('|')[0]

def update_a(tag, given):
	""" update transition frequencies """
	if a.has_key(given):
		if a[given].has_key(tag):
			a[given][tag] = a[given][tag] + 1.0
		else:
			a[given][tag] = 1.0
	else:
		a[given] = { tag:1.0 }

def update_b(tag, word):
	""" update observation frequencies """
	if b.has_key(tag):
		if b[tag].has_key(word):
			b[tag][word] = b[tag][word] + 1.0
		else:
			b[tag][word] = 1.0
	else:
		b[tag] = { word:1.0 }

def update_c(tag):
	""" update prior frequencies """
	if c.has_key(tag):
		c[tag] = c[tag] + 1.0
	else:
		c[tag] = 1.0

def show(list):
	for item in list:
		print('%s' % (item, ))
	print '\n'

def parse(file):
	""" parses the file in the 'Penn Treebank annotation style for POS tags' into a list of sequences of word/tag constructs """
	line_list = file.readlines()
	line_list = filter(filter_unused_lines, line_list)
	sentence_list = []
	sentence = []
	for line in line_list:
		line = filter(filter_unused_strings, line)
		term_list = line.split()
		for term in term_list:
			sentence.append(term)
			if term.startswith(sentence_separator):
				sentence_list.append(sentence)
				sentence = []
	return sentence_list

def train(sentence_list):
	""" processes a list of sentences and updates the transition-, observation- and prior frequency """
	for sentence in sentence_list:
		previous = START
		update_c(previous)
		for term in sentence:
			word = term.split('/')[0]
			vocabulary.add(word)
			tag = term.split('/')[1].split('|')[0]
			update_a(tag, previous)
			update_b(tag, word)
			update_c(tag)
			previous = tag
		update_a(END, previous)
						
def forward_algorithm(sentence):
	""" this function implements the forward algorithm
		the implementation follows the book SPEECH and LANGUAGE PROCESSING 2nd edition by Daniel Jurafsky and James H. Martin
		it initializes a matrix of size N + 2 (where N is the number of states) x T (where T is the sentence length -> # time steps)
		additionally it performs laplace smoothing on the observation in order to consider out-of-vocabulary terms
	"""
	sentence = map(map_extract_word, sentence)
	# count unknown words
	unknown_words = set([])
	for word in sentence:
		if not word in vocabulary:
			unknown_words.add(word)
	# create forward probability matrix
	N = len(c) + 1
	T = len(sentence)
	forward = [[0.0]*N for i in range(T)]
	# check if empty sentence
	if len(sentence) == 0:
		return (transition_probability(END, START), forward)
	# initialization step
	for si, s in enumerate(filter(filter_start_end_states, c.keys()), 1):
		if (transition_probability(s, START) > 0 and observation_probability(len(unknown_words), sentence[0], s) > 0):
			forward[0][si] = log(transition_probability(s, START)) + log(observation_probability(len(unknown_words), sentence[0], s))			
	# recursion step
	for t in range(1, T):
		for si, s in enumerate(filter(filter_start_end_states, c.keys()), 1):
			cell_initialized = False # This flag is needed in order to use log scale probabilities. The cell is initialized to 0.0 which is undefined for log.
			for si_, s_ in enumerate(c.keys(), 1):
				if not cell_initialized:
					if (forward[t-1][si_] != 0 and transition_probability(s, s_) > 0 and observation_probability(len(unknown_words), sentence[t], s) > 0):
						forward[t][si] = forward[t-1][si_] + log(transition_probability(s, s_)) + log(observation_probability(len(unknown_words), sentence[t], s))
						cell_initialized = True
				else:
					if (forward[t-1][si_] != 0 and transition_probability(s, s_) > 0 and observation_probability(len(unknown_words), sentence[t], s) > 0):
						forward[t][si] = forward[t][si] + log(1 + exp((forward[t-1][si_] + log(transition_probability(s, s_)) + log(observation_probability(len(unknown_words), sentence[t], s))) - forward[t][si]))  
					
	show(forward)
	# termination step
	cell_initialized = False
	for si, s in enumerate(filter(filter_start_end_states, c.keys()), 1):
		if not cell_initialized:
			if (forward[T-1][si] != 0 and transition_probability(END, s) > 0):
				forward[T-1][N-1] = forward[T-1][si] + log(transition_probability(END, s))
		else:
			if (forward[T-1][si] != 0 and transition_probability(END, s) > 0):
				forward[T-1][N-1] = forward[T-1][N-1] + log(1 + exp((forward[T-1][si] + log(transition_probability(END, s))) - forward[T-1][N-1]))
	show(forward)
	return (forward[T-1][N-1], forward)

def viterbi_algorithm(sentence):
	""" this function implements the viterbi algorithm
		the implementation follows the book SPEECH and LANGUAGE PROCESSING 2nd edition by Daniel Jurafsky and James H. Martin
		it initializes a matrix of size N + 2 (where N is the number of states) x T (where T is the sentence length -> # time steps)
		additionally it does performs laplace smoothing on the observation in order to consider out-of-vocabulary terms
	"""
	sentence = map(map_extract_word, sentence)
	# count unknown words
	unknown_words = set([])
	for word in sentence:
		if not word in vocabulary:
			unknown_words.add(word)
	# create viterbi probability matrix
	N = len(c) + 1
	T = len(sentence)
	viterbi = [[0.0]*N for i in range(T)]
	backpointer = [[EMPTY]*N for i in range(T)]
	tag_sequence = []
	# check if empty sentence
	if len(sentence) == 0:
		return (transition_probability(END, START), viterbi)
	# initialization step
	for si, s in enumerate(filter(filter_start_end_states, c.keys()), 1):
		if (transition_probability(s, START) != 0 and observation_probability(len(unknown_words), sentence[0], s) > 0):
			viterbi[0][si] = log(transition_probability(s, START)) + log(observation_probability(len(unknown_words), sentence[0], s))
			backpointer[0][si] = EMPTY
	# recursion step
	for t in range(1, T):
		for si, s in enumerate(filter(filter_start_end_states, c.keys()), 1):
			current_max = float('-infinity')
			for si_, s_ in enumerate(c.keys(), 1):
				if (viterbi[t-1][si_] != 0 and transition_probability(s, s_) > 0 and observation_probability(len(unknown_words), sentence[t], s) > 0):
					current_prob = viterbi[t-1][si_] + log(transition_probability(s, s_)) + log(observation_probability(len(unknown_words), sentence[t], s))
					if current_prob > current_max:
						viterbi[t][si] = current_prob
						backpointer[t][si] = (si_, s_)
						current_max = current_prob
	# termination step
	current_max = float('-infinity')
	for si, s in enumerate(filter(filter_start_end_states, c.keys()), 1):
		if (viterbi[T-1][si] != 0 and transition_probability(END, s) > 0):
			current_prob = viterbi[T-1][si] * transition_probability(END, s)
			if current_prob > current_max:
				viterbi[T-1][N-1] = current_prob
				backpointer[T-1][N-1] = (si, s)
				current_max = current_prob
	# trace back
	pointer = backpointer[T-1][N-1]
	while not pointer == EMPTY:
		tag_sequence.append(pointer[1])
		pointer = backpointer[T-len(tag_sequence)][pointer[0]]
	tag_sequence.reverse()
	return (viterbi[T-1][N-1], viterbi, backpointer, tag_sequence)

def transition_probability(e, e_given):
	""" returns the transition probability, if there is no transition or the given element is unknown 0 is returned """
	if a.has_key(e_given):
		if a[e_given].has_key(e):
			return (a[e_given][e] / c[e_given])
	return 0.0
		
def observation_probability(unknown, e, e_given):
	""" performs laplace smoothing by adding 1 to all words in the set of vocabulary and the unknown words encountered in the sentence and increasing the denominator accordingly """ 
	total = len(vocabulary) + unknown
	if b.has_key(e_given):
		if b[e_given].has_key(e):
			return (b[e_given][e] + 1.0)/(c[e_given] + total)
		else: # out of dictionary word
			return 1.0/(total)
	else: # tag is unknown
		print('!!!!')
		return 0

if __name__ == '__main__':	
	file_list = os.listdir(resource_path)
	# training
	print('training...')
	sentences_file = open('sentences.txt', 'w')
	for file in file_list:
		if file.startswith(training_prefix):
			training_file = open(resource_path + file, 'r')
			sentence_list = parse(training_file)
			sentences_file.write('%s\n' % (sentence_list, ))
			train(sentence_list)
	sentences_file.close()
	hmm_file = open('hmm.txt', 'w')
	hmm_file.write('%s\n\n%s\n\n%s' % (str(a), str(b), str(c)))
	hmm_file.close()
	print('training done.')
	
	# computing likelihood
	print('computing likelihood...')
	forward_file = open('forward.txt', 'w')
	for file in file_list:
		if file.startswith(test_prefix):
			training_file = open(resource_path + file, 'r')
			sentence_list = parse(training_file)
			for sentence in sentence_list:
				(p, foward_table) = forward_algorithm(sentence)
				forward_file.write('%s \n %s \n\n' % (sentence, p))
	print('likelihood computed.')
	forward_file.close()
	
	# computing most likely tag sequences
	print('computing most likely tag sequence...')
	viterbi_file = open('viterbi.txt', 'w')
	match_count = 0.0
	total_count = 0.0
	for file in file_list:
		if file.startswith(test_prefix):
			training_file = open(resource_path + file, 'r')
			sentence_list = parse(training_file)
			for sentence in sentence_list:
				(p, viterbi_table, viterbi_backpointer, tagger_sequence) = viterbi_algorithm(sentence)
				human_sequence = map(map_extract_tag, sentence)
				for i in range(min(len(human_sequence), len(tagger_sequence))): # because of underflow it is possible that the tag sequences are not equal in length...s
					if tagger_sequence[i] == human_sequence[i]:
						match_count = match_count + 1.0
				total_count = total_count + max(len(human_sequence), len(tagger_sequence))
				viterbi_file.write('%s\n%s\nProbability: %f\n\n' % (human_sequence, tagger_sequence, p))
	print('most likely tag sequence computed.')
	print('accuracy of tagger is: %f' % (match_count / total_count, ))
	viterbi_file.close()
#	print(c.keys())
#	print(len(c))
			

