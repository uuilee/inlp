﻿import os
import util
import fileparser
from math import log, exp

EMPTY = (0,0)

aa = {} # transition frequency
bb = {} # emission frequency
cc = {} # prior frequency
vocabulary = set([])

def write_dynamic_program(list, file):
	file = open('%s.txt' % (file, ), 'w')
	for sublist in list:
		for i, item in enumerate(filter(fileparser.filter_start_end_states, cc.keys()), 1):
			file.write('%.1f-%s  ' % (sublist[i], item))
		file.write('%.1f  ' % (sublist[45], ))
		file.write('\n')

def update_transitions(transition_map, tag, given):
	""" update transition frequencies """
	if transition_map.has_key(given):
		if transition_map[given].has_key(tag):
			transition_map[given][tag] = transition_map[given][tag] + 1.0
		else:
			transition_map[given][tag] = 1.0
	else:
		transition_map[given] = { tag:1.0 }

def update_emissions(emission_map, tag, word):
	""" update emission frequencies """
	if emission_map.has_key(tag):
		if emission_map[tag].has_key(word):
			emission_map[tag][word] = emission_map[tag][word] + 1.0
		else:
			emission_map[tag][word] = 1.0
	else:
		emission_map[tag] = { word:1.0 }

def update_counts(count_map, tag):
	""" update prior frequencies """
	if count_map.has_key(tag):
		count_map[tag] = count_map[tag] + 1.0
	else:
		count_map[tag] = 1.0

def train(transition_map, emission_map, count_map, sentence_list):
	""" processes a list of sentences and updates the transition-, observation- and prior frequency """
	for sentence in sentence_list:
		previous = fileparser.START
		update_counts(count_map, previous)
		for term in sentence:
			word = term.split('/')[0]
			vocabulary.add(word)
			tag = term.split('/')[1].split('|')[0]
			update_transitions(transition_map, tag, previous)
			update_emissions(emission_map, tag, word)
			update_counts(count_map, tag)
			previous = tag
		update_transitions(transition_map, fileparser.END, previous)

def forward_algorithm(transition_map, emission_map, count_map, sentence):
	""" this function implements the forward algorithm
		the implementation follows the book SPEECH and LANGUAGE PROCESSING 2nd edition by Daniel Jurafsky and James H. Martin
		it initializes a matrix of size N + 2 (where N is the number of states) x T (where T is the sentence length -> # time steps)
		additionally it performs laplace smoothing on the observation in order to consider out-of-vocabulary terms
	"""
	sentence = map(fileparser.map_extract_word, sentence)
	# count unknown words
	unknown_words = set([])
	for word in sentence:
		if not word in vocabulary:
			unknown_words.add(word)
	# create forward probability matrix
	N = len(count_map) + 1
	T = len(sentence)
	forward = [[float('-infinity')]*N for i in range(T)]
	# check if empty sentence
	if len(sentence) == 0:
		return (transition_probability(transition_map, count_map, fileparser.END, fileparser.START), forward)
	# initialization step
	for si, s in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
		if (transition_probability(transition_map, count_map, s, fileparser.START) > 0 and emission_probability(emission_map, count_map, len(unknown_words), sentence[0], s) > 0):
			forward[0][si] = log(transition_probability(transition_map, count_map, s, fileparser.START)) + log(emission_probability(emission_map, count_map, len(unknown_words), sentence[0], s))			
	# recursion step
	for t in range(1, T):
		for si, s in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
			cell_initialized = False # This flag is needed in order to use log scale probabilities. The cell is initialized to 0.0 which is undefined for log.
			for si_, s_ in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
				if not cell_initialized:
					if (forward[t-1][si_] > float('-infinity') and transition_probability(transition_map, count_map, s, s_) > 0 and emission_probability(emission_map, count_map, len(unknown_words), sentence[t], s) > 0):
						forward[t][si] = forward[t-1][si_] + log(transition_probability(transition_map, count_map, s, s_)) + log(emission_probability(emission_map, count_map, len(unknown_words), sentence[t], s))
						cell_initialized = True
				else:
					if (forward[t-1][si_] > float('-infinity') and transition_probability(transition_map, count_map, s, s_) > 0 and emission_probability(emission_map, count_map, len(unknown_words), sentence[t], s) > 0):
						forward[t][si] = forward[t][si] + log(1 + exp((forward[t-1][si_] + log(transition_probability(transition_map, count_map, s, s_)) + log(emission_probability(emission_map, count_map, len(unknown_words), sentence[t], s))) - forward[t][si]))  
	# termination step
	cell_initialized = False
	for si, s in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
		if not cell_initialized:
			if (forward[T-1][si] > float('-infinity') and transition_probability(transition_map, count_map, fileparser.END, s) > 0):
				forward[T-1][N-1] = forward[T-1][si] + log(transition_probability(transition_map, count_map, fileparser.END, s))
				cell_initialized = True
		else:
			if (forward[T-1][si] > float('-infinity') and transition_probability(transition_map, count_map, fileparser.END, s) > 0):
				forward[T-1][N-1] = forward[T-1][N-1] + log(1 + exp((forward[T-1][si] + log(transition_probability(transition_map, count_map, fileparser.END, s))) - forward[T-1][N-1]))
	write_dynamic_program(forward, 'dpf')
	return (forward[T-1][N-1], forward)

def backward_algorithm(transition_map, emission_map, count_map, sentence):
	sentence = map(fileparser.map_extract_word, sentence)
	# count unknown words
	unknown_words = set([])
	for word in sentence:
		if not word in vocabulary:
			unknown_words.add(word)
	# create forward probability matrix
	N = len(count_map) + 1
	T = len(sentence)
	backward = [[float('-infinity')]*N for i in range(T)]
	# check if empty sentence
	if len(sentence) == 0:
		return (transition_probability(transition_map, count_map, fileparser.END, fileparser.START), backward)
	# initialization step
	for si, s in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
		if (transition_probability(transition_map, count_map, fileparser.END, s) > 0):
			backward[T-1][si] = log(transition_probability(transition_map, count_map, fileparser.END, s))
	# recursion step
	for t in reversed(range(0, T-1)):
		for si, s in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
			cell_initialized = False # This flag is needed in order to use log scale probabilities. The cell is initialized to 0.0 which is undefined for log.
			for _si, _s in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
				if not cell_initialized:
					if (backward[t+1][_si] > float('-infinity') and transition_probability(transition_map, count_map, _s, s) > 0 and emission_probability(emission_map, count_map, len(unknown_words), sentence[t+1], _s) > 0):
						backward[t][si] = backward[t+1][_si] + log(transition_probability(transition_map, count_map, _s, s)) + log(emission_probability(emission_map, count_map, len(unknown_words), sentence[t+1], _s))
						cell_initialized = True
				else:
					if (backward[t+1][_si] > float('-infinity') and transition_probability(transition_map, count_map, _s, s) > 0 and emission_probability(emission_map, count_map, len(unknown_words), sentence[t+1], _s) > 0):
						backward[t][si] = backward[t][si] + log(1 + exp((backward[t+1][_si] + log(transition_probability(transition_map, count_map, _s, s)) + log(emission_probability(emission_map, count_map, len(unknown_words), sentence[t+1], _s))) - backward[t][si]))  
	
	# termination step
	cell_initialized = False
	for si, s in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
		if not cell_initialized:
			if (backward[0][si] > float('-infinity') and transition_probability(transition_map, count_map, s, fileparser.START) > 0 and emission_probability(emission_map, count_map, len(unknown_words), sentence[0], s) > 0):
				backward[0][0] = backward[0][si] + log(transition_probability(transition_map, count_map, s, fileparser.START)) + log(emission_probability(emission_map, count_map, len(unknown_words), sentence[0], s))
				cell_initialized = True
		else:
			if (backward[0][si] > float('-infinity') and transition_probability(transition_map, count_map, s, fileparser.START) > 0 and emission_probability(emission_map, count_map, len(unknown_words), sentence[0], s) > 0):
				backward[0][0] = backward[0][0] + log(1 + exp((backward[0][si] + log(transition_probability(transition_map, count_map, s, fileparser.START)) + log(emission_probability(emission_map, count_map, len(unknown_words), sentence[0], s))) - backward[0][0]))
	write_dynamic_program(backward, 'dpb')
	return (backward[0][0], backward)			

def viterbi_algorithm(transition_map, emission_map, count_map, sentence):
	""" this function implements the viterbi algorithm
		the implementation follows the book SPEECH and LANGUAGE PROCESSING 2nd edition by Daniel Jurafsky and James H. Martin
		it initializes a matrix of size N + 2 (where N is the number of states) x T (where T is the sentence length -> # time steps)
		additionally it does performs laplace smoothing on the observation in order to consider out-of-vocabulary terms
	"""
	sentence = map(fileparser.map_extract_word, sentence)
	# count unknown words
	unknown_words = set([])
	for word in sentence:
		if not word in vocabulary:
			unknown_words.add(word)
	# create viterbi probability matrix
	N = len(count_map) + 1
	T = len(sentence)
	viterbi = [[float(-9999)]*N for i in range(T)]
	backpointer = [[EMPTY]*N for i in range(T)]
	tag_sequence = []
	# check if empty sentence
	if len(sentence) == 0:
		return (transition_probability(transition_map, count_map, fileparser.END, fileparser.START), viterbi)
	# initialization step
	for si, s in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
		if (transition_probability(transition_map, count_map, s, fileparser.START) > 0 and emission_probability(emission_map, count_map, len(unknown_words), sentence[0], s) > 0):
			viterbi[0][si] = log(transition_probability(transition_map, count_map, s, fileparser.START)) + log(emission_probability(emission_map, count_map, len(unknown_words), sentence[0], s))
			backpointer[0][si] = EMPTY
	# recursion step
	for t in range(1, T):
		for si, s in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
			current_max = float(-9999)
			for si_, s_ in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
				if (viterbi[t-1][si_] > float(-9999) and transition_probability(transition_map, count_map, s, s_) > 0 and emission_probability(emission_map, count_map, len(unknown_words), sentence[t], s) > 0):
					current_prob = viterbi[t-1][si_] + log(transition_probability(transition_map, count_map, s, s_)) + log(emission_probability(emission_map, count_map, len(unknown_words), sentence[t], s))
					if current_prob > current_max:
						viterbi[t][si] = current_prob
						backpointer[t][si] = (si_, s_)
						current_max = current_prob
	# termination step
	current_max = float(-9999)
	for si, s in enumerate(filter(fileparser.filter_start_end_states, count_map.keys()), 1):
		if (viterbi[T-1][si] > float(-9999) and transition_probability(transition_map, count_map, fileparser.END, s) > 0):
			current_prob = viterbi[T-1][si] + log(transition_probability(transition_map, count_map, fileparser.END, s))
			if current_prob > current_max:
				viterbi[T-1][N-1] = current_prob
				backpointer[T-1][N-1] = (si, s)
				current_max = current_prob
	write_dynamic_program(viterbi, 'dpv')
	# trace back
	pointer = backpointer[T-1][N-1]
	while not pointer == EMPTY:
		tag_sequence.append(pointer[1])
		pointer = backpointer[T-len(tag_sequence)][pointer[0]]
	tag_sequence.reverse()
	return (viterbi[T-1][N-1], viterbi, backpointer, tag_sequence)

def initialize_custom_uniform_hmm(num_states):
	a = {}
	b = {}
	for i in range(num_states):
		a[i] = {}
		b[i] = {}
		for j in range(num_states):
			a[i][j] = 1.0
	return (a, b)

def forward_backward(transition_map, emission_map, count_map):
	# initialize A and B
	
	# TODO: To be done
	pass

def transition_probability(transition_map, count_map, e, e_given):
	""" returns the transition probability, without laplace smoothing """
	if transition_map.has_key(e_given):
		if transition_map[e_given].has_key(e):
			return (transition_map[e_given][e])/(count_map[e_given])
		else:
			return 0.0
	return 0.0
		
def emission_probability(emission_map, count_map, unknown, e, e_given):
	""" performs laplace smoothing by adding 1 to all words in the set of vocabulary and the unknown words encountered in the sentence and increasing the denominator accordingly
		we assume that all out-of-dictionary words in this particular sentence has exactly been counted once.
	"""
	if emission_map.has_key(e_given):
		total = len(vocabulary) + unknown
		if emission_map[e_given].has_key(e):
			return (emission_map[e_given][e] + 1.0)/(count_map[e_given] + total)
		else: # out of dictionary word
			return 1.0/(count_map[e_given] + total)
	else: # tag is unknown
		return 0.0

def run():
	file_list = os.listdir(fileparser.resource_path)
	# training
	print('training...')
	sentences_file = open('sentences.txt', 'w')
	for file in file_list:
		if file.startswith(fileparser.training_prefix):
			training_file = open(fileparser.resource_path + file, 'r')
			sentence_list = fileparser.parse(training_file)
			sentences_file.write('%s\n' % (sentence_list, ))
			train(aa, bb, cc, sentence_list)
	sentences_file.close()
	a_file = open('a.txt', 'w')
	util.prettywrite_nested_map(aa, a_file)
	a_file.close()
	b_file = open('b.txt', 'w')
	util.prettywrite_nested_map(bb, b_file)
	b_file.close()
	c_file = open('c.txt', 'w')
	util.prettywrite_map(cc, c_file)
	c_file.close()
	print('training done.')
	
	# computing likelihood
	print('computing likelihood...')
	forward_file = open('forward.txt', 'w')
	for file in file_list:
		if file.startswith(fileparser.test_prefix):
			training_file = open(fileparser.resource_path + file, 'r')
			sentence_list = fileparser.parse(training_file)
			for sentence in sentence_list:
				(forward_p, forward_table) = forward_algorithm(aa, bb, cc, sentence)
				(backward_p, backward_table) = backward_algorithm(aa, bb, cc, sentence)
				forward_file.write('%s\n %s\n %s\n\n' % (sentence, forward_p, backward_p))
	print('likelihood computed.')
	forward_file.close()
	
	# computing most likely tag sequencesanc accuracy
	print('computing most likely tag sequence and tagger accuracy...')
	viterbi_file = open('viterbi.txt', 'w')
	match_count = 0.0
	total_count = 0.0
	for file in file_list:
		if file.startswith(fileparser.test_prefix):
			training_file = open(fileparser.resource_path + file, 'r')
			sentence_list = fileparser.parse(training_file)
			for sentence in sentence_list:
				(p, viterbi_table, viterbi_backpointer, tagger_sequence) = viterbi_algorithm(aa, bb, cc, sentence)
				human_sequence = map(fileparser.map_extract_tag, sentence)
				# update tagger accuracy information
				for i in range(min(len(human_sequence), len(tagger_sequence))): # because of underflow it is possible that the tag sequences are not equal in length...s
					if tagger_sequence[i] == human_sequence[i]:
						match_count = match_count + 1.0
				total_count = total_count + max(len(human_sequence), len(tagger_sequence))
				viterbi_file.write('%s\n%s\nProbability: %f\n\n' % (human_sequence, tagger_sequence, p))
	print('most likely tag sequence computed.')
	print('accuracy of tagger is: %f' % (match_count / total_count, ))
	viterbi_file.close()
	return (aa, bb, cc)
	# computing new hmm model
	
	
if __name__ == '__main__':	
	run()

