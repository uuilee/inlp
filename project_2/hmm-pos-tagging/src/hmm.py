'''
Created on 20.04.2010
@author: Willy Lai
@contact: laiw@student.ethz.ch
'''

import os

resource_path = '../training/'
test_prefix = ('test', )
training_prefix = ('train', )

line_delimiters = ('*', '\n')
word_delimiters = ('[', ']')
sentence_separator = ('=', )

start = '<s>'
end = '</s>'

a = {} # transition frequency
b = {} # observation frequency
c = {} # prior frequency
vocabulary = set([])

def filter_file(line):
	""" extracts all relevant lines """
	return not line.startswith(line_delimiters)

def filter_line(word):
	""" extracts all word/tag constructs """
	return not (word in word_delimiters)

def filter_states(state):
	return not (state in (start, end))

def update_a(tag, given):
	""" update transition frequencies """
	if a.has_key(given):
		if a[given].has_key(tag):
			a[given][tag] = a[given][tag] + 1
		else:
			a[given][tag] = 1.0
	else:
		a[given] = { tag:1.0 }

def update_b(tag, word):
	""" update observation frequencies """
	if b.has_key(tag):
		if b[tag].has_key(word):
			b[tag][word] = b[tag][word] + 1
		else:
			b[tag][word] = 1
	else:
		b[tag] = { word:1 }

def update_c(tag):
	""" update prior frequencies """
	if c.has_key(tag):
		c[tag] = c[tag] + 1
	else:
		c[tag] = 1

def parse(file):
	""" parses the file in the 'Penn Treebank annotation style for POS tags' into a list of sequences of word/tag constructs """
	sentence_list = []
	sentence = [start]
	line_list = file.readlines()
	line_list = filter(filter_file, line_list)
	sos = False # boolean flag indicating when a sentence has started
	for line in line_list:
		line = filter(filter_line, line)
		if line.startswith(sentence_separator): # sentence separator discovered
			if sos == True: # end of sentence
				sentence.append(end)
				sentence_list.append(sentence)
				sentence = [start]
				sos = False
			else: # start of sentence
				sos = True
		else:
			term_list = line.split()
			for term in term_list:
				vocabulary.add(term)
				sentence.append(term)
	return sentence_list

def train(sentence_list):
	""" processes a list of sentences and updates the transition-, obervation- and prior frequency """
	for sentence in sentence_list:
		previous = sentence[0]
		update_c(previous)
		for term in sentence[1:len(sentence)-1]:
			word = term.split('/')[0]
			tag = term.split('/')[1].split('|')[0]
			update_a(tag, previous)
			update_b(tag, word)
			update_c(tag)
			previous = tag
		update_a(sentence[len(sentence)-1], tag)
						
def forward_algorithm(sentence):
	""" this function implements the forward algorithm
		the implementation follows the book SPEECH and LANGUAGE PROCESSING 2nd edition by Daniel Jurafsky and James H. Martin
		it initializes a matrix of size N + 2 (where N is the number of states) x T (where T is the sentence length -> # time steps)
		additionally it does performs laplace smoothing on the observation in order to consider out-of-vocabulary terms
	"""
	# count unkown words
	unknown_words = set([])
	for word in sentence:
		if not word in vocabulary:
			unknown_words.add(word)
	# create forward probability matrix
	N = len(c) + 1
	T = len(sentence)
	forward = [[0.0]*N]*T
	# initialization step
	for si, s in enumerate(filter(filter_states, c.keys()), 1):
		forward[0][si] = transition_probability(s, start) * laplace_smoothing(len(unknown_words), sentence[1], s)
	# recursion step
	for t in range(2, T):
		for si, s in enumerate(c.keys(), 1):
			for si_, s_ in enumerate(c.keys(), 1):
				forward[t][si] = forward[t][si] + forward[t-1][si_] * transition_probability(s, s_) * laplace_smoothing(len(unknown_words), sentence[t], s)  
	# termination step
	for si, s in enumerate(c.keys(), 1):
		forward[T-1][N-1] = forward[T-1][N-1] + forward[T-1][si] * transition_probability(end, s)
	return forward[T-1][N-1]

def viterby_algorithm(sentence):
	""" this function implements the viterbi algorithm
		the implementation follows the book SPEECH and LANGUAGE PROCESSING 2nd edition by Daniel Jurafsky and James H. Martin
		it initializes a matrix of size N + 2 (where N is the number of states) x T (where T is the sentence length -> # time steps)
		additionally it does performs laplace smoothing on the observation in order to consider out-of-vocabulary terms
	"""
	pass

def transition_probability(e, e_given):
	""" returns the transition probability, if there is no transition or the given element is unknown 0 is returned """
	if a.has_key(e_given):
		if a[e_given].has_key(e):
			return (a[e_given][e] / c[e_given])
	return 0.0
		
def laplace_smoothing(unkown, e, e_given):
	""" performs laplace smoothing by adding 1 to all words in the set of vocabulary and the unknown words encountered in the sentence and increasing the denominator accordingly """ 
	total = len(vocabulary) + unkown
	if b.has_key(e_given):
		if b[e_given].has_key(e):
			return (b[e_given][e] + 1)/(c[e_given] + total)
		else:
			return 1.0/(total)
	else: # tag is unkown
		return 0

if __name__ == '__main__':	
	file_list = os.listdir(resource_path)
	# training
	print('training...')
	for file in file_list:
		if file.startswith(training_prefix):
			training_file = open(resource_path + file, 'r')
			sentence_list = parse(training_file)
			train(sentence_list)
#	file = open('dump.txt', 'w')
#	file.write(str(a))
	print('training done.')
	
	# computing likelihood
	print('computing likelihood...')
	for file in file_list:
		if file.startswith(test_prefix):
			training_file = open(resource_path + file, 'r')
			sentence_list = parse(training_file)
			for sentence in sentence_list:
				print(forward_algorithm(sentence))
	print('likelihood computed.')
			

