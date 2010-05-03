import unittest
import math
import itertools
import random
import copy

from LogProbability import LogProbability
    
START = '<s>'
END = '</s>'
    
# ------------------------------
# Graph access functions
# ------------------------------
def states(a, b):
    return list(b.iterkeys())

def log_graph(a, b):
    """Given a graph (a, b), returns the same graph with all floating point values replaced with LogProbability"""
    a_ = {}
    for key in a.iterkeys():
        a_[key] = LogProbability(a[key])
        
    b_ = {}
    for x in b.iterkeys():
        b_[x] = {}
        for y in b[x].iterkeys():
            b_[x][y] = LogProbability(b[x][y])
            
    return (a_, b_)
    
# ------------------------------
# Brute force algorithm
# ------------------------------
def brute_force_algorithm(seq, a, b, num=LogProbability):
    #a, b = log_graph(a, b)
    def seq_prob(seq, start_state, a, b):
        if len(seq) == 0:
            return a.get((start_state, END), 0.0)
        result = num(0.0)
        for next_state in states(a, b):
            result += a[(start_state, next_state)]*b[next_state][seq[0]]*seq_prob(seq[1:], next_state, a, b)
        return result
    
    result = num(0.0)
    for state in b.iterkeys():
        result += seq_prob(seq[1:], state, a, b) * b[state][seq[0]] * a.get((START, state), 0.0)
    return result

# ------------------------------
# Forward algorithm
# ------------------------------
# Implementation of the forward algorithm, as described in [1]
#
# [1] Speech and language processing, Jurafsky, D. and Martin, J. H.
def forward_algorithm(seq, a, b, num=LogProbability, forward={}):
    # Initialization step
    T = len(seq)
    forward.clear()
    for state in states(a, b):
        forward[(state, 1)] = a[(START, state)]*b[state][seq[0]]
    	
    # Recursion step
    for t in xrange(2, len(seq)+1):
        for s in states(a, b):
            forward[(s, t)] = sum([forward[s_, t-1]*a[(s_, s)]*b[s][seq[t-1]] for s_ in states(a, b)], num(0))
         
    forward[END, T] = sum([a[s, END]*forward[(s, T)] for s in states(a, b)], num(0))
    return forward[END, T]

# ------------------------------
# Backward algorithm
# ------------------------------
def backward_algorithm(seq, a, b, num=LogProbability, backward={}):
    # Initialization step
    backward.clear()
    T = len(seq)
    for state in states(a, b):
        backward[(T, state)] = a[(state, END)]
        
    # Recursion
    for t in xrange(T-1, 0, -1):
        for i in states(a, b):
            backward[(t, i)] = sum([a[(i, j)]*b[j][seq[t]]*backward[(t+1, j)] for j in states(a, b)], num(0))
    
    # Termination
    backward[(T, END)] = sum([a[(START, j)]*b[j][seq[0]]*backward[(1, j)] for j in states(a, b)], num(0))
    return backward[(T, END)]
	
class TestAlgorithm(object):
    def test_weather(self):
        """Tests the example given in [1] on p214"""
        a = {
		    (START, 'HOT'): 0.8, (START, 'COLD'): 0.2,
            ('HOT', 'COLD'): 0.2, ('HOT', 'HOT'): 0.7, ('HOT', END): 0.1,
            ('COLD', 'COLD'): 0.5, ('COLD', 'HOT'): 0.4, ('COLD', END): 0.1
        }
        b = {
            'HOT': {'1': 0.2, '2': 0.4, '3': 0.4},
            'COLD': {'1': 0.5, '2': 0.4, '3': 0.1}
        }
        a, b = log_graph(a, b)
        p = self.algorithm(['3', '1', '3'], a, b)
        expected = (1*8*4*(7*2*(7*4 + 2*1) + 2*5*(5*1 + 4*4)) + 
                    1*2*1*(4*2*(7*4 + 2*1) + 5*5*(5*1 + 4*4))) / 10000000.0
        self.assertAlmostEqual(expected, float(p))

class TestBruteForceAlgorithm(unittest.TestCase, TestAlgorithm):
    def setUp(self):
        self.algorithm = brute_force_algorithm

class TestForwardAlgorithm(unittest.TestCase, TestAlgorithm):
    def setUp(self):
        self.algorithm = forward_algorithm        

class TestBackwardAlgorithm(unittest.TestCase, TestAlgorithm):
    def setUp(self):
        self.algorithm = backward_algorithm        

# ------------------------------
# Forward backward algorithm
# ------------------------------
def forward_backward(observations, vocabulary, state_set, num=LogProbability):
    # Choose initial values for a and b
    a = {}
    for s in state_set:
        a[(START, s)] = num(1.0) / len(state_set)
        a[(s, END)] = num(1.0 / (len(state_set) + 1))
    print a
    
    for i, j in itertools.product(state_set, state_set):
        a[(i, j)] = num(1.0 / (len(state_set) + 1))
    
    b = {}
    for s in state_set:
        b[s] = {}
        for w in vocabulary:
            b[s][w] = num(1.0 / len(vocabulary))
        
    T = len(observations)
    
    print state_set
    print '***'
    
    changed = True
    while changed:
        print '---'
        changed = False
        
        # E-step
        alpha = {}
        beta = {}
        forward_algorithm(observations, a, b, forward=alpha)
        backward_algorithm(observations, a, b, backward=beta)
        
        impl_states = state_set + [START, END]
        
        y = {}
        e = {}
        
        N = state_set[-1]
        
        for t, j in itertools.product(xrange(1, T+1), state_set):
            y[(t, j)] = alpha[(j, t)]*beta[(t, j)] / beta[(T, END)]
        
        for t, i, j in itertools.product(xrange(1, T-1), state_set, state_set):
            e[(t, i, j)] = (alpha[(i, t)]*a[(i, j)]*b[j][observations[t]]*beta[(t+1, j)]) / alpha[(N, T)]
        
        # M-step
        a_ = copy.copy(a)
        b_ = copy.copy(b)
        #print e
        for i, j in itertools.product(state_set, state_set):
            c = sum([e[(t, i, j)] for t in xrange(1, T-1)], num(0.0))
            d = sum([e[(t, i, j)] for j in state_set for t in xrange(1, T-1)], num(0.0))
            a_[(i, j)] = c / d
            
        for j in b.iterkeys():
            b_[j] = {}
            for vk in b[j].iterkeys():
                c = sum([y[(t, j)] for t in xrange(1, T) if observations[t-1] == vk], num(0.0))
                d = sum([y[(t, j)] for t in xrange(1, T)], num(0.0))
                b_[j][vk] = c / d
        
        
        #Compare changes
        sqr = 0.0
        for key in a.iterkeys():
            if key not in a_:
                continue
            sqr += (float(a[key]) - float(a_[key]))**2
        print sqr      

        if sqr > 0.0001:
            changed = True
        
        
        a, b = a_, b_
        
        
            
            
        
    print sum([1 for x in range(3) for y in range(3)])
        
    return (a, b)
		
if __name__ == '__main__':
    #unittest.main()
    a, b = forward_backward(['a', 'b', 'c', 'a', 'a', 'a', 'b', 'c', 'a', 'a', 'b', 'b', 'c', 'a', 'b', 'c', 'a'], ['a', 'b', 'c'], ['X', 'Y', 'Z'])
    print a
    print b