﻿import unittest
import math

# ------------------------------
# LogProbability
# ------------------------------
eps = 1e200

def is_finite(val):
    return val not in (float('infinity'), float('-infinity'), float('nan'))

assert(is_finite(2.0))
assert(is_finite(-134234.0))
assert(not is_finite(34e34000))

class LogProbability(object):
    """A floating point probaility stored in log-format, trying to avoid underflow"""
    def __init__(self, value=None):
        if value is not None:
            assert 0.0 <= value <= 1.0, ("Probabilities should be 0.0 <= p <= 1.0 but was %f" % value)
            self.logv = math.log(value)
        else:
            self.logv = None

    @staticmethod
    def convert(obj):
        if type(obj) == LogProbability:
            return obj
        elif type(obj) == float:
            return LogProbability(obj)
        elif type(obj) == int:
            return LogProbability(float(obj))
        else:
            raise NotImplementedError("There's no conversion defined for this type")        
    
    def __add__(self, other):
        assert(self.is_valid())
        assert(other.is_valid())
        x = self.logv
        y = other.logv
        assert(is_finite(x))
        assert(is_finite(y))
        if x < (y - eps):
            return y
        if y < (x - eps):
            return x
        
        diff = x - y
        assert(is_finite(diff))
        assert(is_finite(x))
        assert(is_finite(y))
        
        if not is_finite(math.exp(diff)):
            return x if x > y else y
        
        logv = y + math.log(1.0 + math.exp(diff))
        result = LogProbability()
        result.logv = logv
        return result
        
    def __mul__(self, obj):
        assert(self.is_valid())
        other = LogProbability.convert(obj)
        assert(other.is_valid())
        return LogProbability(self.logv + other.logv)

    def __div__(self, obj):
        assert(self.is_valid())
        other = LogProbability.convert(obj)
        assert(other.is_valid())
        return LogProbability(self.logv - other.logv)
    
    def __sub__(self, obj):
        assert(self.is_valid())
        other = LogProbability.convert(obj)
        assert(other.is_valid())
        raise NotImplementedError()
    
    def __eq__(self, obj):
        other = LogProbability.convert(obj)
        return self.logv == other.logv
    
    def __str__(self):
        return '%f' % math.exp(self.logv)
    
    def is_valid(self):
        return self.logv is not None
    
lf = LogProbability(0.2)
lf2 = LogProbability(0.2)
assert(lf == lf2)
assert(lf + lf2 == 0.4)
assert(lf == 0.2)
print lf    
    
# ------------------------------
# Graph access functions
# ------------------------------
def states(a, b):
    return list(b.iterkeys())

# ------------------------------
# Brute force algorithm
# ------------------------------
def brute_force_algorithm(seq, a, b, start_vector, num=LogProbability):
    def seq_prob(seq, start_state, a, b):
        if len(seq) == 0:
            return 1.0
        result = 0.0
        for next_state in states(a, b):
            result += a[(start_state, next_state)]*b[next_state][seq[0]]*seq_prob(seq[1:], next_state, a, b)
        return result
    
    result = 0.0
    for state in b.iterkeys():
        result += start_vector[state]*b[state][seq[0]]*seq_prob(seq[1:], state, a, b)
    return result

# ------------------------------
# Forward algorithm
# ------------------------------
# Implementation of the forward algorithm, as described in [1]
#
# [1] Speech and language processing, Jurafsky, D. and Martin, J. H.
def forward_algorithm(seq, a, b, start_vector, num=LogProbability):
    # Initialization step
    forward = {}
    for state in states(a, b):
        forward[(state, 1)] = start_vector[state]*b[state][seq[0]]
    	
    # Recursion step
    for t in xrange(2, len(seq)+1):
        for s in states(a, b):
            forward[(s, t)] = sum([forward[s_, t-1]*a[(s_, s)]*b[s][seq[t-1]] for s_ in states(a, b)])
         
    # TODO: implement termination character
    return sum([forward[(s, len(seq))] for s in states(a, b)])

# ------------------------------
# Backward algorithm
# ------------------------------
def backward_algorithm(seq, a, b, start_vector, num=LogProbability):
    # Initialization step
	backward = {}
	T = len(seq)
	for state in states(a, b):
		#backward[(T, state)] = a[(state, F)]
		# TODO: final state
		backward[(T, state)] = 1.0
		
	# Recursion
	for t in xrange(T-1, 0, -1):
		for i in states(a, b):
			backward[(t, i)] = sum([a[(i, j)]*b[j][seq[t]]*backward[(t+1, j)] for j in states(a, b)])

	# Termination
	return sum([start_vector[j]*b[j][seq[0]]*backward[(1, j)] for j in states(a, b)])
	
class TestAlgorithm(object):
    def test_weather(self):
        """Tests the example given in [1] on p214"""
        a = {
            ('HOT', 'COLD'): 0.3, ('HOT', 'HOT'): 0.7,
            ('COLD', 'COLD'): 0.6, ('COLD', 'HOT'): 0.4
        }
        b = {
            'HOT': {'1': 0.2, '2': 0.4, '3': 0.4},
            'COLD': {'1': 0.5, '2': 0.4, '3': 0.1}
        }
        start_vector = {'HOT': 0.8, 'COLD': 0.2}
        p = self.algorithm(['3', '1', '3'], a, b, start_vector)
        expected = (0.8*0.4*(0.7*0.2*(0.7*0.4 + 0.3*0.1) + 0.3*0.5*(0.6*0.1 + 0.4*0.4)) + 
                    0.2*0.1*(0.4*0.2*(0.7*0.4 + 0.3*0.1) + 0.6*0.5*(0.6*0.1 + 0.4*0.4)))
        self.assertEqual(expected, p)

class TestBruteForceAlgorithm(unittest.TestCase, TestAlgorithm):
    def setUp(self):
        self.algorithm = brute_force_algorithm

class TestForwardAlgorithm(unittest.TestCase, TestAlgorithm):
    def setUp(self):
        self.algorithm = forward_algorithm        

class TestBackwardAlgorithm(unittest.TestCase, TestAlgorithm):
    def setUp(self):
        self.algorithm = backward_algorithm        
		
if __name__ == '__main__':
	unittest.main()