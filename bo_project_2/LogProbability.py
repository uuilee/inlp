# ------------------------------
# LogProbability
# ------------------------------

import math

eps = 1e200

def is_finite(val):
    return val not in (float('infinity'), float('-infinity'), float('nan'))

assert(is_finite(2.0))
assert(is_finite(-134234.0))
assert(not is_finite(34e34000))

class LogProbability(object):
    """A floating point probaility stored in log-format, trying to avoid underflow"""
    def __init__(self, value=None, logarithmic=False):
        if value is not None:
            if not logarithmic:
                #assert 0.0 <= value <= 1.0, ("Probabilities should be 0.0 <= p <= 1.0 but was %f" % value)
                if value != 0.0:
                    self.logv = math.log(value)
                else:
                    self.logv = float('-infinity')
            else:
                self.logv = value
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
        
        if x == float('-infinity'):
            return LogProbability(y, logarithmic=True)
            
        if y == float('-infinity'):
            return LogProbability(x, logarithmic=True)
        
        assert(is_finite(x))
        assert(is_finite(y))
        if x < (y - eps):
            return LogProbability(y, logarithmic=True)
        if y < (x - eps):
            return LogProbability(x, logarithmic=True)
        
        diff = x - y
        assert(is_finite(diff))
        assert(is_finite(x))
        assert(is_finite(y))
        
        if not is_finite(math.exp(diff)):
            return x if x > y else y
        
        logv = y + math.log(1.0 + math.exp(diff))
        return LogProbability(logv, logarithmic=True)        
        
    def __mul__(self, obj):
        assert(self.is_valid())
        other = LogProbability.convert(obj)
        assert(other.is_valid())
        result = LogProbability()
        result.logv = self.logv + other.logv 
        return result

    def __div__(self, obj):
        assert(self.is_valid())
        other = LogProbability.convert(obj)
        assert(other.is_valid())
        result = LogProbability()
        result.logv = self.logv - other.logv 
        return result
    
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
    
    def __repr__(self):
        return self.__str__()
    
    def __float__(self):
        return math.exp(self.logv)
    
    def is_valid(self):
        return self.logv is not None
    
lf = LogProbability(0.2)
lf2 = LogProbability(0.2)
assert(lf == lf2)
assert(lf + lf2 == 0.4)
assert(lf == 0.2)
print lf    