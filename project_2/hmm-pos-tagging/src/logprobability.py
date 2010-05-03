# ------------------------------
# LogProbability
# ------------------------------

import math

eps = 1e200

inf = float('infinity')
neg_inf = float('-infinity')
nan = float('nan')

def is_finite(val):
    return val not in (inf, neg_inf, nan)

class LogProbability(object):
    """A floating point probaility stored in log-format, trying to avoid underflow"""
    def __init__(self, value=None, logarithmic=False):
        if value is not None:
            if not logarithmic:
                #assert 0.0 <= value <= 1.0, ("Probabilities should be 0.0 <= p <= 1.0 but was %f" % value)
                if value != 0.0:
                    self.logv = math.log(value)
                else:
                    self.logv = neg_inf
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
    
    def __add__(self, obj):
        other = LogProbability.convert(obj)
    
        assert(self.is_valid())
        assert(other.is_valid())
        x = self.logv
        y = other.logv
        
        if x == neg_inf:
            return other
            
        if y == neg_inf:
            return self
        
        assert(is_finite(x))
        assert(is_finite(y))
        if x < (y - eps):
            return other
        if y < (x - eps):
            return self
        
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
        return 'E%f' % self.logv
    
    def __repr__(self):
        return self.__str__()
    
    def __float__(self):
        return math.exp(self.logv)
        
    def __cmp__(self, obj):
        other = LogProbability.convert(obj)
        return cmp(self.logv, other.logv)
    
    def is_valid(self):
        return self.logv is not None
    
    
