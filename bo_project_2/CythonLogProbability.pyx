# ------------------------------
# LogProbability
# ------------------------------

import math

eps = 1e200

cdef double inf, neg_inf, nan

inf = float('infinity')
neg_inf = float('-infinity')
nan = float('nan')

cdef bool is_finite(double val):
    return val not in (inf, neg_inf, nan)

assert(is_finite(2.0))
assert(is_finite(-134234.0))
assert(not is_finite(34e34000))    
    
cdef class LogProbability(object):
    """A floating point probaility stored in log-format, trying to avoid underflow"""
    cdef public double logv
    
    def __init__(LogProbability self, value=None, bool logarithmic=False):
        if value is not None:
            if not logarithmic:
                #assert 0.0 <= value <= 1.0, ("Probabilities should be 0.0 <= p <= 1.0 but was %f" % value)
                if abs(value) != 0.0:
                    self.logv = math.log(value)
                else:
                    self.logv = neg_inf
            else:
                self.logv = value
        else:
            raise NotImplementedError("Currently, None is not supported")

    def convert(LogProbability self, obj):
        if type(obj) == LogProbability:
            return obj
        elif type(obj) == float:
            return LogProbability(obj)
        elif type(obj) == int:
            return LogProbability(float(obj))
        else:
            raise NotImplementedError("There's no conversion defined for this type: %s" % type(obj))        
    
    def __add__(LogProbability self, obj):
        other = self.convert(obj)
    
        assert(self.is_valid())
        assert(other.is_valid())
        
        cdef double x, y
        
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
        
        cdef double logv
        
        logv = y + math.log(1.0 + math.exp(diff))
        return LogProbability(logv, logarithmic=True)        
        
    def __mul__(LogProbability self, obj):
        assert(self.is_valid())
        other = self.convert(obj)
        assert(other.is_valid())
        return LogProbability(self.logv + other.logv, logarithmic=True)

    def __div__(LogProbability self, obj):
        assert(self.is_valid())
        other = self.convert(obj)
        assert(other.is_valid())
        return LogProbability(self.logv - other.logv, logarithmic=True)
    
    def __sub__(LogProbability self, obj):
        assert(self.is_valid())
        other = self.convert(obj)
        assert(other.is_valid())
        raise NotImplementedError()
    
    #def __eq__(LogProbability self, obj, int op):
    #    other = self.convert(obj)
    #    return self.logv == other.logv
    
    def __str__(LogProbability self):
        return 'E%f' % self.logv
    
    def __repr__(LogProbability self):
        return self.__str__()
    
    def __float__(LogProbability self):
        return math.exp(self.logv)
        
    def __richcmp__(LogProbability self, obj, int op):
        cdef LogProbability other
        other = self.convert(obj)
        if op == 0: # <
            return self.logv < other.logv
        elif op == 1: # <=
            return self.logv <= other.logv
        elif op == 2: # ==
            return self.logv == other.logv
        elif op == 3: # !=
            return self.logv != other.logv
        elif op == 4: # >
            return self.logv > other.logv
        elif op == 5: # >=
            return self.logv >= other.logv
        
    def __cmp__(LogProbability self, obj):
        other = self.convert(obj)
        return cmp(self.logv, other.logv)
    
    def is_valid(LogProbability self):
        return self.logv is not None