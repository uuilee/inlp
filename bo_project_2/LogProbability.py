try:
    import CythonLogProbability
    from CythonLogProbability import LogProbability
except:
    print '*** WARNING *** Could not load cython module, this means that code might be much slower'
    from PythonLogProbability import LogProbability
    
lf = LogProbability(0.2)
lf2 = LogProbability(0.2)
lf3 = LogProbability(0.3)
assert(lf == lf2)
assert(lf != lf3)
assert(lf + lf2 == 0.4)
assert(lf == 0.2)
assert(lf3 > lf)
assert(lf3 >= lf)
assert(lf < lf3)
assert(lf <= lf3)
