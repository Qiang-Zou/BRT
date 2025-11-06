import math
from .sampling import randn_uvgrid
import numpy as np

def chordErrorCheck(t0,t1,t2,eval_fn,tol=10/math.sqrt(101)):
    return chordError(t0,t1,t2,eval_fn)>tol
def chordError(t0,t1,t2,eval_fn):
    dist=eval_fn
    return dist(t0,t2)/(dist(t0,t1)+dist(t1,t2))
