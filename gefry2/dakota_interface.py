import numpy as np
import os
import cPickle as pickle

from gefry2 import Source

try:
    PSPEC = os.environ['GEFRY_PSPEC']
except KeyError:
    print("$GEFRY_PSPEC not defined, falling back to default ./spec.pkl")
    PSPEC = './spec.pkl'

with open(PSPEC) as f:
    P, _, _ = pickle.load(f)

def respFn(**kwargs):
    try:
        n_var = kwargs['variables']
        n_resp = kwargs['functions']
        eid = kwargs['currEvalId']

        X, Y, I = kwargs['av']

        src = Source((X, Y), I)
        resp = P(src, uncertain=False).astype(np.float64)

        print('{}: '.format(eid), resp)

        return {'fns': resp, 'failure': 0}

    except:
        return {'fns': np.zeros(n_resp), 'failure': 1}
