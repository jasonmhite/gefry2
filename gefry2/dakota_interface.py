import numpy as np
import os
import cPickle as pickle

from gefry2 import Source
from gefry2.util import chunk_list

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

def respFnN(**kwargs):
    try:
        n_var = kwargs['variables']
        n_resp = kwargs['functions']
        eid = kwargs['currEvalId']

        var_list = kwargs['av']
        assert(len(var_list) % 3 == 0)

        sources = [
            Source((X, Y), I)
            for (X, Y, I) in chunk_list(var_list, 3)
        ]

        resp = P(sources, uncertain=False).astype(np.float64)

        print('{}: {}'.format(eid, resp))
        return {'fns': resp, 'failure': 0}

    except:
        return {'fns': np.zeros(n_resp), 'failure': 1}
