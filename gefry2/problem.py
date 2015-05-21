from numpy import *
from shapely.ops import cascaded_union
from shapely import geometry as g

__all__ = ['Domain', 'Detector', 'Source', 'Problem']

class Domain(object):
    def __init__(self, geometry, sigmas):
        assert len(geometry) == len(sigmas)
        self.geometry = geometry
        self.sigmas = array(sigmas)
        self.NObj = len(self.geometry)

        self.all_obj = cascaded_union(geometry)

        self.dom = g.box(*self.all_obj.bounds)

    def __call__(self, a, b):
        # Simple linear search over the objects
        L = g.LineString([a, b])

        path = zeros(self.NObj)

        for i, obj in enumerate(self.geometry):
            inter = L.intersection(obj)

            if not inter.is_empty:
                path[i] = inter.length

        return exp(-(path * self.sigmas).sum()) # alpha

## Replace these with namedtuples?

class Detector(object):
    def __init__(self, loc, dwell, epsilon, A):
        self.loc = array(loc)
        self.epsilon = epsilon
        self.dwell = dwell
        self.A = A

class Source(object):
    def __init__(self, loc, intensity):
        self.loc = array(loc)
        self.intensity = intensity

class Problem(object):
    def __init__(self, domain, detectors, source, background):
        self.domain = domain
        self.detectors = detectors
        self.source = source # Note: only allowing 1
        self.background = background

        # Cache values
        self.r_d = array([i.loc for i in self.detectors])
        self.A_d = array([i.A for i in self.detectors])
        self.dwell_d = array([i.dwell for i in self.detectors])
        self.epsilon_d = array([i.epsilon for i in self.detectors])

        self.I0 = source.intensity * self.dwell_d * self.epsilon_d / (4 * pi)

        # Compute reference response
        self.nominal = self(uncertain=False)

    def __call__(self, src_hyp=None, uncertain=True):
        if src_hyp is None: # Computing reference values
            source = self.source
        else:
            source = src_hyp

        # vector and distance to source
        r = self.r_d - source.loc
        dr = linalg.norm(r, axis=1)

        # Compute fractional solid angle for detectors
        # 4pi factor already included in I0
        # Assumes distances >> dimension of detector
        omega = self.A_d / (dr ** 2.)

        # Compute attenuation coefficients
        alpha = empty(len(self.detectors))

        for (i, r_i) in enumerate(self.r_d):
            alpha[i] = self.domain(self.source.loc, r_i)

        # Mean detector count rate
        I = self.I0 * alpha * omega

        if uncertain:
            return random.poisson(I) + random.poisson(self.background)
        else:
            return rint(I).astype(int64) + self.background
