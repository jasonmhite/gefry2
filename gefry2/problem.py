from numpy import *
from shapely.ops import cascaded_union
from shapely import geometry as g
from gefry2 import background_terms

from collections import Iterable

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

        # TODO insert buildup correction here
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
    @classmethod
    def upgrade_problem(cls, old_problem, domain=None, detectors=None, source=None, background=None):
        # Upgrade an existing problem object to the new internals
        return cls(
            domain if domain is not None else old_problem.domain,
            detectors if detectors is not None else old_problem.detectors,
            source if source is not None else old_problem.source,
            background if background is not None else old_problem.background,
        )

    def __init__(self, domain, detectors, source, background):
        self.domain = domain
        self.detectors = detectors
        if isinstance(background, background_terms.BackgroundTermBase):
            self.background = background
        else:
            # Is a scalar
            self.background = background_terms.ConstantPoissonBackground(background)

        # Can pass either one or many sources
        if isinstance(source, Iterable):
            self.source = source
            self._ns = len(self.source)
        else: # Single source
            assert isinstance(source, Source) # Just to make sure
            self.source = [source]
            self._ns = 1

        # Cache values
        self.r_d = array([i.loc for i in self.detectors])
        self.A_d = array([i.A for i in self.detectors])
        self.dwell_d = array([i.dwell for i in self.detectors])
        self.epsilon_d = array([i.epsilon for i in self.detectors])

        self._nd = len(self.detectors)

        self.I0 = [
            i.intensity * self.dwell_d * self.epsilon_d / (4 * pi)
            for i in self.source
        ]

        self._expected_background = around([self.background(d).mean() for d in self.detectors])

        # Compute reference response
        self.nominal = self(uncertain=False)

    def __call__(self, src_hyp=None, uncertain=True, background=True):
        # Note: this allows for computing the response with a different
        # number of sources than actual
        if src_hyp is None: # Computing reference values
            sources = self.source
        else:
            # Again, check for single source and listify it if needed
            if isinstance(src_hyp, Iterable):
                sources = src_hyp
            else:
                assert isinstance(src_hyp, Source)
                sources = [src_hyp]

        I = zeros(self._nd)

        for s_i, source in enumerate(sources):
            # vector and distance to source
            r = self.r_d - source.loc
            dr = linalg.norm(r, axis=1)

            # Compute fractional solid angle for detectors
            # 4pi factor already included in I0
            # Assumes distances >> dimension of detector
            omega = self.A_d / (dr ** 2.)

            alpha = empty(self._nd)

            for (i, r_i) in enumerate(self.r_d):
                alpha[i] = self.domain(source.loc, r_i)

            # Mean detector count rate
            Ip = self.I0[s_i] * alpha * omega

            if uncertain:
                I += random.poisson(Ip)
            else:
                I += rint(Ip).astype(int64)


            if background and uncertain:
                v = around(array([self.background(d).rvs() for d in self.detectors]))
                I += v
            elif background and not uncertain:
                I += self._expected_background

        return I
