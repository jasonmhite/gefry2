import abc
import scipy.stats as st

class BackgroundTermBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        return NotImplemented

    @abc.abstractmethod
    def __call__(self, source):
        return NotImplemented


class PartialRandomField(BackgroundTermBase):
    def __init__(self, hash):
        self.hash = hash

    def __call__(self, source):
        rv = self.hash[source]
        return rv

    @classmethod
    def from_detectors(cls, detectors, dists):
        hash = {det: dist for det, dist in zip(detectors, dists)}

        return cls(hash)

class ConstantPoissonBackground(BackgroundTermBase):
    def __init__(self, B):
        self._dist = st.poisson(B)

    def __call__(self, source):
        return self._dist


# Placeholders for future variations
class GaussianRandomField(BackgroundTermBase): pass
class ArbitraryRandomField(BackgroundTermBase): pass
