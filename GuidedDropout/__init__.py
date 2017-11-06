# define raw functions for guided dropout and guided dropconnect
from .GuidedDropout import SpecificGDCEncoding, SpecificGDOEncoding

try:
    # define the more advance function, to use with tensorflowHelpers
    from .ANNGuidedDrop import DenseLayerwithGD, ComplexGraphWithGD
except:
    pass