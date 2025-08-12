import pymc as pm

from sample import mcModel

pm.model_to_graphviz(mcModel).view()
