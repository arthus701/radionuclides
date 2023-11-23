import pymc as pm

# from arch_pyMC import mcModel
from radio_pyMC import mcModel

pm.model_to_graphviz(mcModel).view()
