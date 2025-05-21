from settings import *
from model_class_builder import *
import numpy as np
from math import sqrt
import openseespy.opensees as ops
import opsvis as opsv
import matplotlib
import matplotlib.pyplot as plt

# Building Dimension
numBayX = 3
numBayY = 3
numFloor = 3
bayWidthX = 4.0 * m
bayWidthY = 4.0 * m
storyHeight = 3 * m

# Starting an instance of OpenSees Model
ModelA = Model(numBayX, numBayY, bayWidthX, bayWidthY, numFloor, storyHeight)
nodes = ModelA.make_nodes()
columns = ModelA.build_columns(650 * mm, 650 * mm)
beams = ModelA.build_beams(230 * mm, 425 * mm)
ModelA.apply_rigid_diaphragm_constraint()
ModelA.distribute_the_mass()
ModelA.distribute_the_self_weight()  # Distributed to all nodes of a particular floor
ModelA.eigen_values(6)

ModelA.run_static_analysis(Z=0.36, I=1, R=5)  # Applied to center of mass node
print("\n Output Results: \n")
ModelA.print_results()

# ModelA.perform_rsa(Z=0.36, I=1, R=5, num_modes=6)
