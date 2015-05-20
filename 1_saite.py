from numpy import *
from scipy.linalg import eig, eigh
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
from matplotlib.pyplot import *

# Homogeneous case

L = 2
N = 513

#################################
#                               #
# TODO: Unteraufgaben a) bis f) #
#                               #
#################################


# Inhomogeneous case

drho = 0.99
rho1 = 1 - drho
rho2 = 1 + drho

#########################
#                       #
# TODO: Unteraufgabe g) #
#                       #
#########################
