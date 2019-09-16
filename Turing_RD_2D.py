"""
@author: Amit Landge
This script is used to simulate the activator-inhibitor system from Alan Turing's 1952 paper.
"""
from fipy import *
import numpy as np
from matplotlib import pyplot as plt

#First, I define the mesh
nx=300
dx=0.01
mesh = Grid2D(nx=nx, ny=nx, dx=dx, dy=dx)

#Define initial noise
noise1 = GaussianNoiseVariable(mesh = mesh, mean = 0, variance = 0.001)
noise2 = GaussianNoiseVariable(mesh = mesh, mean = 0, variance = 0.001)

#Define cell variables ()
act = CellVariable(name="X", mesh=mesh, value=0.5+ noise1)
inh = CellVariable(name="Y", mesh=mesh, value=0.5+ noise2)

#Define diffusion parameters
Da= 0.005
Di= 0.03

#Define the equations
eqAct = TransientTerm(var=act) == DiffusionTerm(coeff=Da, var=act) +1 + 5*act - 6*inh
eqInh = TransientTerm(var=inh) == DiffusionTerm(coeff=Di, var=inh) +1 + 6*act - 7*inh
eq = eqAct & eqInh

#Define Zero-flux Boundary Conditions.
act.faceGrad.constrain(0., where= mesh.exteriorFaces)
inh.faceGrad.constrain(0., where= mesh.exteriorFaces)
dt = 0.002
steps = int(8/dt)
# print steps

#Define viewer
viewerAct = MatplotlibViewer(vars=(act,), datamin=0, datamax=2, cmap =plt.get_cmap('gist_yarg'))
viewerInh = MatplotlibViewer(vars=(inh,), datamin=0, datamax=2, cmap =plt.get_cmap('gist_yarg'))

#Loop to solve each parameter case.
for step in range(steps):
    eq.solve(dt = dt)
    #nake and save plots every 100 steps
    if step%100==0:
        viewerAct.plot("Plots2D/Sim3/plotAct_%st.pdf"%(step*dt))
        viewerInh.plot("Plots2D/Sim3/plotInh_%st.pdf"%(step*dt))
        TSVViewer(vars=(act,inh)).plot(filename="Plots2D/Sim3/dataActInh_%st.tsv"%(step*dt))
        #Track progress of the simulation
        print("Time passed= %s"%(dt*step))
