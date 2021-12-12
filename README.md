# JCHamSim_Py
Python Code to calculate the response of a Dispersive Jaynes-Cummings Hamiltonian at the Strongly Dispersive Regime. Requires NumPy, SciPy, and Matplotlib.
Also requires QuTiP (Quantum Toolbox in Python) for comparison and benchmarking.


The code was part of a Term Project for PH-354 'Computational Physics' course at the Indian Institute of Science during Spring 2020.
The aim of the project is to determine the response of the dispersive Jaynes-Cummings Hamiltonian under a Strong Drive using the method of Quantum Trajectory Simulations. Time evolution of the system is performed using Eigenvalue Decomposition of the time-dependent Hamiltonian.
The calculations are also performed using 'Rotating Wave Approximation' (RWA) for comparison purposes.

'QuTiP', an open-source Python-based library for simulating Open Quantum Systems is used for benchmarking performance and accuracy.


Based on the following paper:

Lev S. Bishop, Eran Ginossar, S. M. Girvin, *Response of the Strongly-Driven Jaynes-Cummings Oscillator*, Phys. Rev. Lett. 105, 100505.

### About the Repository
This Repository is purely for **Archival Purposes**. It mostly consists of Jupyter Notebooks containing results of calculations, all importing routines in `qu_eig.py`.
Project Report and Presentation also present for reference. Data generated from the calculations is not stored in this repo.
