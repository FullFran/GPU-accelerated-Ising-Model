GPU-Accelerated Ising Model Simulation
This repository contains an implementation of the 3D Ising model using GPU acceleration to improve computational efficiency. The Ising model is a fundamental model in statistical mechanics, widely used to study phase transitions and critical phenomena in condensed matter physics.


The Ising model describes a system of discrete spins arranged on a lattice, where each spin can take values of +1 or -1. The interactions between neighboring spins determine the system's energy, which evolves dynamically through Monte Carlo simulations using the Metropolis-Hastings algorithm.

This repository implements GPU acceleration using pytorch to significantly improve performance, making it possible to simulate large lattice sizes efficiently.

Features
✅ 3D Ising Model Implementation on a cubic lattice.
✅ Metropolis-Hastings Algorithm for Monte Carlo simulations.
✅ GPU Acceleration using Numba for faster computations.
✅ Lattice Visualization with Matplotlib.
✅ Statistical Analysis of physical observables (energy, magnetization, susceptibility).
