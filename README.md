# proliferation_rate

Simulation of proliferation behavior in the DFK formalism.
The simulation is run for a radially symmetric cell colony with cells being either in a growing or in a dividing state. 
The default is for the simulation to be in 2D, as indicated by the choice of the laplace operator. 

Boundary conditions are generally assumed to be absorbing at some large radius/infinity, but stronger conditions like linearity towards the outer edge can be enforced. 

# Installation 

The program is provided as python scripts in the `scripts/` directory. 
Below this, further subdirectories are provided. They are:

- `simulator/`: Here we provide a delayed differential equation version of the DFK formalism in `dde_dfk.py` as well as a version without finite time of the division phase in `simple_dfk.py` where the numerical stability is higher but mechanosensitivity of proliferation is not accurately reproduced. 
- `analysis/`: Here you can find scripts to analyze the data in trajectories produced by either of the simulators.
- `trajectory_transform/`: Whether you want to cut out only part of a trajectory or whether you want to reduce the sampling rate of a trajectory for speeding up the analysis procedure, the scripts within this directory provide you with the necessary tools to edit and change trajectories.

As python scripts do not need to be compiled, you only need to have a python3 interpreter installed (ask your search engine of choice for an installation guide if you need it), as well as a few packages for which we recommend using the python package manager `pip`. 

To install the necessary dependencies, run 

```pip install numpy scipy matplotlib```

In addition, if you want to use the DDE version of the DFK simulator, you will have to install the ddesolver project [https://github.com/Reshief/ddesolver](https://github.com/Reshief/ddesolver). To do so, please follow the installation instructions on the DDEsolver project page. 

# Running

To run the simulator, on a command line just invoke, e.g.:

```python scripts/simulator/simple_dfk.py <further parameters>```

Where `<further parameters>`  denotes the command line settings you desire for your simulation run. 
Each script in this project will divulge the available options as well as additional information upon using the option `-h` or `--help` if you need help. 
Additionally, the tools will inform you about missing parameters on the command line. Please be aware that you will not see the error output if you double click the python script on windows. You have to run it through the command line.

For example, a recent run configuration was:
```python ./scripts/simulator/dfk_dde_chemical.py -o ./output/test_1/ -g -t 40 -rs 1 -D_chem 2 -b_chem 4 -D_c 4 -D_0 0.2 -dt 0.001```