# plexus-simulate
**Overview**: Multivariate Hawkes process simulations for modeling excititory neuronal activity dynamics

# Installation
```bash
git clone https://github.com/pgrosjean/plexus-simulate.git
cd plexus
bash setup_mamba.sh
conda activate plexus_simulate
pip install -e .
```

# Overview
plexus-simulate contains utilities to simulate multivariate hawkes processes, convert depolarization process events to GCaMP6m signals, and plot them, as described in the paper: "Network-aware self-supervised learning enables high-content phenotypic screening for genetic modifiers of neuronal activity dynamics"

### Multivariate Hawkes process simulation tool

```python
from plexus_simulate.simulation_utils import simulate_hawkes_discrete

# Example simulation
n_nodes = 60
alpha = 0
connectivity=0.2
alpha_base = alpha/n_nodes
events, intensities, network_graph = simulate_hawkes_discrete(n_nodes=n_nodes,
                                                              k=int(n_nodes*connectivity),
                                                              p=0.1,
                                                              T=90*2,
                                                              delta_t=1/25,
                                                              mu=0.2,
                                                              alpha_base=alpha_base,
                                                              beta=2.0,
                                                              gamma=0.2,
                                                              seed=42,
                                                              t_feedback_beta=1,
                                                              t_feedback_gamma=20,
                                                              mu_std=0.2)
```

### Converting simulated depolarization events to GCaMP signal
```python
from plexus_simulate.simulation_utils import calcium_dynamics
CB, C, F = calcium_dynamics(events)
```

### Plotting and saving the simulated data
```python
from plexus_simulate.plotting_utils import plot_calcium_traces, plot_calcium_traces_and_save

# Starting from 90 seconds to avoid the initial transients in the simulation
start_idx = 90*25
sampling_freq = 25
plot_calcium_traces(F[:, start_idx:], "#8a508f", sampling_freq)
plot_calcium_traces_and_save(F[:, start_idx:], "#8a508f", sampling_freq, './simulation_plots/simulation_1.pdf')
```
### Resulting plots look like this:
![image](https://github.com/user-attachments/assets/8abfabfa-b192-4f08-ba49-e0a1726c9067)


### Simulating example high-content screening plate
Example of generating a zarr file to match the input to the plexus model can be found in the notebook [here](https://github.com/pgrosjean/plexus-simulate/blob/main/notebooks/simulate_multivariate_hawkes_zarr_file.ipynb)
