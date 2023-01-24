# gsdn
GCNScheduler for Dynamic Networks

This repo contains code for training a GCN to imitate the HEFT scheduling algorithm, resulting in the ability to rapidly compute schedules for distributing complex tasks across large, dynamic networks.

The simulation environment simulates robots patrolling the perimeter of arbitrary polygons with noisy inter-robot communication.

# Basic Usage
[![Open In Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/ANRGUSC/gsdn/tree/master)

```bash
python preprocess.py    # Generate dataset - data saved to ./data/data.pkl
python train.py         # Train GCNScheduler - weights saved to ./data/model.pt
python simulate.py      # Run simulations - data saved to ./data/results
python plot.py          # Generate plots - plots saved to ./data/plots
```

# Acknowledgements
This work was supported in part by Army Research Laboratory under Cooperative Agreement W911NF-17-2-0196.
