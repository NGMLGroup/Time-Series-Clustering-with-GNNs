# Time-Series-Clustering-with-GNNs

To do list:

- [ ] Maybe update code for newest version of tsl
- [ ] Add citation guidelines
- [ ] Maybe try to reduce the number of packages to the minimum required
- [ ] Add paper name plus links to paper


Implementation of...

Contains code for synthetic data generation, adjacency construction, layers and
model, and a script for reproducing the results of the synthetic data
experiment.

 ## Python prerequisites
 The implementation is done with [Pytorch](https://pytorch.org/) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/) (with optional dependencies). Moreover, [Torch Spatiotemporal](https://torch-spatiotemporal.readthedocs.io/) is used heavily for the implementation of the spatio-temporal GNN model, the synthetic data generation, and training of the model. In full, the code is verified for the following packages with dependencies:

    python=3.12.8
    torch=2.5.1
    torchvision=0.20.1
    torchaudio=2.5.1
    torch-geometric=2.6.1
    pyg_lib=0.4.0
    torch_scatter=2.1.2
    torch_sparse=0.6.18
    torch_cluster=1.6.3
    torch_spline_conv=1.2.2
    torch-spatiotemporal=0.9.4
    pygsp=0.5.1
    matplotlib=3.10.0



 ## Data generation
Code for generating synthetic time series data is in synth_data.py in the `data_generation` directory. An example of how to generate data is found at the end of the script, and if run it will generate and save the Balanced dataset to the `data` directory. The following files are saved:

- `series.npz`: Numpy file containing the time series data.
- `cluster_index.npy`: Numpy file containing the ground truth cluster labels.
- `edge_index.npy`: Numpy file containing the edge indices of the graph.
- `dataset_params.npy`: Numpy file containing the parameters used to generate the data (intended for reproducing the data).
- `dataset_params.txt`: Text file containing the parameters used to generate the data (intended for human readability).

 ## Data loading


Below is code of how to load the synthetic data

```python
from data_generation.synth_data import SyntheticSpatioTemporalDataset
dataset = SyntheticSpatioTemporalDataset(load_from='{path_to_data}')
```

Numpy files containing the parameters used to generate the different synthetic datasets can be found at `data/synthetic/{dataset_name}/dataset_params.npy`. These parameters can be used to generate the data as follows:

```python
from data_generation.synth_data import setup_dataset_with_params
dataset = setup_dataset_with_params('{path_to_params}', '{path_to_save_or_load_data}')
```

 ## Synthetic data experiment
 The training and evaluation of the model on the synthetic data is done in the `main.py`. The script is currently set to execute the experiment on the Balanced dataset by default. The script can be run with the following command:

```bash
python main.py --dataset={dataset_name}
```

The following datasets are available: `balanced`, `balanced_u`, `mostlyseries`, `mostlygraph`, `onlyseries`, and `onlygraph`.


 ## Adjacency construction
 The adjacency matrix construction methods used in the real world data experiment is given in
 `adj_construction.py` in the `modules` folder. At the end of this script is an example of how to apply the function, using the `Elergone` dataset from Torch Spatiotemporal.

 ## Citation
