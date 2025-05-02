# Time-Series-Clustering-with-GNNs

This repository contains the code for the reproducibility of the experiments presented in the paper "On Time Series Clustering with Graph Neural Networks".


## 📂 Repository structure

The repository is structured as follows:

```
./
├── datasets/                           # Datasets used in the experiments
│   ├── cer/                            # CER dataset
│   |   └── subset_idx.npy              # Indices for the subset of the CER dataset
│   ├── synthetic/                      # Synthetic datasets
│   |   ├── {dataset_name}/             # Dataset folder
│   |   |   └── dataset_params.npy      # Parameters used to generate the data
├── source/                             # Source code
|   |   ├── data/                       # Folder related to data handling
|   |   |   ├── adj_construction.py     # Adjacency matrix construction methods
|   |   |   ├── cer_data.py             # Class for loading the CER dataset
|   |   |   └── synth_data.py           # Synthetic data generation
|   |   ├── modules/                    # Folder related to the model and training
|   |   |   ├── layers.py               # Layers used in the model
|   |   |   ├── model.py                # Model implementation
|   |   |   ├── pooling_functions.py    # Functions for the different pooling methods
|   |   |   ├── predictor               # Predictor class for training the model
|   |   |   └── utils.py                # Utility functions
├── run_and_log_results.py              # Script for running and logging results of the experiments
└── run_experiment.py                   # Minimal example script for training and evaluating the model
```

## 📝 Requirements
The implementation is done with [Pytorch](https://pytorch.org/) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/). Moreover, [Torch Spatiotemporal](https://torch-spatiotemporal.readthedocs.io/) is used heavily for the implementation of the spatio-temporal GNN model, data handling and generation, and training of the model. In full, the code is verified for the following packages with their dependencies:

    python=3.12.8
    torch=2.5.1
    torch-geometric=2.6.1
    pyg_lib=0.4.0
    torch_scatter=2.1.2
    torch_sparse=0.6.18
    torch_cluster=1.6.3
    torch_spline_conv=1.2.2
    torch-spatiotemporal=0.9.4
    pygsp=0.5.1
    matplotlib=3.10.0
    openpyxl=3.1.5

## 📦 Datasets

The datasets are stored in the folder `datasets`, which by default is empty with the exception of files to reproduce the synthetic data generation and the subset sampling of CER.

### Synthetic datasets
Numpy files containing the parameters used to generate the different synthetic datasets can be found at `data/synthetic/{dataset_name}/dataset_params.npy`. Code for generating the synthetic time series data is in `synth_data.py` in the `source/data` directory. An example of how to generate data without the use of the saved params can be found at the end of the script, and if run it will generate and save the Balanced dataset to the `datasets` directory. After generation the following files will be created:

- `series.npz`: Numpy file containing the time series data.
- `cluster_index.npy`: Numpy file containing the ground truth cluster labels.
- `edge_index.npy`: Numpy file containing the edge indices of the graph.
- `dataset_params.npy`: Numpy file containing the parameters used to generate the data.
- `dataset_params.txt`: Text file containing the parameters used to generate the data (intended for human readability).

The given parameters can be used to generate the data as follows:

```python
from source.data.synth_data import setup_dataset_with_params
dataset = setup_dataset_with_params('{path_to_params}', '{path_to_data_storage_location}')
```

Below is code of how to load the synthetic data once generated

```python
from source.data.synth_data import SyntheticSpatioTemporalDataset
dataset = SyntheticSpatioTemporalDataset(load_from='{path_to_data}')
```

### CER dataset

The CER dataset is loaded using the `FilteredCER` class in `cer_data.py` in the `source/data` directory. The download URL is not provided in the script and should be requested [here](https://www.ucd.ie/issda/data/commissionforenergyregulationcer/). To successfully download the dataset, the url in the class must be replaced (it is None by default). The `datasets/cer` directory contains a file named `subset_idx.npy` which is used to extract the subset of the dataset used in the experiments.

The adjacency matrix construction methods used with CER is given in
 `adj_construction.py` in the `source/data` directory. At the end of the script is an example of how to apply the function, using the `Elergone` dataset from Torch Spatiotemporal.




## 🧪 Experiments

The training and evaluation of the model (with MinCutPool) can be done by running the file `run_experiment.py`. The script is very minimal and executes only one run with the given configuration and prints the clustering metrics. By default the script is set to execute the experiment on the Balanced dataset. The script can be run with different configurations with the following command:

```bash
python main.py --dataset={dataset_name} --n_clusters={n_clusters} --adj_type={adjacency construction method}
```

The following datasets are available: `balanced`, `balanced_u`, `mostlyseries`, `mostlygraph`, `onlyseries`, and `onlygraph`.

The following adjacency types (for CER) are available: `identity`, `full`, `random`, `euclidean`, `pearson`, and `correntropy`.

The full experiments with logging of metrics can be run by executing the `run_and_log_results.py` script with the following command:

```bash
python run_and_log_results.py --experiment={experiment_name}
```

where the available experiments are `synthetic` and `cer`.
