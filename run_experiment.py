import os
import argparse
import torch
import pprint
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from sklearn.metrics.cluster import (normalized_mutual_info_score,
                                     homogeneity_score,
                                     completeness_score)
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.metrics.torch import MaskedMAE
from source.data import (setup_dataset_with_params,
                         FilteredCER,
                         FilteredCER,
                         construct_adjacency)
from source.modules import TTSModel
from source.modules import CustomPredictor

# Device setup
accelerator = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Argument parsing for dataset
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='balanced_uniform')
parser.add_argument('--n_clusters', type=int, default=5)
parser.add_argument('--adj_type', type=str, default='euclidean')
parser.add_argument('--pool_loss', type=str, default='mincut')
args = parser.parse_args()

# Dataset and clustering parameters
dataset_name = args.dataset
n_clusters = args.n_clusters
adj_type = args.adj_type
scaler_axis = (0, 1)


## PARAMETERS

base_path = os.getcwd()

# Load in loss weights
if args.pool_loss == 'nopoolloss':
    topo_w = 0.0
    qual_w = 0.0

elif dataset_name == 'cer':
    coeff_path = os.path.join(base_path, 'datasets', 'cer',
                        'loss_coeffs.csv')
    loss_weights = pd.read_csv(coeff_path)

    # Extract based on adj_type and n_clusters
    topo_w, qual_w = loss_weights.loc[
        (loss_weights['adj_type'] == adj_type) &
        (loss_weights['n_clusters'] == n_clusters)
        ][['topo_w', 'qual_w']].values[0]

else:
    coeff_path = os.path.join(base_path, 'datasets', 'synthetic',
                              'loss_coeffs.csv')
    loss_weights = pd.read_csv(coeff_path)

    # Extract based on pool_loss and dataset_name
    topo_w, qual_w = loss_weights.loc[
        (loss_weights['pool_loss'] == args.pool_loss) &
        (loss_weights['dataset_name'] == dataset_name)
        ][['topo_w', 'qual_w']].values[0]


# Other model parameters
hidden_size = 16
temporal_layers = 2
kernel_size = 3
dilation = 2
exponential_dilation = True
skip_connection = True
gcn_layers = 2
n_pool_units = n_clusters
n_latent_channels = 1
pool_method = 'mincut'
softmax_temp = 1.0
temp_step_size = (softmax_temp - 0.01) / 100
temp_min = 0.01
lift_softmax = ('temperature' if dataset_name != 'cer'
                else 'straight_through')


# Training parameters
n_epochs = 250
batch_size = 16 if dataset_name != 'cer' else 8
starting_lr = 1e-3
scale_target = True
gradient_clip_val = 5
lr_scheduler = 'multistep'
lr_milestone_dist = 50
lr_num_milestones = 5
weight_decay = 1e-4

# Loss and metrics
loss_fn = MaskedMAE()


## READY DATASET

# Load and setup synthetic dataset
if dataset_name != 'cer':
    dataset_path = os.path.join(base_path, 'datasets', 'synthetic',
                                dataset_name)
    dataset_params_path = os.path.join(dataset_path,
                                    'dataset_params.npy')

    dataset = setup_dataset_with_params(dataset_params_path, dataset_path)

    window = 16
    horizon = 1
    covariates = None
    X, idx = dataset.numpy(return_idx=True)
    labels = dataset.cluster_index
    adj = dataset.connectivity
    dataset_params = np.load(os.path.join(dataset_path, 'dataset_params.npy'),
                            allow_pickle='TRUE').item()
    print("Dataset params:")
    pprint.pprint(dataset_params)

# Load and setup CER dataset
else:
    dataset_path = os.path.join(base_path, 'datasets', 'cer')
    dataset = FilteredCER(dataset_path,
                        missing_threshold=0.05,
                        remove_other=True,
                        resample_hourly=True)
    window = 72
    horizon = 1
    labels = dataset.codes - 1

    subset_idx = np.load(os.path.join(dataset_path, 'subset_idx.npy'))
    dataset.target = dataset.target.iloc[:, subset_idx]
    dataset.mask = dataset.mask[:, subset_idx, :]
    labels = labels[subset_idx]

    covariates = {'u': dataset.datetime_encoded(['day', 'week', 'year']).values}

    X, idx = dataset.numpy(return_idx=True)

    adj = construct_adjacency(dataset, adj_type=adj_type)


print(dataset)

torch_dataset = SpatioTemporalDataset(target = X,
                                    index = idx,
                                    connectivity = adj,
                                    covariates = covariates,
                                    mask = dataset.mask,
                                    window = window,
                                    horizon = horizon,
                                    delay = 0,
                                    stride = 1)
exog_size = False if covariates is None else torch_dataset.input_map.u.shape[-1]
print(torch_dataset)

# Rescale and split the data
scalers = {'target': StandardScaler(axis=scaler_axis)}
splitter = dataset.get_splitter(val_len=0.1, test_len=0.1)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    scalers=scalers,
    mask_scaling=True,
    splitter=splitter,
    batch_size=batch_size,
    pin_memory=True,
    workers=0
)
dm.setup()
print(dm)


## SETUP MODEL
model_kwargs = {
    'input_size': dm.n_channels,
    'horizon': horizon,
    'exog_size': exog_size,
    'hidden_size': hidden_size,
    'temporal_layers': temporal_layers,
    'kernel_size': kernel_size,
    'dilation': dilation,
    'exponential_dilation': exponential_dilation,
    'skip_connection': skip_connection,
    'gcn_layers': gcn_layers,
    'n_nodes': torch_dataset.n_nodes,
    'n_clusters':  n_clusters,
    'topo_w': topo_w,
    'qual_w': qual_w,
    'pool_method': pool_method,
    'softmax_temp': softmax_temp,
    'lift_softmax': lift_softmax,
}

# Optimizer config
optim_class = torch.optim.Adam
optim_kwargs = {'lr': starting_lr, 'weight_decay': weight_decay}

scheduler_class = torch.optim.lr_scheduler.MultiStepLR
scheduler_kwargs = {'gamma': 0.5,
                    'milestones': [
                        lr_milestone_dist*j
                        for j in range(1, lr_num_milestones+1)
                        ]
                    }

# Setup lightning module
predictor = CustomPredictor(
    temp_step_size=temp_step_size,
    temp_min=temp_min,
    model_class=TTSModel,
    model_kwargs=model_kwargs,
    optim_class=optim_class,
    optim_kwargs=optim_kwargs,
    scheduler_class=scheduler_class,
    scheduler_kwargs=scheduler_kwargs,
    scale_target=scale_target,
    loss_fn=loss_fn,
    metrics=None
)


## TRAINING
trainer = pl.Trainer(max_epochs=n_epochs,
                    logger=False,
                    devices="auto",
                    accelerator=accelerator,
                    limit_train_batches=100,
                    limit_val_batches=50,
                    callbacks=None,
                    gradient_clip_val=gradient_clip_val,
                    gradient_clip_algorithm='norm',
                    enable_checkpointing=False)
trainer.fit(predictor, datamodule=dm)


## EVALUATION
predictor.freeze()

s = predictor.model.pooling_layer.assignments().detach().cpu()
hard_assignments = np.argmax(s.numpy(), axis=-1)
cluster_ids, cluster_sizes = np.unique(hard_assignments,
                                        return_counts=True)
print(f'Tot clusters: {len(cluster_ids)}\n'
    f'IDs: {cluster_ids}\n'
    f'sizes: {cluster_sizes}')

nmi = normalized_mutual_info_score(labels, hard_assignments)
hs = homogeneity_score(labels, hard_assignments)
cs = completeness_score(labels, hard_assignments)

print(f'NMI: {nmi} \nHS: {hs} \nCS: {cs}')