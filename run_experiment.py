import os
import argparse
import torch
import pprint
import numpy as np
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
parser.add_argument('--dataset', type=str, default='balanced')
parser.add_argument('--n_clusters', type=int, default=5)
parser.add_argument('--adj_type', type=str, default='euclidean')
args = parser.parse_args()

# Dataset and clustering parameters
dataset_name = args.dataset
n_clusters = args.n_clusters
adj_type = args.adj_type
scaler_axis = (0, 1)

# Dicts of auxliary loss weights
loss_weights_synth = {'balanced': [0.1, 0.1], 'balanced_u': [1.06, 0.1],
                'mostlyseries': [0.1, 0.1], 'mostlygraph': [0.58, 0.1],
                'onlyseries': [0.1, 0.1], 'onlygraph': [1.54, 0.58]}

loss_weights_cer = {'2': {'correntropy': [0.58, 0.1],
                                   'euclidean': [0.58, 0.1],
                                   'full': [2.02, 1.06],
                                   'identity': [1.06, 0.1],
                                   'pearson': [2.5, 0.1],
                                   'random': [0.1, 0.1]},
                    '10': {'correntropy': [2.5, 0.58],
                                    'euclidean': [2.5, 0.58],
                                    'full': [0.1, 1.54],
                                    'identity': [1.06, 0.1],
                                    'pearson': [0.1, 0.1],
                                    'random': [0.1, 0.1]},
}

if dataset_name == 'cer':
    is_cer = True
else:
    is_cer = False


## PARAMETERS

# Model parameters
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

if not is_cer and dataset_name in loss_weights_synth.keys():
    topo_w, qual_w = loss_weights_synth[dataset_name]
elif is_cer and str(n_clusters) in loss_weights_cer.keys():
    topo_w, qual_w = loss_weights_cer[str(n_clusters)][adj_type]
else:
    topo_w = 0.1
    qual_w = 0.1

# Training parameters
n_epochs = 250
batch_size = 16 if not is_cer else 8
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
base_path = os.getcwd()

# Load and setup synthetic dataset
if not is_cer:
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
    'softmax_temp': softmax_temp
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