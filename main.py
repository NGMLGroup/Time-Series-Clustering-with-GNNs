import argparse, sys
import traceback
import os
import gc
import numpy as np
import torch
import pprint
# from sklearn.metrics import homogeneity_score
from sklearn.metrics.cluster import (normalized_mutual_info_score,
                                     homogeneity_score,
                                     completeness_score)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.metrics.torch import MaskedMAE, MaskedMRE
from data_generation.synth_data import SyntheticSpatioTemporalDataset
from modules.torch_model import TTSModel
from modules.lightning_module import CustomPredictor
from modules.utils import (
    find_devices
)

torch.set_float32_matmul_precision('high')

parser=argparse.ArgumentParser()

parser.add_argument("--dataset_name", help="Give dataset name")
parser.add_argument("--sweep_id", help="Sweep ID for the sweep")
parser.add_argument("--n_clusters", help="Number of clusters to use")

args=parser.parse_args()

dataset_name = args.dataset_name if args.dataset_name else 'balanced'

## EXPERIMENTAL PARAMETERS
dataset_name = 'balanced'
n_clusters = 5
scaler_axis = (0, 1)

## HYPERPARAMETERS

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
topo_w = 0.1
qual_w = 0.1

# Training parameters
n_epochs = 250
batch_size = 16
starting_lr = 1e-3
scale_target = True
gradient_clip_val = 5
lr_scheduler = 'multistep'
lr_milestone_dist = 50
lr_num_milestones = 5
early_stop_patience = 50
weight_decay = 1e-4
load_best_model = True # TODO: Check if it works with True here

# Loss and metrics
loss_fn = MaskedMAE()
# metrics = {'mae': MaskedMAE(), 'mre': MaskedMRE()}
metrics = None

## LOAD DATASET
base_path = os.getcwd()
data_path = os.path.join(base_path, 'data')


dataset_path = os.path.join(data_path, dataset_name)
dataset = SyntheticSpatioTemporalDataset(
load_from= dataset_path
)
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
print("Number of NaNs: ", np.sum(np.isnan(dm.scalers['target'].bias.numpy()))) # TODO: Maybe remove this

## SETUP MODEL
model_kwargs = {
    'input_size': dm.n_channels,
    'horizon': horizon,
    'exog_size': False if covariates is None else torch_dataset.input_map.u.shape[-1],
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
    model_class=TTSModel,
    model_kwargs=model_kwargs,
    optim_class=optim_class,
    optim_kwargs=optim_kwargs,
    scheduler_class=scheduler_class,
    scheduler_kwargs=scheduler_kwargs,
    scale_target=scale_target,
    loss_fn=loss_fn,
    metrics=metrics,
    log_lr=True,
    log_grad_norm=True
)


## TRAINING
logger = None# TODO: Change or remove logger
# avoid logging gradients, parameter histogram and model topology
# logger.watch(predictor.model, log=None) 

checkpoint_callback = ModelCheckpoint(
    # dirpath='logs/model_with_pooling',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
)

early_stop_callback = EarlyStopping(
    monitor='val_mae',
    patience=early_stop_patience,
    mode='min'
)

trainer = pl.Trainer(max_epochs=n_epochs,
                    logger=logger,
                    devices=find_devices(1),
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    limit_train_batches=100,
                    limit_val_batches=50,
                    callbacks=[checkpoint_callback, early_stop_callback],
                    gradient_clip_val=gradient_clip_val,
                    gradient_clip_algorithm='norm')
trainer.fit(predictor, datamodule=dm)

## EVALUATION
predictor.freeze()


s = predictor.model.pooling_layer.assignments().detach().cpu()
hard_assignments = np.argmax(s.numpy(), axis=-1)
cluster_ids, cluster_sizes = np.unique(hard_assignments[-1], 
                                        return_counts=True)
print(f'Tot clusters: {len(cluster_ids)} 
        \nIDs: {cluster_ids} 
        \nsizes: {cluster_sizes}')

nmi = normalized_mutual_info_score(labels, hard_assignments)
hs = homogeneity_score(labels, hard_assignments)
cs = completeness_score(labels, hard_assignments)

print(f'NMI: {nmi} \nHS: {hs} \nCS: {cs}')

