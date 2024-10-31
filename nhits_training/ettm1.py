import os
import csv

import pandas as pd
from datasetsforecast.long_horizon import LongHorizon

from ray import tune
from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast

from neuralforecast.losses.numpy import mae, mse

horizon = 96

Y_df, _, _ = LongHorizon.load(directory='./', group='ETTm1')
Y_df['ds'] = pd.to_datetime(Y_df['ds'])

total_size = len(Y_df.ds.unique())
val_size = int(.1 * total_size)
test_size = int(.2 * total_size)

import matplotlib.pyplot as plt

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--seed', help="Run deterministic training", type=int, default=None)
args = ap.parse_args()

# Use your own config or AutoNHITS.default_config
nhits_config = {
       "learning_rate": tune.choice([0.001]),                                     # Initial Learning rate
       "max_steps": tune.choice([300]),                                         # Number of SGD steps
       "input_size": tune.choice([horizon]),                                 # input_size = multiplier * horizon
       "batch_size": tune.choice([7]),                                           # Number of series in windows
       "windows_batch_size": tune.choice([256]),                                 # Number of windows in batch
       "n_pool_kernel_size": tune.choice([[8, 4, 2]]),               # MaxPool's Kernelsize
       "n_freq_downsample": tune.choice([[2, 1, 1]]), # Interpolation expressivity ratios
       "activation": tune.choice(['ReLU']),                                      # Type of non-linear activation
       "n_blocks":  tune.choice([[1, 1, 1]]),                                    # Blocks per each 3 stacks
       "mlp_units":  tune.choice([[[512, 512]]]),        # 2 512-Layers per block for each stack
       "interpolation_mode": tune.choice(['linear']),                            # Type of multi-step interpolation
       "val_check_steps": tune.choice([100]),                                    # Compute validation every 100 epochs
       "random_seed": tune.randint(1, 10) if args.seed is None else args.seed,
    }

models = [AutoNHITS(h=horizon,
                    config=nhits_config, 
                    num_samples=1)]

nf = NeuralForecast(
    models=models,
    freq='15min')

Y_hat_df = nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size, n_windows=None)

nf.models[0].results.get_best_result().config

y_true = Y_hat_df.y.values
y_hat = Y_hat_df['AutoNHITS'].values

with open('sota.csv', 'a') as f:
    w = csv.writer(f)
    w.writerow([os.environ["START"], os.environ["STEP"], mae(y_hat, y_true), mse(y_hat, y_true)])

print('MAE: ', mae(y_hat, y_true))
print('MSE: ', mse(y_hat, y_true))

