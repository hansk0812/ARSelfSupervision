import subprocess
import os
import csv
import argparse

import numpy as np

from visualize import plot_self_supervision_bar_graph

NUM_RESULTS = 10

def print_iqr(vals):
    assert isinstance(vals, np.ndarray)

    print ("\tMin:", np.min(vals))
    print ("\tMedian:", np.median(vals))
    print ("\tMax:", np.max(vals))
    q1 = np.percentile(vals, 25, method='midpoint')
    q3 = np.percentile(vals, 75, method='midpoint')
    print ("\tQ1:", q1)
    print ("\tQ3:", q3)
    print ("\t\tIQR:", q3-q1)

    print ("\n\tMean:", np.mean(vals))
    print ("\tSTD:", np.std(vals))

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="Directory paths with metrics.csv for IQR based on top 10")
    ap.add_argument("--mse", action="store_true", help="Use MSE instead of MAE for sorting")
    args = ap.parse_args()

    with open(os.path.join(args.folders[0], 'sota.csv'), 'r') as f:
        R = csv.reader(f)
        mse_original, mae_original = [], []
        for line in R:
            # Order of metrics relatively defined! Change here if different!
            mse_original.append(float(line[-1]))
            mae_original.append(float(line[-2]))

    mse_original = np.array(mse_original)[:NUM_RESULTS]
    mae_original = np.array(mae_original)[:NUM_RESULTS]

    print ("Original MSE IQR:")
    print_iqr(mse_original)

    print ("Original MAE IQR:")
    print_iqr(mae_original)

    samples = [[1, 1, mae_original.mean() if not args.mse else mse_original.mean(), 0.5]]

    mse_algorithm, mae_algorithm, params, nets, lambdas = [], [], [], [], []
    for folder in args.folders:
        # extract lambda from shell files
        out = "x=$(cat %srun.sh | grep LAMBDA); IFS=\" \" read -ra ADDR <<< \"$x\"; echo \"${ADDR[2]}\"" % folder
        process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        out, err = process.communicate(out)

        with open(os.path.join(folder, 'metrics.csv'), 'r') as f:
            R = csv.reader(f)
            for idx, line in enumerate(R):
                if idx == 0:
                    continue
                # Order of metrics relatively defined! Change here if different!
                mse_algorithm.append(float(line[-1]))
                mae_algorithm.append(float(line[-2]))

                params.append([float(x) for x in line[2:4]])
                nets.append(line[0])
                lambdas.append(float(out.split('\n')[0].split('=')[-1]))
                
                window = int(line[1])

    indices = list(range(len(mse_algorithm)))
    indices = sorted(indices, key=lambda x: mse_algorithm[x] if args.mse else mae_algorithm[x])

    mse_algorithm = np.array(mse_algorithm)[indices][:NUM_RESULTS]
    mae_algorithm = np.array(mae_algorithm)[indices][:NUM_RESULTS]
    params = np.array(params)[indices][:NUM_RESULTS]
    nets = np.array(nets)[indices][:NUM_RESULTS]
    lambdas = np.array(lambdas)[indices][:NUM_RESULTS]

    print ("\nAlgorithm MSE IQR:")
    print_iqr(mse_algorithm)

    print ("Algorithm MAE IQR:")
    print_iqr(mae_algorithm)
    
    min_params, min_val = [], 10000
    
    metric = mae_algorithm if not args.mse else mse_algorithm
    for (start, step), LAMBDA, m in zip(params, lambdas, metric):
        samples.append([start, step, m, LAMBDA])

        if m < min_val:
            min_val = m
            min_params = [start, step, m, LAMBDA]
    
    print ("Best model parameters: START=%.2f STEP=%.2f LAMBDA=%.2f METRIC=%.5f" % (min_params[0], min_params[1], min_params[-1], min_params[-2]))
    samples = sorted(samples, key=lambda x: x[-2], reverse=True)

    assert len(np.unique(nets)) == 1
    
    # samples: [[start, step, metric, lambda]...]
    plot_self_supervision_bar_graph(samples, window, "net%s_window%d_mae%.4f.png" % (nets[0], window, float(mae_algorithm[0])), mse=args.mse)
