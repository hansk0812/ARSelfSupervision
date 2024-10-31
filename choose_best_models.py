import os
import csv
import argparse
from pprint import pprint

from visualize import plot_self_supervision_bar_graph

ap = argparse.ArgumentParser()
ap.add_argument("model_dir", help="Directory of metrics.csv file")
args = ap.parse_args()

with open(os.path.join(args.model_dir, "metrics.csv"), 'r') as f:
    R = csv.reader(f)
    
    best = {}
    for line in R:
        if not line[1] in best:
            best[line[1]] = [line]
        else:
            best[line[1]].append(line)
    
    top_k = 10
    for key in best:
        final = []
        final.append([x for x in best[key] if x[3] == "1"])
        final.extend(sorted(best[key], key=lambda x: float(x[5]))[:top_k])
        best[key] = final
        
        if len(final[0]) == 1:
            final[0] = final[0][0]
        
        samples = []
        for idx in range(len(final)):
            net, window, start, step, mse, mae = final[idx]
            sample = [float(start), float(step), float(mae)]
            samples.append(sample)
        
        plot_self_supervision_bar_graph(samples, "net%s_window%d_mae%.4f.png" % (net, int(window), float(mae)))

pprint (best)
