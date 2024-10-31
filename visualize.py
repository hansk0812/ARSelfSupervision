import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint

# sample: [[start, step, MAE, lambda]...]
def plot_self_supervision_bar_graph(sample, save_path="fig.png"):

    #xlabels = ["START=%.2f\nSTEP=%.2f" % (x[0], x[1]) for x in sample]
    xlabels = ["S=%.2f\ns=%.2f\nλ=%.2f" % (x[0], x[1], x[-1]) for x in sample]

    weight_counts = {
        "START": np.array([x[2] * x[0] for x in sample]),
    }

    n = 0
    while (not all([x[0]+n*x[1] >= 1 for x in sample])):
        array = []
        for idx in range(len(sample)):
            if sample[idx][0] + n * sample[idx][1] >= 1:
                array.append(0)
            else:
                if sample[idx][0] + (n+1)*sample[idx][1] > 1:
                    array.append(sample[idx][2] * (1-(sample[idx][0]+n*sample[idx][1])))
                else:
                    array.append(sample[idx][2] * sample[idx][1])
        
        weight_counts.update({"STEP_%d" % n: np.array(array)})
        n += 1

    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(sample))

    colors = [(0.6,0.6,0.6,1), (0.3,0.3,0.3,1)]
    idx = 0

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(xlabels, weight_count, width, label=boolean, bottom=bottom, color=colors[idx])
        idx = (idx+1) % 2
        bottom += weight_count

    ax.set_title("Self-Supervision Time Windows vs Mean Average Error")
    #ax.legend(loc="upper right")
    
    plt.xticks(fontsize = 8) 
    props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
    plt.text(0.02, 0.5, "S: START\ns: STEP", fontsize=8, transform=plt.gcf().transFigure, bbox=props)
    
    print ("Saving %s!" % save_path)
    plt.savefig(save_path)

if __name__ == "__main__":
    
    sample = [[0.1, 0.2, 0.3], [0.2, 0.4, 0.28], [0.4, 0.1, 0.25], [0.2, 0.15, 0.4], [0.3,0.12,0.5]]
    plot_self_supervision_bar_graph(sample)
