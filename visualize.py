import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint

from collections import OrderedDict

# sample: [[start, step, MAE, lambda]...]
def plot_self_supervision_bar_graph(sample, window_size, save_path="fig.png", save=True):

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

    ax.set_title("Self-Supervision Time Windows (size=%d) vs Mean Average Error\n(Ascending order of performance)" % window_size)
    #ax.legend(loc="upper right")
    
    plt.ylabel("Mean Average Error (MAE)")

    plt.xticks(fontsize = 8) 
    props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
    plt.text(0.9, 0.036, "S: START\ns: STEP", fontsize=8, transform=plt.gcf().transFigure, bbox=props)
   
    if save:
        print ("Saving %s!" % save_path)
        plt.savefig(save_path)

def compare_best(window_size_samples_dict, dataset="NAMEOFDATASET"):
    # window_size_samples_dict (OrderedDict): Dictionary of window size keys: samples values, samples: [original, sota]
    # original, sota: [start, step, MAE, lambda]
    
    keys = list(window_size_samples_dict.keys())

    xlabels = ["Window Size = %d ; S=%.2f, s=%.2f, λ=%.2f" % (keys[idx], 
                                                             window_size_samples_dict[keys[idx]][1][0],
                                                             window_size_samples_dict[keys[idx]][1][1],
                                                             window_size_samples_dict[keys[idx]][1][-1],
                                                             ) 
                                                             for idx in range(len(keys))]

    metrics = {
        'NHITS': [window_size_samples_dict[key][0][2] for key in keys],
        'NHITS + AR Self-Supervision': [window_size_samples_dict[key][1][2] for key in keys],
    }

    x = np.arange(len(xlabels))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 1

    fig, ax = plt.subplots(layout='constrained')
    
    colors = [(0.3,0.3,0.3,1), (0.6,0.6,0.6,1)]
    for idx, (attribute, measurement) in enumerate(metrics.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[idx])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Average Error (MAE)')
    ax.set_xlabel('Window Sizes, AR Self-Supervision Parameters: S-Start, s-Step (%s)' % dataset)
    ax.set_title('Window sizes vs Metric')
    ax.set_xticks(x + width*1.5, xlabels)
    ax.legend(loc='upper left', ncols=len(metrics))

    ax.set_ylim([0,1])

    #props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
    #plt.text(0.924, 0.028, "S: START\ns: STEP", fontsize=8, transform=plt.gcf().transFigure, bbox=props)
    plt.show()

if __name__ == "__main__":
    
    #sample = [[0.1, 0.2, 0.3], [0.2, 0.4, 0.28], [0.4, 0.1, 0.25], [0.2, 0.15, 0.4], [0.3,0.12,0.5]]
    #plot_self_supervision_bar_graph(sample)

    D = OrderedDict({
            96: [[1,1,0.4,0.5], [0.3,0.2,0.3,0.2]],
            192: [[1,1,0.6,0.5], [0.4,0.1,0.5,0.8]],
            336: [[1,1,0.8,0.5], [0.1,0.1,0.7,0.3]],
            720: [[1,1,0.95,0.5], [0.3,0.4,0.85,0.4]],
        })
    compare_best(D)
