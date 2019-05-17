from __future__ import print_function
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


if __name__ == "__main__":
    with open("results.yaml", "r") as f:
        all_data = yaml.load(f, Loader=yaml.CLoader)

    col_names = ["B_sigma_scaling", "bbox_expand_ratio", "diff_in_color_space", "dist_lambda", "label_erosion_ratio"]

    train_results = []
    test_results = []

    for data in all_data:
        col = [data["params"][name] for name in col_names] + [data["iou"]]
        if data["prefix"] == "train":
            train_results.append(col)
        elif data["prefix"] == "test":
            test_results.append(col)
        else:
            raise ValueError("Bad prefix")

    train_results = np.vstack(train_results)
    test_results = np.vstack(test_results)
    all_results = np.vstack([train_results, test_results])

    unique_pts = [np.unique(train_results[:, k]) for k in range(len(col_names))]
    every_combination = [x.ravel() for x in np.meshgrid(*unique_pts)]
    n_combinations = len(every_combination[0])

    ious_and_combinations = []

    # Make bar plot of all conditions
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.yaxis.grid(True)
    width = 0.15       # the width of the bars

    group_inds_by_Bss_and_dl = {}
    curr_group_ind = 0
    within_group_colors = ['darkred', 'orange', 'darkblue', 'teal']
    within_group_names = ['Color + FG Label', 'No Color, FG Label', 'Color, Grabcut', 'No Color, Grabcut']
    all_condition_names = []
    for k in range(n_combinations):
        values = [every_combination[i][k] for i in range(len(col_names))]
        # Compute IOU at this combination
        target = np.tile(values, [all_results.shape[0], 1])
        mask = np.all(all_results[:, :-1] == target, axis=1)
        num_good_samples = np.sum(np.isfinite(all_results[mask, -1]))
        if num_good_samples > 0:
            iou_mean = np.nanmean(all_results[mask, -1])
            iou_std = np.nanstd(all_results[mask, -1], ddof=1)
            ious_and_combinations.append([values, iou_mean, iou_std, all_results[mask, -1]])

            Bss_and_dll = (values[0], values[3])
            if Bss_and_dll not in group_inds_by_Bss_and_dl.keys():
                group_inds_by_Bss_and_dl[Bss_and_dll] = curr_group_ind
                curr_group_ind += 1
                all_condition_names.append("K_sigma %1.1f\nK_lambda %1.3f" % Bss_and_dll)
            group_ind = group_inds_by_Bss_and_dl[Bss_and_dll]
            within_group_ind = int(values[2]*1. + (values[4] > 0.5)*2.)
            stuff = ax.boxplot(all_results[mask, -1], positions=[group_ind + width*within_group_ind],
                               widths=[width*0.75], showfliers=False, whis=0.0)
            [l.set_linewidth(2) for l in stuff["medians"]]
            #ax.bar(group_ind + width*within_group_ind, iou_mean, width=width, yerr=iou_std,
            #       color=within_group_colors[within_group_ind], zorder=0, alpha=0.5)
            plt.scatter(
                np.ones(num_good_samples)*(group_ind + width*within_group_ind),
                all_results[mask, -1],
                color=within_group_colors[within_group_ind],
                alpha=0.05, zorder=1.)

        else:
            print("Rejecting condition ", values, " for not having enough samples.")

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=4)
                    for c in within_group_colors]
    ax.legend(custom_lines, within_group_names)
    plt.xticks(np.array(range(len(all_condition_names)))+width*2, all_condition_names, rotation=90)
    plt.ylabel("IOU")
    plt.xlim(-0.5, len(all_condition_names)+0.5)
    plt.tight_layout()

    # Print in sorted order
    #ious_and_combinations = sorted(ious_and_combinations, key=lambda x: x[1])
    #for iou_and_combination in ious_and_combinations:
    #    values, iou, iou_std, masked_results = iou_and_combination
    #    print("IOU %4.4f +/- %4.4f: " % (iou, iou_std))
    #    for i, col_name in enumerate(col_names):
    #        print("\t%s: %f" % (col_name, values[i]))
    plt.show()
