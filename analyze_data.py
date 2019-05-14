from __future__ import print_function
import os
import sys
import numpy as np
import yaml

if __name__ == "__main__":
    with open("results.yaml", "r") as f:
        all_data = yaml.load(f)

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
            assert(False)

    train_results = np.vstack(train_results)
    test_results = np.vstack(train_results)

    unique_pts = [np.unique(train_results[:, k]) for k in range(len(col_names))]
    every_combination = [x.ravel() for x in np.meshgrid(*unique_pts)]
    n_combinations = len(every_combination[0])

    ious_and_combinations = []

    for k in range(n_combinations):
        values = [every_combination[i][k] for i in range(len(col_names))]
        # Compute IOU at this combination
        target = np.tile(values, [train_results.shape[0], 1])
        mask = np.all(train_results[:, :-1] == target, axis=1)
        num_good_samples = np.sum(np.isfinite(train_results[mask, -1]))
        if num_good_samples > 0:
            ious_and_combinations.append([np.nanmean(train_results[mask, -1]), values])

    # Print in sorted order
    ious_and_combinations = sorted(ious_and_combinations, key=lambda x: x[0])
    for iou_and_combination in ious_and_combinations:
        iou, values = iou_and_combination
        print("IOU %4.4f: " % iou)
        for i, col_name in enumerate(col_names):
            print("\t%s: %f" % (col_name, values[i]))