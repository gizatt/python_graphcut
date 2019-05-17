from __future__ import print_function
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from PIL import Image
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

if __name__ == "__main__":

    np.random.seed(44)
    with open("results.yaml", "r") as f:
        all_data = yaml.load(f, Loader=yaml.CLoader)

    startpath = "/home/gizatt/data/6865_output/"

    # Pick example at random matching a target
    target_condition = {
        "dist_lambda": 0.1,
        "diff_in_color_space": 0.,
        "B_sigma_scaling": 0.5,
        "label_erosion_ratio": 0.1
    }

    inds = np.random.permutation(range(len(all_data)))
    example_i = 0
    n_examples = 15
    n_plots = 5
    example_search_i = 0

    # oh god I need serious refactoring to make this not disgusting
    ious_by_params = {}
    def freeze_dict(params):
        return tuple([params[k] for k in sorted(params.keys())])

    for data in all_data:
        ious_by_params[freeze_dict(data["params"])] = data["iou"]

    plt.figure().set_size_inches(32, 32)
    while example_i < n_examples:
        data = all_data[inds[example_search_i]]
        example_search_i += 1
        prefix = data["prefix"]

        match_fail = False
        for k in target_condition.keys():
            if data["params"][k] != target_condition[k]:
                match_fail = True
                break
        if match_fail:
            continue

        condition_prefix = os.path.join(
            "dl%1.4f" % data["params"]["dist_lambda"],
            "color%d" % data["params"]["diff_in_color_space"],
            "Bss%2.2f" % data["params"]["B_sigma_scaling"],
            "bbox_%2.2f" % data["params"]["bbox_expand_ratio"],
            "labelerosion_%2.2f" % data["params"]["label_erosion_ratio"])
        color_image_name = data["params"]["color_image"]
        label_image_name = data["params"]["label_image"]
        dirnames = color_image_name.split('/')
        basename = os.path.splitext(dirnames[-1])[0]
        result_prefix = os.path.join(
            startpath, condition_prefix, prefix, os.path.join(*dirnames[2:-1]),
            basename)

        print("here0")
        plt.subplot(n_examples, n_plots, example_i*n_plots + 1)
        color_image = Image.open(color_image_name)
        plt.imshow(color_image)
        plt.axis('off')
        if example_i == 0:
            plt.title("Original image")

        print("here1")
        plt.subplot(n_examples, n_plots, example_i*n_plots + 2)
        label_image = Image.open(label_image_name)
        label_image_arr = np.array(label_image)
        if len(label_image_arr.shape) > 2 and label_image_arr.shape[2] > 1:
            label_image_arr = np.sum(label_image_arr, axis=2)
        plt.imshow(label_image_arr == np.max(label_image_arr))
        plt.axis('off')
        if example_i == 0:
            plt.title("GT Mask")

        print("here2")
        plt.subplot(n_examples, n_plots, example_i*n_plots + 3)
        label_image_hint = Image.open(result_prefix + "_img_label_array_crop.png")
        plt.imshow(np.array(label_image_hint).astype(float))
        plt.axis('off')
        if example_i == 0:
            plt.title("FG Hint")

        print("here3")
        plt.subplot(n_examples, n_plots, example_i*n_plots + 4)
        segmentation = Image.open(result_prefix + "_mask_fg.png")
        plt.imshow(segmentation)
        plt.xticks([])
        plt.yticks([])
        if example_i == 0:
            plt.title("GraphCut Mask")
        plt.gca().yaxis.set_label_position("right")
        plt.gca().set_ylabel("IOU %1.2f" % data["iou"],
            fontsize=12)

        print("here4")
        # Cludge in the labelerosion_1.0 version
        condition_prefix = os.path.join(
            "dl%1.4f" % data["params"]["dist_lambda"],
            "color%d" % data["params"]["diff_in_color_space"],
            "Bss%2.2f" % data["params"]["B_sigma_scaling"],
            "bbox_%2.2f" % data["params"]["bbox_expand_ratio"],
            "labelerosion_%2.2f" % 1.0)
        plt.subplot(n_examples, n_plots, example_i*n_plots + 5)
        result_prefix = os.path.join(
            startpath, condition_prefix, prefix, os.path.join(*dirnames[2:-1]),
            basename)
        segmentation = Image.open(result_prefix + "_mask_fg.png")
        plt.imshow(segmentation)
        plt.xticks([])
        plt.yticks([])
        plt.gca().yaxis.set_label_position("right")
        if example_i == 0:
            plt.title("GrabCut Mask")
        params_mutated = deepcopy(data['params'])
        params_mutated["label_erosion_ratio"] = 1.0
        print(data["params"], params_mutated)
        plt.gca().set_ylabel("IOU %1.2f" % (ious_by_params[freeze_dict(params_mutated)]),
            fontsize=12)
        example_i += 1
        print("here5")
        plt.pause(1E-3)
        print("here6")

    plt.show()
    plt.tight_layout()