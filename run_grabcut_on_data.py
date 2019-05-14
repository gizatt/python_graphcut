from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
import multiprocessing
try:  # Python 2
    import Queue as queue
except ImportError:  # Python 3
    import queue
import numpy as np
import scipy as sp
import scipy.misc
import scipy.ndimage.morphology
from PIL import Image
import time
import traceback
import yaml

from graphcut import build_affinity_graph, perform_min_cut


# https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


# https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


class Worker(object):
    """Multiprocess worker."""

    def __init__(self, input_queue, output_queue,
                 termination_event, error_queue=None):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.termination_event = termination_event
        self.error_queue = error_queue

    def HandleCase(self, prefix, params):
        # FIXED PARAMS
        border_width = 10
        # Max size of CROPPED image
        max_image_size = 100

        color_image = params["color_image"]
        label_image = params["label_image"]

        condition_prefix = os.path.join(
            "dl%1.4f" % params["dist_lambda"],
            "color%d" % params["diff_in_color_space"],
            "Bss%2.2f" % params["B_sigma_scaling"],
            "bbox_%2.2f" % params['bbox_expand_ratio'],
            "labelerosion_%2.2f" % params["label_erosion_ratio"])

        save_prefix = os.path.join(
            "/tmp/Output", condition_prefix, prefix, *splitall(color_image)[2:-1])

        os.system("mkdir -p %s" % save_prefix)
        save_prefix = os.path.join(
            save_prefix,
            os.path.splitext(os.path.basename(color_image))[0])
        print("Save prefix: ", save_prefix)

        # Load image
        img = Image.open(color_image)
        img_orig_rows = img.height
        img_orig_cols = img.width
        img_lum = img.convert('L')
        img_lum_array = np.array(img_lum)/255.
        img_array = np.array(img)/255.

        # Load in the ground truth label image
        img_label = Image.open(label_image).convert('L').resize(
            (img_orig_cols, img_orig_rows))

        assert(img_label.height == img_orig_rows)
        assert(img_label.width == img_orig_cols)
        img_label_array = (np.array(img_label) > 250)*1.0

        # Get outer bounding box of the true image segment by expanding
        # by the desired ratio (or at least the border width)
        bbox_expand_ratio = params["bbox_expand_ratio"]
        xmin, xmax, ymin, ymax = bbox(img_label_array)
        expand_amt_x = max(border_width, (xmax-xmin)*bbox_expand_ratio)
        expand_amt_y = max(border_width, (ymax-ymin)*bbox_expand_ratio)
        avg_bbox_size = ((xmax - xmin) + (ymax - ymin))/2.
        xmin = max(int(xmin - expand_amt_x), 0)
        xmax = min(int(xmax + expand_amt_x), img_array.shape[0])
        ymin = max(int(ymin - expand_amt_y), 0)
        ymax = min(int(ymax + expand_amt_y), img_array.shape[1])

        if params["label_erosion_ratio"] >= 1.0:
            img_label_array_eroded = img_label_array * 0.
        else:
            label_erosion_iters = int(params["label_erosion_ratio"]*avg_bbox_size)
            img_label_array_eroded = sp.ndimage.morphology.binary_erosion(
                img_label_array, iterations=label_erosion_iters)

        # Crop the input image appropriately.
        img_array_crop = img_array[xmin:xmax, ymin:ymax]
        img_lum_array_crop = img_lum_array[xmin:xmax, ymin:ymax]
        img_label_array_gt_crop = img_label_array[xmin:xmax, ymin:ymax]
        # And construct the label image, taking the outer images
        # to be the background.
        img_label_array_crop = np.stack([
            img_label_array_eroded[xmin:xmax, ymin:ymax],
            img_lum_array_crop*0.,
            img_lum_array_crop*0.], axis=-1)
        if border_width > 0:
            img_label_array_crop[0:border_width, :, 2] = 1.0
            img_label_array_crop[-border_width:, :, 2] = 1.0
            img_label_array_crop[:, 0:border_width, 2] = 1.0
            img_label_array_crop[:, -border_width:, 2] = 1.0

        # Figure out scaling and scale the cropped images down if
        # necessary
        image_scale = max(max(img_array_crop.shape[0] / max_image_size,
                              img_array_crop.shape[1] / max_image_size),
                          1.0)
        target_size = np.array(img_array_crop.shape) / image_scale
        img_array_crop = sp.misc.imresize(img_array_crop, target_size) / 255.
        img_lum_array_crop = sp.misc.imresize(img_lum_array_crop, target_size) / 255.
        img_label_array_crop = sp.misc.imresize(img_label_array_crop, target_size, interp='nearest') / 255.
        img_label_array_gt_crop = sp.misc.imresize(img_label_array_gt_crop, target_size, interp='nearest') / 255.

        plt.imsave(save_prefix + "_img_array_crop.png", img_array_crop)
        plt.imsave(save_prefix + "_img_lum_array_crop.png", img_lum_array_crop)
        plt.imsave(save_prefix + "_img_label_array_crop.png", img_label_array_crop)
        plt.imsave(save_prefix + "_img_label_array_gt_crop.png", img_label_array_gt_crop)

        graph_matrix, source_node_ind, sink_node_ind = build_affinity_graph(
            img_array_crop, img_lum_array_crop, img_label_array_crop,
            dist_lambda=params["dist_lambda"],
            n_hist_bins=20,
            neighbor_inds=[-1, 0, 1],
            make_plots=False, connect_node_labels=True,
            connect_color_priors=True,
            diff_in_color_space=params["diff_in_color_space"],
            B_sigma_scaling=params["B_sigma_scaling"])

        foreground_inds, background_inds = perform_min_cut(
            graph_matrix, source_node_ind, sink_node_ind)
        background_inds.remove(sink_node_ind)
        foreground_inds.remove(source_node_ind)

        mask_bg = (img_lum_array_crop*0).reshape(-1)
        mask_bg[list(background_inds)] = 1.0
        mask_bg = mask_bg.reshape(img_lum_array_crop.shape)
        mask_fg = (img_lum_array_crop*0).reshape(-1)
        mask_fg[list(foreground_inds)] = 1.0
        mask_fg = mask_fg.reshape(img_lum_array_crop.shape)
        combined_masks = np.stack([img_lum_array_crop*0.6 + mask_fg*0.4,
                                   img_lum_array_crop*0.6,
                                   img_lum_array_crop*0.6 + mask_bg*0.4], axis=-1)
        print(np.min(combined_masks), np.max(combined_masks))
        plt.imsave(save_prefix + "_mask_bg.png", mask_bg)
        plt.imsave(save_prefix + "_mask_fg.png", mask_fg)
        plt.imsave(save_prefix + "_mask_combined.png", combined_masks)

        # Calculate IOU
        intersection = np.logical_and(mask_fg, img_label_array_gt_crop)
        union = np.logical_or(mask_fg, img_label_array_gt_crop)
        iou = float(np.sum(intersection)) / float(np.sum(union))
        print("IOU: ", iou)
        self.output_queue.put((params, prefix, iou))

    def __call__(self, worker_index):
        while ((not self.input_queue.empty()) or
               (not self.termination_event.is_set())):
            try:
                new_data = None
                try:
                    prefix, params = self.input_queue.get(False)
                    self.HandleCase(prefix, params)
                except queue.Empty:
                    pass

                if new_data is None:
                    time.sleep(0)
                    continue

                print(new_data)

            except Exception as e:
                if self.error_queue:
                    self.error_queue.put((worker_index, e))
                else:
                    print("Unhandled exception in Worker #%d" % worker_index)
                    traceback.print_exc()


if __name__ == "__main__":
    with open("./Input/ObjectDiscoverySubset/train.yaml", 'r') as f:
        train_pairs = yaml.load(f)
    with open("./Input/ObjectDiscoverySubset/test.yaml", 'r') as f:
        test_pairs = yaml.load(f)

    print("Loaded %d train pairs, %d test pairs." %
          (len(train_pairs), len(test_pairs)))

    #fake_worker = Worker(None, None)
    #for k in range(2):
    #    fake_worker.HandleCase("dummy",
    #        {
    #            "dist_lambda": 1/50.,
    #            "diff_in_color_space": False,
    #            "B_sigma_scaling": 1.,
    #            "bbox_expand_ratio": 0.5,
    #            "label_erosion_ratio": 0.1,
    #            "color_image": test_pairs[k]["color_image"],
    #            "label_image": test_pairs[k]["label_image"],
    #        })
    #sys.exit(0)

    worker_pool = multiprocessing.Pool(processes=10)
    worker_manager = multiprocessing.Manager()
    input_queue = worker_manager.Queue()
    output_queue = worker_manager.Queue()
    termination_event = worker_manager.Event()
    result = worker_pool.map_async(
        Worker(input_queue=input_queue,
               output_queue=output_queue,
               termination_event=termination_event),
        range(worker_pool._processes))

    for diff_in_color_space in [True, False]:
        for dist_lambda in [1/10., 1/50., 1/250.]:
            for B_sigma_scaling in [0.5, 1., 2.]:
                for bbox_expand_ratio in [1.5]:
                    for label_erosion_ratio in [0.1, 1.0]:
                        params = {
                            "dist_lambda": dist_lambda,
                            "diff_in_color_space": diff_in_color_space,
                            "B_sigma_scaling": B_sigma_scaling,
                            "bbox_expand_ratio": bbox_expand_ratio,
                            "label_erosion_ratio": label_erosion_ratio
                        }

                        for train_pair in train_pairs:
                            params["color_image"] = train_pair["color_image"]
                            params["label_image"] = train_pair["label_image"]
                            input_queue.put(("train", params))

                        for test_pair in test_pairs:
                            params["color_image"] = test_pair["color_image"]
                            params["label_image"] = test_pair["label_image"]
                            input_queue.put(("test", params))

    termination_event.set()
    while (True):
        if result.ready() and output_queue.empty():
            break
        if not output_queue.empty():
            params, prefix, iou = output_queue.get(timeout=0)
            with open("results.yaml", 'a') as f:
                yaml.dump([{
                        "params": params,
                        "prefix": prefix,
                        "iou": float(iou)
                    }], f, default_flow_style=False)
        time.sleep(0)

    print("All done!")
