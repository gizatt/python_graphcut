from __future__ import print_function
import os
import sys
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import networkx as nx
import numpy as np
import scipy.sparse
from scipy.interpolate import interp1d
import time
from PIL import Image

from graphcut import build_affinity_graph, perform_min_cut
import networkx
import networkx.algorithms.flow as flow


class GrabCutManager():
    def __init__(self, params, interactive=True, make_plots=False):
        self.params = params
        color_image = self.params["color_image"]
        image_scale = self.params["image_scale"]
        self.make_plots = make_plots
        self.interactive = interactive

        # Load image + the label image.
        img = Image.open(color_image)
        img_orig_rows = img.height
        img_orig_cols = img.width
        img = img.resize((img_orig_cols/image_scale,
                          img_orig_rows/image_scale))
        img_lum = img.convert('L')
        self.img_lum_array = np.array(img_lum)/255.
        self.img_array = np.array(img)/255.

        if self.interactive:
            plt.figure().set_size_inches(10, 10)
            plt.imshow(img)
            self.RS = RectangleSelector(
                plt.gca(), self.HandleLineSelect,
                drawtype='box', useblit=True,
                button=[1, 3],  # don't use middle button
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True)
            plt.connect('key_press_event', self.HandleKeyPress)
            plt.show()

    def PerformGrabCut(self, xlim, ylim, border_width=5, center_radius=0):
        # Crop the input image appropriately.
        img_array_crop = self.img_array[ylim[0]:ylim[1],
                                        xlim[0]:xlim[1]]
        img_lum_array_crop = self.img_lum_array[ylim[0]:ylim[1],
                                                xlim[0]:xlim[1]]
        # Take the outer pixels to be background
        img_label_array_crop = np.zeros((img_lum_array_crop.shape[0],
                                         img_lum_array_crop.shape[1], 3))

        if border_width > 0:
            img_label_array_crop[0:border_width, :, 2] = 1.0
            img_label_array_crop[-border_width:, :, 2] = 1.0
            img_label_array_crop[:, 0:border_width, 2] = 1.0
            img_label_array_crop[:, -border_width:, 2] = 1.0

        # Assume box at center is foreground
        if center_radius > 0:
            center_x = img_label_array_crop.shape[0] / 2
            center_y = img_label_array_crop.shape[1] / 2
            img_label_array_crop[
                (center_x-center_radius):(center_x+center_radius),
                (center_y-center_radius):(center_y+center_radius), 0] = 1.

        graph_matrix, source_node_ind, sink_node_ind = build_affinity_graph(
            img_array_crop, img_lum_array_crop, img_label_array_crop,
            self.params["dist_lambda"],
            self.params["n_hist_bins"],
            self.params["neighbor_inds"],
            make_plots=True, connect_node_labels=True,
            connect_color_priors=True,
            diff_in_color_space=self.params["diff_in_color_space"],
            B_sigma_scaling=self.params["B_sigma_scaling"])

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
        return mask_bg, mask_fg, combined_masks

    # https://matplotlib.org/examples/widgets/rectangle_selector.html
    def HandleLineSelect(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        print("lb(%d, %d) --> ub(%d, %d)" % (x1, y1, x2, y2))
        # print(" The button you used were: %s %s" % (
        #  eclick.button, erelease.button))
        self.RS.set_active(False)
        mask_bg, mask_fg, combined_masks = self.PerformGrabCut((x1, x2), (y1, y2))

        plt.figure().set_size_inches(10, 10)
        plt.subplot(1, 3, 1)
        plt.imshow(mask_bg)
        plt.title("BG mask")
        plt.subplot(1, 3, 2)
        plt.imshow(mask_fg)
        plt.title("FG mask")

        plt.subplot(1, 1, 1)
        plt.imshow(combined_masks)
        plt.title("Combined mask + underlying image")
        plt.show()

        self.RS.set_active(True)

    def HandleKeyPress(self, event):
        print(' Key pressed.')
        if event.key in ['Q', 'q']:
            sys.exit(0)


def main():
    params = {
        "color_image": "Input/ycb_blender.jpg",
        "label_image": "Input/ycb_blender_labels.bmp",
        "image_scale": 2,
        "n_hist_bins": 50,
        "dist_lambda": 1/10.,
        "neighbor_inds": [-1, 0, 1],
        "diff_in_color_space": True,
        "B_sigma_scaling": 0.5
    }
    #params = {
    #    "color_image": "Input/bird.jpg",
    #    "image_scale": 4,
    #    "n_hist_bins": 50,
    #    "dist_lambda": 1/50.,
    #    "neighbor_inds": [-1, 0, 1],
    #    "diff_in_color_space": True,
    #    "B_sigma_scaling": 1.
    #}
    #params = {
    #    "color_image": "Input/staff.jpg",
    #    "image_scale": 4,
    #    "n_hist_bins": 50,
    #    "dist_lambda": 1/10.,
    #    "neighbor_inds": [-1, 0, 1],
    #    "diff_in_color_space": True,
    #    "B_sigma_scaling": 0.5
    #}
    #params = {
    #    "color_image": "Input/squirrel.jpg",
    #    "image_scale": 4,
    #    "n_hist_bins": 50,
    #    "dist_lambda": 1/50.,
    #    "neighbor_inds": [-1, 0, 1]
    #}
    #params = {
    #    "color_image": "Input/seagulls.jpg",
    #    "image_scale": 4,
    #    "n_hist_bins": 50,
    #    "dist_lambda": 1/50.,
    #    "neighbor_inds": [-1, 0, 1]
    #}
    #params = {
    #    "color_image": "Input/gundam.jpg",
    #    "label_image": "Input/gundam_labels.bmp",
    #    "image_scale": 4,
    #    "n_hist_bins": 50,
    #    "dist_lambda": 1/10.,
    #    "neighbor_inds": [-1, 0, 1],
    #    "diff_in_color_space": True,
    #    "B_sigma_scaling": 0.5,
    #}


    global grabcut_manager
    grabcut_manager = GrabCutManager(params)



if __name__ == "__main__":
    main()
