from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse
from scipy.interpolate import interp1d
import time
from PIL import Image

import networkx
import networkx.algorithms.flow as flow


def perform_min_cut(cgraph, s_index, t_index, residual=None):
    print("Starting min cut with networkx library.")
    start_time = time.time()
    G = networkx.from_scipy_sparse_matrix(cgraph, edge_attribute="capacity")
    print("... converted graph in %f seconds." % (time.time() - start_time))
    start_time = time.time()
    cut_value, partition = flow.minimum_cut(
        G, s_index, t_index,
        flow_func=flow.preflow_push, residual=residual)
    print("... final cut value %f. " % (cut_value))
    # Sanity-check which is foreground / background?
    if s_index in partition[0]:
        foreground_inds = partition[0]
        background_inds = partition[1]
    else:
        foreground_inds = partition[1]
        background_inds = partition[0]

    print("... foreground has %d nodes, background has %d nodes." %
          (len(foreground_inds), len(background_inds)))
    print("... done in %f seconds." % (time.time() - start_time))
    return foreground_inds, background_inds


class HistogramDistribution():
    def __init__(self, data, **kwargs):
        hist, bin_edges = np.histogram(data, density=True, **kwargs)
        self.pdf_approx = interp1d(bin_edges[:-1], hist,
                                   kind='previous', bounds_error=False,
                                   fill_value="extrapolate")

    def pdf(self, x):
        return self.pdf_approx(x)

    def log_pdf(self, x):
        # Avoid -infs
        return np.log(self.pdf_approx(x) + 1E-6)


def build_affinity_graph(img_array, img_lum_array, img_labels_array,
                         dist_lambda, n_hist_bins, neighbor_inds,
                         make_plots=False,
                         B_sigma_scaling=1.0,
                         connect_node_labels=True,
                         connect_color_priors=True,
                         diff_in_color_space=False):
    rows, cols = img_lum_array.shape

    # Generate background and foreground index lists.
    # Mask foreground in red (taking anything close
    # in case image compression does weird things)
    fg_label_mask = img_labels_array[:, :, 0] > 0.9
    fg_label_inds = np.vstack(np.nonzero(fg_label_mask))
    n_fg_label_inds = fg_label_inds.shape[1]
    # Mask foreground in blue (taking anything close
    # in case image compression does weird things)
    bg_label_mask = img_labels_array[:, :, 2] > 0.9
    bg_label_inds = np.vstack(np.nonzero(bg_label_mask))
    n_bg_label_inds = bg_label_inds.shape[1]
    print("Number of labels: %d foreground, %d background" %
          (n_fg_label_inds, n_bg_label_inds))
    # And generate a not-labeled index list.
    not_labeled_mask = np.logical_not(bg_label_mask + fg_label_mask)
    not_labeled_inds = np.vstack(np.nonzero(not_labeled_mask))
    n_not_labeled_inds = not_labeled_inds.shape[1]

    # Generate a histogram of itensities over the labeled regions
    # and precompute the log-likelihoods at the not-labeled positions
    if n_bg_label_inds > 0:
        bg_intensity_hist = HistogramDistribution(
            img_lum_array, weights=1.*bg_label_mask,
            bins=n_hist_bins, range=(0., 1.))
        bg_nll_all = -bg_intensity_hist.log_pdf(img_lum_array)
        bg_nll = bg_nll_all[not_labeled_inds[0, :],
                            not_labeled_inds[1, :]]
        print("BG NLL range: %f to %f" % (np.min(bg_nll), np.max(bg_nll)))
    else:
        bg_nll = 1.

    if n_fg_label_inds > 0:
        fg_intensity_hist = HistogramDistribution(
            img_lum_array, weights=1.*fg_label_mask,
            bins=n_hist_bins, range=(0., 1.))
        fg_nll_all = -fg_intensity_hist.log_pdf(img_lum_array)
        fg_nll = fg_nll_all[not_labeled_inds[0, :],
                            not_labeled_inds[1, :]]

        print("FG NLL range: %f to %f" % (np.min(fg_nll), np.max(fg_nll)))
    else:
        fg_nll = 1.

    if make_plots:
        plt.figure()
        plt.title("Foreground and background nlls over image")
        plt.subplot(1, 2, 1)
        if (n_fg_label_inds > 0):
            plt.imshow(fg_nll_all)
        plt.title("FG NLL")
        plt.subplot(1, 2, 2)
        if (n_bg_label_inds > 0):
            plt.imshow(bg_nll_all)
        plt.title("BG NLL")
        plt.pause(1E-3)

    if make_plots:
        plt.figure()
        plt.subplot(1, 2, 1)
        xi = np.linspace(0.0, 1.0, 255)
        if (n_fg_label_inds > 0):
            plt.plot(xi, fg_intensity_hist.log_pdf(xi), label="log_pdf")
            plt.plot(xi, fg_intensity_hist.pdf(xi), label="pdf")
        plt.legend()
        plt.title("Foreground histogram")
        plt.subplot(1, 2, 2)
        if (n_bg_label_inds > 0):
            plt.plot(xi, bg_intensity_hist.log_pdf(xi), label="log_pdf")
            plt.plot(xi, bg_intensity_hist.pdf(xi), label="pdf")
        plt.legend()
        plt.title("Background histogram")
        plt.pause(0.1)

    # Build segmentation graph from GraphCut
    N_nodes = rows*cols + 2
    graph_matrix = scipy.sparse.csr_matrix(((N_nodes, N_nodes)))
    sink_node_ind = N_nodes - 1
    source_node_ind = N_nodes - 2

    def uv_to_ind(u, v):
        return u * cols + v

    # Compute pairwise relationships terms
    start_Bpq = time.time()
    if make_plots:
        plt.figure()
    nn = len(neighbor_inds)
    for ui, u_neighbor_dir in enumerate(neighbor_inds):
        us, vs = np.meshgrid(range(rows), range(cols), indexing='ij')
        inds = uv_to_ind(us, vs)
        for vi, v_neighbor_dir in enumerate(neighbor_inds):
            if u_neighbor_dir == v_neighbor_dir == 0:
                dist = 1.
            else:
                dist = np.sqrt(float(u_neighbor_dir**2 + v_neighbor_dir**2))

            if diff_in_color_space:
                diff_im = (img_array -
                           np.roll(img_array,
                                   (u_neighbor_dir, v_neighbor_dir), range(2)))
                diff_im = np.sum(np.square(diff_im).astype(float), axis=2)
            else:
                diff_im = (img_lum_array -
                           np.roll(img_lum_array,
                                   (u_neighbor_dir, v_neighbor_dir), range(2)))
                diff_im = np.square(diff_im).astype(float)

            # Calculate sigma via Eq 5 in the GrabCut paper as scaled
            # to the expected contrast across the whole image
            B_sigma = B_sigma_scaling * (np.mean(diff_im)) + 1E-6
            # Luminance smoothness -- higher scores for closeness
            # of neighboring colors.
            Bpq = np.exp(-diff_im / (2. * B_sigma))/dist
            rolled_inds = np.roll(
                inds, (u_neighbor_dir, v_neighbor_dir), range(2))

            # Ugly logic for cutting off edges that aren't valid
            # in this cycle:
            valid_inds = inds.copy()
            if u_neighbor_dir >= 1:
                Bpq = Bpq[u_neighbor_dir:, :]
                rolled_inds = rolled_inds[u_neighbor_dir:, :]
                valid_inds = valid_inds[u_neighbor_dir:, :]
            elif u_neighbor_dir <= -1:
                Bpq = Bpq[:u_neighbor_dir, :]
                rolled_inds = rolled_inds[:u_neighbor_dir, :]
                valid_inds = valid_inds[:u_neighbor_dir, :]
            if v_neighbor_dir >= 1:
                Bpq = Bpq[:, v_neighbor_dir:]
                rolled_inds = rolled_inds[:, v_neighbor_dir:]
                valid_inds = valid_inds[:, v_neighbor_dir:]
            elif v_neighbor_dir <= -1:
                Bpq = Bpq[:, :v_neighbor_dir]
                rolled_inds = rolled_inds[:, :v_neighbor_dir]
                valid_inds = valid_inds[:, :v_neighbor_dir]
            graph_matrix += scipy.sparse.coo_matrix(
                (Bpq.flatten(), (valid_inds.flatten(), rolled_inds.flatten())),
                shape=graph_matrix.shape)
            if make_plots:
                plt.subplot(nn, nn, ui*nn + vi + 1)
                plt.title("Offset %d, %d" % (u_neighbor_dir, v_neighbor_dir))
                plt.imshow(Bpq)
    if make_plots:
        plt.tight_layout()

    # Find a K that exceeds the sum (across pixels) of the max connection
    # going in to each pixel.
    K_per_pixel = graph_matrix[:-2, :-2].max(axis=1).toarray().squeeze()
    K = np.max(K_per_pixel) + 1
    # That's a little burdensome to calculate around, and only
    # matters as a "sufficiently big" number, so I take the max of that
    # over all pixels here instead.
    print("Pairwise relationship terms added in %f seconds." % (
        time.time() - start_Bpq))

    start_sink_and_source = time.time()
    if connect_node_labels:
        # Background nodes connect to the source for free..
        if (n_bg_label_inds > 0):
            graph_matrix += scipy.sparse.coo_matrix(
                (np.zeros(n_bg_label_inds),
                 (np.ones(n_bg_label_inds, dtype=np.long)*source_node_ind,
                  uv_to_ind(bg_label_inds[0, :], bg_label_inds[1, :]))),
                shape=graph_matrix.shape)
            # Background nodes connect to the sink with exceeding large weight.
            graph_matrix += scipy.sparse.coo_matrix(
                (K * np.ones(n_bg_label_inds),
                 (np.ones(n_bg_label_inds, dtype=np.long)*sink_node_ind,
                  uv_to_ind(bg_label_inds[0, :], bg_label_inds[1, :]))),
                shape=graph_matrix.shape)

        if (n_fg_label_inds > 0):
            # Foreground nodes connect to the sink for free.
            graph_matrix += scipy.sparse.coo_matrix(
                (np.zeros(n_fg_label_inds),
                 (np.ones(n_fg_label_inds, dtype=np.long)*sink_node_ind,
                  uv_to_ind(fg_label_inds[0, :], fg_label_inds[1, :]))),
                shape=graph_matrix.shape)
            # Foreground nodes connect to source with larger weight than
            # any other edge going to that node.
            graph_matrix += scipy.sparse.coo_matrix(
                (K * np.ones(n_fg_label_inds),
                 (np.ones(n_fg_label_inds, dtype=np.long)*source_node_ind,
                  uv_to_ind(fg_label_inds[0, :], fg_label_inds[1, :]))),
                shape=graph_matrix.shape)

    if connect_color_priors:
        if not connect_node_labels:
            # All nodes should connect to the background and foreground
            # with weight on how they fit into the OPPOSITE intensity histograms.
            if n_bg_label_inds > 0:
                graph_matrix[source_node_ind, :-2] += bg_nll_all.flatten()
            if n_fg_label_inds > 0:
                graph_matrix[sink_node_ind, :-2] += fg_nll_all.flatten()
        else:
            # Unlabled nodes should connect to the background and foreground
            # with weight on how they fit into the OPPOSITE intensity histograms.
            graph_matrix += scipy.sparse.coo_matrix(
                (np.ones(n_not_labeled_inds)*bg_nll*dist_lambda,
                 (np.ones(n_not_labeled_inds, dtype=np.long)*source_node_ind,
                  uv_to_ind(not_labeled_inds[0, :], not_labeled_inds[1, :]))),
                shape=graph_matrix.shape)

            graph_matrix += scipy.sparse.coo_matrix(
                (np.ones(n_not_labeled_inds)*fg_nll*dist_lambda,
                 (np.ones(n_not_labeled_inds, dtype=np.long)*sink_node_ind,
                  uv_to_ind(not_labeled_inds[0, :], not_labeled_inds[1, :]))),
                shape=graph_matrix.shape)

    # Duplicate all of the above into the transpose part of
    # the matrix.
    graph_matrix[:, -2:] += graph_matrix[-2:, :].T
    if make_plots:
        plt.figure()
        plt.title("Weights to the sink and source")
        plt.subplot(2, 2, 1)
        plt.imshow(np.asarray(graph_matrix[:-2, -2].todense()).reshape(img_lum_array.shape))
        plt.title("To source")
        plt.subplot(2, 2, 2)
        plt.title("To sink")
        plt.imshow(np.asarray(graph_matrix[:-2, -1].todense()).reshape(img_lum_array.shape))
        plt.subplot(2, 2, 3)
        plt.title("To source")
        plt.imshow(np.asarray(graph_matrix[-2, :-2].todense()).reshape(img_lum_array.shape))
        plt.subplot(2, 2, 4)
        plt.title("To sink")
        plt.imshow(np.asarray(graph_matrix[-1, :-2].todense()).reshape(img_lum_array.shape))
        plt.pause(1E-3)

    print("Bg and fg relationship terms added in %f seconds." %
          (time.time() - start_sink_and_source))
    return graph_matrix, source_node_ind, sink_node_ind


def main():
    # CONFIG PARAMS

    # n_hist_bins: # of histogram bins used for computing prior over
    # foreground + background colors.
    # Image_scale: Ratio by which the input image is scaled down.
    # Dist_lambda: Multiplicative weight given to edges
    # connecting unlabeled nodes to the background
    # based on their negative log likelihood under
    # the background/foreground intensity distributions.

    do_min_cut = True
    do_laplacian = False

    # Load interesting image
    image_set = []

    #params = {
    #    "color_image": "Input/jeb.png",
    #    "label_image": "Input/jeb_labels.bmp",
    #    "image_scale": 1,
    #    "n_hist_bins": 50,
    #    "dist_lambda": 1/50.,
    #    "neighbor_inds": [-1, 0, 1],
    #   "diff_in_color_space": True
    #}

    #params = {
    #    "color_image": "Input/gundam.jpg",
    #    "label_image": "Input/gundam_labels.bmp",
    #    "image_scale": 8,
    #    "n_hist_bins": 50,
    #    "dist_lambda": 1/50.,
    #    "neighbor_inds": [-1, 0, 1]
    #}
#
    params = {
        "color_image": "Input/ycb_blender.jpg",
        "label_image": "Input/ycb_blender_labels.bmp",
        "image_scale": 4,
        "n_hist_bins": 50,
        "dist_lambda": 1/50.,
        "neighbor_inds": [-1, 0, 1],
        "diff_in_color_space": True
    }

    #params = {
    #    "color_image": "Input/bird.jpg",
    #    "label_image": "Input/bird_labels.bmp",
    #    "image_scale": 4,
    #    "n_hist_bins": 50,
    #    "dist_lambda": 1/50.,
    #    "neighbor_inds": [-1, 0, 1],
    #    "diff_in_color_space": True
    #}

    #params = {
    #    "color_image": "Input/squirrel.jpg",
    #    "label_image": "Input/squirrel_labels.bmp",
    #    "image_scale": 4,
    #    "n_hist_bins": 50,
    #    "dist_lambda": 1/50.,
    #    "neighbor_inds": [-1, 0, 1],
    #    "diff_in_color_space": True
    #}


    n_hist_bins = params["n_hist_bins"]
    dist_lambda = params["dist_lambda"]
    color_image = params["color_image"]
    label_image = params["label_image"]
    image_scale = params["image_scale"]
    neighbor_inds = params["neighbor_inds"]
    diff_in_color_space = params["diff_in_color_space"]

    img = Image.open(color_image)
    img_orig_rows = img.height
    img_orig_cols = img.width
    img = img.resize((img_orig_cols/image_scale, img_orig_rows/image_scale))
    img_lum = img.convert('L')
    img_lum_array = np.array(img_lum)/255.
    img_array = np.array(img)/255.
    img_labels = Image.open(label_image).resize(
        (img_orig_cols/image_scale, img_orig_rows/image_scale))
    img_labels_array = np.array(img_labels)/255.

    rows = img_array.shape[0]
    cols = img_array.shape[1]
    print("Input resolution: %d rows x %d cols" % (rows, cols))

    plt.figure().set_size_inches(12, 12)
    plt.subplot(3, 1, 1)
    plt.imshow(img_lum)
    plt.title("Luminance image")
    plt.subplot(3, 1, 2)
    plt.imshow(img_labels)
    plt.title("Label image")
    plt.subplot(3, 1, 3)
    plt.imshow(img_labels_array*0.75 +
               np.stack([img_lum_array]*3, axis=-1))
    plt.title("Labels over luminance image")

    plt.pause(0.1)

    graph_matrix, source_node_ind, sink_node_ind = build_affinity_graph(
        img_array, img_lum_array, img_labels_array, dist_lambda, n_hist_bins,
        neighbor_inds, make_plots=True, connect_node_labels=True,
        connect_color_priors=True, diff_in_color_space=diff_in_color_space,
        B_sigma_scaling=B_sigma_scaling)

    if do_min_cut:
        foreground_inds, background_inds = perform_min_cut(
            graph_matrix, source_node_ind, sink_node_ind)
        background_inds.remove(sink_node_ind)
        foreground_inds.remove(source_node_ind)
        plt.figure().set_size_inches(10, 10)
        plt.subplot(1, 3, 1)
        mask_bg = (img_lum_array*0).reshape(-1)
        mask_bg[list(background_inds)] = 1.0
        mask_bg = mask_bg.reshape(img_lum_array.shape)
        plt.imshow(mask_bg)
        plt.title("BG mask")
        plt.subplot(1, 3, 2)
        mask_fg = (img_lum_array*0).reshape(-1)
        mask_fg[list(foreground_inds)] = 1.0
        mask_fg = mask_fg.reshape(img_lum_array.shape)
        plt.imshow(mask_fg)
        plt.title("FG mask")

        plt.subplot(1, 3, 3)
        combined_masks = np.stack([img_lum_array*0.4 + mask_fg*0.6,
                                   img_lum_array*0.4,
                                   img_lum_array*0.4 + mask_bg*0.6], axis=-1)
        plt.imshow(combined_masks)
        plt.title("Combined mask + underlying image")
        plt.pause(1e-2)

    if do_laplacian:
        # Calculate graph Laplacian and its eigendecomp
        start_laplacian = time.time()
        affinity_part = graph_matrix.copy()
        # Normalize affinity
        #affinity_part.data -= np.min(affinity_part.data)
        #affinity_part.data /= np.max(affinity_part.data)

        # Eq 3, Semantic Soft Segmentation, approx laplacian
        # Graph is undirectional, so axis of sum should not matter.
        D = scipy.sparse.diags(np.asarray(affinity_part.sum(axis=0)).squeeze())
        D_isqrt = scipy.sparse.diags(
            np.reciprocal(np.sqrt(
                affinity_part.dot(np.ones(affinity_part.shape[0])))))
        laplacian = D_isqrt.dot(D - affinity_part).dot(D_isqrt)
        print("Computed laplacian %f seconds." % (
              (time.time() - start_laplacian)))
        #laplacian = scipy.sparse.csgraph.laplacian(affinity_part, normed=True)
        N_eigvecs = 10
        start_laplacian_eigs = time.time()
        lap_eigs, lap_eigvecs = scipy.sparse.linalg.eigs(
            laplacian, k=N_eigvecs)
        print("Computed laplacian  eigs in %f seconds." % (
              (time.time() - start_laplacian_eigs)))

        plt.figure().set_size_inches(12, 12)
        for k in range(N_eigvecs):
            plt.subplot(np.ceil(N_eigvecs/2.), 2, k+1)
            eigvec = np.real(lap_eigvecs[:rows*cols, k])
            eigvec /= np.max(np.abs(eigvec))
            eigvec = eigvec.reshape(rows, cols)
            img_recolored_by_eigvec = np.stack([img_lum_array*0.2,
                                                np.clip(-eigvec, 0., 1.),
                                                np.clip(-eigvec, 0., 1.)],
                                                axis=-1)
            plt.imshow(img_recolored_by_eigvec)
            plt.title("Eigenvec #%d" % k)

    plt.show()

if __name__ == "__main__":
    main()
