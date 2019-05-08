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


# Ref
# https://carlonicolini.github.io/sections/science/2018/09/12/weighted-graph-from-adjacency-matrix-in-graph-tool.html
def image_to_graphcut_graph(adj):
    g = gt.Graph(directed=False)
    edge_weights = g.new_edge_property('float')
    g.edge_properties['weight'] = edge_weights
    nnz = np.nonzero(np.triu(adj, 1))
    nedges = len(nnz[0])
    g.add_edge_list(
        np.hstack(
            [np.transpose(nnz),
             np.reshape(adj[nnz], (nedges, 1))]),
        eprops=[edge_weights])
    return g


class HistogramDistribution():
    def __init__(self, data, **kwargs):
        hist, bin_edges = np.histogram(data, density=True, **kwargs)
        self.pdf_approx = interp1d(bin_edges[:-1], hist,
                                   kind='previous', bounds_error=False,
                                   fill_value=0)

    def pdf(self, x):
        return self.pdf_approx(x)

    def log_pdf(self, x):
        # Avoid -infs
        return np.log(self.pdf_approx(x) + 1E-5)


def main():
    # CONFIG PARAMS
    B_sigma = 0.25
    n_hist_bins = 25
    image_scale = 4
    # Load interesting image
    img = Image.open("Input/ycb_blender.jpg")
    img_orig_rows = img.height
    img_orig_cols = img.width
    img = img.resize((img_orig_cols/image_scale, img_orig_rows/image_scale))
    img_lum = img.convert('L')
    img_lum_array = np.array(img_lum)/255.
    img_array = np.array(img)/255.
    img_labels_gt = Image.open("Input/ycb_blender_labels_gt.bmp").resize((img_orig_cols/image_scale, img_orig_rows/image_scale))
    img_labels_gt_array = np.array(img_labels_gt)/255.
    img_labels = Image.open("Input/ycb_blender_labels.bmp").resize((img_orig_cols/image_scale, img_orig_rows/image_scale))
    img_labels_array = np.array(img_labels)/255.

    rows = img_array.shape[0]
    cols = img_array.shape[1]
    print("Input resolution: %d rows x %d cols" % (rows, cols))

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
    fg_intensity_hist = HistogramDistribution(
        img_lum_array, weights=1.*fg_label_mask,
        bins=n_hist_bins, range=(0., 1.))
    bg_intensity_hist = HistogramDistribution(
        img_lum_array, weights=1.*bg_label_mask,
        bins=n_hist_bins, range=(0., 1.))

    # Precompute the log-likelihoods at the not-labeled positions
    fg_nll = -fg_intensity_hist.log_pdf(img_lum_array)
    fg_nll = fg_nll[not_labeled_inds[0, :],
                    not_labeled_inds[1, :]]
    print("FG NLL range: %f to %f" % (np.min(fg_nll), np.max(fg_nll)))
    bg_nll = -bg_intensity_hist.log_pdf(img_lum_array)
    bg_nll = bg_nll[not_labeled_inds[0, :],
                    not_labeled_inds[1, :]]
    print("BG NLL range: %f to %f" % (np.min(bg_nll), np.max(bg_nll)))

    # Build segmentation graph from GraphCut
    N_nodes = rows*cols + 2
    graph_matrix = scipy.sparse.csr_matrix(((N_nodes, N_nodes)))
    sink_node_ind = N_nodes - 1
    source_node_ind = N_nodes - 2

    def uv_to_ind(u, v):
        return u * cols + v

    # Compute pairwise relationships terms
    start_Bpq = time.time()
    for u_neighbor_dir in [-1, 0, 1]:
        us, vs = np.meshgrid(range(rows), range(cols), indexing='ij')
        inds = uv_to_ind(us, vs)
        valid_inds = inds[1:-1, 1:-1]
        for v_neighbor_dir in [-1, 0, 1]:
            if u_neighbor_dir == v_neighbor_dir == 0:
                continue
            dist = np.sqrt(float(u_neighbor_dir**2 + v_neighbor_dir**2))
            diff_im = (img_lum_array -
                       np.roll(img_lum_array,
                               (u_neighbor_dir, v_neighbor_dir), range(2)))[1:-1, 1:-1]
            # Luminance smoothness -- higher scores for closeness
            # of neighboring colors.
            Bpq = np.exp(-np.square(diff_im).astype(float) / (2. * B_sigma**2.))/dist
            # Same thing, but in full color space.
            # color_diff_im = (img_array - np.roll(img_array, (u_neighbor_dir, v_neighbor_dir), range(2)))[1:-1, 1:-1, :]
            # Bpq = np.exp(-np.sum(np.square(color_diff_im), axis=2).astype(float)/(2.*B_sigma**2.))/dist
            rolled_inds = np.roll(
                inds, (u_neighbor_dir, v_neighbor_dir), range(2))[1:-1, 1:-1]
            graph_matrix += scipy.sparse.coo_matrix(
                (Bpq.flatten(), (valid_inds.flatten(), rolled_inds.flatten())),
                shape=graph_matrix.shape)

    # In the paper, this weight is calculated per pixel as
    #  1 + the max weight connecting that pixel to another pixel.
    # That's a little burdensome to calculate around, and only
    # matters as a "sufficiently big" number, so I take the max of that
    # over all pixels here instead.
    max_Bpq = np.max(graph_matrix.data)
    print("Pairwise relationship terms added in %f seconds. Max B_pq = %f" %
          (time.time()-start_Bpq, max_Bpq))

    start_sink_and_source = time.time()
    K = 1. + max_Bpq
    # Background nodes connect to the source for free..
    graph_matrix += scipy.sparse.coo_matrix(
        (np.zeros(n_bg_label_inds),
         (np.ones(n_bg_label_inds, dtype=np.long)*source_node_ind,
          uv_to_ind(bg_label_inds[0, :], bg_label_inds[1, :]))),
        shape=graph_matrix.shape)
    # Background nodes connect to the sink with exceeding large weight.
    graph_matrix += scipy.sparse.coo_matrix(
        (np.ones(n_bg_label_inds)*K,
         (np.ones(n_bg_label_inds, dtype=np.long)*sink_node_ind,
          uv_to_ind(bg_label_inds[0, :], bg_label_inds[1, :]))),
        shape=graph_matrix.shape)

    # Foreground nodes connect to the sink for free.
    graph_matrix += scipy.sparse.coo_matrix(
        (np.zeros(n_fg_label_inds),
         (np.ones(n_fg_label_inds, dtype=np.long)*sink_node_ind,
          uv_to_ind(fg_label_inds[0, :], fg_label_inds[1, :]))),
        shape=graph_matrix.shape)
    # Foreground nodes connect to source with larger weight than
    # any other edge going to that node.
    graph_matrix += scipy.sparse.coo_matrix(
        (np.ones(n_fg_label_inds)*K,
         (np.ones(n_fg_label_inds, dtype=np.long)*source_node_ind,
          uv_to_ind(fg_label_inds[0, :], fg_label_inds[1, :]))),
        shape=graph_matrix.shape)

    # Unlabled nodes should connect to the background and foreground
    # with weight on how they fit into the respective intensity histograms.
    graph_matrix += scipy.sparse.coo_matrix(
        (np.ones(n_not_labeled_inds)*fg_nll,
         (np.ones(n_not_labeled_inds, dtype=np.long)*sink_node_ind,
          uv_to_ind(not_labeled_inds[0, :], not_labeled_inds[1, :]))),
        shape=graph_matrix.shape)
    graph_matrix += scipy.sparse.coo_matrix(
        (np.ones(n_not_labeled_inds)*bg_nll,
         (np.ones(n_not_labeled_inds, dtype=np.long)*source_node_ind,
          uv_to_ind(not_labeled_inds[0, :], not_labeled_inds[1, :]))),
        shape=graph_matrix.shape)

    print("Bg and fg relationship terms added in %f seconds." %
          (time.time() - start_sink_and_source))

    start_laplacian = time.time()
    affinity_part = graph_matrix[:, :].copy()
    affinity_part.data /= np.max(affinity_part.data)
    affinity_part.data = affinity_part.data
    affinity_part.setdiag(1.)
    # Eq 3, Semantic Soft Segmentation, approx laplacian
    D = scipy.sparse.diags(affinity_part.dot(np.ones(affinity_part.shape[0])))
    D_isqrt = scipy.sparse.diags(
        np.reciprocal(np.sqrt(
            affinity_part.dot(np.ones(affinity_part.shape[0])))))
    laplacian = D_isqrt.dot(D - affinity_part).dot(D_isqrt)
    print("Computed laplacian %f seconds." % (
          (time.time() - start_laplacian)))
    #laplacian = scipy.sparse.csgraph.laplacian(affinity_part, normed=True)
    N_eigvecs = 4
    start_laplacian_eigs = time.time()
    lap_eigs, lap_eigvecs = scipy.sparse.linalg.eigs(
        laplacian, k=N_eigvecs)
    print("Computed laplacian  eigs in %f seconds." % (
          (time.time() - start_laplacian_eigs)))

    plt.figure().set_size_inches(12, 12)
    plt.subplot(3, 1, 1)
    plt.imshow(img_lum)
    plt.title("Luminance image")
    plt.subplot(3, 1, 2)
    plt.imshow(img_labels)
    plt.title("Label image")
    plt.subplot(3, 1, 3)
    plt.imshow(img_labels_gt)
    plt.title("GT labels")

    plt.figure().set_size_inches(12, 12)
    for k in range(N_eigvecs):
        plt.subplot(np.ceil(N_eigvecs/2.), 2, k+1)
        eigvec = np.real(lap_eigvecs[:rows*cols, k])
        eigvec /= np.max(np.abs(eigvec))
        eigvec = eigvec.reshape(rows, cols)
        img_recolored_by_eigvec = np.stack([img_lum_array,
                                            np.clip(-eigvec, 0., 1.),
                                            np.clip(-eigvec, 0., 1.)],
                                            axis=-1)
        plt.imshow(img_recolored_by_eigvec)
        plt.title("Eigenvec #%d" % k)

    plt.show()

if __name__ == "__main__":
    main()
