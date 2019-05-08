from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse
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


def main():
    # CONFIG PARAMS
    B_sigma = 0.25

    # Load interesting image
    img = Image.open("Input/ycb_blender.jpg").resize((160, 120))
    img_lum = img.convert('L')
    img_lum_array = np.array(img_lum)/255.
    img_array = np.array(img)/255.
    img_labels_gt = Image.open("Input/ycb_blender_labels_gt.bmp").resize((160, 120))
    img_labels_gt_array = np.array(img_labels_gt)/255.
    img_labels = Image.open("Input/ycb_blender_labels.bmp").resize((160, 120))
    img_labels_array = np.array(img_labels)/255.

    rows = img_array.shape[0]
    cols = img_array.shape[1]
    print("Input resolution: %d rows x %d cols" % (rows, cols))

    # Generate background and foreground index lists.
    # Mask foreground in red (taking anything > 250)
    fg_inds = np.vstack(np.nonzero(img_labels_array[:, :, 0] > 250))
    n_fg_inds = fg_inds.shape[1]
    # Mask foreground in blue (taking anything > 250)
    bg_inds = np.vstack(np.nonzero(img_labels_array[:, :, 2] > 250))
    n_bg_inds = bg_inds.shape[1]
    print("Number of labels: %d foreground, %d background" %
          (n_fg_inds, n_bg_inds))

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
        (np.zeros(n_bg_inds),
         (np.ones(n_bg_inds, dtype=np.long)*source_node_ind,
          uv_to_ind(bg_inds[0, :], bg_inds[1, :]))),
        shape=graph_matrix.shape)
    # Background nodes connect to the sink with exceeding large weight.
    graph_matrix += scipy.sparse.coo_matrix(
        (np.ones(n_bg_inds)*K,
         (np.ones(n_bg_inds, dtype=np.long)*sink_node_ind,
          uv_to_ind(bg_inds[0, :], bg_inds[1, :]))),
        shape=graph_matrix.shape)
    # Foreground nodes connect to the sink for free.
    graph_matrix += scipy.sparse.coo_matrix(
        (np.zeros(n_fg_inds),
         (np.ones(n_fg_inds, dtype=np.long)*sink_node_ind,
          uv_to_ind(fg_inds[0, :], fg_inds[1, :]))),
        shape=graph_matrix.shape)
    # Foreground nodes connect to source with larger weight than
    # any other edge going to that node.
    graph_matrix += scipy.sparse.coo_matrix(
        (np.ones(n_fg_inds)*K,
         (np.ones(n_fg_inds, dtype=np.long)*source_node_ind,
          uv_to_ind(fg_inds[0, :], fg_inds[1, :]))),
        shape=graph_matrix.shape)

    print("TODO: background histogram term.")
    print("Bg and fg relationship terms added in %f seconds." %
          (time.time() - start_sink_and_source))

    start_laplacian = time.time()
    affinity_part = graph_matrix[:-2, :-2].copy()
    affinity_part.data /= np.max(affinity_part.data)
    affinity_part.data = affinity_part.data
    affinity_part.setdiag(1.)
    # Eq 3, Semantic Soft Segmentation, approx laplacian
    D = scipy.sparse.diags(affinity_part.dot(np.ones(rows*cols)))
    D_isqrt = scipy.sparse.diags(
        np.reciprocal(np.sqrt(
            affinity_part.dot(np.ones(rows*cols)))))
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
        eigvec = np.real(lap_eigvecs[:, k])
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
