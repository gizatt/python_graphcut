from __future__ import print_function
import os
import sys
import numpy as np
import yaml

if __name__ == "__main__":
    # Traverse Input/ObjectDiscoverySubset until a folder with
    # "GroundTruth" is found. Then list all images in the folder
    # + their corresponding ground truth image.

    np.random.seed(42)

    all_pairs = []
    startpath = "./Input/ObjectDiscoverySubset"
    for root, dirs, files in os.walk(startpath):
        if "GroundTruth" in dirs:
            for file in files:
                # Find the right extension name for the label image
                for extension in [".png", ".bmp"]:
                    label_image_name = os.path.join(root, "GroundTruth", file[:-4] + extension)
                    if os.path.exists(label_image_name):
                        all_pairs.append({
                            "color_image": os.path.join(root, file),
                            "label_image": label_image_name
                        })
                        break
    n_train = 100
    n_test = 20

    print("%d pairs, picking %d train and %d test at random." %
          (len(all_pairs), n_train, n_test))
    assert(n_train + n_test < len(all_pairs))

    inds = np.random.permutation(range(len(all_pairs)))
    train_inds = inds[:n_train]
    test_inds = inds[n_train:(n_train+n_test)]

    train_pairs = [all_pairs[x] for x in train_inds]
    test_pairs = [all_pairs[x] for x in test_inds]

    with open("./Input/ObjectDiscoverySubset/train.yaml", 'w') as f:
        yaml.dump(train_pairs, f, default_flow_style=False)

    with open("./Input/ObjectDiscoverySubset/test.yaml", 'w') as f:
        yaml.dump(test_pairs, f, default_flow_style=False)
