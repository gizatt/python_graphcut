from __future__ import print_function
import os
import sys
import multiprocessing
try:  # Python 2
    import Queue as queue
except ImportError:  # Python 3
    import queue
import Queue
import time
import traceback
import yaml


class Worker(object):
    """Multiprocess worker."""

    def __init__(self, input_queue, termination_event, error_queue=None):
        self.input_queue = input_queue
        self.termination_event = termination_event
        self.error_queue = error_queue

    def HandleCase(self, prefix, params):
        print("Prefix %s" % prefix)

    def __call__(self, worker_index):
        while ((not self.input_queue.empty()) or
               (not self.termination_event.is_set())):
            try:
                new_data = None
                try:
                    prefix, params = self.input_queue.get(False)
                    self.HandleCase(prefix, params)
                except Queue.Empty:
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

    worker_pool = multiprocessing.Pool(processes=10)
    worker_manager = multiprocessing.Manager()
    input_queue = worker_manager.Queue()
    termination_event = worker_manager.Event()
    result = worker_pool.map_async(
        Worker(input_queue=input_queue,
               termination_event=termination_event),
        range(worker_pool._processes))

    for diff_in_color_space in [True, False]:
        for dist_lambda in [1/10., 1/50., 1/250.]:
            for B_sigma_scaling in [0.5, 1., 2.]:
                params = {
                    "dist_lambda": dist_lambda,
                    "diff_in_color_space": diff_in_color_space,
                    "B_sigma_scaling": B_sigma_scaling
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
        if result.ready():
            break
        time.sleep(1E-6)

    print("All done!")
