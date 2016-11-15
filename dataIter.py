import os
import numpy as np


class Batch_data_from_file_iter():
    """
    An iterator over a dataset file, which converts each
    line of the file into an example.

    The option ``'load_line'`` is a function which, given
    a string (a line in the file) outputs an example.
    """

    def __init__(self, filename, batch_size=80, sentence_max_size=50):
        self.filename = filename
        self.batch_size = batch_size
        self.sentence_max_size = sentence_max_size
        self.f = open(os.path.expanduser(self.filename))

    def __iter__(self):
        return self

    def load_line(self, line):
        return np.array(map(int, line.split()), dtype=np.int64)

    def next(self):
        batch_x = []
        i = 0
        while i < self.batch_size:
            line = self.f.readline()
            if not line:  # if line is empty
                if i == 0:
                    self.f.close()
                    self.f = open(os.path.expanduser(self.filename))
                    raise StopIteration()
                else:
                    break
            x = self.load_line(line)
            if x.shape[0] <= self.sentence_max_size:
                batch_x += [x]
                i += 1

        return batch_x


def prepare_data(x_list):
    length_max = 0
    for example in x_list:
        ex_length = example.shape[0]
        if length_max < ex_length:
            length_max = ex_length

    # x = np.zeros([length_max + 1, len(x_list)], dtype=np.int64)
    # x_mask = np.zeros_like(x, dtype=np.float32)
    # for i, example in enumerate(x_list):
    #     x[0:example.shape[0], i] = example
    #     x_mask[0:example.shape[0] + 1, i] = np.ones([example.shape[0] + 1], dtype=np.int32)

    x = np.zeros([length_max, len(x_list)], dtype=np.int64)
    x_mask = np.zeros_like(x, dtype=np.float32)
    for i, example in enumerate(x_list):
        x[0:example.shape[0], i] = example
        x_mask[0:example.shape[0], i] = np.ones([example.shape[0]], dtype=np.int32)

    return x, x_mask
