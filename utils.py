import csv
import cv2
import numpy as np
import pdb
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size


def tensor2img(x,opt):
    if opt.no_mean_norm:
        x = x.copy() * 255.
        return x.transpose((1,2,0)).astype(np.uint8)

    x[0] = x[0] + opt.mean[0]
    x[1] = x[1] + opt.mean[1]
    x[2] = x[2] + opt.mean[2]
    x = x.transpose((1,2,0)).astype(np.uint8)
    return x

def cvimg2tensor(src):
    out = src.copy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out = out.transpose((2,0,1)).astype(np.float64)
    # out = out / 255
    return out

def DxDy(x):
    # shift one pixel and get difference (for both x and y direction)
    return x[:,:,:,:,:-1] - x[:,:,:,:,1:], x[:,:,:,:-1,:]-x[:,:,:,1:,:]