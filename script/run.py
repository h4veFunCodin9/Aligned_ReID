"""Run to extract features"""
from __future__ import print_function

import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse
from PIL import Image

from aligned_reid.dataset import create_dataset
from aligned_reid.dataset.PreProcessImage import PreProcessIm
from aligned_reid.model.Model import Model
from aligned_reid.model.TripletLoss import TripletLoss
from aligned_reid.model.loss import global_loss
from aligned_reid.model.loss import local_loss

from aligned_reid.utils.utils import time_str
from aligned_reid.utils.utils import str2bool
from aligned_reid.utils.utils import tight_float_str as tfs
from aligned_reid.utils.utils import may_set_mode
from aligned_reid.utils.utils import load_state_dict
from aligned_reid.utils.utils import load_ckpt
from aligned_reid.utils.utils import save_ckpt
from aligned_reid.utils.utils import set_devices
from aligned_reid.utils.utils import AverageMeter
from aligned_reid.utils.utils import to_scalar
from aligned_reid.utils.utils import ReDirectSTD
from aligned_reid.utils.utils import set_seed
from aligned_reid.utils.utils import adjust_lr_exp
from aligned_reid.utils.utils import adjust_lr_staircase


class Config(object):
    def __init__(self):
        self.set_seed = False
        self.trianset_part = 'trainval' # ['trainval', 'train']
        self.dataset = 'market1501' # ['market501', 'cuhk03', 'duke', 'combined']

        self.model_weight_file = '/Users/luzhoutao/study_abroad/7-DeeCamp/workspace/Aligned_ReID/Dataset/model_weight_all_norm.pth'

        self.sys_device_ids = ()

        if self.set_seed:
            self.seed = 1
        else:
            self.seed = None

        # Whether to scale by 1/255
        self.scale_im = True
        self.resize_h_w = (256, 128)
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]

        # no need to build dataset
        # no need for train argument
        # no need for test argument

        self.im_dir = '/Users/luzhoutao/study_abroad/7-DeeCamp/' \
                      'workspace/Aligned_ReID/Dataset/duke/images'

        dataset_kwargs = dict(
            resize_h_w=self.resize_h_w,
            scale=self.scale_im,
            im_mean=self.im_mean,
            im_std=self.im_std,
            batch_dims='NCHW',
            prng=np.random
        )
        self.pre_process_im = PreProcessIm(**dataset_kwargs)

        # ReID model
        self.normalize_feature = True

        # no need to train
        # no need to log

class ExtractFeature(object):
    def __init__(self, model, TVT):
        self.model = model
        self.TVT = TVT

    def __call__(self, im):
        old_train_eval_model= self.model.training
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model.eval()
        im = Variable(self.TVT(torch.from_numpy(im).float()))
        global_feat, local_feat = self.model(im)[:2]
        global_feat = global_feat.data.cpu().numpy()
        # Restore the model to its old train/val mode.
        self.model.train(old_train_eval_model)
        return global_feat

def get_extractor():
    cfg = Config()

    TVT, TMO = set_devices(cfg.sys_device_ids)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    #######################
    # print configuration #
    #######################

    import pprint
    print('-' * 60)
    print('cfg.__dict__')
    pprint.pprint(cfg.__dict__)
    print('-' * 60)

    ##########
    # Models #
    ##########

    model = Model() # num_class is None, no need for local feature
    model_w = DataParallel(model)

    print(model.state_dict().keys())

    # Transfer model to specify devices
    TMO([model])

    assert(cfg.model_weight_file != '')
    map_location = (lambda storage, loc: storage)
    sd = torch.load(cfg.model_weight_file, map_location=map_location)
    if 'state_dicts' in sd:        # load from checkpoint
        print(len(sd['state_dicts']))
        sd = sd['state_dicts'][0]
    load_state_dict(model, sd)
    print('Loaded model weights from {}'.format(cfg.model_weight_file))

    ###########
    # Extract #
    ###########
    extractor = ExtractFeature(model_w, TVT)
    return extractor

if __name__ == '__main__':
    # terrace
    video_path = "Dataset/epfl/terrace2-c0.avi"

    import cv2
    cap = cv2.VideoCapture(video_path)
    fid, frame = 0, None # frame start with index 1


    # extractor
    extractor = get_extractor()

    # read bbox
    import numpy as np
    bbox_path = "Dataset/epfl/bbox.txt"
    # frame-id, x, y, w, h, conf, features
    records = []

    start, count = time.time(), 0
    with open(bbox_path, 'r') as f:
        line = f.readline()
        while line:
            id, x, y, w, h, conf = line.strip().split()
            id, x, y, w, h, conf = int(id), float(x), float(y), float(w), float(h), float(conf)
            x1, y1 = int(np.round(max(0, x))), int(np.round(max(0, y)))
            x2, y2 = int(np.round(max(0, x+w))), int(np.round(max(0, y+h)))

            while fid < id:
                succ, frame = cap.read()
                fid += 1

            p = frame[y1:y2, x1:x2]

            new_p = np.zeros([3, p.shape[0], p.shape[1]])
            new_p[0], new_p[1], new_p[2] = p[:,:,0], p[:,:,1], p[:,:,2]
            im = np.stack([new_p], axis=0)

            '''if feat is not None:
                next_feat = extractor(im)
                print(np.sqrt(np.sum(np.square(np.subtract(feat, next_feat)))))
                feat = next_feat
            else:
                feat = extractor(im)'''

            '''print(feat.shape)'''
            feat = extractor(im)

            rec = np.zeros([6 + feat.shape[1]])
            rec[0], rec[1], rec[2], rec[3], rec[4], rec[5] = id, x, y, w, h, conf
            rec[6:] = feat[0]

            records.append(rec)

            count += 1
            line = f.readline()

            print(str(count), end='\r\n')

    end = time.time()
    print(end - start, (end - start) / count)

    records = np.stack(records, axis=0)
    np.save('bbox_feats.npy', records)
