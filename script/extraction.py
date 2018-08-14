from __future__ import print_function

import sys
import os
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
from torch.nn.parallel import DataParallel

import numpy as np
from PIL import Image

from aligned_reid.dataset.PreProcessImage import PreProcessIm
from aligned_reid.model.Model import Model
from aligned_reid.utils.utils import load_state_dict
from aligned_reid.utils.utils import set_devices

sys_device_ids = (0,)
TVT, TMO = set_devices(sys_device_ids)

scale_im = True
im_mean = [0.486, 0.459, 0.408]
im_std = [0.229, 0.224, 0.225]

resize_h_w = (256, 128)

normalize_featre = True

model_weight_file = ""
local_conv_out_channels = 128


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
   #print(ims.shape)
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    global_feat, local_feat = self.model(ims)[:2]
    global_feat = global_feat.data.cpu().numpy()
    local_feat = local_feat.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return global_feat

def prepare():
    model = Model(local_conv_out_channels)
    model_w = DataParallel(model)

    map_location = (lambda storage, loc: storage)
    sd = torch.load(model_weight_file, map_location=map_location)
    if 'state_dicts' in sd:
        sd = sd['state_dicts'][0]
    load_state_dict(model, sd)
    print('Loaded model weight from {}'.format(model_weight_file))

    TMO([model])

    extractor = ExtractFeature(model_w, TVT)
    preprocessor = PreProcessIm(resize_h_w, scale=scale_im, im_mean=im_mean, im_std=im_std)

    return extractor, preprocessor

def run(preprocessor, extractor, im_path):
    im = np.asarray(Image.open(im_path))
    im, _ = preprocessor(im)
    im = np.stack([im], axis=0)
    feat = extractor(im)
    return feat

if __name__ == '__main__':
    im_dir = '/Users/luzhoutao/study_abroad/7-DeeCamp/workspace/beyond-part-models/eval/people/s1c0/0'  # '/mnt/md1/lztao/deecamp/ReID-beyond_part_model/eval/people/s2c0/'
    feat_dir = '/Users/luzhoutao/study_abroad/7-DeeCamp/workspace/beyond-part-models/eval/feats/s1c0/0'

    preprocessor, extractor = prepare()
    for im_name in os.listdir(im_dir):
        feat = run(preprocessor, extractor, os.path.join(im_dir, im_name))
        np.save(os.path.join(feat_dir, im_name.split('.')[0]), feat)
        print(im_name, end='\r')