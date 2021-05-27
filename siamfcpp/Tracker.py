# -*- coding: utf-8 -*
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import time
from .utils.bbox import cxywh2xywh, xywh2cxywh, xyxy2cxywh
from .utils.crop import get_crop, get_subwindow_tracking
from .utils.misc import imarray_to_tensor, tensor_to_numpy

from .tracking_utils import postprocess_box, postprocess_score, restrict_box, cvt_box_crop2frame

class SiamFCppTracker(object):
    r"""
    Basic SiamFC++ tracker

    Hyper-parameters
    ----------------
        total_stride: int
            stride in backbone
        context_amount: float
            factor controlling the image patch cropping range. Set to 0.5 by convention.
        test_lr: float
            factor controlling target size updating speed
        penalty_k: float
            factor controlling the penalization on target size (scale/ratio) change
        window_influence: float
            factor controlling spatial windowing on scores
        windowing: str
            windowing type. Currently support: "cosine"
        z_size: int
            template image size
        x_size: int
            search image sizPipelineBasee
        num_conv3x3: int
            number of conv3x3 tiled in head
        min_w: float
            minimum width
        min_h: float
            minimum height
        phase_init: str
            phase name for template feature extraction
        phase_track: str
            phase name for target search

    Hyper-parameters (to be calculated at runtime)
    ----------------------------------------------
    score_size: int
        final feature map
    score_offset: int
        final feature map
    """
    default_hyper_params = dict(
        total_stride=8,
        context_amount=0.5,
        test_lr=0.52,
        penalty_k=0.04,
        window_influence=0.3,
        windowing="cosine",
        z_size=127,
        x_size=303,
        num_conv3x3=3,
        min_w=10,
        min_h=10,
        phase_init="feature",
        phase_track="track",
    )

    def __init__(self, debug=False):
        super().__init__()
        self._hyper_params = self.default_hyper_params  # mapping-like object
        self._state = dict()  # pipeline state

        self.update_params()

        # set underlying model to device
        self.model = None
        self.device = torch.device("cuda")
        self.debug = debug
        self._state['is_lost']=False

    def set_model(self, model):
        self.model = model.to(self.device)
        self.model.eval()

    def to_device(self, device):
        self.device = device
        self.model = self.model.to(device)

    def update_params(self):
        hps = self._hyper_params
        hps['score_size'] = (
            hps['x_size'] -
            hps['z_size']) // hps['total_stride'] + 1 - hps['num_conv3x3'] * 2
        hps['score_offset'] = (
            hps['x_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps

    def feature(self, im, target_pos, target_sz, avg_chans=None):
        r"""
        Extract feature
        :param im: initial frame
        :param target_pos: target position (x, y)
        :param target_sz: target size (w, h)
        :param avg_chans: channel mean values
        :return:
        """
        if avg_chans is None:
            avg_chans = np.mean(im, axis=(0, 1))

        z_size = self._hyper_params['z_size']
        context_amount = self._hyper_params['context_amount']

        im_z_crop, _ = get_crop(
            im,
            target_pos,
            target_sz,
            z_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        phase = self._hyper_params['phase_init']
        with torch.no_grad():
            features = self.model(imarray_to_tensor(im_z_crop).to(self.device),
                                  phase=phase)

        return features, im_z_crop, avg_chans

    def init(self, im, state):
        r"""
        Initialize tracker
            Internal target state representation: self._state['state'] = (target_pos, target_sz)
        :param im: initial frame image
        :param state: bbox, format: xywh
        :return: None
        """
        rect = state  # bbox in xywh format is given for initialization in case of tracking
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]

        self._state['im_h'] = im.shape[0]
        self._state['im_w'] = im.shape[1]

        # extract template feature
        features, im_z_crop, avg_chans = self.feature(im, target_pos, target_sz)

        score_size = self._hyper_params['score_size']
        if self._hyper_params['windowing'] == 'cosine':
            window = np.outer(np.hanning(score_size), np.hanning(score_size))
            window = window.reshape(-1)
        elif self._hyper_params['windowing'] == 'uniform':
            window = np.ones((score_size, score_size))
        else:
            window = np.ones((score_size, score_size))

        self._state['z_crop'] = im_z_crop
        self._state['avg_chans'] = avg_chans
        self._state['features'] = features
        self._state['window'] = window
        # self.state['target_pos'] = target_pos
        # self.state['target_sz'] = target_sz
        self._state['state'] = (target_pos, target_sz)

    def track(self,
              im_x,
              target_pos,
              target_sz,
              features,
              update_state=False,
              **kwargs):
        if 'avg_chans' in kwargs:
            avg_chans = kwargs['avg_chans']
        else:
            avg_chans = self._state['avg_chans']

        z_size = self._hyper_params['z_size']
        x_size = self._hyper_params['x_size']
        context_amount = self._hyper_params['context_amount']
        phase_track = self._hyper_params['phase_track']
        im_x_crop, scale_x = get_crop(
            im_x,
            target_pos,
            target_sz,
            z_size,
            x_size=x_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        with torch.no_grad():
            score, box, cls, ctr, *args = self.model(
                imarray_to_tensor(im_x_crop).to(self.device),
                *features,
                phase=phase_track)

        box = tensor_to_numpy(box[0])
        score = tensor_to_numpy(score[0])[:, 0]
        cls = tensor_to_numpy(cls[0])
        ctr = tensor_to_numpy(ctr[0])
        box_wh = xyxy2cxywh(box)

        # score post-processing
        is_lost=(score.max()<0.1)
        self._state['is_lost']=is_lost
        best_pscore_id, pscore, penalty = postprocess_score(
            score, box_wh, target_sz, scale_x, self._hyper_params["penalty_k"], self._state['window'], self._hyper_params["window_influence"])
        # box post-processing
        new_target_pos, new_target_sz = postprocess_box(
            best_pscore_id, score, box_wh, target_pos, target_sz, scale_x,
            x_size, penalty, self._hyper_params["test_lr"])

        if self.debug:
            box = cvt_box_crop2frame(box_wh, target_pos, x_size, scale_x)

        # restrict new_target_pos & new_target_sz
        new_target_pos, new_target_sz = restrict_box(
            new_target_pos, new_target_sz, self._state['im_w'], self._state['im_h'], self._hyper_params['min_h'], self._hyper_params['min_w'])

        # record basic mid-level info
        self._state['x_crop'] = im_x_crop
        bbox_pred_in_crop = np.rint(box[best_pscore_id]).astype(np.int)
        self._state['bbox_pred_in_crop'] = bbox_pred_in_crop
        # record optional mid-level info
        if update_state:
            self._state['score'] = score
            self._state['pscore'] = pscore
            self._state['all_box'] = box
            self._state['cls'] = cls
            self._state['ctr'] = ctr

        return new_target_pos, new_target_sz,target_pos

    def update(self, im):
        # get track
        # target_pos_prior, target_sz_prior = self.state['target_pos'], self.state['target_sz']
        target_pos_prior, target_sz_prior = self._state['state']
        features = self._state['features']
        # forward inference to estimate new state
        target_pos, target_sz,pos= self.track(im,
                                           target_pos_prior,
                                           target_sz_prior,
                                           features,
                                           update_state=True)
        # save underlying state
        # self.state['target_pos'], self.state['target_sz'] = target_pos, target_sz
        self._state['state'] = target_pos, target_sz

        # return rect format
        track_rect = cxywh2xywh(np.concatenate([target_pos, target_sz],
                                               axis=-1))
        return track_rect,self._state['is_lost'], pos

    # ======== tracking processes ======== #


