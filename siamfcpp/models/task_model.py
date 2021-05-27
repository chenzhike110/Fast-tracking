import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_block import (conv_bn_relu, xcorr_depthwise)
from .module_base import ModuleBase

class SiamTrack(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(
        pretrain_model_path="./siamfcpp/models/siamfcpp-tinyconv-vot.pkl",
        # pretrain_model_path="./siamfcpp/models/siamfcpp-alexnet-vot.pkl",
        head_width=64,
    )

    def __init__(self, backbone, head, loss):
        super(SiamTrack, self).__init__()
        self.basemodel = backbone.eval()
        # head
        self.head = head.eval()
        # loss
        self.loss = loss
        self._hyper_params = self.default_hyper_params

    def forward(self, *args, phase="track"):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        # phase: train
        if phase == 'train':
            target_img, search_img = args
            # backbone feature
            f_z = self.basemodel(target_img)
            f_x = self.basemodel(search_img)
            # feature adjustment
            c_z_k = self.c_z_k(f_z)
            r_z_k = self.r_z_k(f_z)
            c_x = self.c_x(f_x)
            r_x = self.r_x(f_x)
            # feature matching
            r_out = xcorr_depthwise(r_x, r_z_k)
            c_out = xcorr_depthwise(c_x, c_z_k)
            # head
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final = self.head(
                c_out, r_out)
            # fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            # fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # output
            out_list = fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final
        # phase: feature
        elif phase == 'feature':
            target_img, = args
            # backbone feature
            f_z = self.basemodel(target_img)
            # template as kernel
            c_z_k = self.c_z_k(f_z)
            r_z_k = self.r_z_k(f_z)
            # output
            out_list = [c_z_k, r_z_k]

        # phase: track
        elif phase == 'track':
            if len(args) == 3:
                search_img, c_z_k, r_z_k = args
                # backbone feature
                f_x = self.basemodel(search_img)
                # feature adjustment
                c_x = self.c_x(f_x)
                r_x = self.r_x(f_x)
            elif len(args) == 4:
                # c_x, r_x already computed
                c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))

            # feature matching
            r_out = xcorr_depthwise(r_x, r_z_k)
            c_out = xcorr_depthwise(c_x, c_z_k)
            # head
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final = self.head(
                c_out, r_out)
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
            # register extra output
            extra = dict(c_x=c_x, r_x=r_x)
            # output
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self, device):
        r"""
        Load model parameters
        """
        self._make_convs()

        if self._hyper_params["pretrain_model_path"] != "":
            model_path = self._hyper_params["pretrain_model_path"]
            try:
                state_dict = torch.load(model_path,
                                        map_location=torch.device("cuda:"+str(device)))
            except:
                state_dict = torch.load(model_path,
                                        map_location=torch.device("cpu"))
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            try:
                self.load_state_dict(state_dict, strict=True)
            except:
                self.load_state_dict(state_dict, strict=False)

    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        self.r_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.c_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

        # initialze head
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight, std=0.01)