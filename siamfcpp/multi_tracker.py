import torch
import numpy as np
from torch import multiprocessing
from .utils.crop import get_crop, get_subwindow_tracking
from .utils.bbox import cxywh2xywh, xywh2cxywh, xyxy2cxywh
from .utils.misc import imarray_to_tensor, tensor_to_numpy
from .model_build import build_alex, build_model
from .tracking_utils import postprocess_box, postprocess_score, restrict_box, cvt_box_crop2frame

class Multi_Tracker(torch.multiprocessing.Process):
    hyper_params = dict(
        total_stride=8,
        context_amount=0.5,
        test_lr=0.52,
        penalty_k=0.8,
        window_influence=0.2,
        windowing="cosine",
        z_size=127,
        x_size=303,
        num_conv3x3=3,
        min_w=10,
        min_h=10,
        phase_init="feature",
        phase_track="track",
    )
    def __init__(self, index, image, dataqueue, resultqueue):
        super(Multi_Tracker, self).__init__()
        # self.model = build_model("siamfcpp/models/siamfcpp-tinyconv-vot.pkl")
        self.model = build_alex("siamfcpp/models/siamfcpp-alexnet-vot.pkl", 0)
        self.model.cuda()
        self.model.eval()
        self.index = index
        self.tracking_index = {}
        self.device = torch.device("cuda")
        self.dataqueue = dataqueue
        self.resultqueue = resultqueue
        self.avg_chans = np.mean(image, axis=(0, 1))
        self.z_size = self.hyper_params['z_size']
        self.x_size = self.hyper_params['x_size']
        self.context_amount = self.hyper_params['context_amount']
        self.phase = self.hyper_params['phase_init']
        self.phase_track = self.hyper_params['phase_track']
        self.score_size = (self.hyper_params['x_size'] -self.hyper_params['z_size']) // self.hyper_params['total_stride'] + 1 - self.hyper_params['num_conv3x3'] * 2
        self.window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))
        self.window = self.window.reshape(-1)
        self.im_h, self.im_w = image.shape[0], image.shape[1]
        self.total_num = 0
        print("Process",index," Already!")
    
    def prepare(self, state, im_x):
        for i in range(len(state)):
            self.tracking_index[self.index*100+self.total_num+i] = [state[i]]
            im_z_crop, _ = get_crop(im_x, state[i][:2], state[i][2:4], self.z_size, avg_chans=self.avg_chans, context_amount=self.context_amount, func_get_subwindow=get_subwindow_tracking)
            array = torch.from_numpy(np.ascontiguousarray(im_z_crop.transpose(2, 0, 1)[np.newaxis, ...], np.float32)).to(self.device)
            with torch.no_grad():
                self.tracking_index[self.index*100+self.total_num+i].append(self.model(array,phase=self.phase))
            self.tracking_index[self.index*100+self.total_num+i].append(0)
    
    def delete_index(self, index):
        try:
            del self.tracking_index[index]
        except Exception as err:
            print("delete error: ", err)

    def run(self):
        while True:
            try: 
                im_x, state, delete = self.dataqueue.get(timeout=1)
            except Exception as error:
                # print(error)
                continue
            else:

                if len(state) > 0:
                    self.prepare(state, im_x)
                    print("init success")
                    self.total_num += len(state)
                    continue

                delete_list = []
                for i in self.tracking_index.keys():
                    if self.tracking_index[i][2] > 10 or i in delete:
                        delete_list.append(i)
                for i in delete_list:
                    self.delete_index(i)
                    print("delete",i)
                
                result = []
                for i in self.tracking_index.keys():
                    im_x_crop, scale_x = get_crop(im_x, self.tracking_index[i][0][:2], self.tracking_index[i][0][2:4], self.z_size, x_size=self.x_size, avg_chans=self.avg_chans,context_amount=self.context_amount, func_get_subwindow=get_subwindow_tracking)
                    array = torch.from_numpy(np.ascontiguousarray(im_x_crop.transpose(2, 0, 1)[np.newaxis, ...], np.float32)).to(self.device)
                    with torch.no_grad():
                        score, box, cls, ctr, *args = self.model(array, *self.tracking_index[i][1], phase=self.phase_track)
                    
                    box = tensor_to_numpy(box[0])
                    score = tensor_to_numpy(score[0])[:, 0]
                    cls = tensor_to_numpy(cls[0])
                    ctr = tensor_to_numpy(ctr[0])
                    box_wh = xyxy2cxywh(box)

                    # lost goal
                    if score.max()<0.2:
                        self.tracking_index[i][2] += 1
                        result.append([cxywh2xywh(np.concatenate([self.tracking_index[i][0][:2], self.tracking_index[i][0][2:4]],axis=-1)), self.tracking_index[i][2]])
                        continue
                    elif self.tracking_index[i][2] > 0:
                        self.tracking_index[i][2] -= 1
                    best_pscore_id, pscore, penalty = postprocess_score(score, box_wh, self.tracking_index[i][0][2:4], scale_x, self.hyper_params["penalty_k"], self.window, self.hyper_params["window_influence"])
                    # box post-processing
                    new_target_pos, new_target_sz = postprocess_box(best_pscore_id, score, box_wh, self.tracking_index[i][0][:2], self.tracking_index[i][0][2:4], scale_x, self.x_size, penalty, self.hyper_params["test_lr"])
                    new_target_pos, new_target_sz = restrict_box(new_target_pos, new_target_sz, self.im_w, self.im_h, self.hyper_params['min_h'], self.hyper_params['min_w'])

                    # save underlying state
                    self.tracking_index[i][0] = np.append(new_target_pos, new_target_sz)

                    # return rect format
                    track_rect = cxywh2xywh(np.concatenate([new_target_pos, new_target_sz],axis=-1))
                    result.append([track_rect, self.tracking_index[i][2]])
                
                self.resultqueue.put([result, list(self.tracking_index.keys())])
                # print("process",self.index,"send: ",list(self.tracking_index.keys()))
            torch.cuda.empty_cache() 
            