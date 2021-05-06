import numpy as np
import numba

@numba.jit(nopython=True)
def change(r):
    return np.maximum(r, 1. / r)

@numba.jit(nopython=True)
def sz(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return np.sqrt(sz2)

@numba.jit(nopython=True)
def sz_wh(wh):
    pad = (wh[0] + wh[1]) * 0.5
    sz2 = (wh[0] + pad) * (wh[1] + pad)
    return np.sqrt(sz2)

@numba.jit(nopython=True)
def postprocess_score(score, box_wh, target_sz, scale_x, penalty_k, _state, window_influence):
    r"""
    Perform SiameseRPN-based tracker's post-processing of score
    :param score: (HW, ), score prediction
    :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
    :param target_sz: previous state (w & h)
    :param scale_x:
    :return:
        best_pscore_id: index of chosen candidate along axis HW
        pscore: (HW, ), penalized score
        penalty: (HW, ), penalty due to scale/ratio change
    """

    # size penalty
    target_sz_in_crop = target_sz * scale_x
    s_c = change(
        sz(box_wh[:, 2], box_wh[:, 3]) /
        (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                    (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
    penalty = np.exp(-(s_c - 1))*penalty_k
    pscore = penalty + score

    # ipdb.set_trace()
    # cos window (motion model)
    pscore = pscore * (
        1 - window_influence) + _state * window_influence
    best_pscore_id = np.argmax(pscore)

    return best_pscore_id, pscore, penalty

@numba.jit(nopython=True)
def postprocess_box(best_pscore_id, score, box_wh, target_pos,
                        target_sz, scale_x, x_size, penalty, test_lr):
    r"""
    Perform SiameseRPN-based tracker's post-processing of box
    :param score: (HW, ), score prediction
    :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
    :param target_pos: (2, ) previous position (x & y)
    :param target_sz: (2, ) previous state (w & h)
    :param scale_x: scale of cropped patch of current frame
    :param x_size: size of cropped patch
    :param penalty: scale/ratio change penalty calculated during score post-processing
    :return:
        new_target_pos: (2, ), new target position
        new_target_sz: (2, ), new target size
    """
    pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x)
    # about np.float32(scale_x)
    # attention!, this casting is done implicitly
    # which can influence final EAO heavily given a model & a set of hyper-parameters

    # box post-postprocessing
    lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
    res_x = pred_in_crop[0] + target_pos[0] - (x_size // 2) / scale_x
    res_y = pred_in_crop[1] + target_pos[1] - (x_size // 2) / scale_x
    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    new_target_pos = np.array([res_x, res_y])
    new_target_sz = np.array([res_w, res_h])

    return new_target_pos, new_target_sz

@numba.jit(nopython=True)
def restrict_box(target_pos, target_sz, im_w, im_h, min_h, min_w):
    r"""
    Restrict target position & size
    :param target_pos: (2, ), target position
    :param target_sz: (2, ), target size
    :return:
        target_pos, target_sz
    """
    target_pos[0] = max(0, min(im_w, target_pos[0]))
    target_pos[1] = max(0, min(im_h, target_pos[1]))
    target_sz[0] = max(min_w, min(im_w, target_sz[0]))
    target_sz[1] = max(min_h, min(im_h, target_sz[1]))

    return target_pos, target_sz

@numba.jit(nopython=True)
def cvt_box_crop2frame(box_in_crop, target_pos, scale_x, x_size):
    r"""
    Convert box from cropped patch to original frame
    :param box_in_crop: (4, ), cxywh, box in cropped patch
    :param target_pos: target position
    :param scale_x: scale of cropped patch
    :param x_size: size of cropped patch
    :return:
        box_in_frame: (4, ), cxywh, box in original frame
    """
    x = (box_in_crop[..., 0]) / scale_x + target_pos[0] - (x_size //
                                                            2) / scale_x
    y = (box_in_crop[..., 1]) / scale_x + target_pos[1] - (x_size //
                                                            2) / scale_x
    w = box_in_crop[..., 2] / scale_x
    h = box_in_crop[..., 3] / scale_x
    box_in_frame = np.stack([x, y, w, h], axis=-1)

    return box_in_frame