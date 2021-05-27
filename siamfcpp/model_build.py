from .models.module_base import ModuleBase
from .models.DenseboxHead import DenseboxHead
from .models.tinyconv import TinyConv
from .models.alexnet import AlexNet
from .models.task_model import SiamTrack

def build_model(path):
    
    backbone = TinyConv()
    head = DenseboxHead()
    task_model = SiamTrack(backbone, head, None)
    task_model._hyper_params["pretrain_model_path"] = path
    head.update_params()
    task_model.update_params(0)
    # head.update_params()
    return task_model

def build_alex(path, device):

    backbone = AlexNet()
    head = DenseboxHead()
    head._hyper_params["head_width"] = 256
    task_model = SiamTrack(backbone, head, None)
    task_model._hyper_params["pretrain_model_path"] = path
    task_model._hyper_params["head_width"] = 256
    head.update_params()
    task_model.update_params(device)
    # head.update_params()
    return task_model
