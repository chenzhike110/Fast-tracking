from .models.module_base import ModuleBase
from .models.DenseboxHead import DenseboxHead
from .models.tinyconv import TinyConv
from .models.task_model import SiamTrack

def build_model(path):
    
    backbone = TinyConv()
    head = DenseboxHead()
    task_model = SiamTrack(backbone, head, None)
    task_model.default_hyper_params["pretrain_model_path"] = path
    head.update_params()
    task_model.update_params()
    # head.update_params()
    return task_model
