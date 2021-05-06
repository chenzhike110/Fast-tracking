from siamfcpp.models.alexnet import AlexNet
from siamfcpp.models.tinyconv import TinyConv
import torch
from torchvision import models
import onnx

model = TinyConv()
model.update_params("siamfcpp/models/siamfcpp-tinyconv-vot.pkl")
model.eval()

alex = AlexNet()
alex.update_params("siamfcpp/models/siamfcpp-alexnet-vot.pkl")
alex.eval()

x = torch.randn(1, 3, 303, 303, requires_grad=True)

ONNX_ALEX = 'alexnet-vot.onnx'
torch.onnx.export(alex, x, ONNX_ALEX, input_names=['input'],
                  output_names=['output'], export_params=True)

onnx_model = onnx.load(ONNX_ALEX)
onnx.checker.check_model(onnx_model)

ONNX_ALEX = 'tinyconv-vot.onnx'
torch.onnx.export(model, x, ONNX_ALEX, input_names=['input'],
                  output_names=['output'], export_params=True)

onnx_model = onnx.load(ONNX_ALEX)
onnx.checker.check_model(onnx_model)

