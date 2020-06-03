import torch
from torch.autograd import Variable
import os
from LRCN.lrcn_model import ConvLstm
from pytorch2keras import pytorch_to_keras
import numpy as np
import onnx



model_dir = '../20200510-193055/Saved_model_checkpoints'
# model_dir = './20200510-193055/Saved_model_checkpoints'
model_name = 'epoch_95.pth.tar'
onx_model_filename = 'model.onnx'
latent_dim = 512
hidden_size = 256
lstm_layers = 2
bidirectional = True
num_class = 55
num_frames_video = 5
cnn_input_shape = (1, num_frames_video, 3, 224, 224)   # batch_size=1, timesteps=5, channel_x=3, h_x=224, w_x=224
lstm_input_shape = (1, num_frames_video, latent_dim)   # batch_size=1, timesteps=5, channel_x=3, h_x=224, w_x=224

def load_pytorch_model():
    print('Loading model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Creating Pytorch ConvLSTM model object')
    model = ConvLstm(latent_dim, hidden_size, lstm_layers, bidirectional, num_class)
    model = model.to(device)

    print('Loading checkpoint')
    checkpoint = torch.load(os.path.join(model_dir, model_name), map_location=device)

    print('Loading model state dictionary')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def convert_to_keras(pytorch_model, input_shape):
    input_np = np.random.uniform(0, 1, input_shape)
    input_var = Variable(torch.FloatTensor(input_np))

    input_var = torch.randn(*input_shape, requires_grad=True)

    shape = [(input_shape[1:])]

    k_model = pytorch_to_keras(pytorch_model, input_var, input_shapes=shape, verbose=True, do_constant_folding=True)


def conver_to_onnx(pytorch_model, input_shape, filename):
    print(*input_shape)
    input_var = torch.randn(*input_shape, requires_grad=True)

    torch.onnx.export(pytorch_model,  # model being run
                      input_var,  # model input (or a tuple for multiple inputs)
                      filename,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})


def load_onnx_model(filename):
    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def main():
    pytorch_model = load_pytorch_model()

    cnn_params = pytorch_model.conv_model.parameters()
    cnn_params = list(cnn_params)

    lstm_params = pytorch_model.Lstm.parameters()
    lstm_params = list(lstm_params)

    # weights = [p.data.numpy() for p in lstm_params]
    
    # keras_model = convert_to_keras(pytorch_model.Lstm, lstm_input_shape)
    conver_to_onnx(pytorch_model.Lstm, lstm_input_shape, onx_model_filename)

    onnx_model = load_onnx_model(onx_model_filename)

    x = 5

if __name__ == '__main__':
    main()

