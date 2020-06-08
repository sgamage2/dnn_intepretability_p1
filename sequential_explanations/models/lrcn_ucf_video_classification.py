import logging, time, os

import models.lrcn
import utility


exp_params = {}
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'lrcn_1'
exp_params['latent_dim'] = 512  # No. of features of LSTM input
exp_params['lstm_layer_units'] = [256, 256]
exp_params['output_nodes'] = 55  # No. of classes
exp_params['lstm_time_steps'] = 5   # No. of frames to classify at one time (LSTM unrolling)

def create_lrcn(params):
    model = models.lrcn.LRCNClassifier()
    model.initialize(params)
    return model


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    utility.initialize(exp_params)

    model = create_lrcn(exp_params)

    # 2 Options for here:
    # 1) Transfer weights from pytorch model to keras LRCN model (models.lrcn)
    # 2) Train LRCN from scratch

    utility.save_model(model, exp_params['results_dir'])

    model = utility.load_model(exp_params['results_dir'])  # For testing

    print(model)

if __name__ == "__main__":
    main()
