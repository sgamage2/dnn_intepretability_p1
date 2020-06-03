import logging, time, os

import models.lrcn
import utility


exp_params = {}
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'lrcn_1'


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    utility.initialize(exp_params)

    # 2 Options:
    # 1) Transfer weights from pytorch model to keras LRCN model (models.lrcn)
    # 2) Train LRCN from scratch
