import torch
from torch.nn.parallel.data_parallel import DataParallel
from model import get_pose_net
from config import cfg
import os.path as osp

class rootnet:
    def lood_model(path):
        model_path = path
        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        print('Load checkpoint from {}'.format(model_path))
        model = get_pose_net(cfg, False)
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval();
        return model

