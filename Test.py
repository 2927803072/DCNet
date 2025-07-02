import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from model.DCNet import DCNet
from utils.dataloader import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=704, help='testing size default 352')
opt = parser.parse_args()

# for _data_name in ['CAMO', 'COD10K', 'CHAMELEON',NC4K]:
for _data_name in ['LVScene4K']:
    # data_path = '/youtu_action_data/xiaobinhu/dataset_hitnet_cod/TestDataset/{}/'.format(_data_name)
    # save_path = '/youtu_action_data/xiaobinhu/dataset_hitnet_cod/res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    data_path = 'C:/Users/ROG/Desktop/{}/Test'.format(_data_name)
    save_path = './results/DCNet_{}/'.format(_data_name)
    model = DCNet()
    model.load_state_dict(torch.load(opt.pth_path))
    # torch.cuda.set_device(1)
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    print('root', image_root, gt_root)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    print('****', test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        print('***name', name)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        x_in = image

        P1, P2 = model(x_in)
        res = F.upsample(P1[-1]+P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        print('> {} - {}'.format(_data_name, name))
        # misc.imsave(save_path+name, res)
        # If `mics` not works in your environment, please comment it and then use CV2
        cv2.imwrite(save_path+name, res)
