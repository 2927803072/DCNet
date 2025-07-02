import torch
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse
import os
import numpy as np

from model.DCNet import DCNet
from utils.dataloader import get_loader, test_dataset
from torch.utils.tensorboard import SummaryWriter
from utils.utils import adjust_lr, clip_gradient

best_mae = 1
best_epoch = 1


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def train(train_loader, model, optim, epoch, opt, total_step, writer):
    total_loss = 0
    model.train()
    for step, data in enumerate(train_loader):

        imgs, gts = data
        imgs = imgs.cuda()
        gts = gts.cuda()
        input_fea = imgs #concatenate

        optim.zero_grad()
        stage_pre, pre = model(input_fea)
        stage_loss_list = [structure_loss(out, gts) for out in stage_pre]
        stage_loss = 0
        gamma = 0.2
        for iteration in range(len(stage_pre)):
            stage_loss += (gamma * iteration) * stage_loss_list[iteration]

        map_loss = structure_loss(pre, gts)
        loss = stage_loss + map_loss
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optim.step()

        total_loss += loss

        if step % 20 == 0 or step == total_step:
            print(
                '[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss: {:0.4f}]'.
                format(datetime.now(), epoch, opt.epoch, step, total_step, loss.item()))
            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}'.
                format(epoch, opt.epoch, step, total_step, loss.item()))

    writer.add_scalar("Train_Loss", total_loss, global_step=epoch)

    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        torch.save(model.state_dict(), save_path + str(epoch) + '_DCNet.pth')


def val(val_loader, model, epoch, save_path, writer):
    global best_mae, best_epoch
    model.eval()

    with torch.no_grad():
        mae_sum = 0
        for i in range(val_loader.size):
            image, gt, name = val_loader.load_data()
            gt = np.asarray(gt, np.float32)

            gt /= (gt.max() + 1e-8)

            image = image.cuda()
            in_fea = image


            res, res1 = model(in_fea)
            res = F.upsample(res[-1] + res1, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / val_loader.size
        writer.add_scalar('Val_MAE', mae, global_step=epoch)

        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch

                torch.save(model.state_dict(), save_path + '/DCNet_best.pth')

        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        logging.info('#VAL#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer Adam')
    parser.add_argument('--augmentation', default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size,candidate=352,704,1056')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='train from snapshot')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--root_dir', type=str, default='C:/Users/ROG/Desktop/LVScene4K', help='path to train dataset')
    # parser.add_argument('--val_path', type=str, default='C:/Users/ROG/Desktop/demo/Test', help='path to validate dataset')
    parser.add_argument('--save_path', type=str, default='./Snapshot/DCNet/')
    parser.add_argument('--save_model', type=str, default='./Snapshot/DCNet/Best_DCNet/')
    parser.add_argument('--epoch_save', type=int, default=5, help='every n epochs to save model')
    opt = parser.parse_args()

    os.makedirs(opt.save_path, exist_ok=True)
    os.makedirs(opt.save_model, exist_ok=True)

    logging.basicConfig(filename=opt.save_path+'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level = logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("ADCOD-Train")

    # torch.cuda.set_device(1)
    model = DCNet().cuda()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    train_image_root = '{}/Train/Imgs/'.format(opt.root_dir)
    train_gt_root = '{}/Train/GT/'.format(opt.root_dir)
    train_loader = get_loader(train_image_root, train_gt_root, batch_size=opt.batchsize, image_size=opt.trainsize, num_workers=8)

    val_image_root = '{}/Test/Imgs/'.format(opt.root_dir)
    val_gt_root = '{}/Test/GT/'.format(opt.root_dir)
    val_loader = test_dataset(val_image_root, val_gt_root, test_size=opt.trainsize)

    total_step = len(train_loader)

    writer = SummaryWriter(opt.save_path + "SummaryWriter")

    print('-----------------Ready for Training!----------------------')

    for epoch in range(1, opt.epoch+1):

        adjust_lr(optimizer, epoch, opt.decay_rate, opt.decay_epoch)

        train(train_loader, model, optimizer, epoch, opt, total_step, writer)
        val(val_loader, model, epoch, opt.save_model, writer)

    writer.close()