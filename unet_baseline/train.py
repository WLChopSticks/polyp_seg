import sys
sys.path.extend(['../', '../../','../models/deeplab3_plus'])
import os, time, argparse
import logging
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import RandomAffine, RandomRotation, RandomHorizontalFlip, ColorJitter, Resize, ToTensor,\
    Normalize, RandomResizedCrop, RandomOrder, RandomApply, Compose, RandomVerticalFlip, RandomChoice, RandomGrayscale,\
    RandomSizedCrop
from datasets_own import poly_seg, Compose_own
from models import UNet, fcn, unet_plus
from models.deeplab3_plus.deeplab import *
from utils import CrossEntropyLoss2d, dice_fn, UnionLossWithCrossEntropyAndDiceLoss, UnionLossWithCrossEntropyAndSize, Boundary_Loss
from utils import DiceLoss, Size_Loss_naive, UnionLossWithCrossEntropyAndDiceAndBoundary
from datetime import datetime
import plot.单纯测试验证集 as rstest
import update_mask
import time
from threading import Thread
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Segmeantation for polyp',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default='deeplabV3+', type=str, help='unet,unet_plus，fcn, deeplabV3+')
    parser.add_argument('--fold_num', default='0', type=str, help='fold number')
    parser.add_argument('--train_root', default=r'', type=str, help='train dataset absolute path')
    parser.add_argument('--test_root', default=r'', type=str, help='test or validation dataset absolute path')
    parser.add_argument('--train_csv', default=r'', type=str, help='train csv file absolute path')
    parser.add_argument('--test_csv', default=r'',  type=str, help='test csv file absolute path')
    parser.add_argument('--event_prefix', default='deeplabV3+', type=str, help='tensorboard logdir prefix')
    parser.add_argument('--tensorboard_name', default='lossunit5')
    parser.add_argument('--batch_size', default=3, type=int, help='batch_size')
    parser.add_argument('--gpu_order', default='0', type=str, help='gpu order')
    parser.add_argument('--torch_seed', default=2, type=int, help='torch_seed')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=200, type=int, help='num epoch')
    parser.add_argument('--ite_start_time', default=0, type=int, help='iteration start epoch')
    parser.add_argument('--ite_end_time', default=1, type=int, help='iteration end times')
    parser.add_argument('--loss', default='ce+boundary', type=str, help='ce, union, ce+size, ce+boundary')
    parser.add_argument('--img_size', default=(288,384), type=tuple, help='(512,512)')
    parser.add_argument('--lr_policy', default='StepLR', type=str, help='StepLR')
    parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--checkpoint', default='checkpoint/')
    parser.add_argument('--params_name', default='init_0.7668.pkl')
    parser.add_argument('--log_name', default='unet.log', type=str, help='log name')
    parser.add_argument('--history', default='history/')
    parser.add_argument('--style', default='aug', help='none or aug')
    parser.add_argument('--send_result', default=False, help='send test result or not')
    parser.add_argument('--size_loss_epoch', default=50, type=int, help='size loss need to train for epoches')
    parser.add_argument('--boundary_co', default=-1, type=float, help='size loss need to train for epoches')
    args = parser.parse_args()
    return args

def build_model(model_name, num_classes):
    if model_name == 'unet':
        net = UNet(colordim=3, n_classes=num_classes)
    elif model_name == 'fcn':
        vgg_model = fcn.VGGNet(requires_grad=True, remove_fc=True)
        net = fcn.FCNs(pretrained_net=vgg_model, n_class=num_classes)
    elif model_name == 'deeplabV3+':
        net = DeepLab(num_classes=num_classes,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=True,
                        freeze_bn=False)
    elif model_name == 'unet_plus':
        net = unet_plus.NestedUNet(input_channels=3,output_channels=num_classes, deepsupervision=False)
    else:
        print('wait a minute')
    return net

load_checkpoint = False
loss_type = 'ce'

def Train(train_root, train_csv, test_root, test_csv, iter_time, checkpoint_name=None):
    # record
    localtime = time.asctime(time.localtime(time.time()))
    logging.info('Seg polyp (Data: %s)' % localtime)
    logging.info('\n')

    # parameters
    args = parse_args()
    logging.info('Parameters: ')
    logging.info('model name: %s' % args.model_name)
    logging.info('torch seed: %d' % args.torch_seed)
    logging.info('gpu order: %s' % args.gpu_order)
    logging.info('batch size: %d' % args.batch_size)
    logging.info('num epoch: %d' % args.num_epoch)
    logging.info('ite_start_time: %d' % args.ite_start_time)
    logging.info('ite_end_time: %d' % args.ite_end_time)
    logging.info('learing rate: %f' % args.lr)
    logging.info('loss: %s' % args.loss)
    logging.info('img_size: %s' % str(args.img_size))
    logging.info('lr_policy: %s' % args.lr_policy)
    logging.info('resume: %s' % args.resume)
    logging.info('log_name: %s' % args.fold_num+args.log_name)
    logging.info('params_name: %s' % args.fold_num+args.params_name)
    logging.info('\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_order
    torch.manual_seed(args.torch_seed)
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 2
    net = build_model(args.model_name, num_classes)

    # resume
    if checkpoint_name is None:
        checkpoint_path = os.path.join(args.checkpoint, args.model_name)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        checkpoint_name = os.path.join(checkpoint_path, args.fold_num+args.params_name)
    if load_checkpoint:
        logging.info('Resuming from checkpoint...')
        checkpoint = torch.load(checkpoint_name)
        best_loss = checkpoint['loss']
        # best_dice = checkpoint['dice']
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        net.load_state_dict(checkpoint['net'])
    else:
        best_loss = float('inf')
        best_dice = float('inf')
        start_epoch = 0
        history = {'train_loss': [], 'test_loss': [], 'test_dice': []}
    start_epoch = 0
    end_epoch = start_epoch + args.num_epoch

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = nn.DataParallel(net)
    net.to(device)

    # data
    img_size = args.img_size

    ## train
    # train_aug = Compose([
    #     Resize(size=(img_size, img_size)),
    #     ToTensor(),
    #     Normalize(mean=(0.5, 0.5, 0.5),
    #               std=(0.5, 0.5, 0.5))])
    # RandomOrder
    if args.style == 'aug':
        train_img_aug = Compose_own([
            # RandomAffine(90, shear=45),
            # RandomRotation(90),
            # RandomHorizontalFlip(),
            # ColorJitter(brightness=0.05),
            Resize(img_size),
            ToTensor()])

        train_mask_aug = Compose_own([
            # RandomAffine(90, shear=45),
            # RandomRotation(90),
            # RandomHorizontalFlip(),
            # ColorJitter(brightness=0.05),
            Resize(img_size),
            ToTensor()])
    else:
        train_img_aug = Compose_own([
            Resize(img_size),
            ToTensor()])

        train_mask_aug = Compose_own([
            Resize(img_size),
            ToTensor()])



    ## test
    test_img_aug = Compose_own([Resize(size=img_size), ToTensor()])
    test_mask_aug = Compose_own([Resize(size=img_size), ToTensor()])

    train_dataset = poly_seg(root=train_root, csv_file=train_csv, img_transform=train_img_aug, mask_transform=train_mask_aug, iter_time=iter_time)
    test_dataset = poly_seg(root=test_root, csv_file=test_csv, img_transform=test_img_aug, mask_transform=test_mask_aug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=8, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=8, shuffle=False, drop_last=True)

    # loss function, optimizer and scheduler
    if loss_type == 'ce':
        criterion = CrossEntropyLoss2d().to(device)
    elif loss_type == 'union':
        criterion = UnionLossWithCrossEntropyAndDiceLoss().to(device)
    elif loss_type == 'ce+size':
        criterion = UnionLossWithCrossEntropyAndSize().to(device)
    elif loss_type == 'ce+boundary':
        criterion = UnionLossWithCrossEntropyAndSize().to(device)
        criterion2 = UnionLossWithCrossEntropyAndDiceAndBoundary().to(device)
    else:
        print('Do not have this loss')
    optimizer = Adam(net.parameters(), lr=args.lr, amsgrad=True)
    # optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.lr_policy == 'StepLR':
        scheduler = StepLR(optimizer, step_size=args.size_loss_epoch, gamma=0.5)

    # training process
    logging.info('Start Training For Polyp Seg')
    from skimage.segmentation import slic, mark_boundaries
    import numpy as np
    size_loss_epoch = args.size_loss_epoch
    for epoch in range(start_epoch, end_epoch):
        ts = time.time()
        scheduler.step()

        # train
        net.train()
        train_loss = 0.

        for batch_idx, (inputs, gts, img_names) in tqdm(enumerate(train_loader),
                                                 total=int(len(train_loader.dataset) / args.batch_size) + 1):

            if epoch >= 0:
                seg_labs = []
                images = []
                for input in inputs:
                    input_tem = input.clone().numpy().astype(np.double)
                    input_tem = (((input_tem * 0.5) + 0.5) * 255).astype(np.uint8)
                    input_tem = np.transpose(input_tem,(1,2,0))
                    # import cv2
                    # input_tem = cv2.cvtColor(input_tem, cv2.COLOR_RGB2BGR)
                    images.append(input_tem)
                    # seg_map = segmentation.felzenszwalb(input_tem, scale=32, sigma=0.5, min_size=128)
                    # image_path = os.path.join('../data/CVC-912/train/images', img_names)
                    seg_map = slic(input_tem, n_segments=100, compactness=10)
                    out = mark_boundaries(input_tem, seg_map)
                    seg_map = seg_map.flatten()
                    seg_lab = [np.where(seg_map == u_label)[0]
                               for u_label in np.unique(seg_map)]
                    seg_labs.append(seg_lab)

            inputs = inputs.to(device)
            gts = gts.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # if len(torch.where(torch.isnan(outputs))[0]) != 0:
            #     print('app nan:' + str(batch_idx))
            if epoch >= 0:
                #outputs 为输出结果
                temps = []
                output_softmax = F.softmax(outputs, dim=1)
                for i in range(int(args.batch_size)):
                    if epoch < args.size_loss_epoch:
                        #gt用超像素变化
                        gt = gts[i].view(-1, 1).cpu().detach().numpy()
                        for inds in seg_labs[i]:
                            u_labels, hist = np.unique(gt[inds], return_counts=True)
                            gt[inds] = u_labels[np.argmax(hist, 0)]
                        target = torch.from_numpy(gt).long()
                        target = torch.reshape(target,(288,384))
                        if target.sum().numpy() == 0:
                            target = gts[i].cpu()
                        target = target.unsqueeze(0)
                        temps.append(target)
                        cv2.imwrite(os.path.join('123', img_names[i]), gt.reshape((288,384)) * 255)
                    else:
                        output = outputs[i]
                        output = output.permute(1, 2, 0).view(-1, 2)#一共2类
                        #output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
                        target = torch.argmax(output, 1)
                        target = target * (gts[i].view(-1))
                        im_target = target.data.cpu().numpy()
                        a = np.resize(im_target, (288, 384))
                        cv2.imwrite(os.path.join('234', img_names[i]), a * 255)

                        '''refine'''
                        # if im_target.sum() == 0:
                        #     im_target = gts[i].view(-1,1).cpu().detach().numpy()
                        #     for inds in seg_labs[i]:
                        #         if im_target[inds].sum() != len(inds):
                        #             im_target[inds] = 0


                        if im_target.sum() < 128:
                            for inds in seg_labs[i]:
                                # u_labels, hist = np.unique(im_target[inds], return_counts=True)
                                if im_target[inds].sum() > 0:
                                    im_target[inds] = 1

                        else:
                            for inds in seg_labs[i]:
                                u_labels, hist = np.unique(im_target[inds], return_counts=True)
                                im_target[inds] = u_labels[np.argmax(hist,0)]

                        if im_target.sum() == 0:
                            im_target = gts[i].view(-1, 1).cpu().detach().numpy()
                        im_target = np.resize(im_target, (288,384))


                        # print(img_names[i])
                        target = torch.from_numpy(im_target).long()
                        # gt = gts[i].cpu().clone().detach()
                        cv2.imwrite(os.path.join('123', img_names[i]), target.numpy() * 255)
                        # target = torch.reshape(target,(288,384))
                        target = target.unsqueeze(0)
                        temps.append(target)

                new_gt = torch.cat([x for x in temps],0)
                #new_gt = torch.cat((temps[0],temps[1]),0)
                new_gt = new_gt.to(device)

            # from utils.crf import dense_crf
            # crf_res = []
            # for i in range(len(images)):
            #     out = dense_crf(images[i], output_softmax[i][1].cpu().detach().numpy())
            #     crf_res.append(torch.from_numpy(out).unsqueeze(0))
            # crf_results = torch.cat([x for x in crf_res]).to(device)


            if epoch < size_loss_epoch:
                loss = criterion(outputs, new_gt)
            else:
                loss = criterion2(outputs,new_gt,1, args.boundary_co)
                optimizer.lr = args.lr/10

            # if args.loss == 'ce-dice':
            #     # loss = 2 * criterion1(outputs, targets) + criterion2(outputs, targets)
            #     pass
            # elif loss_type == 'ce+boundary':
            #     loss1 = criterion(outputs, new_gt)
            #     loss2 = criterion2(outputs, new_gt)
            #     loss = loss1 + 0.0001 * loss2
            # else:
            #     loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_epoch = train_loss / (batch_idx + 1)
        history['train_loss'].append(train_loss_epoch)

        writer.add_scalar('training_loss', train_loss_epoch, epoch)

        # test
        net.eval()
        test_loss = 0.
        test_dice = 0.

        for batch_idx, (inputs, targets, images) in tqdm(enumerate(test_loader),
                                                 total=int(len(test_loader.dataset) / args.batch_size) + 1):
            with torch.no_grad():
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = net(inputs)
                if args.loss == 'ce-dice':
                    # loss = 2 * criterion1(outputs, targets) + criterion2(outputs, targets)
                    pass
                else:
                    loss = criterion(outputs, targets)
                dice = dice_fn(outputs, targets)
            test_loss += loss.item()
            test_dice += dice.item()
        test_loss_epoch = test_loss / (batch_idx + 1)
        test_dice_epoch = test_dice / len(test_loader.dataset)
        history['test_loss'].append(test_loss_epoch)
        history['test_dice'].append(test_dice_epoch)
        writer.add_scalar('validation_loss', test_loss_epoch, epoch)
        writer.add_scalar('validation_dice', test_dice_epoch, epoch)
        time_cost = time.time() - ts
        logging.info('epoch[%d/%d]: train_loss: %.3f | test_loss: %.3f  | test_dice: %.3f || time: %.1f'
                     % (epoch + 1, end_epoch, train_loss_epoch, test_loss_epoch, test_dice_epoch, time_cost))

        # save checkpoint
        if test_loss_epoch < best_loss:
            logging.info('Checkpoint Saving...')

            save_model = net
            # if torch.cuda.device_count() > 1:
            #     save_model = list(net.children())[0]
            state = {
                'net': save_model.state_dict(),
                'loss': test_loss_epoch,
                'dice': test_dice_epoch,
                'epoch': epoch + 1,
                'history': history
            }
            torch.save(state, checkpoint_name)
            best_loss = test_loss_epoch

        # if test_dice_epoch > best_dice:
        #     logging.info('Checkpoint Saving...')
        #
        #     save_model = net
        #     # if torch.cuda.device_count() > 1:
        #     #     save_model = list(net.children())[0]
        #     state = {
        #         'net': save_model.state_dict(),
        #         'loss': test_loss_epoch,
        #         'dice': test_dice_epoch,
        #         'epoch': epoch + 1,
        #         'history': history
        #     }
        #     checkpoint_name_dice = os.path.join(checkpoint_path, 'dice_' + args.fold_num + args.params_name)
        #     torch.save(state, checkpoint_name_dice)
        #     best_dice = test_dice_epoch

    writer.close()
    return net

args = parse_args()
if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)
if not os.path.exists(args.history):
    os.mkdir(args.history)
logging_save = os.path.join(args.history, args.fold_num+args.log_name)
logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(logging_save)
                    ])





if __name__ == "__main__":

    train_root = args.train_root
    train_csv = args.train_csv
    test_root = args.test_root
    test_csv = args.test_csv
    if len(train_root) == 0:
        train_root = os.path.join(sys.path[0], '../data/CVC-912/train')
    if len(test_root) == 0:
        test_root = os.path.join(sys.path[0], '../data/CVC-912/val')
    if len(train_csv) == 0:
        train_csv = os.path.join(sys.path[0], '../data/fixed-csv/train.csv')
    if len(test_csv) == 0:
        test_csv = os.path.join(sys.path[0], '../data/fixed-csv/val.csv')

    #all train data csv, including train and val
    train_data_csv = os.path.join(sys.path[0], '../data/fixed-csv/train_data.csv')
    load_checkpoint
    loss_type
    if args.resume == 1: load_checkpoint = True
    start = args.ite_start_time
    if start != 0: load_checkpoint = True
    while start < args.ite_end_time:
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        if len(args.tensorboard_name) == 0:
            log_dir = os.path.join('./Graph', args.event_prefix, TIMESTAMP)
        else:
            log_dir = os.path.join('./Graph', args.event_prefix, args.tensorboard_name, str(start))
        writer = SummaryWriter(log_dir)
        checkpoint_path = os.path.join(args.checkpoint, args.model_name)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        checkpoint_name = os.path.join(checkpoint_path, str(start) + args.params_name)
        loss_type = args.loss
        model = Train(train_root, train_csv, test_root, test_csv,start, checkpoint_name)

        # update_mask.updateMask(train_root,train_data_csv,'../data/CVC-912/train/masks',model, start)
        start += 1
        load_checkpoint = False
        loss_type = 'union'

        # test
        dataset_root = os.path.join(sys.path[0], '../data/CVC-912/test')
        val_csv_path = os.path.join(sys.path[0], '../data/fixed-csv/test.csv')
        checkpoint_path = os.path.join(sys.path[0], '../unet_baseline/checkpoint/deeplabV3+',args.fold_num+args.params_name)
        checkpoint_path = checkpoint_name
        result =rstest.validate(val_csv_path, dataset_root, checkpoint_path)
        print(result)

        #send result to wechat
        import requests
        if args.send_result:
            url = "https://sc.ftqq.com/SCU28703Te109f3ff3fede315f4017d79786274ab5b35cf275612b.send?"
            url2 = "https://sc.ftqq.com/SCU87403Tdd9ec9b4572930aee59a144326d0f5e15e5c9a4163f6a.send?"

            result_str = 'val_dice: {0}\n\n' \
                         'Recall: {1}\n\n' \
                         'Specificity: {2}\n\n' \
                         'Precision: {3}\n\n' \
                         'Dice: {4}\n\n' \
                         'F2: {5}\n\n' \
                         'IoU_p: {6}\n\n' \
                         'IoU_b: {7}\n\n' \
                         'IoU_m: {8}\n\n' \
                         'Acc: {9}\n\n'.format(result['val_dice'], result['Recall'], result['Specificity']
                                               , result['Precision'], result['Dice'], result['F2'], result['IoU_p'],
                                               result['IoU_b'], result['IoU_m'], result['Acc'], )

            params = {"text": 'linux: ' + args.tensorboard_name,
                      'desp': result_str + '\n\nthe infomation is to wl'}

            res = requests.get(url=url, params=params)
            params2 = {"text": 'ubuntu: ' + args.tensorboard_name,
                       'desp': result_str + '\n\nthis message is to ljx'}
            # res2 = requests.get(url=url2, params=params2)
            print(res.text)
            # print(res2.text)

