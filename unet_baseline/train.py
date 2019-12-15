import sys
sys.path.extend(['../', '../../'])
import os, time, argparse
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import RandomAffine, RandomRotation, RandomHorizontalFlip, ColorJitter, Resize, ToTensor,\
    Normalize, RandomResizedCrop, RandomOrder, RandomApply, Compose, RandomVerticalFlip, RandomChoice
from datasets_own import poly_seg, Compose_own
from models import UNet, fcn
from utils import CrossEntropyLoss2d, dice_fn
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Segmeantation for polyp',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default='unet', type=str, help='unet, ...')
    parser.add_argument('--fold_num', default='0', type=str, help='fold number')
    parser.add_argument('--train_root', default=r'', type=str, help='train dataset absolute path')
    parser.add_argument('--test_root', default=r'', type=str, help='test or validation dataset absolute path')
    parser.add_argument('--train_csv', default=r'', type=str, help='train csv file absolute path')
    parser.add_argument('--test_csv', default=r'',  type=str, help='test csv file absolute path')
    parser.add_argument('--event_prefix', default='unet', type=str, help='tensorboard logdir prefix')
    parser.add_argument('--tensorboard_name', default='deconv_init')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--gpu_order', default='0', type=str, help='gpu order')
    parser.add_argument('--torch_seed', default=2, type=int, help='torch_seed')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=200, type=int, help='num epoch')
    parser.add_argument('--loss', default='ce', type=str, help='ce, dice')
    parser.add_argument('--img_size', default=256, type=int, help='512')
    parser.add_argument('--lr_policy', default='StepLR', type=str, help='StepLR')
    parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--checkpoint', default='checkpoint/')
    parser.add_argument('--params_name', default='unet_params.pkl')
    parser.add_argument('--log_name', default='unet.log', type=str, help='log name')
    parser.add_argument('--history', default='history/')
    args = parser.parse_args()
    return args

def build_model(model_name, num_classes):
    if model_name == 'unet':
        net = UNet(colordim=3, n_classes=num_classes)
    elif model_name == 'fcn':
        vgg_model = fcn.VGGNet(requires_grad=True, remove_fc=True)
        net = fcn.FCNs(pretrained_net=vgg_model, n_class=num_classes)
    else:
        print('wait a minute')
    return net

def Train(train_root, train_csv, test_root, test_csv):
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
    logging.info('learing rate: %f' % args.lr)
    logging.info('loss: %s' % args.loss)
    logging.info('img_size: %d' % args.img_size)
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
    checkpoint_name = os.path.join(args.checkpoint, args.fold_num+args.params_name)
    if args.resume != 0:
        logging.info('Resuming from checkpoint...')
        checkpoint = torch.load(checkpoint_name)
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        net.load_state_dict(checkpoint['net'])
    else:
        best_loss = float('inf')
        start_epoch = 0
        history = {'train_loss': [], 'test_loss': [], 'test_dice': []}
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
    train_img_aug = Compose_own([
        RandomAffine(90, shear=45),
        RandomRotation(90),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.05),
        Resize((img_size, img_size)),
        ToTensor()])

    train_mask_aug = Compose_own([
        RandomAffine(90, shear=45),
        RandomRotation(90),
        RandomHorizontalFlip(),
        # ColorJitter(brightness=0.05),
        Resize((img_size, img_size)),
        ToTensor()])
    ## test
    test_img_aug = Compose_own([Resize(size=(img_size, img_size)), ToTensor()])
    test_mask_aug = Compose_own([Resize(size=(img_size, img_size)), ToTensor()])

    train_dataset = poly_seg(root=train_root, csv_file=train_csv, img_transform=train_img_aug, mask_transform=train_mask_aug)
    test_dataset = poly_seg(root=test_root, csv_file=test_csv, img_transform=test_img_aug, mask_transform=test_mask_aug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=0, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=0, shuffle=True)

    # loss function, optimizer and scheduler
    if args.loss == 'ce':
        criterion = CrossEntropyLoss2d().to(device)
    else:
        print('Do not have this loss')
    optimizer = Adam(net.parameters(), lr=args.lr, amsgrad=True)
    # optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.lr_policy == 'StepLR':
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    # training process
    logging.info('Start Training For Polyp Seg')
    for epoch in range(start_epoch, end_epoch):
        ts = time.time()
        scheduler.step()

        # train
        net.train()
        train_loss = 0.
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader),
                                                 total=int(len(train_loader.dataset) / args.batch_size) + 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            if args.loss == 'ce-dice':
                # loss = 2 * criterion1(outputs, targets) + criterion2(outputs, targets)
                pass
            else:
                loss = criterion(outputs, targets)
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

        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader),
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
            if torch.cuda.device_count() > 1:
                save_model = list(net.children())[0]
            state = {
                'net': save_model.state_dict(),
                'loss': test_loss_epoch,
                'epoch': epoch + 1,
                'history': history
            }
            torch.save(state, checkpoint_name)
            best_loss = test_loss_epoch
    writer.close()

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
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    if len(args.tensorboard_name) == 0:
        log_dir = os.path.join('./Graph', args.event_prefix, TIMESTAMP)
    else:
        log_dir = os.path.join('./Graph', args.event_prefix, args.tensorboard_name)
    writer = SummaryWriter(log_dir)
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


    Train(train_root, train_csv, test_root, test_csv)
