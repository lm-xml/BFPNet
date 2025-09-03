import os
import datetime

import torch
import math
import torch.optim.lr_scheduler as scheduler
import matplotlib.pyplot as plt

import transforms
from my_dataset import VOCDataSet
from src.model import SSD300, Backbone
import train_utils.train_eval_utils as utils
from src.utils import dboxes300_coco
from train_utils import get_coco_api_from_dataset
from torch.utils.tensorboard import SummaryWriter
import shutil

def create_model(num_classes=2,pretrained=False):
    pre_ssd_path = ""
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)
    if pretrained == True:
        if os.path.exists(pre_ssd_path) is False:
            raise FileNotFoundError("nvidia.pt not find in {}".format(pre_ssd_path))
        pre_model_dict = torch.load(pre_ssd_path, map_location='cpu')
        pre_weights_dict = pre_model_dict

        del_conf_loc_dict = {}
        for k, v in pre_weights_dict.items():
            split_key = k.split(".")
            if "conf" in split_key or "additional_blocks" in split_key or "loc" in split_key:
                continue
            del_conf_loc_dict.update({k: v})

        missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def main(parser_data):
    tb_writer = SummaryWriter(comment="")

    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([
                                    transforms.Resize(),
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    # SSDD
                                    transforms.Normalization(mean=[0.15543534, 0.15543921, 0.15541409],
                                                             std=[0.1619891, 0.16199365, 0.16195092]),
                                    # # RSDD
                                    # transforms.Normalization(mean=[0.07768455, 0.07768455, 0.07768455],
                                    #                          std=[0.11129927, 0.11129927, 0.11129927]),
                                    # # SAR-ship
                                    # transforms.Normalization(mean=[0.080467544, 0.080467544, 0.080467544],
                                    #                          std=[0.15530987, 0.15530987, 0.15530987]),
                                    transforms.AssignGTtoDefaultBox()
        ]),
        "val": transforms.Compose([transforms.Resize(),
                                   transforms.ToTensor(),
                                   # SSDD
                                   transforms.Normalization(mean=[0.15543534, 0.15543921, 0.15541409],
                                                            std=[0.1619891, 0.16199365, 0.16195092]),
                                   # # RSDD
                                   # transforms.Normalization(mean=[0.07768455, 0.07768455, 0.07768455],
                                   #                          std=[0.11129927, 0.11129927, 0.11129927]),
                                   # # SAR-ship
                                   # transforms.Normalization(mean=[0.080467544, 0.080467544, 0.080467544],
                                   #                          std=[0.15530987, 0.15530987, 0.15530987]),
                                   ])
    }

    VOC_root = parser_data.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    train_dataset = VOCDataSet(VOC_root, "2012", data_transform['train'], train_set='train.txt', mode='train')
    batch_size = parser_data.batch_size
    assert batch_size > 1, "batch size must be greater than 1"
    drop_last = True if len(train_dataset) % batch_size == 1 else False
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 2])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn,
                                                    drop_last=drop_last)
    val_dataset = VOCDataSet(VOC_root, "2012", data_transform['val'], train_set='test.txt', mode='train')
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes+1, pretrained=False)
    model.to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    lf = lambda x: ((1 + math.cos(x * math.pi / parser_data.epochs)) / 2) * (1 - args.lf) + args.lf
    lr_scheduler = scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    learning_rate = []

    val_map = []
    val_data = get_coco_api_from_dataset(val_data_loader.dataset)
    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        m_loc_loss, m_con_loss, m_cos_loss, mean_loss, lr = utils.train_one_epoch(model=model, optimizer=optimizer,
                                              data_loader=train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=500, warmup=True)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        tags = ["train_loss", "local_loss","con_loss","cos_loss","learn_rate"]
        tb_writer.add_scalar(tags[0],mean_loss, epoch)
        tb_writer.add_scalar(tags[1],m_loc_loss, epoch)
        tb_writer.add_scalar(tags[2],m_con_loss, epoch)
        tb_writer.add_scalar(tags[3],m_cos_loss, epoch)
        tb_writer.add_scalar(tags[4],optimizer.param_groups[0]["lr"], epoch)

        lr_scheduler.step()
        temp_epoch = parser_data.val_epoch
        if (epoch + 1) % temp_epoch == 0:
            if epoch >= 0:   # args.epochs / 2:
                coco_info = utils.evaluate(model=model, data_loader=val_data_loader,
                                           device=device, data_set=val_data)
                # write into txt
                with open(results_file, "a") as f:
                    result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
                    if epoch == temp_epoch - 1:
                        txt = parser.parse_args()
                        argsDict = txt.__dict__
                        f.writelines('------------------ start ------------------' + '\n')
                        for eachArg, value in argsDict.items():
                            f.writelines(eachArg + ' : ' + str(value) + '\n')
                        f.writelines('------------------- end -------------------')
                        f.write("\n")
                    txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                    f.write(txt + "\n")

                val_map.append(coco_info[1])  # pascal mAP

        # save weights
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, "./save_weights/BFPNet-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)
    tb_writer.close()

    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('--num_classes', default=1, type=int, help='num_classes')
    parser.add_argument('--data-path', default='', help='dataset')
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--val_epoch', default=5, type=int, metavar='N',
                        help='number of epochs to val')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--lr', default=0.005, type=int, metavar='N',
                        help='learning rate.')
    parser.add_argument('--lf', default=0.001, type=int, metavar='N',
                        help='the rate of learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='Weight decay for SGD')

    args = parser.parse_args()
    print(args)
    main(args)
