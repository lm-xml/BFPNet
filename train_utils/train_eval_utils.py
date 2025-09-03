import math
import sys
import time

import torch

from train_utils import get_coco_api_from_dataset, CocoEvaluator
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # torch.autograd.set_detect_anomaly(True)
    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 5.0 / 10000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    m_loc_loss = torch.zeros(1).to(device)
    m_con_loss = torch.zeros(1).to(device)
    m_cos_loss = torch.zeros(1).to(device)
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):  # log_every输出具体信息
        # batch inputs information
        images = torch.stack(images, dim=0)

        boxes = []
        labels = []
        img_id = []
        Truth_boxes = []
        glabel_local = []
        ground_truth = []
        for t in targets:
            boxes.append(t['boxes'])
            labels.append(t['labels'])
            img_id.append(t["image_id"])
            Truth_boxes.append(t["Truth_boxes"])
            ground_truth.append(t["ground_truth"])
            glabel_local.append(t["glabel_local"])
        targets = {"boxes": torch.stack(boxes, dim=0),
                   "labels": torch.stack(labels, dim=0),
                   "image_id": torch.as_tensor(img_id)}

        images = images.to(device)
        Truth_boxes = [v.to(device) for v in Truth_boxes]
        ground_truth = [v.to(device) for v in ground_truth]
        glabel_local = [v.to(device) for v in glabel_local]

        targets = {k: v.to(device) for k, v in targets.items()}
        losses_dict, local_loss, class_loss, cos_sim_loss = model(images, targets, ground_truth, glabel_local, epoch)
        losses = losses_dict["total_losses"]

        # reduce losses over all GPUs for logging purpose
        losses_dict_reduced = utils.reduce_dict(losses_dict)
        losses_reduce = losses_dict_reduced["total_losses"]

        loss_value = losses_reduce.detach()

        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        loc_dic_r = utils.reduce_dict(local_loss)
        loc_dic = loc_dic_r["local_loss"]
        loc_loss_value = loc_dic.detach()

        con_dic_r = utils.reduce_dict(class_loss)
        con_dic = con_dic_r["con_loss"]
        con_loss_value = con_dic.detach()

        cos_dic_r = utils.reduce_dict(cos_sim_loss)
        cos_dic = cos_dic_r["cos_sim_loss"]
        cos_loss_value = cos_dic.detach()

        m_loc_loss = (m_loc_loss * i + loc_loss_value) / (i + 1)
        m_con_loss = (m_con_loss * i + con_loss_value) / (i + 1)
        m_cos_loss = (m_cos_loss * i + cos_loss_value) / (i + 1)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(losses_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # metric_logger.update(loss=losses, **loss_dict_reduced)
        metric_logger.update(**local_loss)
        metric_logger.update(**class_loss)
        metric_logger.update(**cos_sim_loss)
        metric_logger.update(**losses_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return m_loc_loss, m_con_loss, m_cos_loss, mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device, data_set=None):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    if data_set is None:
        data_set = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(data_set, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = torch.stack(images, dim=0).to(device)

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        results = model(images, targets=None)
        model_time = time.time() - model_time

        outputs = []
        for index, (bboxes_out, labels_out, scores_out) in enumerate(results):

            height_width = targets[index]["height_width"]

            bboxes_out[:, [0, 2]] = bboxes_out[:, [0, 2]] * height_width[1]
            bboxes_out[:, [1, 3]] = bboxes_out[:, [1, 3]] * height_width[0]

            info = {"boxes": bboxes_out.to(cpu_device),
                    "labels": labels_out.to(cpu_device),
                    "scores": scores_out.to(cpu_device)}
            outputs.append(info)

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
