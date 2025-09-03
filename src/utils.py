from math import sqrt
import itertools

import torch
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List
from torch import nn, Tensor
import numpy as np


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def calc_iou_tensor(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# This function is from https://github.com/kuangliu/pytorch-ssd.
class Encoder(object):
    def __init__(self, dboxes):
        self.dboxes = dboxes(order='ltrb')
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        ious = calc_iou_tensor(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)
        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)  # dim, index, value
        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.int64)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]

        temp_label_in = torch.arange(1, labels_in.shape[0]+1)
        glabel_local = torch.zeros(self.nboxes, dtype=torch.int64)
        glabel_local[masks] = temp_label_in[best_dbox_idx[masks]]

        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]

        # Transform format to xywh format
        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])  # x
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])  # y
        w = bboxes_out[:, 2] - bboxes_out[:, 0]  # w
        h = bboxes_out[:, 3] - bboxes_out[:, 1]  # h
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out, glabel_local

    def scale_back_batch(self, bboxes_in, scores_in):
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        # Returns a view of the original tensor with its dimensions permuted.
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)
        # print(bboxes_in.is_contiguous())

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, criteria=0.45, max_output=200):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, criteria, max_output))
        return outputs

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output=200):
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        bboxes_in = bboxes_in[:, 1:, :]
        scores_in = scores_in[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)
        scores_in = scores_in.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[inds], scores_in[inds], labels[inds]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 0.1 / 300) & (hs >= 0.1 / 300)
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only topk scoring predictions
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]
        # print(bboxes_in)

        return bboxes_out, labels_out, scores_out

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        # Reference to https://github.com/amdegroot/ssd.pytorch
        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            if i == 0:
                continue

            score = score.squeeze(1)

            mask = score > 0.05
            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()

                # we only need iou < criteria
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        if not bboxes_out:
            return [torch.empty(size=(0, 4)), torch.empty(size=(0,), dtype=torch.int64), torch.empty(size=(0,))]

        bboxes_out = torch.cat(bboxes_out, dim=0).contiguous()
        scores_out = torch.cat(scores_out, dim=0).contiguous()
        labels_out = torch.as_tensor(labels_out, dtype=torch.long)

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        self.fig_size = fig_size
        # [38, 19, 10, 5, 3, 1]
        self.feat_size = feat_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        # [8, 16, 32, 64, 100, 300]
        self.steps = steps

        # [21, 45, 99, 153, 207, 261, 315]
        self.scales = scales

        fk = fig_size / np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.as_tensor(self.default_boxes, dtype=torch.float32)
        self.dboxes.clamp_(min=0, max=1)

        # For IoU calculation
        # ltrb is left top coordinate and right bottom coordinate
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]   # xmin   dboxes(x_center,y_center,width,height)
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]   # ymin
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]   # xmax
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]   # ymax

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order='ltrb'):
        if order == 'ltrb':
            return self.dboxes_ltrb

        if order == 'xywh':
            return self.dboxes


def dboxes300_coco():
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2]]
    figsize = 512
    steps = [4, 8, 16, 32]
    feat_size = [128, 64, 32, 16]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    Parameters
    ----------
    boxes : Tensor[N, 4])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super(PostProcess, self).__init__()
        # [num_anchors, 4] -> [1, num_anchors, 4]
        self.dboxes_xywh = nn.Parameter(dboxes(order='xywh').unsqueeze(dim=0),
                                        requires_grad=False)
        self.scale_xy = dboxes.scale_xy  # 0.1
        self.scale_wh = dboxes.scale_wh  # 0.2

        self.criteria = 0.5
        self.max_output = 100

    def scale_back_batch(self, bboxes_in, scores_in):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        # Returns a view of the original tensor with its dimensions permuted.
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output):
        # type: (Tensor, Tensor, float, int) -> Tuple[Tensor, Tensor, Tensor]
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        bboxes_in = bboxes_in.clamp(min=0, max=1)

        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        bboxes_in = bboxes_in[:, 1:, :]
        scores_in = scores_in[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)
        scores_in = scores_in.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        # inds = torch.nonzero(scores_in > 0.05).squeeze(1)
        inds = torch.where(torch.gt(scores_in, 0.05))[0]
        bboxes_in, scores_in, labels = bboxes_in[inds, :], scores_in[inds], labels[inds]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 1 / 300) & (hs >= 1 / 300)
        # keep = keep.nonzero().squeeze(1)
        keep = torch.where(keep)[0]
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]
        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only topk scoring predictions
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]
        # print("1")
        return bboxes_out, labels_out, scores_out


    def forward(self, bboxes_in, scores_in, mode='test', max_num=100):
        if mode == 'train':
            bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

            outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
            for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):  # split_size, split_dim
                bbox = bbox.squeeze(0)
                prob = prob.squeeze(0)
                outputs.append(self.decode_single_new(bbox, prob, self.criteria, max_num))
        elif mode == 'test':
            bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

            for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):  # split_size, split_dim
                bbox = bbox.squeeze(0)
                prob = prob.squeeze(0)
                outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))
        return outputs
