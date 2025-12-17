import torch
import torch.nn.functional as func

def get_edge_mask(gt, kernel=3):
    gt = gt.float()
    dilation = func.max_pool2d(input=gt, kernel_size=kernel, stride=1, padding=kernel//2)
    erosion = 1.0 - func.max_pool2d(1.0 - gt, kernel_size=kernel, stride=1, padding=kernel//2)
    edge = (dilation - erosion).clamp(min=0, max=1)

    return edge


class IOU:
    def __init__(self, threshold=0.5, epsilon=1e-6):
        self.threshold = threshold
        self.epsilon = epsilon

    def compute_iou_pos(self, logits, gt):
        probs = logits.sigmoid()
        predict = (probs > self.threshold).float().flatten()
        gt = gt.float().flatten(1)

        intersection = (predict * gt).sum(dim=1)
        union = (predict + gt - predict * gt).sum(dim=1)
        iou = (intersection + self.epsilon) / (union + self.epsilon)

        mean_iou_all = iou.mean().item()
        position = (gt.sum(dim=1) > 0)

        if position.any():
            mean_iou_pos = iou[position].mean().item()
        else:
            mean_iou_pos = 0.0

        return mean_iou_all, mean_iou_pos

    def __call__(self, predict, target):
        predicts = (torch.sigmoid(predict)> self.threshold).float().flatten(1)
        targets = target.flatten(1)
        intersection = (predicts * targets).sum(dim=1)
        union = (predicts + targets - predicts * targets).sum(dim=1)

        return ((intersection + self.epsilon) / (union + self.epsilon)).mean()
