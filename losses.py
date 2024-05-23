import torch

import torch.nn as nn

class Dice(nn.Module):
    def __init__(self, n_classes=3, reduction="mean"):
        super(Dice, self).__init__()
        self.n_classes = n_classes
        self.reduction = reduction
        self.log_vars = nn.Parameter(torch.zeros(self.n_classes))

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        inputs = self._one_hot_encoder(inputs)

        dtype = inputs.dtype
        device = inputs.device
        losses = torch.zeros(self.n_classes, device=device)

        for i in range(self.n_classes):
            losses[i] += self._dice_loss(inputs[:, i, ...], target[:, i, ...])

        multi_task_losses = losses

        if self.reduction == 'sum':
            multi_task_losses = multi_task_losses.sum()
        elif self.reduction == 'mean':
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses

class L2Squared:

    def __call__(self, y_pred, y_true):
        return torch.mean(((y_true - y_pred)**2).sum(dim=1))
