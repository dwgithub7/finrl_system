import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal Loss for Binary Classification (Entry)
    Sigmoid + Focal Loss
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probas = torch.sigmoid(inputs)
        focal_weight = (1 - probas) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SoftmaxFocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss for Multi-class Classification (Direction)
    Softmax + Focal Loss + Label Smoothing
    """
    def __init__(self, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        targets = F.one_hot(targets, num_classes).float()
        targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma

        loss = -targets * focal_weight * log_probs
        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ✅ Usage Example (Dummy Input)
if __name__ == "__main__":
    entry_criterion = BinaryFocalLossWithLogits(gamma=2.0)
    direction_criterion = SoftmaxFocalLossWithLabelSmoothing(gamma=2.0, label_smoothing=0.1)

    entry_preds = torch.randn(32, 1)
    entry_labels = torch.randint(0, 2, (32, 1)).float()

    direction_preds = torch.randn(32, 2)
    direction_labels = torch.randint(0, 2, (32,))

    entry_loss = entry_criterion(entry_preds, entry_labels)
    direction_loss = direction_criterion(direction_preds, direction_labels)

    print(f"Entry Loss: {entry_loss.item():.4f}")
    print(f"Direction Loss: {direction_loss.item():.4f}")
