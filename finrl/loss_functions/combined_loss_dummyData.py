import torch
import torch.nn as nn
from finrl.loss_functions.focal_loss_dummyData import BinaryFocalLossWithLogits, SoftmaxFocalLossWithLabelSmoothing


class CombinedLoss(nn.Module):
    """
    Combines Entry Loss and Direction Loss with adjustable weights (alpha, beta).
    """
    def __init__(self, 
                 entry_loss_fn, 
                 direction_loss_fn, 
                 alpha=1.0, 
                 beta=2.0):
        super().__init__()
        self.entry_loss_fn = entry_loss_fn
        self.direction_loss_fn = direction_loss_fn
        self.alpha = alpha
        self.beta = beta

    def forward(self, entry_preds, entry_labels, direction_preds, direction_labels):
        # Compute individual losses
        entry_loss = self.entry_loss_fn(entry_preds, entry_labels)
        direction_loss = self.direction_loss_fn(direction_preds, direction_labels)

        # Weighted sum
        total_loss = self.alpha * entry_loss + self.beta * direction_loss

        return total_loss, entry_loss, direction_loss

# ✅ Usage Example (Dummy Input)
if __name__ == "__main__":

    # Define loss functions
    entry_criterion = BinaryFocalLossWithLogits(gamma=2.0)
    direction_criterion = SoftmaxFocalLossWithLabelSmoothing(gamma=2.0, label_smoothing=0.1)

    # Combine them
    combined_criterion = CombinedLoss(entry_criterion, direction_criterion, alpha=1.0, beta=2.0)

    # Dummy data
    entry_preds = torch.randn(32, 1)
    entry_labels = torch.randint(0, 2, (32, 1)).float()

    direction_preds = torch.randn(32, 2)
    direction_labels = torch.randint(0, 2, (32,))

    total_loss, entry_loss, direction_loss = combined_criterion(entry_preds, entry_labels, direction_preds, direction_labels)

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Entry Loss: {entry_loss.item():.4f}")
    print(f"Direction Loss: {direction_loss.item():.4f}")
