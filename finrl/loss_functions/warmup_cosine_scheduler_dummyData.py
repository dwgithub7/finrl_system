import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingScheduler(_LRScheduler):
    """
    Warmup + CosineAnnealing Scheduler.
    초기 warmup_steps 동안 선형 증가, 이후 cosine 방식으로 감소.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            # Warmup 단계: 선형 증가
            warmup_factor = (step + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine Annealing 단계
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

# ✅ Usage Example
if __name__ == "__main__":
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = WarmupCosineAnnealingScheduler(
        optimizer=optimizer,
        warmup_steps=10,
        total_steps=100,
        min_lr=1e-6
    )

    for epoch in range(100):
        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:03d}: lr={current_lr:.6f}")
