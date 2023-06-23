class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, init_lr, peak_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.init_lr + (self.peak_lr - self.init_lr) * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr