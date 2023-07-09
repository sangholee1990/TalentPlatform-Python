import torch
from torch.optim.lr_scheduler import LambdaLR


class ConstantLRWithWarmup(LambdaLR):
    """
    Increase lr to optimizer.lr linearly with ratio cur_step/num_warmup_steps
    If cur_step >= num_warmup_steps, the ratio will be 1.0
    Example:
        >>> scheduler = ConstantLRWithWarmup(optimizer, num_warmup_steps, last_epoch)
        ...
        >>> loss.backward()
        >>> optimizer.step()
        >>> scheduler.step()
    """
    def __init__(self, optimizer, num_warmup_steps, last_epoch=-1):
        def lr_lambda(cur_step):
            if cur_step < num_warmup_steps:
                return float(cur_step) / float(max(num_warmup_steps, 1.0))
            return 1.0
        super(ConstantLRWithWarmup, self).__init__(optimizer, lr_lambda, last_epoch)
        optimizer.zero_grad()
        optimizer.step()


class GradientAccumulator(object):
    """
    Update parameter after accumulate gradient by given accumulation_steps \n
    Example:
        >>> accumulator = GradientAccumulator(accumulation_steps=3)
        >>> optimizer = torch.optim.Adam(model.parameters())
        ...
        >>> loss = loss_func(predictions, target)
        >>> with accumulator(loss, optimizer) as accu:
                accu.backward()
                accu.step()
    """

    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self._cur_step = 0
        self._loss = None

    def __call__(self, loss, optimizer, lr_scheduler=None):
        self._loss = loss
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        return self

    def __enter__(self):
        self._cur_step += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def backward(self):
        self._loss = self._loss / self.accumulation_steps
        self._loss.backward()

    def step(self):
        if self._cur_step % self.accumulation_steps == 0:
            # change lr before apply gradients to weights
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
            self._optimizer.step()
            self._optimizer.zero_grad()

    @property
    def cur_step(self):
        return self._cur_step


class FGM(object):
    """
    From paper: Explaining and Harnessing Adversarial Examples
    Attack model's embedding layer as regularization for robustness
    """
    def __init__(self, model):
        self.backup = {}
        self.model = model

    def attack(self, epsilon=1., emb_name='emb'):
        # if self.emb = nn.Embedding(5000, 100) => layer_name="emb"
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # if self.emb = nn.Embedding(5000, 100) => layer_name="emb"
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
