import numpy as np
from layering import Parameters

# ─────────────────────────────────────────────────────────────────────────────
#  SGD with optional momentum and weight decay.
#  Mirrors the interface of the other codebase but adds gradient clipping.
# ─────────────────────────────────────────────────────────────────────────────

class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Args:
        parameters: Either a list of dicts  [{'params': [...], 'lr': ...}, ...]
                    or a list of Parameters objects.
        lr          (float): Default learning rate.
        momentum    (float): Momentum coefficient.
        weight_decay(float): L2 penalty coefficient.
        grad_clip   (float): Global gradient L2-norm clip (0 = disabled).
    """

    def __init__(self, parameters, lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 grad_clip: float = 10.0):

        if isinstance(parameters, list) and all(isinstance(p, dict) for p in parameters):
            self.param_groups = parameters
        elif all(isinstance(p, Parameters) for p in parameters):
            self.param_groups = [{"params": parameters, "lr": lr}]
        else:
            raise ValueError(
                "parameters must be a list of dicts or a list of Parameters.")

        self.defaults = {
            "lr": lr, "momentum": momentum,
            "weight_decay": weight_decay, "grad_clip": grad_clip,
        }
        # Velocity buffers (one per parameter, per group)
        self.velocities = [
            [np.zeros_like(p.values) for p in g["params"]]
            for g in self.param_groups
        ]

    def step(self):
        """Perform one SGD update step."""
        for group, vel_group in zip(self.param_groups, self.velocities):
            lr           = group.get("lr",           self.defaults["lr"])
            momentum     = group.get("momentum",     self.defaults["momentum"])
            wd           = group.get("weight_decay", self.defaults["weight_decay"])
            grad_clip    = group.get("grad_clip",    self.defaults["grad_clip"])

            for param, vel in zip(group["params"], vel_group):
                g = param.gradients.copy()
                if g.ndim == 1:
                    g = g.reshape(-1, 1)

                # Weight decay
                if wd != 0.0:
                    g = g + wd * param.values

                # Global gradient clip
                if grad_clip > 0.0:
                    gnorm = np.linalg.norm(g)
                    if gnorm > grad_clip:
                        g = g * (grad_clip / gnorm)

                vel *= momentum
                vel -= lr * g
                param.values += vel

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for group in self.param_groups:
            for param in group["params"]:
                param.gradients.fill(0)


# ─────────────────────────────────────────────────────────────────────────────
#  Learning-rate schedulers
# ─────────────────────────────────────────────────────────────────────────────

class MultiStepLR:
    """
    Multiply lr by `gamma` at each epoch listed in `milestones`.

    Args:
        optimizer:   An SGD (or compatible) optimizer instance.
        milestones:  List of epoch indices (1-based) at which to decay.
        gamma:       Multiplicative factor.
    """

    def __init__(self, optimizer, milestones: list, gamma: float = 0.1):
        self.optimizer   = optimizer
        self.milestones  = milestones
        self.gamma       = gamma
        self.last_epoch  = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            for group in self.optimizer.param_groups:
                group["lr"] = group.get("lr", self.optimizer.defaults["lr"]) * self.gamma


class PatienceLR:
    """
    Halve the learning rate after `patience` consecutive epochs without
    a sufficient decrease in the monitored loss.

    This avoids the single-bump decay that kills training too early with
    noisy mini-batch SGD loss estimates.

    Args:
        optimizer:  An SGD (or compatible) optimizer instance.
        patience:   Number of consecutive non-improving epochs before decay.
        factor:     Multiplicative factor (default 0.5 = halving).
        min_lr:     Floor below which lr is never reduced further.
        threshold:  Minimum improvement to count as progress.
    """

    def __init__(self, optimizer, patience: int = 3,
                 factor: float = 0.5, min_lr: float = 1e-5,
                 threshold: float = 1e-6):
        self.optimizer   = optimizer
        self.patience    = patience
        self.factor      = factor
        self.min_lr      = min_lr
        self.threshold   = threshold
        self.best_loss   = np.inf
        self.no_improve  = 0

    def step(self, loss: float):
        """
        Call once per epoch with the current training loss.

        Args:
            loss (float): Current epoch loss.

        Returns:
            bool: True if the lr was decayed this epoch.
        """
        if loss < self.best_loss - self.threshold:
            self.best_loss  = loss
            self.no_improve = 0
            return False

        self.no_improve += 1
        if self.no_improve >= self.patience:
            self.no_improve = 0
            decayed = False
            for group in self.optimizer.param_groups:
                old_lr = group.get("lr", self.optimizer.defaults["lr"])
                if old_lr > self.min_lr:
                    group["lr"] = max(old_lr * self.factor, self.min_lr)
                    decayed = True
            # Reset momentum buffers after decay for clean restart
            if decayed:
                for vel_group in self.optimizer.velocities:
                    for vel in vel_group:
                        vel.fill(0)
            return decayed
        return False
