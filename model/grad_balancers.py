#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:38:17 2024

@author: louis
"""

import torch
from torch import nn
from torch.autograd import grad


class AbsGradBalancer(nn.Module):
    def __init__(self, apply_at_backward=True):
        super().__init__()
        self.register_buffer("buffer_nan", torch.tensor(torch.nan))
        self.apply_at_backward = apply_at_backward
        # apply at backward: If gradient wrt last layer is longer to compute than gradient of all other layers

    def _compute_grad_wrt_last_layer(self, losses):
        return tuple(grad(loss, self.last_layer, retain_graph=True)[0].detach() for loss in losses)

    def _all_requires_grad(self, losses):
        return all(loss.requires_grad for loss in losses)

    def _all_finite_losses(self, losses):
        return all(loss.isfinite() for loss in losses)

    def forward(self, losses, model_parameters=None):
        new_grads = None
        if not self._all_finite_losses(losses):
            return losses, new_grads
        if self._all_requires_grad(losses):
            if self.apply_at_backward:
                extracted_grads, new_grads = self.run_backward(losses, model_parameters)
            else:
                extracted_grads = self._compute_grad_wrt_last_layer(losses)
            # Check grads are finite
            if not all(g.isfinite().all() for g in extracted_grads):
                return (l if g.isfinite().all() else self.buffer_nan for l, g in zip(losses, extracted_grads)), None
            self.update_internal_state(extracted_grads)
        return self.balance_losses(losses), new_grads

    def update_internal_state(self, grads):
        raise NotImplementedError()

    def balance_losses(self, losses):
        raise NotImplementedError()

    def balance_grads(self, grads):
        return torch.sum(torch.stack(self.balance_losses(grads), dim=0), dim=0)

    def run_backward(self, losses, model_parameters):
        if not isinstance(model_parameters, (list, tuple)):
            model_parameters = list(model_parameters)
        grads_losses = []
        for i_loss, loss in enumerate(losses):
            for p in model_parameters:
                p.grad = None
            loss.backward(retain_graph=i_loss % len(losses) != len(losses) - 1)
            grads_losses.append(list(p.grad for p in model_parameters))
        # -1 is not guaranteed to give us the last layer but it should give one of the latest
        # (model.parameters is populated in the order in which the modules are defined)
        new_grads = []
        for grad_losses_p in zip(*grads_losses):
            new_grads.append(self.balance_grads(grad_losses_p))
        last_layer_grads = [gl[-1] for gl in grads_losses]  # still necessary to check that grads are finite
        return last_layer_grads, new_grads


class CheckFiniteGrads(AbsGradBalancer):
    def balance_losses(self, losses):
        return losses

    def update_internal_state(self, grads):
        pass


class GradNorm(AbsGradBalancer):
    def __init__(
        self,
        ema_rate: float = 0.999,
        initial_model_and_signal_loss_weight: float = 1.0,
        initial_physical_properties_loss_weight: float = 0.0,
        apply_at_backward: bool = True,
    ):
        super().__init__(apply_at_backward=apply_at_backward)
        self.ema_rate = ema_rate
        self.initial_model_and_signal_loss_weight = initial_model_and_signal_loss_weight
        self.initial_physical_properties_loss_weight = initial_physical_properties_loss_weight
        weights = torch.tensor(
            (self.initial_model_and_signal_loss_weight, self.initial_physical_properties_loss_weight)
        )
        self.register_buffer("grad_norms", 1 - weights)

    def update_internal_state(self, grads):
        current_grad_norms = torch.stack([g.norm() for g in grads])
        new_grad_norms = self.ema_rate * self.grad_norms + (1 - self.ema_rate) * current_grad_norms
        self.grad_norms = new_grad_norms

    def balance_losses(self, inputs):
        weights = 1 - self.grad_norms / self.grad_norms.sum().clip(min=1e-7)
        return tuple(w * input for w, input in zip(weights, inputs))


class PCGrad:
    # from Yu et al., “Gradient Surgery for Multi-Task Learning.”
    ...


if __name__ == "__main__":
    grad_balancer = GradNorm(ema_rate=1 / 2, apply_at_backward=True)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_layer = nn.Parameter(torch.arange(2.0, requires_grad=True))

    def f(t):
        return (2 * t).sum()

    def g(t):
        return t.sum()

    model = Model()

    for i in range(20):
        l1 = f(model.last_layer)
        l2 = g(model.last_layer)
        if i % 3 == 0:
            l1 = l1.detach()
            l2 = l2.detach()
        (l1_balanced, l2_balanced), params = grad_balancer((l1, l2), model_parameters=model.parameters())
        p = params[0].grad if params is not None else params
        print(p)
        model.parameters._u
