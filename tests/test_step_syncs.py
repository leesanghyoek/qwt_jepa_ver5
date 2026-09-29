"""Fewer GPU syncs per update, same numbers.

Phase 2 read every logged term back with its own float() (~20 syncs per microbatch)
and checked gradients one parameter tensor at a time (hundreds of syncs per update),
each stopping the CPU from queueing work while the GPU finished. Both now read back
once; these pin that nothing they report changed meaning.
"""

from __future__ import annotations

import math

import torch

from qjepa.training.phase1 import _finite_gradients


def _parameters(*gradients):
    parameters = []
    for gradient in gradients:
        parameter = torch.nn.Parameter(torch.zeros(3))
        parameter.grad = gradient
        parameters.append(parameter)
    return parameters


def test_one_check_finds_any_non_finite_gradient():
    assert _finite_gradients(_parameters(torch.ones(3), None, torch.zeros(3)))
    assert _finite_gradients(_parameters(None)) and _finite_gradients([])
    for bad in (math.inf, -math.inf, math.nan):
        assert not _finite_gradients(_parameters(torch.ones(3), torch.tensor([0.0, bad, 1.0])))
