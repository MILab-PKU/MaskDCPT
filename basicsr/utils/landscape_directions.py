import math

import torch


def create_random_directions(model):
    x_direction = create_random_direction(model)
    y_direction = create_random_direction(model)
    return [x_direction, y_direction]


def create_random_direction(model):
    weights = get_weights(model)
    direction = get_random_weights(weights)
    normalize_directions_for_weights(direction, weights)
    return direction


def get_weights(model):
    return [p.data for p in model.parameters()]


def get_random_weights(weights):
    random_weights = []
    for w in weights:
        if w.numel() == 1:
            random_weights.append(w)
        else:
            random_weights.append(
                torch.normal(mean=0, std=w.std(), size=w.shape).type_as(w)
            )
    return random_weights


def normalize_direction(direction, weights):
    for d, w in zip(direction, weights):
        d.mul_(w.norm() / (d.norm() + 1e-10))


def normalize_directions_for_weights(direction, weights):
    assert len(direction) == len(weights)
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            d.fill_(0)
        normalize_direction(d, w)
