import copy
import math
from functools import partial

import torch
import torch.nn.functional as F

from basicsr.utils import get_root_logger

from .ntk_util import check_window_size


def zero_init_delta_dict(delta_dict, rho):
    for param in delta_dict:
        delta_dict[param] = torch.zeros_like(param).cuda()

    delta_norm = torch.cat(
        [delta_param.flatten() for delta_param in delta_dict.values()]
    ).norm()
    for param in delta_dict:
        delta_dict[param] *= rho / delta_norm

    return delta_dict


def random_init_on_sphere_delta_dict(delta_dict, rho, **unused_kwargs):
    for param in delta_dict:
        delta_dict[param] = torch.randn_like(param).cuda()

    delta_norm = torch.cat(
        [delta_param.flatten() for delta_param in delta_dict.values()]
    ).norm()
    for param in delta_dict:
        delta_dict[param] *= rho / delta_norm

    return delta_dict


def random_gaussian_dict(delta_dict, rho):
    n_el = 0
    for param_name, p in delta_dict.items():
        delta_dict[param_name] = torch.randn_like(p).cuda()
        n_el += p.numel()

    for param_name in delta_dict.keys():
        delta_dict[param_name] *= rho / (n_el**0.5)

    return delta_dict


def random_init_lw(delta_dict, rho, orig_param_dict, norm="l2", adaptive=False):
    assert norm in ["l2", "linf"], f"Unknown perturbation model {norm}."

    for param in delta_dict:
        if norm == "l2":
            delta_dict[param] = torch.randn_like(delta_dict[param]).cuda()
        elif norm == "linf":
            delta_dict[param] = (
                2 * torch.rand_like(delta_dict[param], device="cuda") - 1
            )

    for param in delta_dict:
        param_norm_curr = orig_param_dict[param].abs() if adaptive else 1
        delta_dict[param] *= rho * param_norm_curr

    return delta_dict


def weight_ascent_step_momentum(
    opt,
    model,
    x,
    y,
    orig_param_dict,
    delta_dict,
    prev_delta_dict,
    step_size,
    rho,
    momentum=0.75,
    adaptive=False,
    norm="linf",
    rgb_range=255,
):
    """
    model:              w[k]
    orig_param_dict:    w[0]
    delta_dict:         w[k]-w[0]
    prev_delta_dict:    w[k-1]-w[0]
    1-alpha:            momentum coefficient

    -----------------------------------------------
    z[k+1] = P(w[k] + step_size*Grad F(w[k]))
    w[k+1] = P(w[k] + alpha*(z[k+1]-w[k])+(1-alpha)*(w[k]-w[k-1]))

    versions
    - old           -> L2-bound on all parameters, layer-wise rescaling
    - lw_l2_indep   -> L2-bound on each layer of rho * norm of the layer (if adaptive)
    """

    delta_dict_backup = {
        param: delta_dict[param].clone() for param in delta_dict
    }  # copy of perturbation dictionary  (w[k]-w[0])
    # curr_params = {name_p: p.clone() for name_p, p in model.named_parameters()}

    model.train()
    model.zero_grad()

    _, _, h, w = x.size()
    if "window_size" not in opt["network_g"]:
        x_ = x
    else:
        # FIXME: this is only supported when the shape of lq's H == W
        window_size, _ = check_window_size(
            [opt["network_g"].get("window_size", h), False]
        )
        mod_pad_h, mod_pad_w = 0, 0
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        x_ = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    x_ = x_.requires_grad_()
    output = model(x_)
    _, _, h, w = output.size()
    output = output[
        :,
        :,
        0 : h - mod_pad_h * opt.get("scale", 1),
        0 : w - mod_pad_w * opt.get("scale", 1),
    ]

    obj = F.l1_loss(output, y)

    obj.backward()
    del output
    torch.cuda.empty_cache()

    # Gradient ascent step, calculating perturbations
    with torch.no_grad():
        if norm == "l2":
            grad_norm = 0.0
            for name, param in model.named_parameters():
                if "weight" in name and param.requires_grad:
                    grad_norm += param.grad.norm() ** 2
            grad_norm = grad_norm**0.5

            for name, param in model.named_parameters():
                if "weight" in name and param.requires_grad:
                    delta_dict[param] += (
                        step_size
                        / (grad_norm + 1e-12)
                        * param.grad
                        * (1 if not adaptive else orig_param_dict[param].abs())
                    )
        elif norm == "linf":
            for name, param in model.named_parameters():
                if "weight" in name and param.requires_grad:
                    grad_sign_curr = param.grad.sign()
                    delta_dict[param] += (
                        step_size
                        * grad_sign_curr
                        * (1 if not adaptive else orig_param_dict[param].abs())
                    )
        else:
            raise ValueError("wrong norm")

    model.zero_grad()
    model.eval()

    with torch.no_grad():
        # Projection step I, rescaling perturbations
        if norm == "l2":  # Project onto L2-ball of radius rho (* ||w|| if adaptive)

            def weighted_norm(delta_dict):
                return (
                    sum(
                        [
                            (
                                (
                                    delta_dict[param]
                                    / (orig_param_dict[param].abs() if adaptive else 1)
                                )
                                ** 2
                            ).sum()
                            for param in delta_dict
                        ]
                    )
                    ** 0.5
                )

            if not adaptive:  # standard projection on the sphere
                delta_norm = weighted_norm(delta_dict)
                if delta_norm > rho:
                    for param in delta_dict:
                        delta_dict[param] *= rho / delta_norm
            else:  # projection on the ellipsoid
                lmbd = 0.1  # weighted_norm(delta_dict_tmp) / 2 / rho - 0.5
                max_lmbd_limit = 10.0
                min_lmbd, max_lmbd = 0.0, max_lmbd_limit
                delta_dict_tmp = {
                    param: delta_dict[param].clone() for param in delta_dict
                }

                curr_norm = new_norm = weighted_norm(delta_dict_tmp)
                if curr_norm > rho:
                    while (new_norm - rho).abs() > 10**-5:
                        curr_norm = new_norm
                        for param in delta_dict:
                            c = 1 / orig_param_dict[param].abs() if adaptive else 1
                            delta_dict_tmp[param] = delta_dict[param] / (
                                1 + 2 * lmbd * c**2
                            )
                        new_norm = weighted_norm(delta_dict_tmp)
                        if (
                            new_norm > rho
                        ):  # if the norm still exceeds rho, increase lmbd and set a new min_lmbd
                            lmbd, min_lmbd = (lmbd + max_lmbd) / 2, lmbd
                        else:
                            lmbd, max_lmbd = (min_lmbd + lmbd) / 2, lmbd
                        if (max_lmbd_limit - max_lmbd) < 10**-2:
                            max_lmbd_limit, max_lmbd = max_lmbd_limit * 2, max_lmbd * 2
                        # print(lmbd, weighted_norm(delta_dict_tmp))
                delta_dict = {
                    param: delta_dict_tmp[param].clone() for param in delta_dict_tmp
                }
        elif norm == "linf":
            # Project onto Linf-ball of radius rho (* |w| if adaptive)
            for param in delta_dict:
                param_curr = (
                    orig_param_dict[param].abs()
                    if adaptive
                    else torch.ones_like(orig_param_dict[param])
                )
                delta_dict[param] = torch.max(
                    torch.min(delta_dict[param], param_curr * rho),
                    -1.0 * param_curr * rho,
                )
        else:
            raise ValueError("wrong norm")

        # Average and Applying perturbations (apply momentum)
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                delta_dict[param] = (
                    momentum * delta_dict[param]
                    + (1 - momentum) * prev_delta_dict[param]
                )
                param.data = orig_param_dict[param] + delta_dict[param]
                prev_delta_dict[param] = delta_dict_backup[param]

    return delta_dict, prev_delta_dict


def eval_APGD_sharpness(
    opt,
    model,
    test_loader,
    rho=0.002,
    step_size_mult=1,
    n_iters=100,
    n_restarts=1,
    min_update_ratio=0.75,
    rand_init=True,
    adaptive=False,
    version="default",
    norm="linf",
    rgb_range=1,
    **kwargs,
):
    """Computes worst-case sharpness for every batch independently, and returns
    the average values.
    """
    logger = get_root_logger()

    assert n_restarts == 1 or rand_init, "Restarts need random init."

    init_fn = partial(random_init_lw, norm=norm, adaptive=adaptive)

    @torch.no_grad()
    def get_loss_and_err(model, x, y):
        """Compute loss and class. error on a single batch."""
        # pad to multiplication of window_size
        _, _, h, w = x.size()
        if "window_size" not in opt["network_g"]:
            x_ = x
        else:
            # FIXME: this is only supported when the shape of lq's H == W
            window_size, _ = check_window_size(
                [opt["network_g"].get("window_size", h), False]
            )
            mod_pad_h, mod_pad_w = 0, 0
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            x_ = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        output = model(x_)
        _, _, h, w = output.size()
        output = output[
            :,
            :,
            0 : h - mod_pad_h * opt.get("scale", 1),
            0 : w - mod_pad_w * opt.get("scale", 1),
        ]

        loss_mae = F.l1_loss(output / rgb_range, y / rgb_range)
        loss_mse = F.mse_loss(output / rgb_range, y / rgb_range)
        return loss_mae.item(), loss_mse.item()

    orig_model_state_dict = copy.deepcopy(model.state_dict())
    orig_param_dict = {param: param.clone() for param in model.parameters()}

    n_batches, delta_norm = 0, 0.0
    avg_mae_loss, avg_mse_loss, avg_init_mae_loss, avg_init_mse_loss = (
        0.0,
        0.0,
        0.0,
        0.0,
    )

    if version == "default":
        p = [0, 0.22]
        w = [0, math.ceil(n_iters * 0.22)]

        while w[-1] < n_iters and w[-1] != w[-2]:
            p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
            w.append(math.ceil(p[-1] * n_iters))

        w = w[1:]  # No check needed at the first iteration.
        step_size_scaler = 0.5
    else:
        raise ValueError(f"Unknown version {version}")

    for i_batch, data in enumerate(test_loader):
        x, y = data["lq"].cuda(), data["gt"].cuda()

        # Loss and err on the unperturbed model.
        init_mae_loss, init_mse_loss = get_loss_and_err(model, x, y)

        # Accumulate over batches.
        avg_init_mae_loss += init_mae_loss
        avg_init_mse_loss += init_mse_loss

        worst_mae_loss_over_restarts = init_mae_loss
        worst_mse_loss_over_restarts = init_mse_loss
        worst_delta_norm_over_restarts = 0.0

        for restart in range(n_restarts):
            delta_dict = {
                param: torch.zeros_like(param) for param in model.parameters()
            }
            delta_dict = init_fn(delta_dict, rho, orig_param_dict=orig_param_dict)
            for name, param in model.named_parameters():
                if "weight" in name and param.requires_grad:
                    param.data += delta_dict[param]

            prev_delta_dict = {param: delta_dict[param].clone() for param in delta_dict}
            worst_model_dict = copy.deepcopy(model.state_dict())

            prev_worst_mae_loss, worst_mae_loss = init_mae_loss, init_mae_loss
            worst_mse_loss = init_mse_loss
            step_size, prev_step_size = (
                2 * rho * step_size_mult,
                2 * rho * step_size_mult,
            )
            prev_cp = 0
            num_of_updates = 0

            for i in range(n_iters):
                torch.cuda.empty_cache()
                delta_dict, prev_delta_dict = weight_ascent_step_momentum(
                    opt,
                    model,
                    x,
                    y,
                    orig_param_dict,
                    delta_dict,
                    prev_delta_dict,
                    step_size,
                    rho,
                    momentum=0.75,
                    adaptive=adaptive,
                    norm=norm,
                    rgb_range=rgb_range,
                )

                with torch.no_grad():
                    curr_mae_loss, curr_mse_loss = get_loss_and_err(model, x, y)
                    delta_norm_total = (
                        torch.cat(
                            [
                                delta_param.flatten()
                                for delta_param in delta_dict.values()
                            ]
                        )
                        .norm()
                        .item()
                    )

                    if curr_mae_loss > worst_mae_loss:
                        worst_mae_loss = curr_mae_loss
                        worst_mse_loss = curr_mse_loss
                        worst_model_dict = copy.deepcopy(model.state_dict())
                        worst_delta_norm = delta_norm_total
                        num_of_updates += 1

                    if i in w:
                        cond1 = num_of_updates < (min_update_ratio * (i - prev_cp))
                        cond2 = (prev_step_size == step_size) and (
                            prev_worst_mae_loss == worst_mae_loss
                        )
                        prev_step_size, prev_worst_mae_loss, prev_cp = (
                            step_size,
                            worst_mae_loss,
                            i,
                        )
                        num_of_updates = 0

                        if cond1 or cond2:
                            step_size *= step_size_scaler
                            model.load_state_dict(worst_model_dict)

                str_to_log = "[batch={} restart={} iter={}] Sharpness: mae loss={:.6f}, mse loss={:.6f}, delta_norm={:.5f} (step={:.5f})".format(
                    i_batch + 1,
                    restart + 1,
                    i + 1,
                    curr_mae_loss - init_mae_loss,
                    curr_mse_loss - init_mse_loss,
                    delta_norm_total,
                    step_size,
                )
                logger.info(str_to_log)

            # Keep the best values over restarts.
            if worst_mae_loss > worst_mae_loss_over_restarts:
                worst_mae_loss_over_restarts = worst_mae_loss
                worst_mse_loss_over_restarts = worst_mse_loss
                worst_delta_norm_over_restarts = worst_delta_norm

            # Reload the unperturbed model for the next restart or batch.
            model.load_state_dict(orig_model_state_dict)

        # Accumulate over batches.
        n_batches += 1
        avg_mae_loss += worst_mae_loss_over_restarts
        avg_mse_loss += worst_mse_loss_over_restarts
        delta_norm = max(delta_norm, worst_delta_norm_over_restarts)

    vals = (
        (avg_mae_loss - avg_init_mae_loss) / n_batches,
        (avg_mse_loss - avg_init_mse_loss) / n_batches,
        delta_norm,
    )

    return vals


def eval_average_sharpness(
    opt,
    model,
    test_loader,
    n_iters=100,
    rho=0.1,
    adaptive=False,
    norm="l2",
    rgb_range=1,
):
    """Average case sharpness with Gaussian noise ~ (0, rho)."""

    logger = get_root_logger()

    @torch.no_grad()
    def get_loss_and_err(model, x, y):
        """Compute loss and class. error on a single batch."""
        # pad to multiplication of window_size
        _, _, h, w = x.size()
        if "window_size" not in opt["network_g"]:
            x_ = x
        else:
            # FIXME: this is only supported when the shape of lq's H == W
            window_size, _ = check_window_size(
                [opt["network_g"].get("window_size", h), False]
            )
            mod_pad_h, mod_pad_w = 0, 0
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            x_ = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        output = model(x_)
        _, _, h, w = output.size()
        output = output[
            :,
            :,
            0 : h - mod_pad_h * opt.get("scale", 1),
            0 : w - mod_pad_w * opt.get("scale", 1),
        ]

        loss_mae = F.l1_loss(output / rgb_range, y / rgb_range)
        loss_mse = F.mse_loss(output / rgb_range, y / rgb_range)
        return loss_mae.item(), loss_mse.item()

    orig_param_dict = {
        param_name: p.clone() for param_name, p in model.named_parameters()
    }  # {param: param.clone() for param in model.parameters()}
    # orig_norm = torch.cat([p.flatten() for p in orig_param_dict.values()]).norm()

    orig_norm, n_el = 0, 0
    for n, p in orig_param_dict.items():
        if "weight" in n and p.requires_grad:
            orig_norm += p.flatten().norm() ** 2.0 * p.numel()
            n_el += p.numel()
    orig_norm = (orig_norm / n_el) ** 0.5
    noisy_model = copy.deepcopy(model)

    delta_dict = {
        param_name: torch.zeros_like(param)
        for param_name, param in model.named_parameters()
    }
    logger.info(f"rho:{rho}, samples: {n_iters}")

    n_batches, avg_mae_loss, avg_mse_loss, avg_init_mae_loss, avg_init_mse_loss = (
        0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            x, y = data["lq"].cuda(), data["gt"].cuda()

            # Loss and err on the unperturbed model.
            init_mae_loss, init_mse_loss = get_loss_and_err(model, x, y)
            avg_init_mae_loss += init_mae_loss
            avg_init_mse_loss += init_mse_loss

            batch_mae_loss, batch_mse_loss = 0.0, 0.0

            for _ in range(n_iters):
                delta_dict = random_init_lw(
                    delta_dict, rho, orig_param_dict, norm=norm, adaptive=adaptive
                )
                for (param_name, delta), (name, param) in zip(
                    delta_dict.items(), noisy_model.named_parameters()
                ):
                    if "weight" in name and param.requires_grad:
                        param.data = orig_param_dict[param_name] + delta

                curr_mae_loss, curr_mse_loss = get_loss_and_err(noisy_model, x, y)
                batch_mae_loss += curr_mae_loss
                batch_mse_loss += curr_mse_loss

            n_batches += 1
            avg_mae_loss += batch_mae_loss / n_iters
            avg_mse_loss += batch_mse_loss / n_iters

            str_to_log = (
                f"[batch={i_batch + 1}] obj={batch_mae_loss / n_iters - init_mae_loss}"
                + f" err={batch_mse_loss / n_iters - init_mse_loss}"
            )
            logger.info(str_to_log)

    vals = (
        (avg_mae_loss - avg_init_mae_loss) / n_batches,
        (avg_mse_loss - avg_init_mse_loss) / n_batches,
        0.0,
    )

    return vals
