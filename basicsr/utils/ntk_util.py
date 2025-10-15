import numpy as np
import torch
import torch.nn.functional as F


def check_window_size(window_size_stats):
    window_size, stats = window_size_stats
    if not (
        isinstance(window_size, tuple) or isinstance(window_size, list) and not stats
    ):
        return [window_size, True]
    return check_window_size([max(window_size), False])


def get_eigenvalue(matrix):
    try:
        eigenvalue, _ = torch.linalg.eigh(matrix, UPLO="L")
    except:
        matrix[matrix == float("Inf")] = matrix.max()
        eigenvalue, _ = torch.linalg.eigh(
            matrix + matrix.mean().item() * torch.eye(matrix.shape[0]).cuda() * 1e-4,
            UPLO="L",
        )  # ascending
    return eigenvalue


def ntk_loader(
    model: torch.nn.Module,
    xloader: torch.utils.data.DataLoader,
    device="cuda",
    opt=None,
):
    """Calculates the Components of the NTK and places into a dictionary whose keys are the named parameters of the model.

    While torch.vmap function is still in development, there appears to me to be an issue with how
    greedy torch.vmap allocates reserved memory. Batching the calls to vmap seems to help. Just like
    with training a model: you can go very fast with high batch size, but it requires an absurd amount
    of memory. Unlike training a model, there is no regularization, so you should make batch size as high
    as possible

    We suggest clearing the cache after running this operation.

        parameters:
            model: a torch.nn.Module object that terminates to a single neuron output
            xloader: a torch.data.utils.DataLoader object whose first value is the input data to the model
            device: a string, either 'cpu' or 'cuda' where the model will be run on

        returns:
            NTKs: a dictionary whose keys are the names parameters and values are said parameters additive contribution to the NTK
    """
    Js = [[]]
    for i, data in enumerate(xloader):
        inputs = data["lq"]

        model.train()
        # FIXME: for DAT, eval the BatchNorm
        for m in model.modules():
            if hasattr(m, "channel_interaction"):
                m.channel_interaction[2].eval()
        model.zero_grad()

        inputs = inputs.to(device, non_blocking=True)
        gt = data["gt"].to(device, non_blocking=True)

        # pad to multiplication of window_size
        _, _, h, w = inputs.size()
        mod_pad_h, mod_pad_w = 0, 0
        if "window_size" not in opt["network_g"]:
            inputs_ = inputs
        else:
            # FIXME: this is only supported when the shape of lq's H == W
            window_size, _ = check_window_size(
                [opt["network_g"].get("window_size", h), False]
            )
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            inputs_ = F.pad(inputs, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        inputs_ = inputs_.requires_grad_()

        y = model(inputs_)
        _, _, h, w = y.size()
        y = y[
            :,
            :,
            0 : h - mod_pad_h * opt.get("scale", 1),
            0 : w - mod_pad_w * opt.get("scale", 1),
        ]
        # loss = F.l1_loss(y, gt, reduction="mean")
        # grad = torch.autograd.grad(loss, backward_params, retain_graph=True, create_graph=True)
        y.backward(torch.ones_like(gt))

        jacobian_params = []
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                jacobian_params.append(param.grad.view(-1).detach())

        Js[0].append(torch.cat(jacobian_params, -1).detach())

        model.zero_grad()
        torch.cuda.empty_cache()

    Js = [torch.stack(_J, 0) for _J in Js]

    Js = [torch.einsum("nc, mc -> nm", [_J, _J]) for _J in Js]

    cond_Js, max_eigen_Js, min_eigen_Js = [], [], []
    for J in Js:
        J_value = get_eigenvalue(J)
        _cond = torch.div(J_value.max(), J_value.min())
        cond_Js.append(_cond.item())
        max_eigen_Js.append(J_value.max().item())
        min_eigen_Js.append(J_value.min().item())

    return (
        sum(cond_Js) / len(cond_Js),
        sum(max_eigen_Js) / len(max_eigen_Js),
        sum(min_eigen_Js) / len(min_eigen_Js),
    )


def hessian_loader(
    model: torch.nn.Module,
    xloader: torch.utils.data.DataLoader,
    device="cuda",
    opt=None,
    top_n=1,
    max_iter=100,
):
    """Calculates the Components of the NTK and places into a dictionary whose keys are the named parameters of the model.

    While torch.vmap function is still in development, there appears to me to be an issue with how
    greedy torch.vmap allocates reserved memory. Batching the calls to vmap seems to help. Just like
    with training a model: you can go very fast with high batch size, but it requires an absurd amount
    of memory. Unlike training a model, there is no regularization, so you should make batch size as high
    as possible

    We suggest clearing the cache after running this operation.

        parameters:
            model: a torch.nn.Module object that terminates to a single neuron output
            xloader: a torch.data.utils.DataLoader object whose first value is the input data to the model
            device: a string, either 'cpu' or 'cuda' where the model will be run on

        returns:
            NTKs: a dictionary whose keys are the names parameters and values are said parameters additive contribution to the NTK
    """

    def normalization(vs):
        """
        normalization of a list of vectors
        return: normalized vectors v
        """
        norms = [torch.sum(v * v) for v in vs]
        norms = [(norm**0.5).cpu().item() for norm in norms]
        vs = [vi / (norms[i] + 1e-6) for (i, vi) in enumerate(vs)]
        return vs

    def orthnormal(ws, vs_list):
        """
        make vector w orthogonal to each vector in v_list.
        afterwards, normalize the output w
        """
        for vs in vs_list:
            for w, v in zip(ws, vs):
                w.data.add_(-v * (torch.sum(w * v)))
        return normalization(ws)

    model.train()
    for i, data in enumerate(xloader):
        inputs = data["lq"]

        inputs = inputs.to(device, non_blocking=True)
        gt = data["gt"].to(device, non_blocking=True)

        # pad to multiplication of window_size
        _, _, h, w = inputs.size()
        if "window_size" not in opt["network_g"]:
            inputs_ = input
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
            inputs_ = F.pad(inputs, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        inputs_ = inputs_.requires_grad_()

        y = model(inputs_)
        _, _, h, w = y.size()
        y = y[
            :,
            :,
            0 : h - mod_pad_h * opt.get("scale", 1),
            0 : w - mod_pad_w * opt.get("scale", 1),
        ]

        loss = F.mse_loss(y, gt) / len(xloader)

        if i == 0:
            loss.backward(create_graph=True)
        else:
            loss.backward()

    weights, grads = [], []
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            weights.append(param)
            grads.append(param.grad)

    topn_eigenvalues = []
    eigenvectors = []
    computed_dim = 0
    while computed_dim < top_n:
        eigenvalues = None
        vs = [
            torch.randn_like(weight).requires_grad_() for weight in weights
        ]  # generate random vector
        vs = normalization(vs)  # normalize the vector

        for _ in range(max_iter):
            vs = orthnormal(vs, eigenvectors)

            model.zero_grad()
            torch.cuda.empty_cache()

            Hvs = torch.autograd.grad(
                grads,
                weights,
                grad_outputs=vs,
                retain_graph=True,
                only_inputs=True,
            )
            tmp_eigenvalues = [
                torch.sum(Hv * v).cpu().item() for (Hv, v) in zip(Hvs, vs)
            ]

            vs = normalization(Hvs)

            if eigenvalues == None:
                eigenvalues = tmp_eigenvalues
            else:
                if (
                    abs(sum(eigenvalues) - sum(tmp_eigenvalues))
                    / (abs(sum(eigenvalues)) + 1e-12)
                    < 1e-8
                ):
                    break
                else:
                    eigenvalues = tmp_eigenvalues

        topn_eigenvalues.append(eigenvalues)
        eigenvectors.append(vs)
        computed_dim += 1

    return topn_eigenvalues[0]
