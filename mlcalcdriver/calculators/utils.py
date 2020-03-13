import torch


def torch_derivative(fx, x, create_graph=False):
    if len(fx.shape) == 2:
        fx = fx.unsqueeze(0)
    dfdx = []
    flat_fx = fx.reshape(-1)
    for i in range(len(flat_fx)):
        (grad_x,) = torch.autograd.grad(
            flat_fx[i],
            x,
            torch.ones_like(flat_fx[i]),
            retain_graph=True,
            create_graph=create_graph,
        )
        dfdx.append(grad_x)
    return torch.stack(dfdx).reshape(fx.shape[2] * x.shape[1], fx.shape[1] * x.shape[2])
