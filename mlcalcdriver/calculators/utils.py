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


def get_derivative_names(property, avail):
    if property not in avail:
        if property == "forces" and "energy" in avail:
            init_property, out_name, derivative = "energy", "forces", -1
            wrt = ["_positions"]
        elif property == "hessian" and "forces" in avail:
            init_property, out_name, derivative = "forces", "hessian", -1
            wrt = ["_positions"]
        elif property == "hessian" and "energy" in avail:
            init_property, out_name, derivative = "energy", "hessian", 2
            wrt = ["_positions", "_positions"]
        else:
            raise ValueError(
                "The property {} is not in the available properties of the model : {}.".format(
                    property, avail
                )
            )
    elif property == "energy" and "energy_U0" in avail:
        init_property, out_name, derivative, wrt = "energy_U0", "", 0, []
    else:
        init_property, out_name, derivative, wrt = property, "", 0, []
    return init_property, out_name, derivative, wrt
