
def register_grad_debug(tensor, name):
    """
    Registers a hook to print the gradient norm of a given tensor
    when backpropagation runs.

    Args:
        tensor (torch.Tensor): The tensor to watch.
        name (str): A label to show which tensor is being debugged.
    """
    if tensor is None:
        print(f"[DEBUG] {name} is None")
        return

    if hasattr(tensor, "requires_grad"):
        if tensor.requires_grad:
            print(f"[DEBUG] {name}.requires_grad = True")
            tensor.register_hook(
                lambda grad: print(f"[DEBUG] {name} grad norm: {grad.norm().item():.6f}")
            )
        else:
            print(f"[DEBUG] {name}.requires_grad = False — tensor not connected to the graph")
    else:
        print(f"[DEBUG] {name} has no 'requires_grad' attribute (not a tensor?)")
