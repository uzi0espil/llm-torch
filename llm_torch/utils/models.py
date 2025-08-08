import torch


def compare_state_dicts(model_a: torch.nn.Module,
                        model_b: torch.nn.Module,
                        atol: float = 1e-6,
                        rtol: float = 1e-5) -> bool:
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    # 1. Same set of keys?
    if sd_a.keys() != sd_b.keys():
        print("State dict keys differ!")
        print("In A not in B:", set(sd_a) - set(sd_b))
        print("In B not in A:", set(sd_b) - set(sd_a))
        return False

    # 2. Tensor‐by‐tensor comparison
    for key in sd_a.keys():
        tensor_a = sd_a[key]
        tensor_b = sd_b[key]
        if not torch.allclose(tensor_a, tensor_b, atol=atol, rtol=rtol):
            diff = (tensor_a - tensor_b).abs().max()
            print(f"Mismatch at '{key}': max abs diff = {diff:.3e}")
            return False

    print("All parameters match within tolerance.")
    return True
