def get_torch_device(verbose=True):
    import torch, platform

    system = platform.system()
    print("System: ", system)
    if system == "Darwin":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    if verbose:
        print(f"[Torch Device] {device} on {system}")
        if device.type == "cuda":
            print("  CUDA:", torch.cuda.get_device_name(0))
        elif device.type == "mps":
            print("  Apple Metal (MPS)")

    return device
