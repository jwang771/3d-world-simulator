"""Compilation helpers for the sparse encoder."""

import torch
from torch import nn


def compile_sparse_encoder(encoder, mode="default", scope="dense"):
    """Compile the sparse encoder's dense submodules without changing state keys.

    The full sparse graph crosses Python wrappers and C++ pybind extensions,
    which can cause Dynamo to repeatedly break and specialize. Dense scope
    compiles normalization and latent-head forwards independently while leaving
    sparse convolution dispatch eager. Compiling each module's bound ``forward``
    keeps the module hierarchy and checkpoint state-dict keys intact.
    """
    if scope == "full":
        return torch.compile(encoder, mode=mode)
    if scope != "dense":
        raise ValueError(f"Unknown encoder compile scope: {scope}")

    for name in ("to_latent", "to_latent_col", "to_latent_col_direct"):
        module = getattr(encoder, name, None)
        if module is not None:
            module.forward = torch.compile(module.forward, mode=mode)

    dense_prefixes = ("to_latent", "to_latent_col", "to_latent_col_direct")
    for path, module in list(encoder.named_modules()):
        if not isinstance(module, nn.GroupNorm) or path.startswith(dense_prefixes):
            continue
        module.forward = torch.compile(module.forward, mode=mode)
    return encoder
