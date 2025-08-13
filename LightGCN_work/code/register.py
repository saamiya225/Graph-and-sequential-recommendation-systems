# register.py — dataset + model registry (robust)

import world
import model
from world import cprint

print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)

# ---- Dataset ----
# Use the concrete loader; BasicDataset is abstract and will raise NotImplementedError.
try:
    from dataloader import Loader
except ImportError as e:
    raise ImportError(
        "Your dataloader.py must define a concrete Loader class. "
        "BasicDataset is abstract and cannot be instantiated."
    ) from e

# Try the two common constructor signatures
try:
    dataset = Loader(world.config)   # many forks accept config
except TypeError:
    dataset = Loader()               # some forks are arg-less and read world.config internally

# ---- Model registry (only add what exists) ----
MODELS = {}

if hasattr(model, 'PureMF'):
    MODELS['mf'] = model.PureMF

if hasattr(model, 'LightGCN'):
    MODELS['lgn'] = model.LightGCN

# Sanity check for the requested model
if world.model_name not in MODELS:
    raise ValueError(
        f"Requested model '{world.model_name}' is not available. "
        f"Available models: {list(MODELS.keys())}. "
        f"Ensure model.py defines the class, or run with --model one of {list(MODELS.keys())}."
    )
