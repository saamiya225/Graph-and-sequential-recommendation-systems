"""
register.py — Dataset & model registry (robust variant)

- Print run-time flags from `world` for quick visibility
- Import a concrete dataset loader (`Loader`) from dataloader.py
- Instantiate the dataset (supports both Loader(config) and Loader())
- Build a MODEL registry from classes present in model.py
- Validate that the requested model (world.model_name) exists
"""

import world
import model
from world import cprint

# Quick run context printout (visible at startup)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)

# ---- Dataset --------------------------------------------------------------
# Expect a concrete Loader in dataloader.py (BasicDataset is abstract).
try:
    from dataloader import Loader
except ImportError as e:
    raise ImportError(
        "Your dataloader.py must define a concrete Loader class. "
        "BasicDataset is abstract and cannot be instantiated."
    ) from e

# Support both common constructor signatures across forks:
#   1) Loader(config) — preferred
#   2) Loader()       — some forks read world.config internally
try:
    dataset = Loader(world.config)
except TypeError:
    dataset = Loader()

# ---- Model registry -------------------------------------------------------
MODELS = {}

# Register only what actually exists to avoid import-time errors.
if hasattr(model, 'PureMF'):
    MODELS['mf'] = model.PureMF

if hasattr(model, 'LightGCN'):
    MODELS['lgn'] = model.LightGCN

# ---- Sanity check ---------------------------------------------------------
if world.model_name not in MODELS:
    raise ValueError(
        f"Requested model '{world.model_name}' is not available. "
        f"Available models: {list(MODELS.keys())}. "
        f"Ensure model.py defines the class, or run with --model one of {list(MODELS.keys())}."
    )
