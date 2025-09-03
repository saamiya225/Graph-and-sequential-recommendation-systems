
"""
register.py — V1 (Global Smoothing)

- Import a concrete dataset Loader (not the abstract BasicDataset)
- Instantiate the dataset (config-aware if your Loader supports it)
- Register available models (e.g., 'lgn' for LightGCN, 'mf' if PureMF present)
- Validate that the requested `world.model_name` exists in the registry
"""

import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book','instacart', 'instacart_small']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}
