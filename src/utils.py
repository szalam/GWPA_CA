#%%
import configparser
from locale import normalize
import pathlib as plib
from collections import defaultdict
import json

def parse_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def get_datafpath():
    config = parse_config()
    data_path = plib.Path(config['path']['data'])
    if not data_path.is_dir:
        raise FileNotFoundError
    return data_path


# %%
