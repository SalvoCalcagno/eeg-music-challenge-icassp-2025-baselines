import os
import json

config_file = "config.json"

def get_config(prefix = ""):
    with open(os.path.join(prefix, config_file), "r") as f:
        config = json.load(f)
    return config

def get_attribute(attribute, prefix = ""):
    config = get_config(prefix)
    return config[attribute]