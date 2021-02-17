"""
Print out contents of PRECOG dataset.
"""

import attrdict
import dill
import json
import glob
import logging
import numpy as np
import os
import pdb

def load_json(json_fn):
    """Load a json datum.

    :param json_fn: <str> the path to the json datum.
    :returns: dict of postprocess json data.
    """
    assert(os.path.isfile(json_fn))
    with open(json_fn, 'r') as f:
        json_datum = json.load(f)
    postprocessed_datum = from_json_dict(json_datum)
    return postprocessed_datum
    
def from_json_dict(json_datum):
    """Postprocess the json datum to ndarray-ify things

    :param json_datum: dict of the loaded json datum.
    :returns: dict of the postprocessed json datum.
    """
    pp = attrdict.AttrDict()
    for k, v in json_datum.items():
        if isinstance(v, list) or isinstance(v, float):
            pp[k] = np.asarray(v)
        elif isinstance(v, dict) or isinstance(v, int) or isinstance(v, str):
            pp[k] = v
        else:
            raise ValueError("Unrecognized type")
    return pp

def get_data():
    data_dir = "/media/external/data/precog_carla_dataset/town02/test/"
    data_file_wildcard = os.path.join(data_dir, "ma_*.json")
    data_file_paths = glob.glob(data_file_wildcard)
    data_file_path = data_file_paths[2]
    print("reading", data_file_path)
    data = load_json(data_file_path)
    return data

def test_1():
    data = get_data()
    
    print(data.keys())
    lidar_params = data["lidar_params"]
    print("lidar_params has keys:", lidar_params.keys())
    
    val_obstacle = lidar_params["val_obstacle"]
    print("val_obstacle", val_obstacle)
    
    hist_max_per_pixel = lidar_params["hist_max_per_pixel"]
    print("hist_max_per_pixel", hist_max_per_pixel)
    
    pixels_per_meter = lidar_params["pixels_per_meter"]
    print("pixels_per_meter", pixels_per_meter)
    
    meters_max = lidar_params["meters_max"]
    print("meters_max", meters_max)
    
    player_future = data["player_future"]
    print("player_future shape", player_future.shape)
    
    agent_futures = data["agent_futures"]
    print("agent_futures shape", agent_futures.shape)
    
    player_past = data["player_past"]
    print("player_past shape", player_past.shape)
    
    agent_pasts = data["agent_pasts"]
    print("agent_pasts shape", agent_pasts.shape)
    
    overhead_features = data["overhead_features"]
    print("overhead_features shape", overhead_features.shape)
    print("overhead_features type", type(overhead_features))

    print("episode", data["episode"])
    print("agent_transforms", data["agent_transforms"])
    print("frame", data["frame"])

def test_2():
    data = get_data()
    overhead_features = data["overhead_features"]

    for i in range(9):
        l = 10*i
        r = l + 10
        print("overhead_features channel 0")
        print(np.array_str(overhead_features[l:r,l:r,0],
            precision=2, suppress_small=True))
        print("overhead_features channel 1")
        print(np.array_str(overhead_features[l:r,l:r,1],
            precision=2, suppress_small=True))
        print("overhead_features channel 2")
        print(np.array_str(overhead_features[l:r,l:r,2],
            precision=2, suppress_small=True))
        print("overhead_features channel 3")
        print(np.array_str(overhead_features[l:r,l:r,3],
            precision=2, suppress_small=True))

test_1()

"""
reading /home/gutturale/code/data/precog_carla_dataset/town02/test/ma_datum_00003828.json
dict_keys(['lidar_params', 'agent_pasts', 'agent_futures', 'agent_transforms', 'overhead_features', 'episode', 'player_past', 'frame', 'player_future', 'player_transform'])
lidar_params has keys: dict_keys(['hist_max_per_pixel', 'val_obstacle', 'meters_max', 'pixels_per_meter'])
player_future shape (20, 3)
agent_futures shape (4, 20, 3)
player_past shape (10, 3)
agent_pasts shape (4, 10, 3)
overhead_features shape (200, 200, 4)
"""










