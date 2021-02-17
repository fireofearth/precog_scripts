"""
Create modified copies of PRECOG dataset.
"""

import attrdict
import dill
import json
import glob
import logging
import numpy as np
import os
import os.path
import pdb

def load_json(json_fn):
    """Load a json datum.

    :param json_fn: <str> the path to the json datum.
    :returns: dict of postprocess json data.
    """
    assert(os.path.isfile(json_fn))
    with open(json_fn, 'r') as f:
        json_datum = json.load(f)
    return json_datum
    #postprocessed_datum = from_json_dict(json_datum)
    #return postprocessed_datum

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

data_dir = "/home/fireofearth/code/data/precog_carla_dataset/town01/test_experiment_0"
data_file_wildcard = os.path.join(data_dir, "ma_datum_*.json")
data_file_paths = glob.glob(data_file_wildcard)

def make_zeroed_past():
    for p in data_file_paths:
        print("Creating modified copies for", p)
        datum = load_json(p)
        # create data with zeroed out past
        print("create data with zeroed out past")
        zeroed_past_datum = datum.copy()
        a = np.asarray(zeroed_past_datum['agent_pasts'])
        a = np.zeros_like(a)
        zeroed_past_datum['agent_pasts'] = a.tolist()
        a = np.asarray(zeroed_past_datum['player_past'])
        a = np.zeros_like(a)
        zeroed_past_datum['player_past'] = a.tolist()
        tail, head = os.path.split(p)
        head, ext = head.split('.')
        code = head.split('_')[-1]
        save_path = os.path.join(data_dir, "ma_zeroed_past_datum_{}.{}".format(code, ext))
        with open(save_path, 'w') as f:
            json.dump(zeroed_past_datum, f)
#make_zeroed_past()

def swap_overhead_0():
    swap_datum = load_json(data_file_paths[0])
    swap_a = np.asarray(swap_datum['overhead_features'])
    print("Swapping past of rest files with ", data_file_paths[0])
    for p in data_file_paths[1:]:
        print("Creating modified copies for", p)
        datum = load_json(p)
        datum = datum.copy()
        datum['overhead_features'] = swap_a.tolist()
        tail, head = os.path.split(p)
        head, ext = head.split('.')
        code = head.split('_')[-1]
        save_path = os.path.join(data_dir, "ma_swap_0_datum_{}.{}".format(code, ext))
        with open(save_path, 'w') as f:
            json.dump(datum, f)
swap_overhead_0()

def swap_overhead_1():
    swap_datum = load_json(data_file_paths[1])
    swap_a = np.asarray(swap_datum['overhead_features'])
    print("Swapping past of rest files with ", data_file_paths[1])
    for p in (data_file_paths[0:1] + data_file_paths[2:]):
        print("Creating modified copies for", p)
        datum = load_json(p)
        datum = datum.copy()
        datum['overhead_features'] = swap_a.tolist()
        tail, head = os.path.split(p)
        head, ext = head.split('.')
        code = head.split('_')[-1]
        save_path = os.path.join(data_dir, "ma_swap_1_datum_{}.{}".format(code, ext))
        with open(save_path, 'w') as f:
            json.dump(datum, f)
swap_overhead_1()

def swap_overhead_2():
    swap_datum = load_json(data_file_paths[2])
    swap_a = np.asarray(swap_datum['overhead_features'])
    print("Swapping past of rest files with ", data_file_paths[2])
    for p in (data_file_paths[0:2] + data_file_paths[3:]):
        print("Creating modified copies for", p)
        datum = load_json(p)
        datum = datum.copy()
        datum['overhead_features'] = swap_a.tolist()
        tail, head = os.path.split(p)
        head, ext = head.split('.')
        code = head.split('_')[-1]
        save_path = os.path.join(data_dir, "ma_swap_2_datum_{}.{}".format(code, ext))
        with open(save_path, 'w') as f:
            json.dump(datum, f)
swap_overhead_2()

def swap_overhead_3():
    swap_datum = load_json(data_file_paths[0])
    swap_a = np.rot90(np.asarray(swap_datum['overhead_features']))
    print("Swapping past of rest files with ", data_file_paths[0])
    for p in data_file_paths[1:]:
        print("Creating modified copies for", p)
        datum = load_json(p)
        datum = datum.copy()
        datum['overhead_features'] = swap_a.tolist()
        tail, head = os.path.split(p)
        head, ext = head.split('.')
        code = head.split('_')[-1]
        save_path = os.path.join(data_dir, "ma_swap_3_datum_{}.{}".format(code, ext))
        with open(save_path, 'w') as f:
            json.dump(datum, f)
swap_overhead_0()

def make_zeroed_overhead():
    for p in data_file_paths:
        print("Creating modified copies for", p)
        datum = load_json(p)
        # create data with no overhead features
        print("create data with no overhead features")
        zeroed_overhead_datum = datum.copy()
        a = np.asarray(zeroed_overhead_datum['overhead_features'])
        a = np.zeros_like(a)
        zeroed_overhead_datum['overhead_features'] = a.tolist()
        tail, head = os.path.split(p)
        head, ext = head.split('.')
        code = head.split('_')[-1]
        save_path = os.path.join(data_dir, "ma_zeroed_overhead_datum_{}.{}".format(code, ext))
        with open(save_path, 'w') as f:
            json.dump(zeroed_overhead_datum, f)
make_zeroed_overhead()

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
