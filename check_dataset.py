import os.path
import glob
import json
import attrdict
import numpy as np

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

def loop_through_files(fs):
    print(len(fs), "files here")
    for idx, item in enumerate(fs):
        if idx % 200 == 0:
            print("read", idx, "files")
        with open(item, 'r') as f:
            try:
                data = json.load(f)
                data = from_json_dict(data)
            except json.decoder.JSONDecodeError as e:
                print("file error at", item)
                print(e)
                print()

town01_train = "/home/fireofearth/code/data/precog_carla_dataset/town01/train"
town01_val = "/home/fireofearth/code/data/precog_carla_dataset/town01/val"
town01_test = "/home/fireofearth/code/data/precog_carla_dataset/town01/test"
town02_test = "/home/fireofearth/code/data/precog_carla_dataset/town02/test"
train_generate850 = "/home/fireofearth/code/data/precog_carla_dataset/town01/train_generate850"
item_wildcard = "ma_*.json"

print("looking at train_generate850")
loc = glob.glob(os.path.join(train_generate850, item_wildcard))
loop_through_files(loc)

# print("looking at town01_train")
# town01 = glob.glob(os.path.join(town01_train, item_wildcard))
# loop_through_files(town01)
# print("looking at town01_val")
# town01 = glob.glob(os.path.join(town01_val, item_wildcard))
# loop_through_files(town01)
# print("looking at town01_test")
# town01 = glob.glob(os.path.join(town01_test, item_wildcard))
# loop_through_files(town01)
# print("looking at town02_test")
# town01 = glob.glob(os.path.join(town02_test, item_wildcard))
# loop_through_files(town01)
