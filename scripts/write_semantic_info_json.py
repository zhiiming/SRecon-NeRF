import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--out_json_path", default="",
                        help="input path to the dataset")
    args = parser.parse_args()
    return args


args = parse_args()
PALETTE = [[0, 0, 0], [0, 0, 170], [51, 0, 0], [0, 255, 0],
           [170, 0, 0], [0, 51, 0], [255, 0, 0], [170, 170, 0],
           [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 170, 0]]
CLASSES = ('background', 'floor', 'wall', 'ceiling',
           'door', 'window', 'cabinet', 'thing',
           'chair', 'computer', 'concrete', 'fireproof')

x = {}
stuff = list(CLASSES)
x["stuff"] = stuff
x["stuff_colors"] = PALETTE
json_object = json.dumps(x)
out_json_path = args.out_json_path
# out_json_path = '/home/ilab/dzm/GitHub/nerf_for_ipm/data_in/video_kb714_low_reflection/semantic_classes.json'
with open(out_json_path, "w") as outfile:
    outfile.write(json_object)
