# File: inspect_calibration.py

import os
import numpy as np
import pickle
import pprint
import cv2

def inspect_calibration(dir_path: str):
    print(f"=== Inspecting directory: {dir_path} ===\n")
    for fname in sorted(os.listdir(dir_path)):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath):
            if fname.endswith('.npy'):
                arr = np.load(fpath)
                print(f"{fname} (np.ndarray) → shape={arr.shape}, dtype={arr.dtype}")
                # print first few elements
                flat = arr.flatten()
                preview = flat[:min(10, flat.size)]
                print("  preview:", preview, "\n")
            elif fname.endswith('.pkl'):
                with open(fpath, 'rb') as f:
                    obj = pickle.load(f)
                print(f"{fname} (pickle) → type={type(obj)}")
                if isinstance(obj, np.ndarray):
                    print(f"  shape={obj.shape}, dtype={obj.dtype}")
                    print("  preview:", obj.flatten()[:min(10, obj.size)])
                else:
                    print("  content:")
                    pprint.pprint(obj, width=80, compact=True)
                print()
            else:
                print(f"{fname} (unknown file type)  —  size={os.path.getsize(fpath)} bytes\n")
        elif os.path.isdir(fpath):
            print(f"{fname}/ (dir) contains: {os.listdir(fpath)}\n")

    # now inspect any images under vis/
    vis_dir = os.path.join(dir_path, 'vis')
    if os.path.isdir(vis_dir):
        print(f"=== Inspecting images in {vis_dir} ===\n")
        for img_name in sorted(os.listdir(vis_dir)):
            img_path = os.path.join(vis_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"{img_name}: failed to load (not an image?)")
            else:
                print(f"{img_name}: shape={img.shape}, dtype={img.dtype}")
        print()

if __name__ == '__main__':
    # adjust this path if you run from somewhere else
    calibration_folder = os.path.join(os.path.dirname(__file__), 'calibration')
    inspect_calibration(calibration_folder)
