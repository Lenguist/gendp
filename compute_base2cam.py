#!/usr/bin/env python3
import os
import pickle
import numpy as np
import cv2

def main():
    # 1) Path to calibration folder
    cal_dir = os.path.join(os.path.dirname(__file__), "calibration")
    print(f"Loading calibration data from: {cal_dir}\n")

    # 2) Load robot‐base → world
    base_pkl = os.path.join(cal_dir, "base.pkl")
    with open(base_pkl, "rb") as f:
        base_data = pickle.load(f)
    R_b2w = base_data["R_base2world"]
    t_b2w = base_data["t_base2world"].reshape(3,1)
    print("R_base2world (robot base → world):")
    print(R_b2w)
    print("t_base2world:")
    print(t_b2w.flatten(), "\n")

    # 3) Load world → camera (rvecs & tvecs)
    rvecs = pickle.load(open(os.path.join(cal_dir, "rvecs.pkl"), "rb"))
    tvecs = pickle.load(open(os.path.join(cal_dir, "tvecs.pkl"), "rb"))

    # Choose one camera ID to inspect
    cam_id = "235422302222"
    print(f"Using camera ID: {cam_id}\n")

    rvec = rvecs[cam_id]          # 3×1
    tvec = tvecs[cam_id].reshape(3,1)
    print("rvec (world → cam Rodrigues vector):", rvec.flatten())
    print("tvec (world → cam translation):", tvec.flatten(), "\n")

    # Convert rvec → R matrix
    R_w2c, _ = cv2.Rodrigues(rvec)
    print("R_world2cam (rotation):")
    print(R_w2c)
    print("t_world2cam (translation):", tvec.flatten(), "\n")

    # 4) Chain: base → world → cam
    R_b2c = R_w2c.dot(R_b2w)
    t_b2c = R_w2c.dot(t_b2w) + tvec
    print("R_base2cam (robot base → camera):")
    print(R_b2c)
    print("t_base2cam (translation):", t_b2c.flatten(), "\n")

    # 5) Load intrinsics
    K_all = np.load(os.path.join(cal_dir, "intrinsics.npy"))
    print("Loaded intrinsics array shape:", K_all.shape)
    # assume this camera is index 0 in intrinsics.npy
    K0 = K_all[0]
    print("K (intrinsics for camera index 0):")
    print(K0)

if __name__ == "__main__":
    main()
