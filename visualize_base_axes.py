#!/usr/bin/env python3
import os
import pickle
import numpy as np
import cv2

def main():
    cal_dir = os.path.join(os.path.dirname(__file__), "calibration")
    vis_dir = os.path.join(cal_dir, "vis")
    cam_id = "235422302222"  # must match the one you used before

    # --- load base→world ---
    b = pickle.load(open(os.path.join(cal_dir, "base.pkl"), "rb"))
    R_b2w = b["R_base2world"]
    t_b2w = b["t_base2world"].reshape(3,1)

    # --- load world→cam for cam_id ---
    rvecs = pickle.load(open(os.path.join(cal_dir, "rvecs.pkl"), "rb"))
    tvecs = pickle.load(open(os.path.join(cal_dir, "tvecs.pkl"), "rb"))
    R_w2c, _ = cv2.Rodrigues(rvecs[cam_id])
    t_w2c = tvecs[cam_id].reshape(3,1)

    # --- compute base→cam once more, for clarity ---
    R_b2c = R_w2c.dot(R_b2w)
    t_b2c = R_w2c.dot(t_b2w) + t_w2c
    print("Computed R_base2cam:\n", R_b2c)
    print("Computed t_base2cam:\n", t_b2c.flatten())

    # --- load intrinsics (assume this cam is at index 0) ---
    K_all = np.load(os.path.join(cal_dir, "intrinsics.npy"))
    K = K_all[0]
    print("Using intrinsics K:\n", K)

    # --- pick an image to draw on ---
    img_path = os.path.join(vis_dir, f"calibration_img_{cam_id}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    # --- define base‐frame points to draw ---
    L = 0.1  # axis length in meters (10 cm)
    pts_base = np.array([
        [0,   0,   0],    # origin
        [L,   0,   0],    # X axis
        [0,   L,   0],    # Y axis
        [0,   0,   L],    # Z axis
    ]).T  # shape = 3×4

    # --- project each point into the camera image ---
    pts_cam = R_b2c.dot(pts_base) + t_b2c   # 3×4
    # normalize
    pts_norm = pts_cam / pts_cam[2:3,:]
    uvw = K.dot(pts_norm)                   # 3×4
    uv = uvw[:2,:].T.astype(int)            # 4×2

    names = ["O", "X", "Y", "Z"]
    for n, (u,v) in zip(names, uv):
        print(f"Pixel for {n}: (u={u}, v={v})")

    # --- draw axes ---
    origin, px, py, pz = uv
    cv2.line(img, tuple(origin), tuple(px), (0,0,255), 2)  # red X
    cv2.line(img, tuple(origin), tuple(py), (0,255,0), 2)  # green Y
    cv2.line(img, tuple(origin), tuple(pz), (255,0,0), 2)  # blue Z

    cv2.imshow("base axes overlay", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
