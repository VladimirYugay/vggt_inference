import numpy as np
import cv2
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation
from PIL import Image


class TUM_RGBD(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, frame_rate=32):
        self.dataset_path = Path(dataset_path)

        # Hardcoded intrinsics
        self.fx, self.fy, self.cx, self.cy = 517.3, 516.5, 318.6, 255.3
        self.distortion = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])
        self.crop_left, self.crop_right = 16, 16
        self.crop_top, self.crop_bottom = 8, 8

        self.intrinsics = np.array([[self.fx, 0.0, self.cx],
                                    [0.0, self.fy, self.cy],
                                    [0.0, 0.0, 1.0]], dtype=np.float32)

        self.color_paths, self.poses = self._load_tum(self.dataset_path, frame_rate)

        # Adjust intrinsics after cropping
        self.cx -= self.crop_left
        self.cy -= self.crop_top

    def __len__(self):
        return len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        return np.loadtxt(filepath, delimiter=' ', dtype=str, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            j = np.argmin(np.abs(tstamp_depth - t))
            k = np.argmin(np.abs(tstamp_pose - t))
            if abs(tstamp_depth[j] - t) < max_dt and abs(tstamp_pose[k] - t) < max_dt:
                associations.append((i, j, k))
        return associations

    def pose_matrix_from_quaternion(self, pvec):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def _load_tum(self, datapath, frame_rate):
        pose_file = datapath / 'groundtruth.txt'
        image_file = datapath / 'rgb.txt'
        depth_file = datapath / 'depth.txt'

        image_data = self.parse_list(image_file)
        depth_data = self.parse_list(depth_file)
        pose_data = self.parse_list(pose_file, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)

        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indices = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indices[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indices.append(i)

        image_paths, poses = [], []
        inv_pose = None
        for ix in indices:
            i, j, k = associations[ix]
            image_path = datapath / image_data[i, 1]
            image_paths.append(str(image_path))
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w
            poses.append(c2w.astype(np.float32))

        return image_paths, poses

    def __getitem__(self, index):
        image_path = self.color_paths[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        return index, image, self.poses[index]
