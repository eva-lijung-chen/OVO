""" Slightly modified code based on Gaussian-SLAM's datasets
"""
import math
import os
from pathlib import Path

import cv2
import glob
import numpy as np
import torch
import json
import imageio


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        resize_ratio = dataset_config.get("resize_ratio", 1.0)
        self.height = int(dataset_config["H"]*resize_ratio)
        self.width = int(dataset_config["W"]*resize_ratio)
        self.fx = dataset_config["fx"]*resize_ratio
        self.fy = dataset_config["fy"]*resize_ratio
        self.cx = dataset_config["cx"]*resize_ratio
        self.cy = dataset_config["cy"]*resize_ratio

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class Replica(BaseDataset):

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(
            list((self.dataset_path / "results").glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        print(f"Loaded {len(self.color_paths)} frames")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)# added
        color_data = color_data.astype(np.uint8)# added
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale

        return index, color_data, depth_data, self.poses[index]

class ScanNet(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list(
            (self.dataset_path / "color").glob("*.jpg")), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(list(
            (self.dataset_path / "depth").glob("*.png")), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(self.dataset_path / "pose")
        depth_th = dataset_config.get("depth_th",0)
        if depth_th >0:
            self.depth_th = depth_th
        else:
            self.depth_th = None

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(path.glob('*.txt'),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                ls.append(list(map(float, line.split(' '))))
            c2w = np.array(ls).reshape(4, 4).astype(np.float32)
            self.poses.append(c2w)

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        lr_color_data = cv2.resize(color_data, (self.dataset_config["W"], self.dataset_config["H"]))

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        if self.depth_th is not None:
            depth_data[depth_data > self.depth_th] = 0
        edge = self.crop_edge
        if edge > 0:
            lr_color_data = lr_color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, lr_color_data, depth_data, self.poses[index], color_data


class ScanNetPP(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.use_train_split = dataset_config["use_train_split"]
        self.train_test_split = json.load(open(f"{self.dataset_path}/dslr/train_test_lists.json", "r"))
        if self.use_train_split:
            self.image_names = self.train_test_split["train"]
        else:
            self.image_names = self.train_test_split["test"]
        self.load_data()

    def load_data(self):
        self.poses = []
        cams_path = self.dataset_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        cams_metadata = json.load(open(str(cams_path), "r"))
        frames_key = "frames" if self.use_train_split else "test_frames"
        frames_metadata = cams_metadata[frames_key]
        frame2idx = {frame["file_path"]: index for index, frame in enumerate(frames_metadata)}
        P = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(np.float32)
        for image_name in self.image_names:
            frame_metadata = frames_metadata[frame2idx[image_name]]
            # if self.ignore_bad and frame_metadata['is_bad']:
            #     continue
            color_path = str(self.dataset_path / "dslr" / "undistorted_images" / image_name)
            depth_path = str(self.dataset_path / "dslr" / "undistorted_projected_depth" / image_name.replace('.JPG', '.png'))
            self.color_paths.append(color_path)
            self.depth_paths.append(depth_path)
            c2w = np.array(frame_metadata["transform_matrix"]).astype(np.float32)
            c2w = P @ c2w @ P.T
            self.poses.append(c2w)

    def __len__(self):
        if self.use_train_split:
            return len(self.image_names) if self.frame_limit < 0 else int(self.frame_limit)
        else:
            return len(self.image_names)

    def __getitem__(self, index):
        color_data = np.asarray(imageio.imread(self.color_paths[index]), dtype=float)
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)

        depth_data = np.asarray(imageio.imread(self.depth_paths[index]), dtype=np.int64)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index]


class Matterport(BaseDataset):

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "rgb").glob("*.png")))
        self.depth_paths = sorted(
            list((self.dataset_path / "depth").glob("*.png")))
        self.load_poses(self.dataset_path / "pose")
        print(f"Loaded {len(self.color_paths)} frames")

    def load_poses(self, path):
        poses = os.listdir(path)
        poses.sort()
        self.poses = []
        C = np.eye(4)
        C[1, 1] = -1
        C[2, 2] = -1
        for pose_file in poses:
            c2w = np.loadtxt(str(path / pose_file)).reshape(4, 4).astype(np.float32)

            c2w = np.matmul(c2w, C)
            self.poses.append(c2w.astype(np.float32))
          
    def _load_depth_intrinsics(self, H, W):
        """
        Load the depth camera intrinsics.

        Returns:
            Depth camera intrinsics as a numpy array (3x3 matrix).
        """        
        H, W = 720, 1080
        hfov = 90 * np.pi / 180
        vfov = 2 * math.atan(np.tan(hfov / 2) * H / W)
        fx = W / (2.0 * np.tan(hfov / 2.0))
        fy = H / (2.0 * np.tan(vfov / 2.0))
        cx = W / 2
        cy = H / 2
        depth_camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return depth_camera_matrix
      
    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)# added
        color_data = color_data.astype(np.uint8)# added
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale

        return index, color_data, depth_data, self.poses[index]



# class Custom(BaseDataset):
#     def __init__(self, dataset_config: dict):
#         super().__init__(dataset_config)
#         self.color_paths = sorted(
#             list((self.dataset_path).glob("rgb*.png")))
#         self.depth_paths = sorted(
#             list((self.dataset_path).glob("depth*.png")))
#         self.load_poses(self.dataset_path / "traj.txt")
#         print(f"Loaded {len(self.color_paths)} frames")

#     def load_poses(self, path):
#         self.poses = []
#         with open(path, "r") as f:
#             lines = f.readlines()
#         for line in lines:
#             c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
#             self.poses.append(c2w.astype(np.float32))

#     def __getitem__(self, index):
#         color_data = cv2.imread(str(self.color_paths[index]))
#         color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)# added
#         color_data = color_data.astype(np.uint8)# added
#         color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
#         depth_data = cv2.imread(
#             str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
#         depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
#         depth_data = depth_data.astype(np.float32) / self.depth_scale

#         return index, color_data, depth_data, self.poses[index]

class Custom(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        
        self.color_paths = sorted(list(self.dataset_path.glob("rgb_*.png")))
        self.depth_paths = sorted(list(self.dataset_path.glob("depth_*.png")))
        self.pose_paths = sorted(list(self.dataset_path.glob("pose_*.json")))

        if len(self.color_paths) == 0:
            if (self.dataset_path / "images").exists():
                self.color_paths = sorted(list((self.dataset_path / "images").glob("*.jpg"))) + \
                                  sorted(list((self.dataset_path / "images").glob("*.png")))
            else:
                self.color_paths = sorted(list(self.dataset_path.glob("*.jpg"))) + \
                                  sorted(list(self.dataset_path.glob("*.png")))

        if len(self.color_paths) == 0:
             raise FileNotFoundError(f"No images found in {self.dataset_path}")

        print(f"Loaded {len(self.color_paths)} frames from Custom Dataset")
        
        if len(self.pose_paths) > 0:
            print(f"Found {len(self.pose_paths)} JSON pose files. Loading...")
            self.load_poses_from_json()
        # elif (self.dataset_path / "traj.txt").exists():
        #     print("Found traj.txt. Loading...")
        #     self.load_poses(self.dataset_path / "traj.txt")
        else:
            print("Warning: No poses found. Returning Identity poses.")
            self.poses = [np.eye(4, dtype=np.float32) for _ in range(len(self.color_paths))]

    def load_poses_from_json(self):
        self.poses = []
        for pose_path in self.pose_paths:
            with open(pose_path, "r") as f:
                data = json.load(f)
                
                # 直接讀取 4x4 矩陣 (這是最原始、最不會錯的數據)
                # 您的 v5.py 已經做了 .T 轉置，存成了 [[R, R, R, t]...]
                # 所以這裡讀進來就是正確的 Column-Major
                c2w = np.array(data["T_world_camera"], dtype=np.float32)
                
                self.poses.append(c2w)

    # def load_poses_from_json(self):
    #     """
    #     從 pose_*.json 讀取位姿
    #     使用 position + quaternion_xyzw 建立 c2w 矩陣 (Column-Major)
    #     這能完美避開 v5.py 中矩陣轉置造成的混亂
    #     """
    #     self.poses = []

    #     for pose_path in self.pose_paths:
    #         with open(pose_path, "r") as f:
    #             data = json.load(f)

    #         # 1. 讀取位置 (Isaac Sim World Coordinates, Y-Up)
    #         pos = np.array(data.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)

    #         # 2. 讀取四元數 (x, y, z, w)
    #         q = data.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
    #         qx, qy, qz, qw = [float(v) for v in q]

    #         # 3. 正規化四元數 (防止數值誤差)
    #         norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    #         if norm > 0:
    #             qx /= norm; qy /= norm; qz /= norm; qw /= norm

    #         # 4. 四元數轉旋轉矩陣 (標準數學公式)
    #         xx, yy, zz = qx*qx, qy*qy, qz*qz
    #         xy, xz, yz = qx*qy, qx*qz, qy*qz
    #         wx, wy, wz = qw*qx, qw*qy, qw*qz

    #         R = np.array([
    #             [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
    #             [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
    #             [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy)],
    #         ], dtype=np.float32)

    #         # 5. 組合成 4x4 矩陣 (Column-Major: R 在左上，t 在最後一欄)
    #         c2w = np.eye(4, dtype=np.float32)
    #         c2w[:3, :3] = R
    #         c2w[:3, 3] = pos
    #         self.poses.append(c2w)

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            try:
                nums = list(map(float, line.split()))
                if len(nums) == 16:
                    c2w = np.array(nums).reshape(4, 4)
                    self.poses.append(c2w.astype(np.float32))
            except ValueError:
                continue

    def __getitem__(self, index):
        # 讀取影像
        color_path = self.color_paths[index]
        color_data = cv2.imread(str(color_path))
        if color_data is None:
            raise RuntimeError(f"Failed to read image: {color_path}")
            
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        
        # 讀取深度圖
        if index < len(self.depth_paths):
            depth_data = cv2.imread(str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
            if depth_data is not None:
                depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                depth_data = depth_data.astype(np.float32) / self.depth_scale
            else:
                depth_data = np.zeros((self.height, self.width), dtype=np.float32)
        else:
            depth_data = np.zeros((self.height, self.width), dtype=np.float32)

        # 確保 pose index 不會越界 (如果 json 數量少於圖片)
        pose_idx = index if index < len(self.poses) else -1
        pose = self.poses[pose_idx] if len(self.poses) > 0 else np.eye(4, dtype=np.float32)

        return index, color_data, depth_data, pose



def get_dataset(dataset_name: str):
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "scannet" :
        return ScanNet
    elif dataset_name == "scannetpp":
        return ScanNetPP
    elif dataset_name == 'matterport':
        return Matterport
    elif dataset_name == 'custom':
        return Custom
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
