# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

# 啟動 Isaac Sim（有 GUI）
simulation_app = SimulationApp({"headless": False})

import os
import sys
import json
import datetime

import carb
import carb.input as cinput   # 其實目前沒用到，但保留也沒關係
import numpy as np
from PIL import Image

import omni
import omni.usd
import omni.appwindow
import omni.replicator.core as rep
from pxr import Gf

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# ======================================================================
# 0. 設定：輸出資料夾 / 是否載入額外 USD 場景
# ======================================================================
OUTPUT_ROOT = "captures"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 每次啟動建立一個不重名的子資料夾
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(OUTPUT_ROOT, run_timestamp)
os.makedirs(RUN_DIR, exist_ok=True)
print("Saving dataset to:", RUN_DIR)

# 若你有自己的場景，可以把這裡改成實際路徑
ENV_USD_PATH = "/home/linser/Desktop/CVDL/final_project/Collected_warehouse_multiple_shelves/warehouse_multiple_shelves.usd"

# ======================================================================
# 1. 準備場景與機器人
# ======================================================================
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)

set_camera_view(
    eye=[5.0, 0.0, 1.5],
    target=[0.00, 0.00, 1.00],
    camera_prim_path="/OmniverseKit_Persp",
)

# (選擇性) 載入自訂 USD 場景到 /World/Env
if ENV_USD_PATH is not None:
    carb.log_info(f"Loading extra USD scene: {ENV_USD_PATH}")
    add_reference_to_stage(usd_path=ENV_USD_PATH, prim_path="/World/Env")
else:
    my_world.scene.add_default_ground_plane()  # 地面
    
# --- 加 Carter ---
# asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Carter/nova_carter/nova_carter.usd"
# add_reference_to_stage(usd_path=asset_path, prim_path="/World/Car")
asset_path = "/home/linser/Desktop/CVDL/cart_0.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Car")
car = Articulation(prim_paths_expr="/World/Car", name="my_car")

car.set_world_poses(positions=np.array([[0.0, -1.5, 0.0]]) / get_stage_units())

# ======================================================================
# 2. 建立 RGBD 相機 + Replicator Annotators
# ======================================================================
with rep.new_layer():
    # 將相機作為 /World/Car 的 child，位置是「相對小車」的座標
    sensor_cam = rep.create.camera(
        position=(0.0, 0.0, 0.5),   # 以車體為原點，可自行調整
        rotation=(0.0, 0.1, 0.0),   # (rx, ry, rz) in rad
        name="rgbd_camera",
        parent="/World/Car/chassis_link"
    )
    render_product = rep.create.render_product(sensor_cam, (640, 480))

# 使用 LdrColor（標準 RGB annotator，對應 LDR color）
rgb_annot = rep.AnnotatorRegistry.get_annotator("LdrColor")
rgb_annot.attach(render_product)

# 深度：距離相機中心（單位：m）
depth_cam_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
depth_cam_annot.attach(render_product)

# 相機內參 annotator（之後用來做 point cloud）
camparams_annot = rep.AnnotatorRegistry.get_annotator("camera_params")
camparams_annot.attach(render_product)
camera_intrinsics = None  # 第一次用時再填

# ======================================================================
# 3. 一些工具：PointCloud、Pose、PLY 儲存
# ======================================================================
stage = omni.usd.get_context().get_stage()
cam_prim = sensor_cam.get_output_prims()["prims"][0]

# def depth_to_pointcloud(depth_m, fx, fy, cx, cy):
#     """
#     depth_m: (H, W) float32, 單位公尺，對應 distance_to_camera
#     回傳: (N, 3) XYZ, N = H*W
#     """
#     H, W = depth_m.shape
#     u, v = np.meshgrid(np.arange(W), np.arange(H))  # u=x, v=y (pixel)

#     Z = depth_m
#     X = (u - cx) * Z / fx
#     Y = (v - cy) * Z / fy

#     pts = np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
#     pts = pts.reshape(-1, 3)           # (N, 3)
#     return pts

# def save_ply_xyzrgb(filename, pts_xyz, rgb):
#     """
#     pts_xyz: (N, 3) float32
#     rgb    : (H, W, 3) uint8 或 (N, 3) uint8
#     存成 ASCII PLY: x y z r g b
#     """
#     if rgb.ndim == 3:
#         H, W, _ = rgb.shape
#         rgb_flat = rgb.reshape(-1, 3)
#     else:
#         rgb_flat = rgb
#     assert pts_xyz.shape[0] == rgb_flat.shape[0]

#     N = pts_xyz.shape[0]
#     header = f"""ply
# format ascii 1.0
# element vertex {N}
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# end_header
# """

#     with open(filename, "w") as f:
#         f.write(header)
#         for (x, y, z), (r, g, b) in zip(pts_xyz, rgb_flat):
#             f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

# def get_camera_pose_world(cam_prim):
#     """
#     回傳:
#       position: [x, y, z]
#       quat_xyzw: [qx, qy, qz, qw]
#       T_world_camera: 4x4 list (row-major)
#     """
#     world_mat = omni.usd.get_world_transform_matrix(cam_prim)
#     translation = world_mat.ExtractTranslation()
#     rotation = world_mat.ExtractRotation().GetQuaternion()

#     pos = [float(translation[0]), float(translation[1]), float(translation[2])]
#     q_xyz = rotation.GetImaginary()
#     q_w = rotation.GetReal()
#     quat_xyzw = [float(q_xyz[0]), float(q_xyz[1]), float(q_xyz[2]), float(q_w)]

#     T = []
#     for r in range(4):
#         row = []
#         for c in range(4):
#             row.append(float(world_mat[r, c]))
#         T.append(row)

#     return pos, quat_xyzw, T

# V5 : change transpose 
def get_camera_pose_world(cam_prim):
    """
    回傳:
      position: [x, y, z]
      quat_xyzw: [qx, qy, qz, qw]
      T_world_camera: 4x4 list (row-major)
    """
    # 取得 Omniverse 的世界變換矩陣（USD Matrix4d）
    world_mat = omni.usd.get_world_transform_matrix(cam_prim)

    # ExtractTranslation() 和 ExtractRotation() 都是正確的
    translation = world_mat.ExtractTranslation()
    rotation = world_mat.ExtractRotation().GetQuaternion()

    pos = [float(translation[0]), float(translation[1]), float(translation[2])]
    q_xyz = rotation.GetImaginary()
    q_w = rotation.GetReal()
    quat_xyzw = [float(q_xyz[0]), float(q_xyz[1]), float(q_xyz[2]), float(q_w)]

    # === MODIFIED: 正確轉換 matrix（不要手工 iterate） ===
    import numpy as np
    world_mat_np = np.array(world_mat)       # column-major
    T_world_camera = world_mat_np.T.tolist() # convert to row-major

    return pos, quat_xyzw, T_world_camera

print("=== Camera Prim ===")
print("Camera prim path:", cam_prim.GetPath().pathString)

# ======================================================================
# 4. 初始化 world，取得小車 DOF 數
# ======================================================================
my_world.reset()

# 先抓一次 joint positions 來看 DOF 數量
car_joint_positions = car.get_joint_positions()
if car_joint_positions is None or len(car_joint_positions.shape) != 2:
    carb.log_error("Failed to get car joint positions.")
    simulation_app.close()
    sys.exit()

num_car_dofs = car_joint_positions.shape[1]
carb.log_info(f"Carter has {num_car_dofs} DOFs.")

# 找出你的小車兩個輪子的 dof index
joint_names = car.dof_names   # 例如 ['joint_wheel_left', 'joint_wheel_right', ...]
wheel_left_idx = joint_names.index("joint_wheel_left")
wheel_right_idx = joint_names.index("joint_wheel_right")

# ------------------------------------------------------------
# 初始化 Keyboard（pynput）
# ------------------------------------------------------------
from pynput import keyboard

keys_down = set()

def on_press(key):
    global keys_down
    try:
        keys_down.add(key.char.upper())
    except:
        pass

def on_release(key):
    global keys_down
    try:
        keys_down.discard(key.char.upper())
    except:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# 移動速度設定
LIN_SPEED = 20.0     # 前進/後退
ANG_SPEED = 10.0     # 旋轉

# ======================================================================
# 5. 主迴圈：WASD 控制小車 + 蒐集資料
# ======================================================================
frame_count = 0
SAVE_EVERY = 10  # 每 60 frame 存一次一整組資料

print("=== 控制說明 ===")
print("W/S : 前進 / 後退")
print("A/D : 左轉 / 右轉")
print("關閉 Isaac Sim 視窗即可結束程式")

while simulation_app.is_running():
    # 物理 + 繪圖
    my_world.step(render=True)
    # 觸發 replicator 生資料
    rep.orchestrator.step()

    frame_count += 1

    # --------------------------------------------------------------
    # 5.1 讀鍵盤，決定小車速度
    # --------------------------------------------------------------
    forward_cmd = 0.0
    turn_cmd = 0.0

    if "W" in keys_down:
        forward_cmd += 1.0
    if "S" in keys_down:
        forward_cmd -= 1.0
    if "A" in keys_down:
        turn_cmd += 1.0
    if "D" in keys_down:
        turn_cmd -= 1.0

    if abs(forward_cmd) > 1e-3 or abs(turn_cmd) > 1e-3:
        v = forward_cmd * LIN_SPEED
        w = turn_cmd * ANG_SPEED

        left_wheel_speed  = v - w
        right_wheel_speed = v + w

        vel = np.zeros((1, num_car_dofs), dtype=np.float32)
        vel[0, wheel_left_idx]  = left_wheel_speed
        vel[0, wheel_right_idx] = right_wheel_speed
        car.set_joint_velocities(vel)
    else:
        car.set_joint_velocities(np.zeros((1, num_car_dofs), dtype=np.float32))

     # --------------------------------------------------------------
    # 5.2 每隔 SAVE_EVERY frame 存一次完整資料：
    #     RGB PNG, Depth PNG, PointCloud PLY, Pose JSON
    # --------------------------------------------------------------
    if frame_count % SAVE_EVERY == 0:
        # ---------- 取 RGB / Depth ----------
        rgb = rgb_annot.get_data()
        depth = depth_cam_annot.get_data()

        rgb_np = np.array(rgb)      # (H, W, 4)
        depth_np = np.array(depth)  # (H, W)

        # ---------- 取 camera_params → 計算 intrinsics ----------
        cam_params = camparams_annot.get_data()

        # 解析解析度（注意：這裡是 [width, height]）
        W, H = cam_params["renderProductResolution"]

        focal_length = cam_params["cameraFocalLength"]       # mm，float
        horiz_aperture = cam_params["cameraAperture"][0]     # mm，水平膠片面

        vert_aperture = H / W * horiz_aperture               # 依比例推垂直膠片面

        fx = W * focal_length / horiz_aperture
        fy = H * focal_length / vert_aperture
        cx = W * 0.5
        cy = H * 0.5

        K = np.eye(3, dtype=float)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        camera_intrinsics = {
            "width": int(W),
            "height": int(H),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "K": K.tolist(),
        }

        # 共同 ID：同一時間點的檔名共用
        base_id = f"{frame_count:06d}"
        rgb_path   = os.path.join(RUN_DIR, f"rgb_{base_id}.png")
        depth_path = os.path.join(RUN_DIR, f"depth_{base_id}.png")
        ply_path   = os.path.join(RUN_DIR, f"cloud_{base_id}.ply")
        pose_path  = os.path.join(RUN_DIR, f"pose_{base_id}.json")

        # ---------- 存 RGB PNG ----------
        if rgb_np.ndim == 3 and rgb_np.shape[2] >= 3:
            rgb_img = Image.fromarray(rgb_np[:, :, :3].astype(np.uint8), mode="RGB")
            rgb_img.save(rgb_path)
            print(f"[{frame_count}] Saved RGB to {rgb_path}")
        else:
            print(f"[{frame_count}] WARNING: Unexpected RGB shape: {rgb_np.shape}")

        # ---------- 處理 depth ----------
        depth_valid = None
        if depth_np.ndim == 2:
            depth_valid = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

            max_depth = 20.0  # for PNG scaling
            depth_clipped = np.clip(depth_valid, 0.0, max_depth)
            depth_16 = (depth_clipped / max_depth * 65535.0).astype(np.uint16)

            depth_img = Image.fromarray(depth_16, mode="I;16")
            depth_img.save(depth_path)
            print(f"[{frame_count}] Saved Depth to {depth_path}")
        else:
            print(f"[{frame_count}] WARNING: Unexpected Depth shape: {depth_np.shape}")

        # v5: 不生成PLY
        # # ---------- 生成 point cloud + 存 PLY ----------
        # if depth_valid is not None:
        #     pts_xyz = depth_to_pointcloud(depth_valid, fx, fy, cx, cy)

        #     if rgb_np.ndim == 3 and rgb_np.shape[2] >= 3:
        #         rgb_3 = rgb_np[:, :, :3].astype(np.uint8)
        #     else:
        #         H, W = depth_valid.shape
        #         rgb_3 = np.full((H, W, 3), 128, dtype=np.uint8)

        #     save_ply_xyzrgb(ply_path, pts_xyz, rgb_3)
        #     print(f"[{frame_count}] Saved PointCloud to {ply_path}")

        # ---------- 取得當下相機世界座標與姿態，存 JSON ----------
        pos, quat_xyzw, T_world_cam = get_camera_pose_world(cam_prim)

        pose_data = {
            "frame_id": frame_count,
            "base_id": base_id,
            # === MODIFIED ===
            # 不再輸出 position/quaternion 自己算的姿態（容易錯）
            # 直接存 4×4 正確姿態矩陣
            # "position": pos,              # [x, y, z]
            # "quaternion_xyzw": quat_xyzw, # [qx, qy, qz, qw]
            "T_world_camera": T_world_cam,
            "intrinsics": camera_intrinsics,
        }

        with open(pose_path, "w") as f:
            json.dump(pose_data, f, indent=2)

        print(f"[{frame_count}] Saved Pose to {pose_path}")

# ======================================================================
# 6. 關閉 SimulationApp
# ======================================================================
simulation_app.close()
