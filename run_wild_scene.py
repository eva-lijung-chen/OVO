import sys
import os

sys.path.append("/mnt/c/ntu_csie/1141/cvpdl/final/OVO/thirdParty/ORB_SLAM3/lib") # 或 python 資料夾

import cv2

import torch
import numpy as np

from typing import Dict
from datetime import datetime
from pathlib import Path
import argparse
import wandb
import torch
import numpy as np
import time
import yaml
import uuid
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import shutil

import json
import glob
import open3d as o3d

from ovo.utils import io_utils, gen_utils, eval_utils
from ovo.entities.ovomapping import OVOSemMap
from ovo.entities.ovo import OVO

def load_representation(scene_path: Path, eval: bool=False) -> OVO:
    config = io_utils.load_config(scene_path / "config.yaml", inherit=False)
    submap_ckpt = torch.load(scene_path /"ovo_map.ckpt" )
    map_params = submap_ckpt.get("map_params", None)
    if map_params is None:
        map_params = submap_ckpt["gaussian_params"]        
        
    ovo = OVO(config["semantic"],None, config["data"]["scene_name"], eval=eval, device=config.get("device", "cuda"))
    ovo.restore_dict(submap_ckpt["ovo_map_params"])
    return ovo, map_params


def compute_scene_labels(scene_path: Path, dataset_name: str, scene_name: str, data_path:str, dataset_info: Dict) -> None:
    print(f"Loading map from {scene_path}...")
    ovo, map_params = load_representation(scene_path, eval=True)
    
    # === 修改開始：不再寫死列表，而是從 YAML 讀取 ===
    # dataset_info 就是從 eval_info.yaml 讀進來的字典
    if "class_names" in dataset_info:
        classes = dataset_info["class_names"]
        print(f"Loaded {len(classes)} classes from config file.")
    else:
        # 萬一 YAML 讀取失敗的備案 (Fallback)
        print("Warning: 'class_names' not found in config! Using default simple list.")
        classes = ["wall", "floor", "chair", "table"] 

    # 執行分類
    print(f"Classifying 3D instances with CLIP using {len(classes)} queries...")
    instances_info = ovo.classify_instances(classes)

    if "masks" not in instances_info:
        instances_info["masks"] = []
    
    # 儲存結果
    if dataset_name == "custom":
        print("Saving custom classification results...")
        save_json_path = scene_path.parent / f"{scene_name}.json"
        
        # 整理要存的資料
        simple_results = {
            "classes": instances_info["classes"].tolist(), # 轉成 list 才能存 JSON
            "conf": instances_info["conf"].tolist(),
            "label_names": [classes[i] if i >= 0 else "unclassified" for i in instances_info["classes"]]
        }
        
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(simple_results, f, indent=4)
            
        print(f"Saved classification results to {save_json_path}")
        
        # 視覺化存成 PLY
        try:
            pcd_xyz = map_params["xyz"].cpu().numpy()
            pcd_obj_ids = map_params["obj_ids"][:, 0].cpu().numpy()
            
            obj_ids = list(ovo.objects.keys())
            pred_classes_indices = instances_info["classes"]
            
            # 建立 ID -> Class Index 的映射
            id_to_class_idx = { -1: -1 }
            for i, obj_id in enumerate(obj_ids):
                id_to_class_idx[obj_id] = pred_classes_indices[i]
            
            # === 顏色映射部分也改成讀取 YAML ===
            # 如果 YAML 裡有定義顏色表 (SCANNET_COLOR_MAP_200)，就用它
            # 否則隨機生成
            color_map = dataset_info.get("SCANNET_COLOR_MAP_200", None)
            
            pcd_colors = np.zeros_like(pcd_xyz)
            
            # 預先生成隨機顏色作為備用
            np.random.seed(42)
            random_colors = np.random.rand(len(classes), 3)

            for i in range(len(pcd_xyz)):
                c_idx = id_to_class_idx.get(pcd_obj_ids[i], -1)
                if c_idx >= 0:
                    # 嘗試從 YAML 顏色表讀取
                    if color_map and c_idx in color_map:
                        # YAML 顏色通常是 0-255，要轉成 0-1
                        pcd_colors[i] = np.array(color_map[c_idx]) / 255.0
                    else:
                        pcd_colors[i] = random_colors[c_idx]
                else:
                    pcd_colors[i] = [0.8, 0.8, 0.8] # 灰色

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
            save_ply_path = scene_path.parent / f"{scene_name}_semantic.ply"
            o3d.io.write_point_cloud(str(save_ply_path), pcd)
            print(f"Saved semantic point cloud to {save_ply_path}")
            
            # 輸出文字報告
            with open(scene_path.parent / f"{scene_name}_labels_report.txt", "w") as f:
                for i, obj_id in enumerate(obj_ids):
                    cls_idx = pred_classes_indices[i]
                    if cls_idx >= 0:
                        cls_name = classes[cls_idx]
                        conf = instances_info["conf"][i]
                        f.write(f"ID {obj_id}: {cls_name} ({conf:.2f})\n")
            
        except Exception as e:
            print(f"Visualization error: {e}")

    ovo.cpu()
    del ovo
    print("Segmentation done! ✨")


def run_scene(scene: str, dataset: str, experiment_name: str, tmp_run: bool = False, depth_filter: bool = None) -> None:
    torch.cuda.empty_cache()
    gc.collect()

    config = io_utils.load_config("data/working/configs/ovo.yaml")
    map_module = config["slam"]["slam_module"]
    if map_module == "orbslam2":
        map_module = "vanilla"
    # map_module = "orbslam2"

    if dataset == "custom":
        config["dataset_name"] = "custom"

        scene_path = Path(f"data/input/Datasets/{dataset}/{scene}")
        config["data"]["scene_name"] = scene
        config["data"]["input_path"] = str(scene_path)

        config["vis"]["stream"] = False
        config["vis"]["show_stream"] = False

        pose_files = sorted(glob.glob(str(scene_path / "pose_*.json")))
        if pose_files:
            print(f"Found {len(pose_files)} pose files.")
            with open(pose_files[0], 'r') as f:
                pose_data = json.load(f)
                intrinsics = pose_data["intrinsics"]
                config["cam"] = {
                    "H": intrinsics["height"],
                    "W": intrinsics["width"],
                    "fx": intrinsics["fx"],
                    "fy": intrinsics["fy"],
                    "cx": intrinsics["cx"],
                    "cy": intrinsics["cy"],
                    # "depth_scale": 3276.75,
                    "depth_scale": 6553.5,
                    "crop_edge": 0,
                    "distortion": None
                }
    else:        
        config_slam = io_utils.load_config(os.path.join(config["slam"]["config_path"],  map_module, dataset.lower()+".yaml"))
        io_utils.update_recursive(config, config_slam)

        config_dataset = io_utils.load_config(f"data/working/configs/{dataset}/{dataset.lower()}.yaml")
        io_utils.update_recursive(config, config_dataset)
        
        if os.path.exists(f"data/working/configs/{dataset}/{scene}.yaml"):
            config_scene = io_utils.load_config(f"data/working/configs/{dataset}/{scene}.yaml")
            io_utils.update_recursive(config, config_scene)
            
        if "data" not in config:
            config["data"] = {}
        config["data"]["scene_name"] = scene
        config["data"]["input_path"] = f"data/input/Datasets/{dataset}/{scene}"

    output_path = Path(f"data/output/{dataset}/")

    if tmp_run:
        output_path = output_path / "tmp"

    output_path = output_path / experiment_name / scene

    if depth_filter is not None:
        config["semantic"]["depth_filter"] = depth_filter

    if os.getenv('DISABLE_WANDB') == 'true':
        config["use_wandb"] = False
    elif config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            config=config,
            dir="data/working/output/wandb",
            group=config["data"]["scene_name"]
            if experiment_name != ""
            else experiment_name,
            name=f'{config["data"]["scene_name"]}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}_{str(uuid.uuid4())[:5]}',
        )

    gen_utils.setup_seed(config["seed"])

    config["vis"]["stream"] = False
    config["vis"]["show_stream"] = False

    gslam = OVOSemMap(config, output_path=output_path)

    # === [終極修正：Z-Up to Y-Up] ===
    if dataset == "custom":
        print("Converting Isaac Sim Z-Up to OVO Y-Up...")
        
        # 這是將 Z軸向上 轉為 Y軸向上 的矩陣
        # X -> X
        # Y -> Z
        # Z -> -Y
        ZUP_TO_YUP = np.array([
            [1,  0,  0,  0],
            [0,  0, -1,  0],
            [0,  1,  0,  0],
            [0,  0,  0,  1]
        ], dtype=np.float32)

        new_poses = []
        for pose in gslam.dataset.poses:
            # 矩陣乘法
            new_pose = pose @ ZUP_TO_YUP
            new_poses.append(new_pose)
        
        gslam.dataset.poses = new_poses

    gslam.run()

    if tmp_run:
        final_path = Path(f"data/output/{dataset}/") / experiment_name / scene
        shutil.move(output_path, final_path)

    if config["use_wandb"]:
        wandb.finish()
    print("Finished run.✨")

def main(args):
    if args.experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M")
        tmp_run = True
    else:
        assert len(args.experiment_name) > 0, "Experiment name cannot be '' "
        experiment_name = args.experiment_name
        tmp_run = False

    experiment_path = Path("data/output") / args.dataset_name / experiment_name

    if args.scenes_list is not None:
        with open(args.scenes_list, "r") as f:
            scenes = f.read().splitlines() 
    else:
        scenes = args.scenes

    if len(scenes) == 0 or args.segment or args.eval:
        path = Path("data/working/configs/") / args.dataset_name / args.dataset_info_file
        with open(path, 'r') as f:
            dataset_info = yaml.full_load(f)

        if len(scenes) == 0:
            scenes = dataset_info["scenes"]

    for scene in scenes:        
        input_path = f"./data/input/Datasets/{args.dataset_name}/{scene}"
        if args.run:
            t0 = time.time()
            run_scene(scene, args.dataset_name, experiment_name, tmp_run = tmp_run)
            t1 = time.time()
            print(f"Scene {scene} took: {t1-t0:.2f}")
        gc.collect()
 
    if args.segment: 
        data_path ="data/input/Datasets/"
        for scene in scenes:    
            scene_path = experiment_path / scene
            compute_scene_labels(scene_path, args.dataset_name, scene, data_path, dataset_info)

    if args.eval:
        if dataset_info["dataset"] == "scannet200":
            gt_path = Path(input_path).parent / "scannet200_gt"
        else:
            gt_path = Path(input_path).parent / "semantic_gt"
        eval_utils.eval_semantics(experiment_path / dataset_info["dataset"], gt_path, scenes, dataset_info, ignore_background=args.ignore_background)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments to run and evaluate over a dataset')
    parser.add_argument('--dataset_name', default="custom", help="Dataset used. Choose either `Replica`, `ScanNet`")
    parser.add_argument('--scenes', nargs="+", type=str, default=[], help=" List of scenes from given dataset to run.  If `--scenes_list` is set, this flag will be ignored.")
    parser.add_argument('--scenes_list',type=str, default=None, help="Path to a txt containing a scene name on each line. If set, `--scenes` is ignored. If neither `--scenes` nor `--scenes_list` are set, the scene list will be loaded from `data/working/config/<dataset_name>/<dataset_info_file>`")
    parser.add_argument('--dataset_info_file',type=str, default="eval_info.yaml")
    parser.add_argument('--experiment_name', default="ovomapping", type=str)
    parser.add_argument('--run', action='store_true', help="If set, compute the final metrics, after running OVO and segmenting.")
    parser.add_argument('--segment', action='store_true', help="If set, use the reconstructed scene to segment the gt point-cloud, after running OVO.")
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ignore_background', action='store_true',help="If set, does not use background ids from eval_info to compute metrics.")
    args = parser.parse_args()
    main(args)