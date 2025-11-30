import torch
import numpy as np
import argparse
import yaml
import json
from pathlib import Path
import open3d as o3d

# 引用 OVO 內部模組
from ovo.entities.visualizer import Visualizer, visualize_3d_points_obj_id_and_obb
from ovo.utils.io_utils import load_config
from run_wild_scene import load_representation

def main(args):
    # 1. 準備路徑
    # run_path 例如: data/output/custom/ovomapping/room0
    run_path = Path(args.run_path)
    scene_name = run_path.name
    
    print(f"Loading data from {run_path}...")

    # 2. 載入 Config 和重建結果 (OVO Map)
    config = load_config(run_path / "config.yaml")
    # 這裡的 load_representation 來自 run_wild_scene.py，已經能處理 custom
    semantic_module, params = load_representation(run_path, eval=True)
    
    pcd_pred = params["xyz"]         # 點雲座標 (Tensor)
    obj_ids = params["obj_ids"]      # 每個點的 Object ID (Tensor)
    
    # 3. 載入分類結果 (JSON)
    # 我們在 run_wild_scene.py 裡存成了 {scene_name}.json
    # 路徑應該是 data/output/custom/ovomapping/room0.json (在 run_path 的上一層)
    json_path = run_path.parent / f"{scene_name}.json"
    
    classes_list = [] # 儲存 Class 名稱
    pred_classes = [] # 儲存每個 Instance 的 Class Index
    
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # data["label_names"] 是每個 Instance 的文字標籤
            # data["classes"] 是每個 Instance 的 Class Index (對應到 eval_info.yaml)
            pred_classes = np.array(data["classes"])
            
            # 我們需要讀取 eval_info.yaml 裡的完整列表來對應 Index
            # 路徑: data/working/configs/custom/eval_info.yaml
            info_path = Path("data/working/configs/custom/eval_info.yaml")
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f2:
                    info = yaml.full_load(f2)
                    classes_list = info.get("class_names", [])
                    color_map = info.get("SCANNET_COLOR_MAP_200", {})
            else:
                print("Warning: eval_info.yaml not found!")
    else:
        print(f"Warning: Classification result {json_path} not found!")

    # 4. 準備顏色 (Semantic Colors)
    # 如果有 features_dc (Gaussian Splatting 的顏色)，優先使用
    # 否則使用分類顏色
    sh_c0 = 0.28209479177387814
    if params.get("features_dc", None) is not None:
        # Gaussian Splatting 的顏色通常需要轉換 SH 系數
        pcd_colors = (params["features_dc"] * sh_c0 + 0.5).clip(0, 1).flatten(0, 1)
    elif params.get("color") is not None:
        pcd_colors = params["color"]
    else:
        # 如果沒有顏色，給預設白色
        pcd_colors = torch.ones_like(pcd_pred) * 0.8

    # 如果要顯示分類顏色 (--visualize_semantic)
    if args.visualize_semantic and len(pred_classes) > 0 and len(color_map) > 0:
        print("Applying semantic colors...")
        semantic_colors = torch.zeros_like(pcd_pred)
        
        # 建立 obj_id -> color 的映射
        # semantic_module.objects.keys() 是所有存在的 ID
        obj_id_list = list(semantic_module.objects.keys())
        
        # 轉成 CPU numpy 方便操作
        pcd_obj_ids_np = obj_ids.cpu().numpy().flatten()
        
        # 建立快速查表 (Lookup Table)
        # 假設最大 ID 不會超過 10000
        max_id = int(pcd_obj_ids_np.max())
        lut = np.zeros((max_id + 1, 3))
        
        # 填入顏色
        for i, oid in enumerate(obj_id_list):
            if oid <= max_id and i < len(pred_classes):
                cls_idx = pred_classes[i]
                if cls_idx in color_map:
                    # YAML 顏色是 0-255，轉 0-1
                    c = np.array(color_map[cls_idx]) / 255.0
                    lut[oid] = c
                else:
                    lut[oid] = [0.8, 0.8, 0.8] # 灰色
        
        # 應用查表
        # 將無法查到的 ID (-1) 設為灰色
        mask_valid = (pcd_obj_ids_np >= 0) & (pcd_obj_ids_np <= max_id)
        semantic_colors_np = np.zeros((len(pcd_pred), 3))
        semantic_colors_np[mask_valid] = lut[pcd_obj_ids_np[mask_valid]]
        
        pcd_colors = torch.from_numpy(semantic_colors_np).float().to(pcd_pred.device)

    # 5. 啟動視覺化
    print("Starting visualizer...")

    while True:
        if args.visualize_obj:
            print("Visualizing Objects (Random Colors per Instance)...")
            visualize_3d_points_obj_id_and_obb(pcd_pred, obj_ids, pcd_colors)
            
        # 模式 2: 顯示語意分類結果
        if args.visualize_semantic:
            print("Visualizing Semantics (Class Colors)...")
            # 這裡我們直接用 Open3D 顯示點雲，因為 OVO 的 visualize_3d_points... 主要是畫 BBox
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_pred.cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors.cpu().numpy())
            o3d.visualization.draw_geometries([pcd], window_name="Semantic Visualization")

        # 模式 3: 互動式查詢 (Interactive Query)
        if args.visualize_interactive_query:
            print("Interactive Query Mode initialized.")
            print("Please enter text queries in the terminal when prompted.")
            vis = Visualizer(semantic_module, scene_name=config["data"]["scene_name"], save_path=run_path.parent)
            
            # 這裡傳入的顏色主要是給背景用的
            vis.visualize_and_query(pcd_pred, obj_ids.squeeze().cpu().numpy(), pcd_colors)

        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Wild Scene Results')
    parser.add_argument('run_path', type=str, help='Path to the output folder (e.g., data/output/custom/ovomapping/room0)')
    parser.add_argument('--visualize_rgb', action='store_true', help='Visualize original RGB texture')
    parser.add_argument('--visualize_obj', action='store_true', help='Visualize Instance Segmentation (Random Colors)')
    parser.add_argument('--visualize_semantic', action='store_true', help='Visualize Semantic Segmentation (Class Colors)')
    parser.add_argument('--visualize_interactive_query', action='store_true', help='Enable text-to-3d query')
    
    args = parser.parse_args()
    main(args)