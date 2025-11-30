import json
import glob
import os
import argparse

def main():
    # 設定參數解析器
    parser = argparse.ArgumentParser(description="將 OVO/Replica 格式的 JSON pose 轉換為 TUM 軌跡檔 (traj.txt)")
    
    # 參數 1: 輸入資料夾 (必填)
    parser.add_argument("--data_folder", type=str, help="包含 RGB、Depth 與 JSON 檔案的資料夾路徑")
    
    # 參數 2: 輸出檔案路徑 (選填)
    # 如果使用者沒輸入，我們會在程式碼中動態設定預設值
    parser.add_argument("--output", type=str, default=None, help="輸出檔案路徑 (預設為輸入資料夾下的 traj.txt)")

    args = parser.parse_args()
    
    # 處理路徑
    data_folder = args.data_folder
    
    # 設定輸出檔案路徑邏輯
    if args.output:
        output_traj = args.output
    else:
        # 預設：在輸入資料夾內建立 traj.txt
        output_traj = os.path.join(data_folder, "traj.txt")

    # 檢查輸入資料夾是否存在
    if not os.path.exists(data_folder):
        print(f"錯誤：找不到資料夾 '{data_folder}'")
        return

    # 找到所有的 pose_*.json 並依照檔名排序
    # 使用 os.path.join 確保路徑在不同作業系統下都正確
    search_path = os.path.join(data_folder, "pose_*.json")
    json_files = sorted(glob.glob(search_path))

    if not json_files:
        print(f"警告：在 '{data_folder}' 內找不到任何 pose_*.json 檔案。")
        return

    print(f"找到 {len(json_files)} 個 JSON 檔案，正在處理中...")

    try:
        with open(output_traj, 'w') as f:
            for json_file in json_files:
                with open(json_file, 'r') as jf:
                    data = json.load(jf)
                    
                    # 1. 取得 Frame ID (時間戳記)
                    timestamp = f"{int(data['frame_id']):06d}" 
                    
                    # 2. 取得位置 (tx, ty, tz)
                    pos = data['position']
                    tx, ty, tz = pos[0], pos[1], pos[2]
                    
                    # 3. 取得旋轉四元數 (qx, qy, qz, qw)
                    quat = data['quaternion_xyzw']
                    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
                    
                    # 4. 寫入 TUM 格式: timestamp tx ty tz qx qy qz qw
                    line = f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n"
                    f.write(line)
        
        print(f"成功！軌跡檔已儲存於：{output_traj}")
        
        # 順便顯示第一個檔案的相機參數供使用者確認
        with open(json_files[0], 'r') as jf:
            first_data = json.load(jf)
            intrinsics = first_data.get('intrinsics', {})
            print(f"--- 參考相機參數 (來自第一個檔案) ---")
            print(f"fx: {intrinsics.get('fx')}")
            print(f"fy: {intrinsics.get('fy')}")
            print(f"cx: {intrinsics.get('cx')}")
            print(f"cy: {intrinsics.get('cy')}")

    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    main()