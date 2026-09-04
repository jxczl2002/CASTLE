import os
import numpy as np
import hashlib
from collections import defaultdict
import random
import gc


SUB_SIZE = 16       
DECIMALS = 5          
VAL_RATIO = 0.2        
FOLDER_A = 'condition'      
FOLDER_B = '3D/temp'         
OUTPUT_DIR = 'Dataset_Subdomains' 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_subdomains(arr, sub_size=SUB_SIZE):
    """
    利用 numpy 步长黑科技，将 3D 大矩阵快速切割为多个 3D 小矩阵
    输入 shape: (128, 128, 64)
    输出 shape: (256, 16, 16, 16)
    """
    nx, ny, nz = arr.shape
    sx, sy, sz = nx // sub_size, ny // sub_size, nz // sub_size
    reshaped = arr.reshape(sx, sub_size, sy, sub_size, sz, sub_size)
    transposed = reshaped.transpose(0, 2, 4, 1, 3, 5)
    return transposed.reshape(-1, sub_size, sub_size, sub_size)

def get_hash(sub_arr):
    """提取子域矩阵的 MD5 哈希指纹"""
    return hashlib.md5(np.round(sub_arr, DECIMALS).tobytes()).hexdigest()

def process_datatype(data_name, base_folder, file_pattern, is_time_series=False):
    """
    独立处理某类数据，完成切分、去重、公共提取、划分集合
    """
    print(f"\n================ 正在处理: {data_name} ================")
    
    subdomain_file_map = defaultdict(set) 
    hash_to_array = {}                    

    case_folders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    total_files = 0

    for case_dir in case_folders:
        file_path = os.path.join(case_dir, file_pattern)
        if not os.path.exists(file_path):
            continue
            
        case_id = os.path.basename(case_dir)
        total_files += 1
        
        data = np.load(file_path) 
        
        if is_time_series:
            num_t = data.shape[0]
            for t in range(num_t):
                subs = extract_subdomains(data[t])
                for sub in subs:
                    h = get_hash(sub)
                    subdomain_file_map[h].add(case_id)  
                    if h not in hash_to_array:
                        hash_to_array[h] = sub         
        else:
            subs = extract_subdomains(data)
            for sub in subs:
                h = get_hash(sub)
                subdomain_file_map[h].add(case_id)
                if h not in hash_to_array:
                    hash_to_array[h] = sub

    if total_files == 0:
        print(f"未找到对应文件 {file_pattern}，跳过该类型。")
        return

    print(f"[{data_name}] 扫描文件数: {total_files}")
    print(f"[{data_name}] 全局去重后的独立子域总数: {len(hash_to_array)}")

    common_hashes = []
    other_hashes = []

    for h, file_set in subdomain_file_map.items():
        if len(file_set) == total_files:
            common_hashes.append(h)
        else:
            other_hashes.append(h)

    print(f"[{data_name}] 提取在所有文件中均存在的公共子域数: {len(common_hashes)}")
    
    random.seed(42)  
    random.shuffle(other_hashes)
    
    split_idx = int(len(other_hashes) * (1.0 - VAL_RATIO))
    train_hashes = other_hashes[:split_idx]
    val_hashes = other_hashes[split_idx:]
    
    train_hashes.extend(common_hashes)

    print(f"[{data_name}] 最终训练集子域数: {len(train_hashes)} (含公共子域)")
    print(f"[{data_name}] 最终验证集子域数: {len(val_hashes)}")

    print(f"[{data_name}] 正在打包数据并写入磁盘 (不要分开保存)...")
    
    if common_hashes:
        np.save(os.path.join(OUTPUT_DIR, f'common_{data_name}.npy'), 
                np.array([hash_to_array[h] for h in common_hashes], dtype=np.float32))
    if train_hashes:
        np.save(os.path.join(OUTPUT_DIR, f'train_{data_name}.npy'), 
                np.array([hash_to_array[h] for h in train_hashes], dtype=np.float32))
    if val_hashes:
        np.save(os.path.join(OUTPUT_DIR, f'val_{data_name}.npy'), 
                np.array([hash_to_array[h] for h in val_hashes], dtype=np.float32))

    del hash_to_array, subdomain_file_map, common_hashes, train_hashes, val_hashes
    gc.collect()


if __name__ == '__main__':
    process_datatype('K', FOLDER_A, 'k_matrix.npy', is_time_series=False)
    
    process_datatype('rhoCp', FOLDER_A, 'rhoCp_matrix.npy', is_time_series=False)
    
    process_datatype('Source', FOLDER_A, 'Q_source.npy', is_time_series=False)
    
    process_datatype('Temp', FOLDER_B, 'T_transient.npy', is_time_series=True)
    
    print("\n所有预处理任务已完成！输出文件位于", OUTPUT_DIR)