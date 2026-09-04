import os
import glob
import numpy as np
import pandas as pd
import gc

NUM_LAYERS = 3

def generate_temperatures():
    input_dir = '3D_3_xy_test/temp_txt'
    output_dir = '3D_3_xy_test/temp'
    os.makedirs(output_dir, exist_ok=True)
    
    nx, ny = 256, 256
    nz = 96 + 32 * NUM_LAYERS
    expected_rows = nx * ny * nz 
    
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))
    print(f"当前模式: {NUM_LAYERS}层堆叠, 预期行数: {expected_rows}.")
    print(f"找到 {len(txt_files)} 个温度文件...")
    
    for count, file_path in enumerate(txt_files, 1):
        case_id = os.path.basename(file_path).split('.')[0]
        case_out_dir = os.path.join(output_dir, f'case_{case_id}')
        os.makedirs(case_out_dir, exist_ok=True)
        
        print(f"[{count}/{len(txt_files)}] 读取 case_{case_id} ... ", end='', flush=True)
        
        df = pd.read_csv(file_path, sep=';', header=None)
        
        if len(df) != expected_rows:
            print(f"\n[警告] case_{case_id} 行数 {len(df)} != {expected_rows}，已跳过！")
            del df; gc.collect()
            continue
            
        df.sort_values(by=[0, 1, 2], ascending=[True, True, True], inplace=True)
        
        temps_flat = df.iloc[:, 3:].values.astype(np.float32)
        num_timesteps = temps_flat.shape[1]
        
        del df
        gc.collect()
        
        T_matrix = temps_flat.reshape((nx, ny, nz, num_timesteps))
        
        T_matrix_4D = np.transpose(T_matrix, (3, 0, 1, 2))
        
        np.save(os.path.join(case_out_dir, 'T_transient.npy'), T_matrix_4D)
        print(f"完成! Shape: {T_matrix_4D.shape}")
        
        del temps_flat, T_matrix, T_matrix_4D
        gc.collect()

    print("所有瞬态温度生成完毕！")

if __name__ == '__main__':
    generate_temperatures()