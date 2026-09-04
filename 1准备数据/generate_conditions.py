import os
import glob
import numpy as np

NUM_LAYERS = 3

def generate_conditions():
    input_dir = '3D_3/xy'
    output_dir = '3D_3/condition'
    os.makedirs(output_dir, exist_ok=True)
    
    nx, ny = 128, 128
    nz = 96 + 32 * NUM_LAYERS 
    
    z_total = 0.024 + 0.008 * NUM_LAYERS
    
    x = np.linspace(-0.05 + 0.1/128/2, 0.05 - 0.1/128/2, nx)
    y = np.linspace(-0.05 + 0.1/128/2, 0.05 - 0.1/128/2, ny)
    z = np.linspace(z_total/nz/2, z_total - z_total/nz/2, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    print(f"当前模式: {NUM_LAYERS}层堆叠. 正在生成全局边界条件...")
    BC_tensor = np.zeros((nx, ny, nz, 6, 2), dtype=np.float32)
    
    BC_tensor[:, :, -1, 0, 0] = 1.0; BC_tensor[:, :, -1, 0, 1] = 1.0
    BC_tensor[:, :, 0, 1, 0] = 1.0;  BC_tensor[:, :, 0, 1, 1] = 0.01
    BC_tensor[0,  :,  :, 2, 0] = 1.0 
    BC_tensor[-1, :,  :, 3, 0] = 1.0 
    BC_tensor[:,  0,  :, 4, 0] = 1.0   
    BC_tensor[:, -1,  :, 5, 0] = 1.0 
    
    np.save(os.path.join(output_dir, f'global_boundary_{NUM_LAYERS}layers.npy'), BC_tensor)
    
    props = {
        'Substrate':    {'k': 0.35, 'rhoCp': 1900 * 950},
        'Interposer':   {'k': 130,  'rhoCp': 2330 * 710},
        'Microbump':    {'k': 55,   'rhoCp': 7100 * 520},
        'Chiplet':      {'k': 130,  'rhoCp': 2330 * 710},
        'HeatSpreader': {'k': 398,  'rhoCp': 8960 * 385},
        'Underfill':    {'k': 0.7,  'rhoCp': 1850 * 840}
    }
    
    chip_specs = {
        'CPU': {'w': 0.03, 'h': 0.02, 'Q': 2.083333333333337e8},
        'M':   {'w': 0.03, 'h': 0.01, 'Q': 8.333333333333355e7},
        'IO1': {'w': 0.02, 'h': 0.02, 'Q': 1.25e8},
        'IO2': {'w': 0.02, 'h': 0.02, 'Q': 1.25e8}
    }
    
    mask_sub = Z <= 0.010 + 1e-6
    mask_int = (Z > 0.010 + 1e-6) & (Z <= 0.014 + 1e-6)
    
    mask_bump_list = []
    mask_chip_list =[]
    z_curr = 0.014
    for _ in range(NUM_LAYERS):
        mask_bump_list.append((Z > z_curr + 1e-6) & (Z <= z_curr + 0.004 + 1e-6))
        z_curr += 0.004
        mask_chip_list.append((Z > z_curr + 1e-6) & (Z <= z_curr + 0.004 + 1e-6))
        z_curr += 0.004
        
    mask_bump_all = np.logical_or.reduce(mask_bump_list)
    mask_chip_all = np.logical_or.reduce(mask_chip_list)
    mask_hs = Z > z_curr + 1e-6
    
    mask_int_xy = (np.abs(X) <= 0.045) & (np.abs(Y) <= 0.045)
    
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))
    print(f"找到 {len(txt_files)} 个条件文件...")
    
    for count, file_path in enumerate(txt_files, 1):
        case_id = os.path.basename(file_path).split('.')[0]
        case_out_dir = os.path.join(output_dir, f'case_{case_id}')
        os.makedirs(case_out_dir, exist_ok=True)
        
        k_matrix     = np.zeros((nx, ny, nz), dtype=np.float32)
        rhoCp_matrix = np.zeros((nx, ny, nz), dtype=np.float32)
        Q_matrix     = np.zeros((nx, ny, nz), dtype=np.float32)
        
        k_matrix[mask_sub] = props['Substrate']['k']
        rhoCp_matrix[mask_sub] = props['Substrate']['rhoCp']
        k_matrix[mask_hs] = props['HeatSpreader']['k']
        rhoCp_matrix[mask_hs] = props['HeatSpreader']['rhoCp']
        
        k_matrix[mask_int & mask_int_xy] = props['Interposer']['k']
        rhoCp_matrix[mask_int & mask_int_xy] = props['Interposer']['rhoCp']
        k_matrix[mask_int & (~mask_int_xy)] = props['Underfill']['k']
        rhoCp_matrix[mask_int & (~mask_int_xy)] = props['Underfill']['rhoCp']
        
        k_matrix[mask_bump_all] = props['Underfill']['k']
        rhoCp_matrix[mask_bump_all] = props['Underfill']['rhoCp']
        k_matrix[mask_chip_all] = props['Underfill']['k']
        rhoCp_matrix[mask_chip_all] = props['Underfill']['rhoCp']
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(';')
            if len(parts) != 3: continue
            name, cx, cy = parts[0], float(parts[1]), float(parts[2])
            spec = chip_specs.get(name)
            if not spec: continue
            
            w, h, Q_val = spec['w'], spec['h'], spec['Q']
            mask_chip_xy = (X >= cx - w/2) & (X <= cx + w/2) & (Y >= cy - h/2) & (Y <= cy + h/2)
            
            mask_this_bump = mask_bump_all & mask_chip_xy
            k_matrix[mask_this_bump] = props['Microbump']['k']
            rhoCp_matrix[mask_this_bump] = props['Microbump']['rhoCp']
            
            mask_this_chip = mask_chip_all & mask_chip_xy
            k_matrix[mask_this_chip] = props['Chiplet']['k']
            rhoCp_matrix[mask_this_chip] = props['Chiplet']['rhoCp']
            Q_matrix[mask_this_chip] = Q_val
            
        np.save(os.path.join(case_out_dir, 'k_matrix.npy'), k_matrix)
        np.save(os.path.join(case_out_dir, 'rhoCp_matrix.npy'), rhoCp_matrix)
        np.save(os.path.join(case_out_dir, 'Q_source.npy'), Q_matrix)
        
        if count % 50 == 0:
            print(f"已处理 {count} / {len(txt_files)}...")

    print("条件特征矩阵生成完毕！")

if __name__ == '__main__':
    generate_conditions()