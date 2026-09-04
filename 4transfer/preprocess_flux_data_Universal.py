import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import torch.multiprocessing as mp
import numpy as np
import gc
from tqdm import tqdm

CASE_FOLDER_NAME = '3D_3_xy' 
DIR_A1 = f'{CASE_FOLDER_NAME}/condition'      
DIR_B1 = f'{CASE_FOLDER_NAME}/temp'            
MODEL_DIR = 'Models_Autoencoders_Final'   
OUTPUT_DIR = f'Flux_Training_Data_Final_{CASE_FOLDER_NAME}' 
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIM = 32
T_INIT_PHYSICAL = 293.15      
MAX_WORKERS = 2                 

class ResBlock3D_WN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            weight_norm(nn.Conv3d(dim, dim, 3, 1, 1)), nn.LeakyReLU(0.2, True),
            weight_norm(nn.Conv3d(dim, dim, 3, 1, 1))
        )
    def forward(self, x): return x + self.block(x)

class SparseDiscreteAutoencoder3D(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            weight_norm(nn.Conv3d(1, 32, 3, 2, 1)), nn.LeakyReLU(0.2, True), ResBlock3D_WN(32),
            weight_norm(nn.Conv3d(32, 64, 3, 2, 1)), nn.LeakyReLU(0.2, True), ResBlock3D_WN(64),
            weight_norm(nn.Conv3d(64, 128, 3, 2, 1)), nn.LeakyReLU(0.2, True),
            nn.Flatten(), weight_norm(nn.Linear(128 * 8, latent_dim))
        )
    def forward_encoder(self, x): return self.encoder(x)

class SmoothAutoencoder3D_UltraTemp(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(1, 64, 3, 2, 0)), nn.LeakyReLU(0.2, True),
            nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(64, 128, 3, 2, 0)), nn.LeakyReLU(0.2, True),
            nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(128, 256, 3, 2, 0)), nn.LeakyReLU(0.2, True),
            nn.Flatten(), weight_norm(nn.Linear(256 * 8, latent_dim))
        )
    def forward_encoder(self, x): return self.encoder(x)

worker_env = {}

def load_ae_model(var_name, device):
    param_path = os.path.join(MODEL_DIR, f'{var_name}_norm_params.json')
    with open(param_path, 'r') as f: params = json.load(f)
        
    model = SmoothAutoencoder3D_UltraTemp() if var_name == 'Temp' else SparseDiscreteAutoencoder3D()
    ckpt = torch.load(os.path.join(MODEL_DIR, f'best_{var_name}_ae.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt, strict=False)
    model.to(device).eval()
    return model, params['v_min'], params['v_max']

def init_worker(nx, ny, nz, sx, sy, sz):
    global worker_env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    enc_K, min_K, max_K = load_ae_model('K', device)
    enc_rhoCp, min_rhoCp, max_rhoCp = load_ae_model('rhoCp', device)
    enc_Q, min_Q, max_Q = load_ae_model('Source', device)
    enc_Temp, min_T, max_T = load_ae_model('Temp', device)
    
    bd_files = glob.glob(os.path.join(DIR_A1, 'global_boundary*.npy'))
    if len(bd_files) == 0:
        raise FileNotFoundError(f"未在 {DIR_A1} 下发现全局边界文件 global_boundary.npy")
    glob_bd = np.load(bd_files[0])
    
    bounds = np.zeros((sx, sy, sz, 12), dtype=np.float32)
    for ix in range(sx):
        for iy in range(sy):
            for iz in range(sz):
                cx, cy, cz = ix * 16 + 8, iy * 16 + 8, iz * 16 + 8
                bounds[ix, iy, iz] = np.concatenate([
                    glob_bd[cx, cy, iz*16+15, 0, :], glob_bd[cx, cy, iz*16, 1, :],
                    glob_bd[ix*16, cy, cz, 2, :], glob_bd[ix*16+15, cy, cz, 3, :],
                    glob_bd[cx, iy*16, cz, 4, :], glob_bd[cx, iy*16+15, cz, 5, :]
                ])
    BOUNDARIES_TENSOR = torch.tensor(bounds, dtype=torch.float32, device=device)
    
    T_init_arr = np.full((1, 16, 16, 16), T_INIT_PHYSICAL, dtype=np.float32)
    T_init_tensor = torch.tensor(((T_init_arr - min_T) / max(max_T - min_T, 1e-8)) * 0.90 + 0.05, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        Z_T_init = enc_Temp.forward_encoder(T_init_tensor).squeeze()

    worker_env = {
        'device': device, 'BOUNDARIES_TENSOR': BOUNDARIES_TENSOR, 'Z_T_init': Z_T_init,
        'enc_K': enc_K, 'enc_rhoCp': enc_rhoCp, 'enc_Q': enc_Q, 'enc_Temp': enc_Temp,
        'min_K': min_K, 'max_K': max_K, 'min_rhoCp': min_rhoCp, 'max_rhoCp': max_rhoCp,
        'min_Q': min_Q, 'max_Q': max_Q, 'min_T': min_T, 'max_T': max_T,
        'nx': nx, 'ny': ny, 'nz': nz, 'sx': sx, 'sy': sy, 'sz': sz
    }

def extract_subs_vectorized(arr, sx, sy, sz):
    """自适应切分"""
    if arr.ndim == 3:
        return arr.reshape(sx, 16, sy, 16, sz, 16).transpose(0, 2, 4, 1, 3, 5)
    elif arr.ndim == 4:
        T = arr.shape[0]
        return arr.reshape(T, sx, 16, sy, 16, sz, 16).transpose(0, 1, 3, 5, 2, 4, 6)

def normalize_to_tensor(arr, v_min, v_max, device):
    rng = max(v_max - v_min, 1e-8)
    return torch.tensor(((arr - v_min) / rng) * 0.90 + 0.05, dtype=torch.float32, device=device).unsqueeze(1)

def process_case(case_id):
    global worker_env
    env = worker_env
    device = env['device']
    sx, sy, sz = env['sx'], env['sy'], env['sz']
    
    out_pt_file = os.path.join(OUTPUT_DIR, f"{case_id}_flux.pt")
    if os.path.exists(out_pt_file): return f"Skipped {case_id}"

    arr_K = np.load(os.path.join(DIR_A1, case_id, 'k_matrix.npy'))
    arr_rhoCp = np.load(os.path.join(DIR_A1, case_id, 'rhoCp_matrix.npy'))
    arr_Q = np.load(os.path.join(DIR_A1, case_id, 'Q_source.npy'))
    arr_Temp = np.load(os.path.join(DIR_B1, case_id, 'T_transient.npy')) 
    num_t = arr_Temp.shape[0]

    with torch.no_grad():
        t_K = normalize_to_tensor(extract_subs_vectorized(arr_K, sx, sy, sz).reshape(-1,16,16,16), env['min_K'], env['max_K'], device)
        Z_K = env['enc_K'].forward_encoder(t_K).view(sx, sy, sz, LATENT_DIM)
        
        t_rhoCp = normalize_to_tensor(extract_subs_vectorized(arr_rhoCp, sx, sy, sz).reshape(-1,16,16,16), env['min_rhoCp'], env['max_rhoCp'], device)
        Z_rhoCp = env['enc_rhoCp'].forward_encoder(t_rhoCp).view(sx, sy, sz, LATENT_DIM)
        
        t_Q = normalize_to_tensor(extract_subs_vectorized(arr_Q, sx, sy, sz).reshape(-1,16,16,16), env['min_Q'], env['max_Q'], device)
        Z_Q = env['enc_Q'].forward_encoder(t_Q).view(sx, sy, sz, LATENT_DIM)

        all_Z_Temp = []
        subs_Temp_flat = extract_subs_vectorized(arr_Temp, sx, sy, sz).reshape(-1, 16, 16, 16) 
        batch_sz = 1024
        for i in range(0, subs_Temp_flat.shape[0], batch_sz):
            batch_t = normalize_to_tensor(subs_Temp_flat[i:i+batch_sz], env['min_T'], env['max_T'], device)
            all_Z_Temp.append(env['enc_Temp'].forward_encoder(batch_t))
        Z_Temp = torch.cat(all_Z_Temp, dim=0).view(num_t, sx, sy, sz, LATENT_DIM)
        X_dataset, Y_dataset, Step_dataset = [], [], []
        Z_pad_grid = torch.zeros((sx, sy, sz, 32), dtype=torch.float32, device=device)
        Z_history = env['Z_T_init'].view(1,1,1,1,32).expand(sx, sy, sz, 4, 32).clone()
        
        for t in range(num_t - 1):
            node_feat = torch.cat([Z_K, Z_rhoCp, Z_Q, Z_pad_grid, Z_history.view(sx, sy, sz, 128)], dim=-1)
            padded_feat = F.pad(node_feat.permute(3, 0, 1, 2), (1, 1, 1, 1, 1, 1), value=0.0).permute(1, 2, 3, 0)
            
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbors.append(padded_feat[dx+1 : dx+sx+1, dy+1 : dy+sy+1, dz+1 : dz+sz+1, :])
            neighbors_tensor = torch.cat(neighbors, dim=-1)
            
            X_t = torch.cat([neighbors_tensor, env['BOUNDARIES_TENSOR']], dim=-1)
            
            X_dataset.append(X_t.reshape(-1, 6924).cpu())
            Y_dataset.append(Z_Temp[t+1].reshape(-1, 32).cpu())
            Step_dataset.append(torch.full((sx*sy*sz, 1), t+1, dtype=torch.float32).cpu())
            
            Z_history = torch.cat([Z_history[:, :, :, 1:, :], Z_Temp[t+1].unsqueeze(-2)], dim=-2)

        torch.save({
            'X': torch.cat(X_dataset, dim=0), 
            'Y': torch.cat(Y_dataset, dim=0),
            'step': torch.cat(Step_dataset, dim=0)
        }, out_pt_file)

    del arr_K, arr_rhoCp, arr_Q, arr_Temp, Z_K, Z_rhoCp, Z_Q, Z_Temp, padded_feat, X_t
    gc.collect(); torch.cuda.empty_cache()
    return f"Processed {case_id}"

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  
    
    case_folders = [f for f in os.listdir(DIR_A1) if os.path.isdir(os.path.join(DIR_A1, f))]
    if len(case_folders) == 0:
        raise ValueError(f"在 {DIR_A1} 中未发现任何子案例文件夹。")
        
    first_case_dir = os.path.join(DIR_A1, case_folders[0])
    sample_K = np.load(os.path.join(first_case_dir, 'k_matrix.npy'))
    
    nx, ny, nz = sample_K.shape
    sx, sy, sz = nx // 16, ny // 16, nz // 16
    del sample_K; gc.collect()
    
    print("\n" + "="*80)
    print(f"[*] 【自适应探针】：成功探测到当前任务尺度")
    print(f"    - 物理尺寸: {nx} x {ny} x {nz} | 子域维度: {sx} x {sy} x {sz} ({sx*sy*sz} Subdomains/Frame)")
    print(f"    - 输入路径: {DIR_A1} & {DIR_B1}")
    print(f"    - 输出路径: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    print(f"[*] 开启 {MAX_WORKERS} 路进程进行多维数据提取...")
    
    with mp.Pool(processes=MAX_WORKERS, initializer=init_worker, initargs=(nx, ny, nz, sx, sy, sz)) as pool:
        for msg in tqdm(pool.imap_unordered(process_case, case_folders), total=len(case_folders)):
            pass
            
    print("\n[+] 目标数据源隐空间预处理完毕！")