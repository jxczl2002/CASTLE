import os
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

DIR_A1 = '3D_3/condition'       
DIR_B1 = '3D_3/temp'            
MODEL_DIR = 'Models_Autoencoders_Final'   
OUTPUT_DIR = 'Flux_Training_Data_Final'   
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIM = 32
T_INIT_PHYSICAL = 293.15      
MAX_WORKERS = 4                 

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

def init_worker():
    global worker_env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    enc_K, min_K, max_K = load_ae_model('K', device)
    enc_rhoCp, min_rhoCp, max_rhoCp = load_ae_model('rhoCp', device)
    enc_Q, min_Q, max_Q = load_ae_model('Source', device)
    enc_Temp, min_T, max_T = load_ae_model('Temp', device)
    
    bd_files = glob.glob(os.path.join(DIR_A1, 'global_boundary*.npy'))
    glob_bd = np.load(bd_files[0])
    bounds = np.zeros((8, 8, 12, 12), dtype=np.float32)
    for ix in range(8):
        for iy in range(8):
            for iz in range(12):
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
        'min_Q': min_Q, 'max_Q': max_Q, 'min_T': min_T, 'max_T': max_T
    }


def extract_subs_vectorized(arr):
    """【黑科技】自动识别 3D 和 4D 数组，瞬间切分子域"""
    if arr.ndim == 3:
        return arr.reshape(8, 16, 8, 16, 12, 16).transpose(0, 2, 4, 1, 3, 5)
    elif arr.ndim == 4:
        T = arr.shape[0]
        return arr.reshape(T, 8, 16, 8, 16, 12, 16).transpose(0, 1, 3, 5, 2, 4, 6)
    else:
        raise ValueError("输入维度不支持")

def normalize_to_tensor(arr, v_min, v_max, device):
    rng = max(v_max - v_min, 1e-8)
    return torch.tensor(((arr - v_min) / rng) * 0.90 + 0.05, dtype=torch.float32, device=device).unsqueeze(1)

def process_case(case_id):
    global worker_env
    env = worker_env
    device = env['device']
    out_pt_file = os.path.join(OUTPUT_DIR, f"{case_id}_flux.pt")
    if os.path.exists(out_pt_file): return f"Skipped {case_id}"

    arr_K = np.load(os.path.join(DIR_A1, case_id, 'k_matrix.npy'))
    arr_rhoCp = np.load(os.path.join(DIR_A1, case_id, 'rhoCp_matrix.npy'))
    arr_Q = np.load(os.path.join(DIR_A1, case_id, 'Q_source.npy'))
    arr_Temp = np.load(os.path.join(DIR_B1, case_id, 'T_transient.npy')) 
    num_t = arr_Temp.shape[0]

    with torch.no_grad():
        t_K = normalize_to_tensor(extract_subs_vectorized(arr_K).reshape(-1,16,16,16), env['min_K'], env['max_K'], device)
        Z_K = env['enc_K'].forward_encoder(t_K).view(8, 8, 12, LATENT_DIM)
        
        t_rhoCp = normalize_to_tensor(extract_subs_vectorized(arr_rhoCp).reshape(-1,16,16,16), env['min_rhoCp'], env['max_rhoCp'], device)
        Z_rhoCp = env['enc_rhoCp'].forward_encoder(t_rhoCp).view(8, 8, 12, LATENT_DIM)
        
        t_Q = normalize_to_tensor(extract_subs_vectorized(arr_Q).reshape(-1,16,16,16), env['min_Q'], env['max_Q'], device)
        Z_Q = env['enc_Q'].forward_encoder(t_Q).view(8, 8, 12, LATENT_DIM)

        all_Z_Temp = []
        subs_Temp_flat = extract_subs_vectorized(arr_Temp).reshape(-1, 16, 16, 16) 
        batch_sz = 1024
        for i in range(0, subs_Temp_flat.shape[0], batch_sz):
            batch_t = normalize_to_tensor(subs_Temp_flat[i:i+batch_sz], env['min_T'], env['max_T'], device)
            all_Z_Temp.append(env['enc_Temp'].forward_encoder(batch_t))
        Z_Temp = torch.cat(all_Z_Temp, dim=0).view(num_t, 8, 8, 12, LATENT_DIM)
        
        X_dataset, Y_dataset, Step_dataset = [], [], []
        Z_pad_grid = torch.zeros((8, 8, 12, 32), dtype=torch.float32, device=device)
        Z_history = env['Z_T_init'].view(1,1,1,1,32).expand(8, 8, 12, 4, 32).clone()
        
        for t in range(num_t - 1):
            node_feat = torch.cat([Z_K, Z_rhoCp, Z_Q, Z_pad_grid, Z_history.view(8, 8, 12, 128)], dim=-1)
            padded_feat = F.pad(node_feat.permute(3, 0, 1, 2), (1, 1, 1, 1, 1, 1), value=0.0).permute(1, 2, 3, 0)
            
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbors.append(padded_feat[dx+1 : dx+9, dy+1 : dy+9, dz+1 : dz+13, :])
            neighbors_tensor = torch.cat(neighbors, dim=-1)
            
            X_t = torch.cat([neighbors_tensor, env['BOUNDARIES_TENSOR']], dim=-1)
            
            X_dataset.append(X_t.reshape(-1, 6924).cpu())
            Y_dataset.append(Z_Temp[t+1].reshape(-1, 32).cpu())
            Step_dataset.append(torch.full((768, 1), t+1, dtype=torch.float32).cpu())
            
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
    
    print(f"[*] 发现 {len(case_folders)} 个待提取案例，开启 {MAX_WORKERS} 路并发引擎进行极限加速...")
    
    with mp.Pool(processes=MAX_WORKERS, initializer=init_worker) as pool:
        for msg in tqdm(pool.imap_unordered(process_case, case_folders), total=len(case_folders)):
            pass
            
    print("\n[+] 隐空间数据集提炼完毕！")