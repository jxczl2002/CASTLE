import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm

AE_MODEL_DIR = 'Models_Autoencoders_Final'            
MPNN_MODEL_PATH = 'Models_Flux_N1/best_N1_mpnn.pth'   
OUTPUT_RL_DIR = 'Models_Seq2Seq_RL_Results_1'          

NUM_CYCLES = 100                    
BATCH_SIZE = 512                     
LEARNING_RATE = 1e-3               
HIDDEN_DIM = 128                  
LATENT_DIM = 32                      
PATIENCE = 20                         
NUM_LAYERS = 3                       
T_INIT_PHYSICAL = 293.15             

SIM_BATCH_SIZE = 32                 
EVAL_BATCH_SIZE = 512               
MPNN_BATCH_SIZE = 1024             

os.makedirs(OUTPUT_RL_DIR, exist_ok=True)

grid_nodes_x = np.linspace(-0.05 + 0.1/128/2, 0.05 - 0.1/128/2, 128)
grid_nodes_y = np.linspace(-0.05 + 0.1/128/2, 0.05 - 0.1/128/2, 128)

CHIP_SPECS = [
    {'name': 'CPU', 'w': 0.03, 'h': 0.02, 'Q': 2.083333333333337e8},
    {'name': 'M',   'w': 0.03, 'h': 0.01, 'Q': 8.333333333333355e7},
    {'name': 'IO1', 'w': 0.02, 'h': 0.02, 'Q': 1.25e8},
    {'name': 'IO2', 'w': 0.02, 'h': 0.02, 'Q': 1.25e8}
]

props = {
    'Substrate':    {'k': 0.35, 'rhoCp': 1900 * 950},
    'Interposer':   {'k': 130,  'rhoCp': 2330 * 710},
    'Microbump':    {'k': 55,   'rhoCp': 7100 * 520},
    'Chiplet':      {'k': 130,  'rhoCp': 2330 * 710},
    'HeatSpreader': {'k': 398,  'rhoCp': 8960 * 385},
    'Underfill':    {'k': 0.7,  'rhoCp': 1850 * 840}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.decoder_fc = nn.Sequential(weight_norm(nn.Linear(latent_dim, 128 * 8)), nn.LeakyReLU(0.2, True))
        self.decoder_conv = nn.Sequential(
            weight_norm(nn.ConvTranspose3d(128, 64, 4, 2, 1)), nn.LeakyReLU(0.2, True), ResBlock3D_WN(64),
            weight_norm(nn.ConvTranspose3d(64, 32, 4, 2, 1)), nn.LeakyReLU(0.2, True), ResBlock3D_WN(32),
            weight_norm(nn.ConvTranspose3d(32, 16, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            weight_norm(nn.Conv3d(16, 1, 3, 1, 1)) 
        )
    def forward_encoder(self, x): return self.encoder(x)
    def forward_decoder(self, z): return self.decoder_conv(self.decoder_fc(z).view(-1, 128, 2, 2, 2))
    def forward(self, x): return self.forward_decoder(self.forward_encoder(x))

class SmoothAutoencoder3D_UltraTemp(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(1, 64, 3, 2, 0)), nn.LeakyReLU(0.2, True),
            nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(64, 128, 3, 2, 0)), nn.LeakyReLU(0.2, True),
            nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(128, 256, 3, 2, 0)), nn.LeakyReLU(0.2, True),
            nn.Flatten(), weight_norm(nn.Linear(256 * 8, latent_dim))
        )
        self.decoder_fc = nn.Sequential(weight_norm(nn.Linear(latent_dim, 256 * 8)), nn.LeakyReLU(0.2, True))
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(256, 128, 3, 1, 0)), nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(128, 64, 3, 1, 0)), nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(64, 16, 3, 1, 0)), nn.LeakyReLU(0.2, True),
            nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(16, 1, 3, 1, 0)) 
        )
    def forward_encoder(self, x): return self.encoder(x) 
    def forward_decoder(self, z): return self.decoder_conv(self.decoder_fc(z).view(-1, 256, 2, 2, 2))
    def forward(self, x): return self.forward_decoder(self.forward_encoder(x))

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), 
            nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.gelu = nn.GELU()
    def forward(self, x): return self.gelu(x + self.net(x))

class N1_FiLM_MPNN(nn.Module):
    def __init__(self):
        super(N1_FiLM_MPNN, self).__init__()
        self.stream_material_base = nn.Sequential(
            nn.Conv3d(76, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.Conv3d(256, 512, 3, padding=1), nn.GroupNorm(32, 512), nn.GELU()
        )
        self.mat_gamma = nn.Conv3d(512, 1024, 3, padding=0)
        self.mat_beta = nn.Conv3d(512, 1024, 3, padding=0)
        self.stream_energy = nn.Sequential(
            nn.Conv3d(160, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.Conv3d(256, 512, 3, padding=1), nn.GroupNorm(32, 512), nn.GELU(),
            nn.Conv3d(512, 1024, 3, padding=0), nn.GroupNorm(32, 1024), nn.GELU()
        )
        self.node_proj = nn.Sequential(nn.Linear(1056, 1024), nn.LayerNorm(1024), nn.GELU())
        self.node_res = nn.Sequential(ResBlock(1024), ResBlock(1024), ResBlock(1024))
        self.node_out = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 32))
        nn.init.normal_(self.node_out[-1].weight, mean=0.0, std=1e-5)
        nn.init.constant_(self.node_out[-1].bias, 0.0)

    def forward(self, x):
        spatial, boundary = x[:, :6912].reshape(-1, 27, 256), x[:, 6912:] 
        clean_z_curr = spatial[:, 13, -32:].clone() 
        
        mat_grid = torch.cat([spatial[:,:,0:64], boundary.unsqueeze(1).expand(-1,27,-1)], dim=-1).permute(0,2,1).contiguous().reshape(-1, 76, 3, 3, 3)
        eng_grid = torch.cat([spatial[:,:,64:96], spatial[:,:,128:256]], dim=-1).permute(0,2,1).contiguous().reshape(-1, 160, 3, 3, 3)
        
        mat_base = self.stream_material_base(mat_grid)
        gated_flux = self.mat_gamma(mat_base).reshape(-1, 1024) * self.stream_energy(eng_grid).reshape(-1, 1024) + self.mat_beta(mat_base).reshape(-1, 1024)
        
        node_inputs = torch.cat([gated_flux, clean_z_curr], dim=-1) 
        delta_z = self.node_out(self.node_res(self.node_proj(node_inputs)))
        return clean_z_curr + delta_z

def extract_subs_vectorized(arr):
    return arr.reshape(8, 16, 8, 16, 12, 16).transpose(0, 2, 4, 1, 3, 5).reshape(-1, 16, 16, 16)

def reassemble_subs_vectorized(subs_flat):
    return subs_flat.reshape(8, 8, 12, 16, 16, 16).transpose(0, 3, 1, 4, 2, 5).reshape(128, 128, 192)

def normalize_t(arr, vmin, vrange):
    return torch.tensor(((arr - vmin) / vrange) * 0.90 + 0.05, dtype=torch.float32, device=device).unsqueeze(1)

def generate_dynamic_boundaries(device):
    bounds = np.zeros((8, 8, 12, 12), dtype=np.float32)
    for ix in range(8):
        for iy in range(8):
            for iz in range(12):
                bd_top = [1.0, 1.0] if iz == 11 else [0.0, 0.0]
                bd_bot = [1.0, 0.01] if iz == 0 else [0.0, 0.0]
                bd_left = [1.0, 0.0] if ix == 0 else [0.0, 0.0]
                bd_right = [1.0, 0.0] if ix == 7 else [0.0, 0.0]
                bd_front = [1.0, 0.0] if iy == 0 else [0.0, 0.0]
                bd_back = [1.0, 0.0] if iy == 7 else [0.0, 0.0]
                bounds[ix, iy, iz] = np.concatenate([bd_top, bd_bot, bd_left, bd_right, bd_front, bd_back])
    return torch.tensor(bounds, dtype=torch.float32, device=device)

def load_ae_model(var_name):
    param_path = os.path.join(AE_MODEL_DIR, f'{var_name}_norm_params.json')
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"未找到 {var_name} 的归一化参数文件: {param_path}")
        
    with open(param_path, 'r') as f: params = json.load(f)
    model = SmoothAutoencoder3D_UltraTemp(latent_dim=LATENT_DIM) if var_name == 'Temp' else SparseDiscreteAutoencoder3D(latent_dim=LATENT_DIM)
    ckpt_path = os.path.join(AE_MODEL_DIR, f'best_{var_name}_ae.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad = False
    return model, params['v_min'], params['v_max'] - params['v_min']

def get_action_mask(placed_list, step):
    mask = np.ones((128, 128), dtype=np.float32)
    spec = CHIP_SPECS[step]
    w_new, h_new = spec['w'], spec['h']
    
    x_min_bound = -0.045 + w_new / 2
    x_max_bound = 0.045 - w_new / 2
    y_min_bound = -0.045 + h_new / 2
    y_max_bound = 0.045 - h_new / 2
    
    mask[grid_nodes_x < x_min_bound, :] = 0.0
    mask[grid_nodes_x > x_max_bound, :] = 0.0
    mask[:, grid_nodes_y < y_min_bound] = 0.0
    mask[:, grid_nodes_y > y_max_bound] = 0.0
    
    for cx_oth, cy_oth, w_oth, h_oth in placed_list:
        min_dx = (w_new + w_oth) / 2 + 0.001
        min_dy = (h_new + h_oth) / 2 + 0.001
        
        dist_grid_x = np.abs(grid_nodes_x[:, None] - cx_oth)
        dist_grid_y = np.abs(grid_nodes_y[None, :] - cy_oth)
        
        collision_mask = (dist_grid_x < min_dx) & (dist_grid_y < min_dy)
        mask[collision_mask] = 0.0
        
    return torch.tensor(mask.flatten(), dtype=torch.bool)

class GPU_Thermal_Environment:
    def __init__(self):
        self.enc_K, self.min_K, self.range_K = load_ae_model('K')
        self.enc_rhoCp, self.min_rhoCp, self.range_rhoCp = load_ae_model('rhoCp')
        self.enc_Q, self.min_Q, self.range_Q = load_ae_model('Source')
        self.enc_Temp, self.min_T, self.range_T = load_ae_model('Temp')
        
        self.mpnn = N1_FiLM_MPNN().to(device)
        self.mpnn.load_state_dict(torch.load(MPNN_MODEL_PATH, map_location=device)['model_state'])
        self.mpnn.eval()
        
        self.bd_tensor = generate_dynamic_boundaries(device)
        
        x = np.linspace(-0.05 + 0.1/128/2, 0.05 - 0.1/128/2, 128)
        y = np.linspace(-0.05 + 0.1/128/2, 0.05 - 0.1/128/2, 128)
        z = np.linspace(0.032/128/2, 0.032 - 0.032/128/2, 192)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        self.mask_sub = Z <= 0.010 + 1e-6
        self.mask_int = (Z > 0.010 + 1e-6) & (Z <= 0.014 + 1e-6)
        self.mask_bump = (Z > 0.014 + 1e-6) & (Z <= 0.018 + 1e-6)
        self.mask_chip = (Z > 0.018 + 1e-6) & (Z <= 0.022 + 1e-6)
        self.mask_hs = Z > 0.022 + 1e-6
        self.mask_int_xy = (np.abs(X) <= 0.045) & (np.abs(Y) <= 0.045)
        self.X, self.Y, self.Z = X, Y, Z
        
        T_init_arr = np.full((1, 16, 16, 16), T_INIT_PHYSICAL, dtype=np.float32)
        T_init_tensor = normalize_t(T_init_arr, self.min_T, self.range_T)
        with torch.no_grad():
            self.Z_T_init = self.enc_Temp.forward_encoder(T_init_tensor).squeeze()

    def step_batch_simulation(self, placed_batches):
        """【双层流式显存控制】：外层对方案级进行切分，内层对节点级进行切分，保证显存永远不超过3G"""
        B_total = len(placed_batches)
        all_T_pred_3D = []
        
        for sim_idx in range(0, B_total, SIM_BATCH_SIZE):
            batch_layouts = placed_batches[sim_idx : sim_idx + SIM_BATCH_SIZE]
            B = len(batch_layouts)
            
            k_flat = np.zeros((B * 768, 16, 16, 16), dtype=np.float32)
            rhoCp_flat = np.zeros((B * 768, 16, 16, 16), dtype=np.float32)
            Q_flat = np.zeros((B * 768, 16, 16, 16), dtype=np.float32)
            
            for b in range(B):
                k_case = np.zeros((128, 128, 192), dtype=np.float32)
                rhoCp_case = np.zeros((128, 128, 192), dtype=np.float32)
                Q_case = np.zeros((128, 128, 192), dtype=np.float32)
                
                k_case[self.mask_sub] = props['Substrate']['k']
                rhoCp_case[self.mask_sub] = props['Substrate']['rhoCp']
                k_case[self.mask_hs] = props['HeatSpreader']['k']
                rhoCp_case[self.mask_hs] = props['HeatSpreader']['rhoCp']
                k_case[self.mask_int & self.mask_int_xy] = props['Interposer']['k']
                rhoCp_case[self.mask_int & self.mask_int_xy] = props['Interposer']['rhoCp']
                k_case[self.mask_int & (~self.mask_int_xy)] = props['Underfill']['k']
                rhoCp_case[self.mask_int & (~self.mask_int_xy)] = props['Underfill']['rhoCp']
                
                k_case[self.mask_bump] = props['Underfill']['k']
                rhoCp_case[self.mask_bump] = props['Underfill']['rhoCp']
                k_case[self.mask_chip] = props['Underfill']['k']
                rhoCp_case[self.mask_chip] = props['Underfill']['rhoCp']
                
                for step, (cx, cy, w, h) in enumerate(batch_layouts[b]):
                    mask_xy = (self.X >= cx - w/2) & (self.X <= cx + w/2) & (self.Y >= cy - h/2) & (self.Y <= cy + h/2)
                    k_case[self.mask_bump & mask_xy] = props['Microbump']['k']
                    rhoCp_case[self.mask_bump & mask_xy] = props['Microbump']['rhoCp']
                    k_case[self.mask_chip & mask_xy] = props['Chiplet']['k']
                    rhoCp_case[self.mask_chip & mask_xy] = props['Chiplet']['rhoCp']
                    Q_case[self.mask_chip & mask_xy] = CHIP_SPECS[step]['Q']
                    
                k_flat[b*768 : (b+1)*768] = extract_subs_vectorized(k_case)
                rhoCp_flat[b*768 : (b+1)*768] = extract_subs_vectorized(rhoCp_case)
                Q_flat[b*768 : (b+1)*768] = extract_subs_vectorized(Q_case)

            with torch.no_grad():
                total_nodes = B * 768
                
                all_Z_K = []
                for i in range(0, total_nodes, EVAL_BATCH_SIZE):
                    t_K_b = normalize_t(k_flat[i : i+EVAL_BATCH_SIZE], self.min_K, self.range_K)
                    all_Z_K.append(self.enc_K.forward_encoder(t_K_b))
                Z_K = torch.cat(all_Z_K, dim=0).view(B, 8, 8, 12, LATENT_DIM)
                
                all_Z_rhoCp = []
                for i in range(0, total_nodes, EVAL_BATCH_SIZE):
                    t_rho_b = normalize_t(rhoCp_flat[i : i+EVAL_BATCH_SIZE], self.min_rhoCp, self.range_rhoCp)
                    all_Z_rhoCp.append(self.enc_rhoCp.forward_encoder(t_rho_b))
                Z_rhoCp = torch.cat(all_Z_rhoCp, dim=0).view(B, 8, 8, 12, LATENT_DIM)
                
                all_Z_Q = []
                for i in range(0, total_nodes, EVAL_BATCH_SIZE):
                    t_Q_b = normalize_t(Q_flat[i : i+EVAL_BATCH_SIZE], self.min_Q, self.range_Q)
                    all_Z_Q.append(self.enc_Q.forward_encoder(t_Q_b))
                Z_Q = torch.cat(all_Z_Q, dim=0).view(B, 8, 8, 12, LATENT_DIM)
                
                Z_pad_grid = torch.zeros((B, 8, 8, 12, 32), dtype=torch.float32, device=device)
                Z_history = self.Z_T_init.view(1,1,1,1,32).expand(B, 8, 8, 12, 4, 32).clone()
                
                for t in range(200):
                    node_feat = torch.cat([Z_K, Z_rhoCp, Z_Q, Z_pad_grid, Z_history.view(B, 8, 8, 12, 128)], dim=-1)
                    padded_feat = F.pad(node_feat.permute(0, 4, 1, 2, 3), (1, 1, 1, 1, 1, 1), value=0.0).permute(0, 2, 3, 4, 1)
                    
                    neighbors = []
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                neighbors.append(padded_feat[:, dx+1 : dx+9, dy+1 : dy+9, dz+1 : dz+13, :])
                    neighbors_tensor = torch.cat(neighbors, dim=-1)
                    
                    X_t = torch.cat([neighbors_tensor, self.bd_tensor.unsqueeze(0).expand(B, -1, -1, -1, -1)], dim=-1).reshape(B * 768, 6924)
                    
                    Z_pred_flat_list = []
                    for i in range(0, X_t.size(0), MPNN_BATCH_SIZE):
                        X_batch = X_t[i : i+MPNN_BATCH_SIZE]
                        Z_pred_flat_list.append(self.mpnn(X_batch))
                    Z_pred_flat = torch.cat(Z_pred_flat_list, dim=0)
                    
                    Z_pred_grid = Z_pred_flat.view(B, 8, 8, 12, LATENT_DIM)
                    Z_history = torch.cat([Z_history[:, :, :, :, 1:, :], Z_pred_grid.unsqueeze(-2)], dim=-2)

                T_pred_norm_list = []
                for i in range(0, Z_pred_flat.size(0), EVAL_BATCH_SIZE):
                    batch_z = Z_pred_flat[i : i+EVAL_BATCH_SIZE]
                    t_p = self.enc_Temp.forward_decoder(batch_z).squeeze(1).cpu().numpy()
                    T_pred_norm_list.append(t_p)
                T_pred_norm = np.concatenate(T_pred_norm_list, axis=0)
                
                T_pred_phys = (T_pred_norm - 0.05) / 0.90 * self.range_T + self.min_T
                T_pred_3D_all_batch = T_pred_phys.reshape(B, 8, 8, 12, 16, 16, 16).transpose(0, 1, 4, 2, 5, 3, 6).reshape(B, 128, 128, 192)
                
                all_T_pred_3D.append(T_pred_3D_all_batch)
                
                del Z_K, Z_rhoCp, Z_Q, Z_pad_grid, Z_history, neighbors_tensor, X_t, Z_pred_flat, Z_pred_grid, T_pred_norm
                del all_Z_K, all_Z_rhoCp, all_Z_Q, Z_pred_flat_list, T_pred_norm_list
                gc.collect(); torch.cuda.empty_cache()

        return np.concatenate(all_T_pred_3D, axis=0)

def non_dominated_sorting(objectives):
    N = objectives.shape[0]
    domination_set = [[] for _ in range(N)]
    dominated_count = np.zeros(N, dtype=np.int32)
    fronts = [[]]
    
    for p in range(N):
        for q in range(N):
            if np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q]):
                domination_set[p].append(q)
            elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)
            
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in domination_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
        
    return fronts[:-1]

def calculate_crowding_distance(objectives, front):
    size = len(front)
    if size == 0: return []
    if size <= 2: return [1.0] * size
    
    distances = np.zeros(size)
    for m in range(3):
        sorted_indices = np.argsort(objectives[front, m])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        obj_range = objectives[front[sorted_indices[-1]], m] - objectives[front[sorted_indices[0]], m]
        if obj_range == 0: obj_range = 1e-8
        
        for i in range(1, size - 1):
            distances[sorted_indices[i]] += (objectives[front[sorted_indices[i+1]], m] - objectives[front[sorted_indices[i-1]], m]) / obj_range
            
    finite_dists = distances[np.isfinite(distances)]
    max_finite = np.max(finite_dists) if len(finite_dists) > 0 else 1.0
    for i in range(size):
        if np.isinf(distances[i]):
            distances[i] = 1.0
        else:
            distances[i] = distances[i] / max_finite
    return distances

def get_pareto_rewards(objectives):
    N = objectives.shape[0]
    fronts = non_dominated_sorting(objectives)
    rewards = np.zeros(N)
    
    for f_idx, front in enumerate(fronts):
        base_reward = 1.0 / (f_idx + 1)
        crowding_dists = calculate_crowding_distance(objectives, front)
        for i, idx in enumerate(front):
            rewards[idx] = base_reward + 0.1 * crowding_dists[i]
            
    return rewards

def calculate_2d_hv(points_2d, ref_2d):
    pts = sorted([p for p in points_2d if p[0] <= ref_2d[0] and p[1] <= ref_2d[1]], key=lambda x: x[0])
    if not pts: return 0.0
    ux = sorted(list(set([p[0] for p in pts] + [ref_2d[0]])))
    area = 0.0
    for i in range(len(ux)-1):
        x_start, x_end = ux[i], ux[i+1]
        active = [p for p in pts if p[0] <= x_start]
        if active:
            min_y = min([p[1] for p in active])
            if min_y < ref_2d[1]:
                area += (x_end - x_start) * (ref_2d[1] - min_y)
    return area

def calculate_3d_hv(points_3d, ref_3d):
    valid_pts = [p for p in points_3d if p[0] <= ref_3d[0] and p[1] <= ref_3d[1] and p[2] <= ref_3d[2]]
    if not valid_pts: return 0.0
    uz = sorted(list(set([p[2] for p in valid_pts] + [ref_3d[2]])))
    volume = 0.0
    for i in range(len(uz)-1):
        z_start, z_end = uz[i], uz[i+1]
        active = [p for p in valid_pts if p[2] <= z_start]
        if active:
            pts_2d = [(p[0], p[1]) for p in active]
            area_2d = calculate_2d_hv(pts_2d, (ref_3d[0], ref_3d[1]))
            volume += area_2d * (z_end - z_start)
    return volume

class Seq2SeqPlacementPolicy(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.chip_emb = nn.Embedding(4, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.coord_emb = nn.Linear(2, hidden_dim)
        self.decoder_cell = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 16384) 
        
    def forward(self, batch_size):
        device = next(self.parameters()).device
        chip_ids = torch.arange(4, device=device).unsqueeze(0).expand(batch_size, -1)
        chip_embedded = self.chip_emb(chip_ids) 
        _, h_state = self.encoder(chip_embedded)
        h_state = h_state.squeeze(0)
        
        prev_coords = torch.zeros(batch_size, 2, device=device)
        actions, log_probs = [], []
        placed_batches = [[] for _ in range(batch_size)]
        
        for step in range(4):
            prev_coord_emb = self.coord_emb(prev_coords)
            dec_in = torch.cat([prev_coord_emb, chip_embedded[:, step]], dim=-1)
            h_state = self.decoder_cell(dec_in, h_state)
            logits = self.output_layer(h_state)
            
            masks = []
            for b in range(batch_size):
                masks.append(get_action_mask(placed_batches[b], step))
            masks_tensor = torch.stack(masks).to(device)
            
            masked_logits = logits.masked_fill(~masks_tensor, -1e9)
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            
            next_coords_list = []
            for b in range(batch_size):
                act_idx = action[b].item()
                ix, iy = act_idx // 128, act_idx % 128
                cx, cy = grid_nodes_x[ix], grid_nodes_y[iy]
                placed_batches[b].append((cx, cy, CHIP_SPECS[step]['w'], CHIP_SPECS[step]['h']))
                next_coords_list.append([cx, cy])
            
            prev_coords = torch.tensor(next_coords_list, dtype=torch.float32, device=device)
                
        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1), placed_batches

    def forward_deterministic(self):
        device = next(self.parameters()).device
        chip_ids = torch.arange(4, device=device).unsqueeze(0) 
        chip_embedded = self.chip_emb(chip_ids) 
        _, h_state = self.encoder(chip_embedded)
        h_state = h_state.squeeze(0)
        
        prev_coords = torch.zeros(1, 2, device=device)
        placed = []
        
        for step in range(4):
            prev_coord_emb = self.coord_emb(prev_coords)
            dec_in = torch.cat([prev_coord_emb, chip_embedded[:, step]], dim=-1)
            h_state = self.decoder_cell(dec_in, h_state)
            logits = self.output_layer(h_state)
            
            mask = get_action_mask(placed, step).to(device)
            masked_logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
            
            action = torch.argmax(masked_logits, dim=-1) 
            
            act_idx = action.item()
            ix, iy = act_idx // 128, act_idx % 128
            cx, cy = grid_nodes_x[ix], grid_nodes_y[iy]
            placed.append((cx, cy, CHIP_SPECS[step]['w'], CHIP_SPECS[step]['h']))
            prev_coords[0] = torch.tensor([cx, cy], device=device)
            
        return placed

def run_reinforcement_learning():
    env = GPU_Thermal_Environment()
    policy = Seq2SeqPlacementPolicy(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=PATIENCE, min_lr=1e-6)
    
    global_mins = np.array([float('inf'), float('inf'), float('inf')])
    global_maxs = np.array([float('-inf'), float('-inf'), float('-inf')])
    is_frozen = False
    best_hv = float('-inf')
    
    global_archive_objs = []      
    global_archive_layouts = []   
    
    print("\n" + "="*80)
    print(f"[*] 强化学习引擎启动！单次循环平行尝试: {BATCH_SIZE} 次 | 最大训练循环数: {NUM_CYCLES}")
    print(f"[*] 学习率自适应衰减耐心值 (Patience): {PATIENCE} 轮")
    print(f"[*] 模型与日志保存文件夹: {OUTPUT_RL_DIR}")
    print("="*80 + "\n")
    
    pareto_csv_path = os.path.join(OUTPUT_RL_DIR, 'pareto_set_history.csv')
    hv_csv_path = os.path.join(OUTPUT_RL_DIR, 'normalized_hv_history.csv')
    ref_csv_path = os.path.join(OUTPUT_RL_DIR, 'reference_point_metrics.csv')
    archive_csv_path = os.path.join(OUTPUT_RL_DIR, 'global_pareto_archive.csv') 
    
    with open(pareto_csv_path, 'w', encoding='utf-8') as f:
        f.write("Cycle,Pareto_Set_Count,Element_Index,CPU_x(m),CPU_y(m),M_x(m),M_y(m),IO1_x(m),IO1_y(m),IO2_x(m),IO2_y(m),Max_Temp(K),Mean_Temp(K),Wirelength(mm)\n")
    with open(hv_csv_path, 'w', encoding='utf-8') as f:
        f.write("Cycle,Normalized_HV\n")
    with open(archive_csv_path, 'w', encoding='utf-8') as f:
        f.write("Archive_Cycle,Pareto_Set_Count,Element_Index,CPU_x(m),CPU_y(m),M_x(m),M_y(m),IO1_x(m),IO1_y(m),IO2_x(m),IO2_y(m),Max_Temp(K),Mean_Temp(K),Wirelength(mm)\n")
        
    for cycle in range(1, NUM_CYCLES + 1):
        current_lr = optimizer.param_groups[0]['lr'] 
        
        actions, log_probs, placed_batches = policy(batch_size=BATCH_SIZE)
        T_pred_3D_all = env.step_batch_simulation(placed_batches)
        
        objs = np.zeros((BATCH_SIZE, 3))
        for b in range(BATCH_SIZE):
            T_field = T_pred_3D_all[b]
            max_t = np.max(T_field)
            mean_t = np.mean(T_field)
            
            coords = np.array([placed_batches[b][k][:2] for k in range(4)])
            dist_sum = 0.0
            for i in range(4):
                for j in range(i+1, 4):
                    dist_sum += np.linalg.norm(coords[i] - coords[j])
                    
            objs[b] = [max_t, mean_t, dist_sum]
            
        if not is_frozen:
            batch_mins = np.min(objs, axis=0)
            batch_maxs = np.max(objs, axis=0)
            global_mins = np.minimum(global_mins, batch_mins)
            global_maxs = np.maximum(global_maxs, batch_maxs)
            
            if cycle == 10:
                is_frozen = True
                ref_absolute = global_maxs * 1.1
                with open(ref_csv_path, 'w', encoding='utf-8') as f:
                    f.write("Max_Temp_Ref(K),Mean_Temp_Ref(K),Wirelength_Ref(mm)\n")
                    f.write(f"{ref_absolute[0]:.4f},{ref_absolute[1]:.4f},{ref_absolute[2]*1000.0:.4f}\n")
                print(f"\n[✓] 锁定参考点并输出！已写入: {ref_csv_path}\n  - Max Temp 限: {ref_absolute[0]:.2f} K\n  - Mean Temp 限: {ref_absolute[1]:.2f} K\n  - Wirelength 限: {ref_absolute[2]*1000.0:.1f} mm\n")

        norm_ref = np.array([1.1, 1.1, 1.1]) 
        if is_frozen:
            norm_objs = (objs - global_mins) / (global_maxs - global_mins + 1e-8)
        else:
            norm_objs = (objs - np.min(objs, axis=0)) / (np.max(objs, axis=0) - np.min(objs, axis=0) + 1e-8)

        rewards = get_pareto_rewards(norm_objs)
        fronts = non_dominated_sorting(norm_objs)
        front_0_points = norm_objs[fronts[0]]
        
        hv_val = calculate_3d_hv(front_0_points, norm_ref)
        with open(hv_csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{cycle},{hv_val:.6f}\n")
            
        with open(pareto_csv_path, 'a', encoding='utf-8') as f:
            for i, idx in enumerate(fronts[0]):
                layout = placed_batches[idx]
                f.write(f"{cycle},{len(fronts[0])},{i},"
                        f"{layout[0][0]:.6f},{layout[0][1]:.6f}," 
                        f"{layout[1][0]:.6f},{layout[1][1]:.6f}," 
                        f"{layout[2][0]:.6f},{layout[2][1]:.6f}," 
                        f"{layout[3][0]:.6f},{layout[3][1]:.6f}," 
                        f"{objs[idx,0]:.4f},{objs[idx,1]:.4f},{objs[idx,2]*1000.0:.4f}\n")

        current_front_0_objs = objs[fronts[0]]
        current_front_0_layouts = [placed_batches[idx] for idx in fronts[0]]
        
        merged_objs = list(global_archive_objs) + list(current_front_0_objs)
        merged_layouts = list(global_archive_layouts) + list(current_front_0_layouts)
        
        if len(merged_objs) > 0:
            merged_objs_np = np.array(merged_objs)
            if is_frozen:
                norm_merged = (merged_objs_np - global_mins) / (global_maxs - global_mins + 1e-8)
            else:
                norm_merged = (merged_objs_np - np.min(merged_objs_np, axis=0)) / (np.max(merged_objs_np, axis=0) - np.min(merged_objs_np, axis=0) + 1e-8)
                
            merged_fronts = non_dominated_sorting(norm_merged)
            
            global_archive_objs = [merged_objs[idx] for idx in merged_fronts[0]]
            global_archive_layouts = [merged_layouts[idx] for idx in merged_fronts[0]]
            
        with open(archive_csv_path, 'a', encoding='utf-8') as f_arch:
            archive_count = len(global_archive_objs)
            for i, idx in enumerate(range(archive_count)):
                layout = global_archive_layouts[i]
                obj = global_archive_objs[i]
                f_arch.write(f"{cycle},{archive_count},{i},"
                             f"{layout[0][0]:.6f},{layout[0][1]:.6f}," 
                             f"{layout[1][0]:.6f},{layout[1][1]:.6f}," 
                             f"{layout[2][0]:.6f},{layout[2][1]:.6f}," 
                             f"{layout[3][0]:.6f},{layout[3][1]:.6f}," 
                             f"{obj[0]:.4f},{obj[1]:.4f},{obj[2]*1000.0:.4f}\n")

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards_norm = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        loss = -torch.mean(log_probs.sum(dim=1) * rewards_norm)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        stoch_best_idx = fronts[0][0]
        print(f"Cycle {cycle:02d} | LR: {current_lr:.1e} | 采样首：最高温 {objs[stoch_best_idx,0]:.1f}K, 平均温 {objs[stoch_best_idx,1]:.1f}K, 走线和 {objs[stoch_best_idx,2]*1000:.1f}mm | 前沿数: {len(fronts[0])} | 采样HV: {hv_val:.4f}")

        if is_frozen:
            det_layout = policy.forward_deterministic()
            T_pred_3D_det = env.step_batch_simulation([det_layout])
            T_field_det = T_pred_3D_det[0]
            
            det_max_t = np.max(T_field_det)
            det_mean_t = np.mean(T_field_det)
            coords_det = np.array([det_layout[k][:2] for k in range(4)])
            det_dist_sum = 0.0
            for i in range(4):
                for j in range(i+1, 4):
                    det_dist_sum += np.linalg.norm(coords_det[i] - coords_det[j])
                    
            det_objs = np.array([det_max_t, det_mean_t, det_dist_sum])
            
            norm_det_objs = (det_objs - global_mins) / (global_maxs - global_mins + 1e-8)
            if np.any(norm_det_objs > 1.1):
                det_hv = 0.0
            else:
                det_hv = (1.1 - norm_det_objs[0]) * (1.1 - norm_det_objs[1]) * (1.1 - norm_det_objs[2])
                
            print(f"         [确定性校验]：最高温 {det_max_t:.1f}K, 平均温 {det_mean_t:.1f}K, 走线和 {det_dist_sum*1000:.1f}mm | 确定性HV: {det_hv:.4f}")
            
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(det_hv) 
            lr_after = optimizer.param_groups[0]['lr']
            if lr_after < lr_before:
                print(f"  [!] 触发学习率衰减！学习率从 {lr_before:.1e} 降低至 {lr_after:.1e}")
            
            if det_hv > best_hv:
                best_hv = det_hv
                best_path = os.path.join(OUTPUT_RL_DIR, 'best_seq2seq_policy.pth')
                torch.save({
                    'model_state': policy.state_dict(),
                    'epoch': cycle,
                    'hv': det_hv,
                    'best_layout': det_layout
                }, best_path)
                print(f"  [✓] 发现更优的确定性测试超体积 (Deterministic HV)，最佳策略已保存至: {best_path}")
            
            del T_pred_3D_det
            
        del T_pred_3D_all, rewards_tensor, rewards_norm, loss; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "="*80)
    print(f"[✓] 强化学习训练与多重数据留存任务圆满成功！")
    print(f"    - 单周期帕累托解集历史已保存至: {pareto_csv_path}")
    print(f"    - 归一化超体积(HV)收敛大表已保存至: {hv_csv_path}")
    print(f"    - 全局外置帕累托前沿进化大表已保存至: {archive_csv_path}")
    print(f"    - 1.1倍冷冻物理边界指标已保存至: {ref_csv_path}")
    print("="*80)

if __name__ == "__main__":
    run_reinforcement_learning()