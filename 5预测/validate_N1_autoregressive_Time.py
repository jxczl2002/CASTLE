import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm

NUM_LAYERS = 3                       
DIR_A1 = f'3D_{NUM_LAYERS}_time_test/condition'  
DIR_B1 = f'3D_{NUM_LAYERS}_time_test/temp'      

AE_MODEL_DIR = 'Models_Autoencoders_Final'
MPNN_MODEL_PATH = f'Models_Flux_Specialized_Flux_Training_Data_Final_3D_{NUM_LAYERS}_time/best_transfer_mpnn.pth'
OUTPUT_DIR = f'Validation_Results_N1_{NUM_LAYERS}L_StartT1_Transfer_Time'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIM = 32
T_INIT_PHYSICAL = 293.15           
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Validating N1 {NUM_LAYERS}-Layer Model (Starting from t=1)")

NZ = 96 + 32 * NUM_LAYERS           
SZ = 6 + 2 * NUM_LAYERS            
NX, NY, SX, SY = 128, 128, 8, 8     

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
        self.decoder_fc = nn.Sequential(weight_norm(nn.Linear(latent_dim, 256 * 8)), nn.LeakyReLU(0.2, True))
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(256, 128, 3, 1, 0)), nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(128, 64, 3, 1, 0)), nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(64, 16, 3, 1, 0)), nn.LeakyReLU(0.2, True),
            nn.ReplicationPad3d(1), weight_norm(nn.Conv3d(16, 1, 3, 1, 0)) 
        )
    def forward_encoder(self, x): return self.encoder(x)
    def forward_decoder(self, z): return self.decoder_conv(self.decoder_fc(z).view(-1, 256, 2, 2, 2))

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.gelu = nn.GELU()
    def forward(self, x): return self.gelu(x + self.net(x))

class N1_FiLM_MPNN(nn.Module):
    def __init__(self):
        super(N1_FiLM_MPNN, self).__init__()
        self.stream_material_base = nn.Sequential(nn.Conv3d(76, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(), nn.Conv3d(256, 512, 3, padding=1), nn.GroupNorm(32, 512), nn.GELU())
        self.mat_gamma = nn.Conv3d(512, 1024, 3, padding=0); self.mat_beta = nn.Conv3d(512, 1024, 3, padding=0)
        self.stream_energy = nn.Sequential(nn.Conv3d(160, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(), nn.Conv3d(256, 512, 3, padding=1), nn.GroupNorm(32, 512), nn.GELU(), nn.Conv3d(512, 1024, 3, padding=0), nn.GroupNorm(32, 1024), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(1056, 1024), nn.LayerNorm(1024), nn.GELU())
        self.node_res = nn.Sequential(ResBlock(1024), ResBlock(1024), ResBlock(1024))
        self.node_out = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 32))

    def forward(self, x):
        spatial, boundary = x[:, :6912].reshape(-1, 27, 256), x[:, 6912:] 
        clean_z_curr = spatial[:, 13, -32:].clone() 
        mat_grid = torch.cat([spatial[:,:,0:64], boundary.unsqueeze(1).expand(-1,27,-1)], dim=-1).permute(0,2,1).contiguous().reshape(-1, 76, 3, 3, 3)
        eng_grid = torch.cat([spatial[:,:,64:96], spatial[:,:,128:256]], dim=-1).permute(0,2,1).contiguous().reshape(-1, 160, 3, 3, 3)
        mat_base = self.stream_material_base(mat_grid)
        gated_flux = self.mat_gamma(mat_base).reshape(-1, 1024) * self.stream_energy(eng_grid).reshape(-1, 1024) + self.mat_beta(mat_base).reshape(-1, 1024)
        delta_z = self.node_out(self.node_res(self.node_proj(torch.cat([gated_flux, clean_z_curr], dim=-1))))
        return clean_z_curr + delta_z


def extract_subs_vectorized(arr):
    if arr.ndim == 3: return arr.reshape(SX, 16, SY, 16, SZ, 16).transpose(0, 2, 4, 1, 3, 5)
    elif arr.ndim == 4:
        T = arr.shape[0]
        return arr.reshape(T, SX, 16, SY, 16, SZ, 16).transpose(0, 1, 3, 5, 2, 4, 6)

def reassemble_subs_vectorized(subs_flat):
    return subs_flat.reshape(SX, SY, SZ, 16, 16, 16).transpose(0, 3, 1, 4, 2, 5).reshape(NX, NY, NZ)

def generate_and_verify_boundaries(device, dir_a1, num_layers):
    bounds = np.zeros((SX, SY, SZ, 12), dtype=np.float32)
    for ix in range(SX):
        for iy in range(SY):
            for iz in range(SZ):
                bd_top = [1.0, 1.0] if iz == SZ - 1 else [0.0, 0.0]
                bd_bot = [1.0, 0.01] if iz == 0 else [0.0, 0.0]
                bd_left = [1.0, 0.0] if ix == 0 else [0.0, 0.0]
                bd_right = [1.0, 0.0] if ix == SX - 1 else [0.0, 0.0]
                bd_front = [1.0, 0.0] if iy == 0 else [0.0, 0.0]
                bd_back = [1.0, 0.0] if iy == SY - 1 else [0.0, 0.0]
                bounds[ix, iy, iz] = np.concatenate([bd_top, bd_bot, bd_left, bd_right, bd_front, bd_back])
    
    dynamic_tensor = torch.tensor(bounds, dtype=torch.float32, device=device)

    search_pattern = os.path.join(dir_a1, f'global_boundary_{num_layers}layers.npy')
    bd_files = glob.glob(search_pattern)
    
    if not bd_files:
        bd_files = glob.glob(os.path.join(dir_a1, 'global_boundary*.npy'))

    if bd_files:
        ref_path = bd_files[0]
        print(f"[*] 找到本地边界文件: {os.path.basename(ref_path)}，正在进行自动推演结果的双向交叉校验...")
        glob_bd = np.load(ref_path)
        
        ref_bounds = np.zeros((SX, SY, SZ, 12), dtype=np.float32)
        for ix in range(SX):
            for iy in range(SY):
                for iz in range(SZ):
                    cx, cy, cz = ix * 16 + 8, iy * 16 + 8, iz * 16 + 8
                    ref_bounds[ix, iy, iz] = np.concatenate([
                        glob_bd[cx, cy, iz*16+15, 0, :], glob_bd[cx, cy, iz*16, 1, :],
                        glob_bd[ix*16, cy, cz, 2, :], glob_bd[ix*16+15, cy, cz, 3, :],
                        glob_bd[cx, iy*16, cz, 4, :], glob_bd[cx, iy*16+15, cz, 5, :]
                    ])
        
        max_diff = np.max(np.abs(bounds - ref_bounds))
        if max_diff < 1e-5:
            print(f"[✓] 边界校验完美通过！自动推演拓扑与物理拓扑 100% 对齐 (Max Error: {max_diff:.1e})。")
        else:
            print(f"[!] 警告: 自动推演边界与硬盘文件不匹配！(Max Error: {max_diff:.1e})")
            print("    已强制覆盖使用本地真实边界进行推演。")
            return torch.tensor(ref_bounds, dtype=torch.float32, device=device)
    else:
        print(f"[-] 未在 {dir_a1} 找到全局参考文件，将安全使用自动推演的理论物理边界。")

    return dynamic_tensor

def load_ae_model(var_name):
    with open(os.path.join(AE_MODEL_DIR, f'{var_name}_norm_params.json'), 'r') as f: params = json.load(f)
    model = SmoothAutoencoder3D_UltraTemp() if var_name == 'Temp' else SparseDiscreteAutoencoder3D()
    ckpt = torch.load(os.path.join(AE_MODEL_DIR, f'best_{var_name}_ae.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt, strict=False)
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad = False
    return model, params['v_min'], max(params['v_max'] - params['v_min'], 1e-8)

def normalize_t(arr, vmin, vrange):
    return torch.tensor(((arr - vmin) / vrange) * 0.90 + 0.05, dtype=torch.float32, device=device).unsqueeze(1)

def validate_N1_autoregressive():
    print("\n" + "="*70)
    print(f"[N1 Validation] 启动 N1 模型全量自回归多维指标推演引擎")
    print(f"-> 目标架构: {NUM_LAYERS} 层 | Z轴总厚度: {NZ} 像素 | Z轴子域层数: {SZ}")
    print("="*70)

    enc_K, min_K, range_K = load_ae_model('K')
    enc_rhoCp, min_rhoCp, range_rhoCp = load_ae_model('rhoCp')
    enc_Q, min_Q, range_Q = load_ae_model('Source')
    enc_Temp, min_T, range_T = load_ae_model('Temp')
    
    mpnn = N1_FiLM_MPNN().to(device)
    mpnn.load_state_dict(torch.load(MPNN_MODEL_PATH, map_location=device)['model_state'])
    mpnn.eval()
    for p in mpnn.parameters(): p.requires_grad = False

    bd_tensor = generate_and_verify_boundaries(device, DIR_A1, NUM_LAYERS)

    if not os.path.exists(DIR_A1): return print(f"Error: 目标条件目录 {DIR_A1} 不存在，请检查配置。")
    case_folders = [f for f in os.listdir(DIR_A1) if os.path.isdir(os.path.join(DIR_A1, f))]
    if len(case_folders) == 0: return print(f"Error: 目录 {DIR_A1} 中未找到验证案例！")

    all_metrics = []
    
    for case_id in tqdm(case_folders, desc="Cases Validation"):
        case_out_dir = os.path.join(OUTPUT_DIR, case_id)
        os.makedirs(case_out_dir, exist_ok=True)
        
        arr_K = np.load(os.path.join(DIR_A1, case_id, 'k_matrix.npy'))
        arr_rhoCp = np.load(os.path.join(DIR_A1, case_id, 'rhoCp_matrix.npy'))
        arr_Q = np.load(os.path.join(DIR_A1, case_id, 'Q_source.npy'))
        arr_Temp = np.load(os.path.join(DIR_B1, case_id, 'T_transient.npy'))
        num_t = arr_Temp.shape[0]
        
        with torch.no_grad():
            Z_K = enc_K.forward_encoder(normalize_t(extract_subs_vectorized(arr_K).reshape(-1,16,16,16), min_K, range_K)).view(SX, SY, SZ, 32)
            Z_rhoCp = enc_rhoCp.forward_encoder(normalize_t(extract_subs_vectorized(arr_rhoCp).reshape(-1,16,16,16), min_rhoCp, range_rhoCp)).view(SX, SY, SZ, 32)
            Z_Q = enc_Q.forward_encoder(normalize_t(extract_subs_vectorized(arr_Q).reshape(-1,16,16,16), min_Q, range_Q)).view(SX, SY, SZ, 32)
            
            Z_true_list = []
            for t_idx in range(num_t):
                sub_t = extract_subs_vectorized(arr_Temp[t_idx]).reshape(-1,16,16,16)
                Z_true_list.append(enc_Temp.forward_encoder(normalize_t(sub_t, min_T, range_T)).view(SX, SY, SZ, 32))
            Z_true_all = torch.stack(Z_true_list)
            
            T_0_const = np.full((NX, NY, NZ), T_INIT_PHYSICAL, dtype=np.float32)
            Z_0 = enc_Temp.forward_encoder(normalize_t(extract_subs_vectorized(T_0_const).reshape(-1,16,16,16), min_T, range_T)).view(SX, SY, SZ, 32)
            
            Z_history = Z_0.unsqueeze(-2).expand(-1, -1, -1, 4, -1).clone()
            
            pred_3D_frames = []
            true_3D_frames = []
            
            pred_3D_frames.append(T_0_const)
            true_3D_frames.append(T_0_const)

            Z_pad_grid = torch.zeros((SX, SY, SZ, 32), dtype=torch.float32, device=device)
            
            for t in tqdm(range(1, num_t), leave=False, desc=f"{case_id} 滚动自回归推演"):
                
                node_feat = torch.cat([Z_K, Z_rhoCp, Z_Q, Z_pad_grid, Z_history.view(SX, SY, SZ, 128)], dim=-1)
                
                padded_feat = F.pad(node_feat.permute(3, 0, 1, 2), (1, 1, 1, 1, 1, 1), value=0.0).permute(1, 2, 3, 0)
                
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            neighbors.append(padded_feat[dx+1 : dx+SX+1, dy+1 : dy+SY+1, dz+1 : dz+SZ+1, :])
                neighbors_tensor = torch.cat(neighbors, dim=-1)
                
                X_t = torch.cat([neighbors_tensor, bd_tensor], dim=-1).reshape(-1, 6924)
                
                Z_pred_flat = mpnn(X_t)
                
                T_pred_norm = enc_Temp.forward_decoder(Z_pred_flat).squeeze(1).cpu().numpy()
                T_pred_phys = (T_pred_norm - 0.05) / 0.90 * range_T + min_T
                T_pred_3D = reassemble_subs_vectorized(T_pred_phys)
                
                T_true_3D = arr_Temp[t]
                
                pred_3D_frames.append(T_pred_3D)
                true_3D_frames.append(T_true_3D)
                
                abs_err_t = np.abs(T_pred_3D - T_true_3D)
                temp_mae_abs = np.mean(abs_err_t)
                temp_mse_abs = np.mean(abs_err_t**2)
                temp_rmse_abs = np.sqrt(temp_mse_abs)
                
                rel_err_t = abs_err_t / T_true_3D
                temp_mae_rel = np.mean(rel_err_t)
                temp_mse_rel = np.mean(rel_err_t**2)
                temp_rmse_rel = np.sqrt(temp_mse_rel)
                
                temp_max_err = np.max(abs_err_t)
                true_max_t = np.max(T_true_3D)
                pred_max_t = np.max(T_pred_3D)

                z_p_np = Z_pred_flat.cpu().numpy()
                z_t_np = Z_true_all[t].reshape(-1, 32).cpu().numpy()
                
                abs_err_z = np.abs(z_p_np - z_t_np)
                latent_mae_abs = np.mean(abs_err_z)
                latent_mse_abs = np.mean(abs_err_z**2)
                latent_rmse_abs = np.sqrt(latent_mse_abs)
                
                rel_err_z = abs_err_z / (np.abs(z_t_np) + 1e-8)
                latent_mae_rel = np.mean(rel_err_z)
                latent_mse_rel = np.mean(rel_err_z**2)
                latent_rmse_rel = np.sqrt(latent_mse_rel)

                all_metrics.append({
                    "Case": case_id, 
                    "Time_Step": t,
                    "Temp_MAE_Abs": temp_mae_abs,
                    "Temp_MSE_Abs": temp_mse_abs,
                    "Temp_RMSE_Abs": temp_rmse_abs,
                    "Temp_MAE_Rel": temp_mae_rel,
                    "Temp_MSE_Rel": temp_mse_rel,
                    "Temp_RMSE_Rel": temp_rmse_rel,
                    "Temp_Max_Err": temp_max_err,
                    "True_Max_T": true_max_t,
                    "Pred_Max_T": pred_max_t,
                    "Latent_MAE_Abs": latent_mae_abs,
                    "Latent_MSE_Abs": latent_mse_abs,
                    "Latent_RMSE_Abs": latent_rmse_abs,
                    "Latent_MAE_Rel": latent_mae_rel,
                    "Latent_MSE_Rel": latent_mse_rel,
                    "Latent_RMSE_Rel": latent_rmse_rel
                })
                
                Z_pred_grid = Z_pred_flat.view(SX, SY, SZ, 32)
                Z_history = torch.cat([Z_history[:, :, :, 1:, :], Z_pred_grid.unsqueeze(-2)], dim=-2)
                
        pred_4D_array = np.stack(pred_3D_frames)
        np.save(os.path.join(case_out_dir, f'T_pred_autoregressive.npy'), pred_4D_array)
        
        plot_frames = [1, num_t // 2, num_t - 1]
        fig, axes = plt.subplots(len(plot_frames), 3, figsize=(16, 5 * len(plot_frames)))
        fig.suptitle(f'Autoregressive Prediction (N1 Model - {NUM_LAYERS}L)\nCase: {case_id}', fontsize=18)
        
        slice_z = NZ // 2 
        for row, frame in enumerate(plot_frames):
            if frame >= num_t: break
            truth = true_3D_frames[frame][:, :, slice_z]
            pred = pred_3D_frames[frame][:, :, slice_z]
            err = np.abs(truth - pred)
            
            vmin, vmax = truth.min(), truth.max()
            
            im0 = axes[row, 0].imshow(truth, cmap='jet', vmin=vmin, vmax=vmax); axes[row, 0].set_title(f'Frame {frame} - Truth', fontsize=14)
            im1 = axes[row, 1].imshow(pred, cmap='jet', vmin=vmin, vmax=vmax); axes[row, 1].set_title(f'Frame {frame} - Prediction', fontsize=14)
            im2 = axes[row, 2].imshow(err, cmap='magma'); axes[row, 2].set_title(f'Frame {frame} - Abs Error', fontsize=14)
            
            fig.colorbar(im0, ax=axes[row, 0], fraction=0.46, pad=0.04)
            fig.colorbar(im1, ax=axes[row, 1], fraction=0.46, pad=0.04)
            fig.colorbar(im2, ax=axes[row, 2], fraction=0.46, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(case_out_dir, 'Autoregressive_CrossSection.png'), dpi=200)
        plt.close()
        
        df_case = pd.DataFrame([m for m in all_metrics if m['Case'] == case_id])
        df_case.to_csv(os.path.join(case_out_dir, 'Autoregressive_Metrics.csv'), index=False)
        
        del arr_K, arr_rhoCp, arr_Q, arr_Temp, pred_3D_frames, true_3D_frames, pred_4D_array, Z_true_all
        gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(OUTPUT_DIR, 'Global_Autoregressive_Metrics.csv')
    df.to_csv(csv_path, index=False)
    
    avg_mae_t = df['Temp_MAE_Abs'].mean()
    avg_rel_mae_t = df['Temp_MAE_Rel'].mean() * 100
    avg_mae_z = df['Latent_MAE_Abs'].mean()
    max_err_t = df['Temp_Max_Err'].max()
    print("\n" + "="*70)
    print(f"[验证任务圆满结束] 全局物理平均 MAE: {avg_mae_t:.4f} K (平均相对误差: {avg_rel_mae_t:.4f}%)")
    print(f"               全局潜在平均 MAE: {avg_mae_z:.4f}")
    print(f"               全局物理极端最大温差: {max_err_t:.4f} K")
    print(f"[!] 4D预测温度场(.npy)、多维指标详细 CSV 报表及 3D 切片图已安全保存至: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    validate_N1_autoregressive()