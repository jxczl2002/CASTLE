import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import gc
from tqdm import tqdm

torch.backends.cudnn.benchmark = True  

DATA_DIR = 'Dataset_Subdomains'
OUTPUT_DIR = 'Models_Autoencoders_Final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIABLES = ['Temp', 'Source', 'K', 'rhoCp']  

BATCH_SIZE = 64                    
EPOCHS = 100                        
LATENT_DIM = 32                      
PATIENCE = 20                       
MIN_EPOCHS = 20                     
TARGET_MSE = 1e-6                   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def spatial_gradient_loss(pred, true):
    """Sobolev 梯度损失：保护温度场的连续性和热传导梯度"""
    dx_pred = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    dx_true = true[:, :, 1:, :, :] - true[:, :, :-1, :, :]
    dy_pred = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    dy_true = true[:, :, :, 1:, :] - true[:, :, :, :-1, :]
    dz_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    dz_true = true[:, :, :, :, 1:] - true[:, :, :, :, :-1]
    return torch.mean((dx_pred - dx_true)**2) + torch.mean((dy_pred - dy_true)**2) + torch.mean((dz_pred - dz_true)**2)

def total_variation_loss(x):
    """TV损失：惩罚条件特征内部的数值波动，强迫产生边缘极其锐利的纯净物理色块"""
    tv_d = torch.mean(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))
    tv_h = torch.mean(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]))
    return tv_d + tv_h + tv_w

class SubdomainFullDataset(Dataset):
    def __init__(self, npy_path, common_data, v_min, range_val):
        self.data = np.load(npy_path, mmap_mode='r')
        self.common_data = common_data
        self.v_min = v_min
        self.range_val = range_val
        self.len_data = len(self.data)
        self.len_common = len(common_data) if common_data is not None else 0

    def __len__(self):
        return self.len_data + self.len_common

    def __getitem__(self, idx):
        if idx < self.len_data:
            sample = np.array(self.data[idx], copy=True, dtype=np.float32)
        else:
            sample = np.array(self.common_data[idx - self.len_data], copy=True, dtype=np.float32)
            
        norm_val = (sample - self.v_min) / self.range_val * 0.90 + 0.05
        tensor_x = torch.from_numpy(norm_val).unsqueeze(0)
        return tensor_x, tensor_x


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

def compute_global_stats(train_path, val_path, common_data):
    v_min, v_max = float('inf'), float('-inf')
    train_mmap = np.load(train_path, mmap_mode='r')
    for i in range(0, len(train_mmap), 50000):
        chunk = train_mmap[i:i+50000]
        v_min, v_max = min(v_min, float(chunk.min())), max(v_max, float(chunk.max()))
    val_mmap = np.load(val_path, mmap_mode='r')
    for i in range(0, len(val_mmap), 50000):
        chunk = val_mmap[i:i+50000]
        v_min, v_max = min(v_min, float(chunk.min())), max(v_max, float(chunk.max()))
    if len(common_data) > 0:
        v_min, v_max = min(v_min, float(common_data.min())), max(v_max, float(common_data.max()))
    return v_min, v_max

def save_checkpoint_atomic(state, filepath):
    temp_filepath = filepath + ".tmp"
    torch.save(state, temp_filepath)
    os.replace(temp_filepath, filepath)

def rewrite_log_from_history(history, log_file_path):
    with open(log_file_path, 'w') as log_f:
        log_f.write("Epoch,LR,Train_TotLoss,Val_MSE,Val_AuxLoss\n")
        for i in range(len(history['train_loss'])):
            log_f.write(f"{history['epochs'][i]},{history['lrs'][i]:.1e},"
                        f"{history['train_loss'][i]:.8f},{history['val_loss'][i]:.8f},"
                        f"{history['val_aux'][i]:.8f}\n")

def train_autoencoder_for_variable(var_name):
    print(f"\n{'='*75}\n[Start] Final Training: {var_name}\n{'='*75}")
    train_path, val_path, common_path = [os.path.join(DATA_DIR, f'{prefix}_{var_name}.npy') for prefix in ['train', 'val', 'common']]
    
    if not os.path.exists(train_path) or not os.path.exists(val_path): 
        print(f"Error: 找不到 {var_name} 的数据集，跳过...")
        return
        
    common_data = np.load(common_path) if os.path.exists(common_path) else np.empty((0,16,16,16), dtype=np.float32)

    param_path = os.path.join(OUTPUT_DIR, f'{var_name}_norm_params.json')
    if os.path.exists(param_path):
        with open(param_path, 'r') as f: norm_params = json.load(f)
        v_min, v_max = norm_params['v_min'], norm_params['v_max']
    else:
        v_min, v_max = compute_global_stats(train_path, val_path, common_data)
        with open(param_path, 'w') as f: json.dump({'v_min': v_min, 'v_max': v_max}, f)
            
    range_val = max(v_max - v_min, 1e-8)
    is_temp = (var_name == 'Temp')
    aux_name = "Val Grad" if is_temp else "Val TV+MAE"

    if is_temp:
        model = SmoothAutoencoder3D_UltraTemp(LATENT_DIM).to(device)
        current_start_lr = 2e-4  
        optimizer = optim.AdamW(model.parameters(), lr=current_start_lr, weight_decay=1e-5)
    else:
        model = SparseDiscreteAutoencoder3D(LATENT_DIM).to(device)
        current_start_lr = 4e-4  

        optimizer = optim.AdamW(model.parameters(), lr=current_start_lr, weight_decay=0.0)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7)
    
    start_epoch, best_val_loss, epochs_no_improve = 1, float('inf'), 0
    history = {'epochs':[], 'lrs':[], 'train_loss':[], 'val_loss':[], 'val_aux':[]}
    latest_ckpt_path = os.path.join(OUTPUT_DIR, f'latest_{var_name}_ae.pth')
    log_file_path = os.path.join(OUTPUT_DIR, f'{var_name}_loss_log.txt')
    
    if os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch, best_val_loss = ckpt['epoch'] + 1, ckpt.get('best_val_loss', float('inf'))
        history = ckpt['history']
        rewrite_log_from_history(history, log_file_path)
    else:
        with open(log_file_path, 'w') as f: f.write("Epoch,LR,Train_TotLoss,Val_MSE,Val_AuxLoss\n")

    train_loader = DataLoader(SubdomainFullDataset(train_path, common_data, v_min, range_val), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(SubdomainFullDataset(val_path, None, v_min, range_val), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    for epoch in range(start_epoch, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']
        model.train()
        train_loss_safe = 0.0
        
        sobolev_weight = 0.1 if (is_temp and epoch > 2) else 0.0
        
        pbar_train = tqdm(train_loader, desc=f"Ep[{epoch}/{EPOCHS}] Train", leave=False, dynamic_ncols=True)
        for batch_x, _ in pbar_train:
            batch_x = batch_x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(batch_x)
            loss_mse = criterion_mse(outputs, batch_x)
            
            if is_temp:
                loss = loss_mse + sobolev_weight * spatial_gradient_loss(outputs, batch_x)
            else:
                loss = loss_mse + 0.2 * criterion_mae(outputs, batch_x) + 0.05 * total_variation_loss(outputs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_safe += loss.item() * batch_x.size(0)
            
        pbar_train.close()
        train_loss_safe /= len(train_loader.dataset)
        
        model.eval()
        val_loss_safe_mse, val_loss_safe_aux = 0.0, 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                outputs = model(batch_x)
                
                v_mse = criterion_mse(outputs, batch_x)
                if is_temp:
                    v_aux = spatial_gradient_loss(outputs, batch_x)
                else:
                    v_aux = 0.2 * criterion_mae(outputs, batch_x) + 0.05 * total_variation_loss(outputs)
                
                val_loss_safe_mse += v_mse.item() * batch_x.size(0)
                val_loss_safe_aux += v_aux.item() * batch_x.size(0)
                
        val_loss_safe_mse /= len(val_loader.dataset)
        val_loss_safe_aux /= len(val_loader.dataset)
        
        train_loss_01 = train_loss_safe / (0.9 ** 2)
        val_loss_01_mse = val_loss_safe_mse / (0.9 ** 2)
        val_loss_01_aux = val_loss_safe_aux / (0.9**2) if is_temp else val_loss_safe_aux / 0.9
        
        scheduler.step(val_loss_01_mse)
        history['epochs'].append(epoch)
        history['lrs'].append(current_lr)
        history['train_loss'].append(train_loss_01)
        history['val_loss'].append(val_loss_01_mse)
        history['val_aux'].append(val_loss_01_aux)
        
        with open(log_file_path, 'a') as f: 
            f.write(f"{epoch},{current_lr:.1e},{train_loss_01:.8f},{val_loss_01_mse:.8f},{val_loss_01_aux:.8f}\n")
            
        print(f"Ep[{epoch:03d}/{EPOCHS}] LR:{current_lr:.1e} | Train:{train_loss_01:.2e} | ValMSE:{val_loss_01_mse:.2e} | {aux_name}:{val_loss_01_aux:.2e}")
        
        ckpt_state = {
            'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 
            'scheduler_state': scheduler.state_dict(), 'best_val_loss': best_val_loss, 'history': history
        }
        save_checkpoint_atomic(ckpt_state, latest_ckpt_path)
        
        if val_loss_01_mse < best_val_loss:
            best_val_loss = val_loss_01_mse
            epochs_no_improve = 0
            save_checkpoint_atomic(ckpt_state, os.path.join(OUTPUT_DIR, f'best_{var_name}_ae.pth'))
            
            if val_loss_01_mse <= TARGET_MSE and epoch >= MIN_EPOCHS: 
                print(f"  --> [✓] 达到目标精度 Val MSE <= 1e-6，提前竣工！")
                break
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE: 
                print(f"  --> [!] 触发 Early Stopping。")
                break

    print(f"[*] {var_name} 训练阶段全部完成。\n")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    for var in VARIABLES: 
        train_autoencoder_for_variable(var)