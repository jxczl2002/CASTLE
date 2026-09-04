import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import gc
import random


TARGET_DATA_DIR = 'Flux_Training_Data_Final_3D_3_xy' 

PRETRAINED_MODEL_PATH = 'Models_Flux_N1/best_N1_mpnn.pth'   
OUTPUT_DIR = f'Models_Flux_Specialized_{os.path.basename(TARGET_DATA_DIR)}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 20                       
BATCH_SIZE = 2                 
CHUNK_SIZE = 2           
START_LR_TRANSFER = 1e-4         
WEIGHT_DECAY = 1e-3        
PATIENCE = 5
LATENT_DIM = 32

SX = None
SY = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Specialized Training on: {TARGET_DATA_DIR}")

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

def format_to_neighbors_batched_transfer(Z_b, B, sz):
    Z_grid = Z_b.reshape(B, SX, SY, sz, 32)
    Z_pad = F.pad(Z_grid.permute(0, 4, 1, 2, 3), (1, 1, 1, 1, 1, 1), value=0.0).permute(0, 2, 3, 4, 1)
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                neighbors.append(Z_pad[:, dx+1:dx+SX+1, dy+1:dy+SY+1, dz+1:dz+sz+1, :])
    return torch.cat(neighbors, dim=-1).reshape(B, -1, 27, 32)

def get_patched_X_batched_transfer(X_base, past_preds_list, B, V, sz):
    if len(past_preds_list) == 0: return X_base
    X_new = X_base.clone()
    spatial = X_new[:, :, :6912].reshape(B, V, 27, 256)
    
    for i, z_pred in enumerate(reversed(past_preds_list)):
        if i >= 4: break
        start_dim = 128 + (3 - i) * 32
        z_neighbors = format_to_neighbors_batched_transfer(z_pred, B, sz) 
        spatial[:, :, :, start_dim : start_dim+32] = z_neighbors
        
    X_new[:, :, :6912] = spatial.reshape(B, V, 6912)
    return X_new

def build_sequence_dataset_latent(chunk_files, S):
    """预分配内存 + 物理硬拷贝"""
    if not chunk_files:
        return None, None, None
        
    dummy = torch.load(chunk_files[0], map_location='cpu')
    steps = dummy['step']
    V = (steps == steps[0]).sum().item()
    SZ = V // (SX * SY) 
    Time = len(steps) // V
    del dummy; gc.collect()
    
    seqs_per_file = Time - S + 1
    total_samples = len(chunk_files) * seqs_per_file
    
    X_stacked = torch.empty((total_samples, S, V, 6924), dtype=torch.float32)
    Y_stacked = torch.empty((total_samples, S, V, 32), dtype=torch.float32)
    
    idx = 0
    for f in chunk_files:
        data = torch.load(f, map_location='cpu')
        X_case = data['X'].reshape(Time, V, 6924)
        Y_case = data['Y'].reshape(Time, V, 32)
        
        for t in range(seqs_per_file):
            X_stacked[idx].copy_(X_case[t : t+S])
            Y_stacked[idx].copy_(Y_case[t : t+S])
            idx += 1
            
        del data, X_case, Y_case
        gc.collect()
        
    return TensorDataset(X_stacked, Y_stacked), SZ, V

def calculate_exact_batches_latent(file_list, batch_size, chunk_size, S):
    if not file_list: return 0
    dummy_data = torch.load(file_list[0], map_location='cpu')
    steps = dummy_data['step']
    V = (steps == steps[0]).sum().item()
    Time = len(steps) // V
    del dummy_data; gc.collect()

    seqs_per_file = Time - S + 1
    if seqs_per_file <= 0: return 0

    total_batches = 0
    num_chunks = math.ceil(len(file_list) / chunk_size)
    for i in range(num_chunks):
        chunk_files = file_list[i*chunk_size : (i+1)*chunk_size]
        seqs_in_chunk = len(chunk_files) * seqs_per_file
        total_batches += math.ceil(seqs_in_chunk / batch_size)
    return total_batches

def save_checkpoint_atomic(state, filepath):
    temp_filepath = filepath + ".tmp"
    torch.save(state, temp_filepath)
    os.replace(temp_filepath, filepath)

def train_flux_mpnn_N1_specialized():
    global SX, SY
    
    pt_files = sorted(glob.glob(os.path.join(TARGET_DATA_DIR, '*.pt')))
    if len(pt_files) == 0: return print(f"Error: No preprocessed flux data found in {TARGET_DATA_DIR}.")
        
    random.seed(42); random.shuffle(pt_files)
    split_idx = int(len(pt_files) * 0.8)
    if split_idx == len(pt_files) and len(pt_files) > 1:
        split_idx = len(pt_files) - 1
    train_files, val_files = pt_files[:split_idx], pt_files[split_idx:]
    
    dummy = torch.load(train_files[0], map_location='cpu')
    V_temp = (dummy['step'] == dummy['step'][0]).sum().item()
    
    SX = 16 if V_temp // 64 > 14 or (V_temp == 3072 or V_temp == 4096) else 8
    SY = SX
    
    SZ_temp = V_temp // (SX * SY)
    NUM_LAYERS_DETECTED = (SZ_temp - 6) // 2
    del dummy; gc.collect()
    print("\n" + "="*80)
    print(f"[-] 【自适应探针】：成功探测到当前物理尺度")
    print(f"    - 网格分辨率: {SX*16} x {SY*16} | Z轴层数: {NUM_LAYERS_DETECTED} 层 (SZ={SZ_temp}, 节点数={V_temp})")
    print("="*80 + "\n")
    
    model = N1_FiLM_MPNN().to(device)
    print(f"[*] 正在从 {PRETRAINED_MODEL_PATH} 载入基础物理预训练权重...")
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device)['model_state'])
    
    print("[*] 正在冻结材料流基础网络，仅对高层能量流和调制网络进行专项微调...")
    for param in model.stream_material_base.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=START_LR_TRANSFER, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE, min_lr=1e-6, verbose=True)
    criterion_mse = nn.MSELoss()
    
    start_epoch, best_val_loss = 1, float('inf')
    history = {'epochs':[], 'lrs_mpnn':[], 'unrolls':[], 'train_loss':[], 'val_loss':[]}
    
    ckpt_path = os.path.join(OUTPUT_DIR, 'latest_specialized_mpnn.pth')
    log_path = os.path.join(OUTPUT_DIR, 'specialized_loss_log.txt')
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch, best_val_loss, history = ckpt['epoch'] + 1, ckpt['best_val_loss'], ckpt['history']
        print(f"[✓] 恢复检查点成功。将从 Epoch {start_epoch} 继续进行专项微调。")
    else:
        with open(log_path, 'w') as f: f.write("Epoch,LR_MPNN,S_Unroll,Train_Latent_MSE,Val_Latent_MSE(S=3)\n")

    VAL_S = 3
    total_val_batches = calculate_exact_batches_latent(val_files, BATCH_SIZE, CHUNK_SIZE, VAL_S)

    for epoch in range(start_epoch, EPOCHS + 1):
        current_lr_mpnn = optimizer.param_groups[0]['lr']
        random.shuffle(train_files)
        
        S = 3 
        
        model.train()
        train_loss_accum, samples = 0.0, 0
        
        total_train_batches = calculate_exact_batches_latent(train_files, BATCH_SIZE, CHUNK_SIZE, S)
        pbar_train = tqdm(total=total_train_batches, desc=f"Ep {epoch:03d} [Train S={S}]", dynamic_ncols=True, leave=False)
        train_chunks = math.ceil(len(train_files) / CHUNK_SIZE)
        
        for i in range(train_chunks):
            chunk = train_files[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
            dataset, SZ, V = build_sequence_dataset_latent(chunk, S)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            for bx, by in loader:
                B = bx.size(0) 
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                
                loss_step_sum = 0
                past_preds = []
                
                for s in range(S):
                    X_base = bx[:, s] 
                    X_input = get_patched_X_batched_transfer(X_base, past_preds, B, V, SZ).reshape(B*V, 6924)
                    
                    z_pred = model(X_input) 
                    z_pred_b = z_pred.reshape(B, V, 32)
                    past_preds.append(z_pred_b) 
                    
                    y_true = by[:, s].reshape(B*V, 32)
                    loss_step_sum += criterion_mse(z_pred, y_true)
                    
                loss_total = loss_step_sum / S  
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss_accum += loss_total.item() * B
                samples += B
                
                pbar_train.update(1)
                pbar_train.set_postfix({'Loss': f"{train_loss_accum/samples:.8f}"})
                
            del dataset, loader; gc.collect(); torch.cuda.empty_cache()
            
        pbar_train.close()
        train_loss_ep = train_loss_accum / samples
        
        model.eval()
        val_loss_accum, val_samples = 0.0, 0
        val_chunks = math.ceil(len(val_files) / CHUNK_SIZE)
        
        pbar_val = tqdm(total=total_val_batches, desc=f"Ep {epoch:03d} [Val S={VAL_S}]", dynamic_ncols=True, leave=False)
        
        for i in range(val_chunks):
            chunk = val_files[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
            dataset, SZ, V = build_sequence_dataset_latent(chunk, VAL_S)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            with torch.no_grad():
                for bx, by in loader:
                    B = bx.size(0)
                    bx, by = bx.to(device), by.to(device)
                    loss_step_sum = 0
                    past_preds = []
                    
                    for s in range(VAL_S):
                        X_base = bx[:, s] 
                        X_input = get_patched_X_batched_transfer(X_base, past_preds, B, V, SZ).reshape(B*V, 6924)
                        z_pred = model(X_input)
                        z_pred_b = z_pred.reshape(B, V, 32)
                        past_preds.append(z_pred_b)
                        
                        y_true = by[:, s].reshape(B*V, 32)
                        loss_step_sum += criterion_mse(z_pred, y_true)
                        
                    val_loss_accum += (loss_step_sum / VAL_S).item() * B
                    val_samples += B
                    pbar_val.update(1)
                    
            del dataset, loader; gc.collect(); torch.cuda.empty_cache()
            
        pbar_val.close()
        val_loss_ep = val_loss_accum / val_samples
        scheduler.step(val_loss_ep)
        
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{current_lr_mpnn:.1e},{S},{train_loss_ep:.8f},{val_loss_ep:.8f}\n")
            
        print(f"-->[Summary] Ep {epoch:03d} | S_Unroll: {S} | LR: {current_lr_mpnn:.1e} | Train Latent MSE: {train_loss_ep:.6e} | Val Latent MSE(S={VAL_S}): {val_loss_ep:.6e}")
        
        ckpt_state = {
            'epoch': epoch, 'model_state': model.state_dict(), 
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 
            'best_val_loss': best_val_loss, 'history': history
        }
        save_checkpoint_atomic(ckpt_state, ckpt_path)
        
        if val_loss_ep < best_val_loss:
            best_val_loss = val_loss_ep
            save_checkpoint_atomic(ckpt_state, os.path.join(OUTPUT_DIR, 'best_transfer_mpnn.pth'))
            print(f"[!] Best N1 model updated! Val Latent MSE(S={VAL_S}): {best_val_loss:.6e}")

if __name__ == "__main__":
    train_flux_mpnn_N1_specialized()