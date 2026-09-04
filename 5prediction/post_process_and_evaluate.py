import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
import gc
from tqdm import tqdm

TASKS = [
    {
        "name": "3D_3_Baseline",
        "pred_dir": "Raw/Validation_Results_N1_3L_StartT1",
        "temp_dir": "3D_3_test/temp",
        "nx": 128, "ny": 128, "nz": 192
    },
    {
        "name": "3D_3_xy_Extrapolation",
        "pred_dir": "Raw/Validation_Results_N1_3L_StartT1_Transfer_xy",
        "temp_dir": "3D_3_xy_test/temp",
        "nx": 256, "ny": 256, "nz": 192
    },
    {
        "name": "3D_3_time_Extrapolation",
        "pred_dir": "Raw/Validation_Results_N1_3L_StartT1_Transfer_Time",
        "temp_dir": "3D_3_time_test/temp",
        "nx": 128, "ny": 128, "nz": 192
    },
    {
        "name": "3D_2_Extrapolation",
        "pred_dir": "Raw/Validation_Results_N1_2L_StartT1_Transfer_Z",
        "temp_dir": "3D_2_test/temp",
        "nx": 128, "ny": 128, "nz": 160
    },
    {
        "name": "3D_4_Extrapolation",
        "pred_dir": "Raw/Validation_Results_N1_4L_StartT1_Transfer_Z",
        "temp_dir": "3D_4_test/temp",
        "nx": 128, "ny": 128, "nz": 224
    }
]

OUTPUT_RESULTS_DIR = 'Smoothing'
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
NUM_WORKERS = 2  


T_INIT_PHYSICAL = 293.15  


def run_scheme2_spr(T_raw, nx, ny, nz):
    """Scheme 2: SPR 局部三次样条插值修复"""
    T_out = T_raw.copy()
    for i in range(16, nx, 16):
        T_out[i-1, :, :] = 0.67 * T_out[i-2, :, :] + 0.33 * T_out[i+2, :, :]
        T_out[i, :, :]   = 0.50 * T_out[i-2, :, :] + 0.50 * T_out[i+2, :, :]
        T_out[i+1, :, :] = 0.33 * T_out[i-2, :, :] + 0.67 * T_out[i+2, :, :]
    for j in range(16, ny, 16):
        T_out[:, j-1, :] = 0.67 * T_out[:, j-2, :] + 0.33 * T_out[:, j+2, :]
        T_out[:, j, :]   = 0.50 * T_out[:, j-2, :] + 0.50 * T_out[:, j+2, :]
        T_out[:, j+1, :] = 0.33 * T_out[:, j-2, :] + 0.67 * T_out[:, j+2, :]
    for k in range(16, nz, 16):
        T_out[:, :, k-1] = 0.67 * T_out[:, :, k-2] + 0.33 * T_out[:, :, k+2]
        T_out[:, :, k]   = 0.50 * T_out[:, :, k-2] + 0.50 * T_out[:, :, k+2]
        T_out[:, :, k+1] = 0.33 * T_out[:, :, k-2] + 0.67 * T_out[:, :, k+2]
    return T_out

def run_scheme4_l2(T_in, W_std):
    """Scheme 4: 空间置信度余弦混合平滑"""
    T_smooth = gaussian_filter(T_in, sigma=1.0, mode='reflect')
    return W_std * T_in + (1.0 - W_std) * T_smooth

def get_3d_confidence_mask(nx, ny, nz):
    x_dist = np.minimum(np.arange(nx) % 16, 16 - (np.arange(nx) % 16))
    y_dist = np.minimum(np.arange(ny) % 16, 16 - (np.arange(ny) % 16))
    z_dist = np.minimum(np.arange(nz) % 16, 16 - (np.arange(nz) % 16))
    x_w = 0.5 * (1.0 - np.cos(np.pi * x_dist / 8.0))
    y_w = 0.5 * (1.0 - np.cos(np.pi * y_dist / 8.0))
    z_w = 0.5 * (1.0 - np.cos(np.pi * z_dist / 8.0))
    X_w, Y_w, Z_w = np.meshgrid(x_w, y_w, z_w, indexing='ij')
    return X_w * Y_w * Z_w

def process_single_case(args):
    """单案例后处理重构：执行 Scheme 2+4 并精确生成物理指标与 Npy 阵"""
    case_id, pred_dir, temp_dir, nx, ny, nz, case_out_dir = args
    pred_path = os.path.join(pred_dir, case_id, 'T_pred_autoregressive.npy')
    true_path = os.path.join(temp_dir, case_id, 'T_transient.npy')
    
    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        return None
        
    T_pred_seq = np.load(pred_path) 
    T_true_seq = np.load(true_path)
    num_t = T_pred_seq.shape[0]
    
    W_std = get_3d_confidence_mask(nx, ny, nz)
    
    smoothed_frames = []
    T_0_const = np.full((nx, ny, nz), T_INIT_PHYSICAL, dtype=np.float32)
    smoothed_frames.append(T_0_const)
    
    case_metrics = []
    
    for t in range(1, num_t):
        T_true = T_true_seq[t]
        t_raw = T_pred_seq[t]
        
        t_s2 = run_scheme2_spr(t_raw, nx, ny, nz)
        t_s24 = run_scheme4_l2(t_s2, W_std)
        smoothed_frames.append(t_s24)
        
        abs_err_t = np.abs(t_s24 - T_true)
        temp_mae_abs = np.mean(abs_err_t)
        temp_mse_abs = np.mean(abs_err_t**2)
        temp_rmse_abs = np.sqrt(temp_mse_abs)
        
        rel_err_t = abs_err_t / T_true
        temp_mae_rel = np.mean(rel_err_t)
        temp_mse_rel = np.mean(rel_err_t**2)
        temp_rmse_rel = np.sqrt(temp_mse_rel)
        
        temp_max_err = np.max(abs_err_t)
        true_max_t = np.max(T_true)
        pred_max_t = np.max(t_s24)
        
        case_metrics.append({
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
            "Pred_Max_T": pred_max_t
        })
        
    np.save(os.path.join(case_out_dir, 'T_pred_autoregressive.npy'), np.stack(smoothed_frames))
    
    df_case = pd.DataFrame(case_metrics)
    df_case.to_csv(os.path.join(case_out_dir, 'Autoregressive_Metrics.csv'), index=False)
    
    plot_frames = [1, num_t // 2, num_t - 1]
    fig, axes = plt.subplots(len(plot_frames), 3, figsize=(16, 5 * len(plot_frames)))
    fig.suptitle(f'Autoregressive Prediction (N1 Model + Scheme 2+4)\nCase: {case_id}', fontsize=18)
    
    slice_z = nz // 2 
    for row, frame in enumerate(plot_frames):
        if frame >= num_t: break
        truth = T_true_seq[frame][:, :, slice_z]
        pred = smoothed_frames[frame][:, :, slice_z]
        err = np.abs(truth - pred)
        vmin, vmax = truth.min(), truth.max()
        
        im0 = axes[row, 0].imshow(truth, cmap='jet', vmin=vmin, vmax=vmax); axes[row, 0].set_title(f'Frame {frame} - Truth', fontsize=14)
        im1 = axes[row, 1].imshow(pred, cmap='jet', vmin=vmin, vmax=vmax); axes[row, 1].set_title(f'Frame {frame} - Smoothed', fontsize=14)
        im2 = axes[row, 2].imshow(err, cmap='magma'); axes[row, 2].set_title(f'Frame {frame} - Abs Error', fontsize=14)
        
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.46, pad=0.04)
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.46, pad=0.04)
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.46, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(case_out_dir, 'Autoregressive_CrossSection.png'), dpi=200)
    plt.close()
    
    del T_pred_seq, T_true_seq, smoothed_frames; gc.collect()
    return case_metrics

def run_global_smoothing_generation():
    print(f"\n{'='*80}\n[Smoothing Generation] 启动全局 Scheme 2+4 离线平滑与对齐引擎")
    print(f"-> 待处理任务数: {len(TASKS)} 个任务")
    print("="*80 + "\n")
    
    for task in TASKS:
        task_name = task["name"]
        pred_dir = task["pred_dir"]
        temp_dir = task["temp_dir"]
        nx, ny, nz = task["nx"], task["ny"], task["nz"]
        
        if not os.path.exists(pred_dir):
            print(f"   [Error] 找不到预测目录 {pred_dir}，已跳过。")
            continue
            
        case_folders = sorted([f for f in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, f))])
        
        task_folder_name = os.path.basename(pred_dir)
        task_out_root = os.path.join(OUTPUT_RESULTS_DIR, task_folder_name)
        os.makedirs(task_out_root, exist_ok=True)
        
        print(f"   [-] 任务: {task_name} | 正在输出至: {task_out_root}")
        print(f"   [-] 装配 2 进程并行后处理...")
        
        args_list = []
        for case_id in case_folders:
            case_out_dir = os.path.join(task_out_root, case_id)
            os.makedirs(case_out_dir, exist_ok=True)
            args_list.append((case_id, pred_dir, temp_dir, nx, ny, nz, case_out_dir))
            
        task_global_metrics = []
        with Pool(processes=NUM_WORKERS) as pool:
            results = list(tqdm(pool.imap(process_single_case, args_list), total=len(args_list), desc="重构进度", leave=False))
            
        for res in results:
            if res is not None:
                task_global_metrics.extend(res)
                
        df_global = pd.DataFrame(task_global_metrics)
        global_csv_path = os.path.join(task_out_root, 'Global_Autoregressive_Metrics.csv')
        df_global.to_csv(global_csv_path, index=False)
        
        avg_mae = df_global['Temp_MAE_Abs'].mean()
        avg_rel_mae = df_global['Temp_MAE_Rel'].mean() * 100
        max_err = df_global['Temp_Max_Err'].max()
        print(f"   [✓] {task_name} 重建成功！物理平均 MAE: {avg_mae:.4f} K (平均相对误差: {avg_rel_mae:.4f}%) | 极端最大温差: {max_err:.2f} K")
        print(f"       全局汇总大表已保存: {global_csv_path}")
        
        del df_global, task_global_metrics; gc.collect()

    print("\n" + "="*80)
    print(f"[!] 后处理 Scheme 2+4 物理重建全部成功完成！")
    print(f"[!] 4D预测矩阵(.npy)、9维指标 CSV 及对比云图已完全保存至: {OUTPUT_RESULTS_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_global_smoothing_generation()