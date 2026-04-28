import numpy as np
import torch
import torchcde
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pickle
import os
import tempfile
from functools import partial
from torchdiffeq import odeint
from utils.utils_ode import find_continue_idx, evaluate_feature, get_data, normalize, seed_everything, sample_data_with_nans


def compute_single_hermite(args):
    """
    单个样本的Hermite系数计算 - 进程池worker函数
    
    Args:
        args: (sample_data, sample_idx) 
              sample_data: numpy array [T, D]
              sample_idx: int, 样本索引
    
    Returns:
        (sample_idx, coeffs_np) 或 (sample_idx, None) 如果失败
    """
    sample_data, sample_idx = args
    
    try:
        # 转换为tensor
        x_tensor = torch.from_numpy(sample_data).float().unsqueeze(0)  # [1, T, D]
        
        # 计算Hermite系数
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_tensor)
        
        # 转换为numpy并返回
        coeffs_np = coeffs.numpy()[0]  # 移除batch维度
        
        # 清理内存
        # del x_tensor, coeffs
        
        return (sample_idx, coeffs_np)
        
    except Exception as e:
        print(f"样本 {sample_idx} 计算失败: {e}")
        return (sample_idx, None)

def compute_hermite_coeffs_multiprocess(x_norm, num_workers=None, chunk_size=50, verbose=True):
    """
    多进程计算Hermite系数
    
    Args:
        x_norm: numpy array [N, T, D]
        num_workers: 进程数，None表示自动选择
        chunk_size: 每个进程一次处理的样本数
        verbose: 是否显示进度
    
    Returns:
        coeffs: numpy array [N, T-1, D*C] 或 None如果全部失败
    """
    N, T, D = x_norm.shape
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # 限制最大进程数避免系统过载
    
    if verbose:
        print(f"多进程计算Hermite系数: {N}样本, {num_workers}进程, chunk_size={chunk_size}")
    
    # 准备任务参数
    tasks = [(x_norm[i], i) for i in range(N)]
    
    # 预分配结果数组
    result_shape = (N, T-1, D*4)  # Hermite系数通常是D*4维
    coeffs_result = np.zeros(result_shape, dtype=np.float32)
    
    failed_samples = []
    completed_count = 0
    
    try:
        # 使用进程池并行计算
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(compute_single_hermite, task): task[1] 
                for task in tasks
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_idx), 
                             total=N, 
                             desc="Computing Hermite coeffs"):
                sample_idx = future_to_idx[future]
                
                try:
                    idx, coeffs_np = future.result()
                    
                    if coeffs_np is not None:
                        coeffs_result[idx] = coeffs_np
                        completed_count += 1
                    else:
                        failed_samples.append(idx)
                        
                except Exception as e:
                    print(f"处理样本 {sample_idx} 结果时出错: {e}")
                    failed_samples.append(sample_idx)
    
    except Exception as e:
        print(f"多进程计算过程中出现错误: {e}")
        return None
    
    if verbose:
        print(f"计算完成: 成功 {completed_count}/{N}, 失败 {len(failed_samples)}")
        if failed_samples:
            print(f"失败的样本索引: {failed_samples[:10]}..." if len(failed_samples) > 10 else f"失败的样本索引: {failed_samples}")
    
    # 如果大部分样本都失败了，返回None
    if completed_count < N * 0.5:  # 少于50%成功
        print("警告: 超过50%的样本计算失败，可能需要检查数据质量")
        return None
    
    return coeffs_result

def compute_hermite_coeffs_chunked_multiprocess(x_norm, num_workers=None, chunk_size=100, verbose=True):
    """
    分块多进程计算Hermite系数 - 更保守的内存管理策略
    
    Args:
        x_norm: numpy array [N, T, D]
        num_workers: 进程数
        chunk_size: 每块的样本数
        verbose: 是否显示进度
    
    Returns:
        coeffs: numpy array [N, T-1, D*C]
    """
    N, T, D = x_norm.shape
    
    if num_workers is None:
        num_workers = min(mp.cpu_count() // 2, 6)  # 更保守的进程数
    
    if verbose:
        print(f"分块多进程计算: {N}样本, {num_workers}进程, 块大小={chunk_size}")
    
    # 预分配结果数组
    result_shape = (N, T-1, D*4)
    coeffs_result = np.zeros(result_shape, dtype=np.float32)
    
    # 分块处理
    num_chunks = (N + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, N)
        chunk_data = x_norm[start_idx:end_idx]
        
        if verbose:
            print(f"处理块 {chunk_idx+1}/{num_chunks}: 样本 {start_idx}-{end_idx-1}")
        
        # 对当前块使用多进程计算
        chunk_coeffs = compute_hermite_coeffs_multiprocess(
            chunk_data, 
            num_workers=num_workers, 
            verbose=False
        )
        
        if chunk_coeffs is not None:
            coeffs_result[start_idx:end_idx] = chunk_coeffs
        else:
            print(f"块 {chunk_idx+1} 计算失败，使用零填充")
        
        # 强制垃圾回收
        gc.collect()
    
    return coeffs_result

def compute_hermite_coeffs_robust(x_norm, strategy='auto', **kwargs):
    """
    鲁棒的Hermite系数计算 - 自动选择最佳策略
    
    Args:
        x_norm: numpy array [N, T, D]
        strategy: 'auto', 'multiprocess', 'chunked', 'sequential'
        **kwargs: 传递给具体计算函数的参数
    
    Returns:
        coeffs: numpy array [N, T-1, D*C]
    """
    N, T, D = x_norm.shape
    data_size_mb = x_norm.nbytes / (1024 * 1024)
    
    print(f"数据大小: {data_size_mb:.1f} MB")
    
    if strategy == 'auto':
        if data_size_mb < 100:  # 小数据集，直接多进程
            strategy = 'multiprocess'
        elif data_size_mb < 500:  # 中等数据集，分块多进程
            strategy = 'chunked'
        else:  # 大数据集，小块处理
            strategy = 'chunked'
            kwargs.setdefault('chunk_size', 50)
    
    print(f"选择策略: {strategy}")
    
    try:
        if strategy == 'multiprocess':
            return compute_hermite_coeffs_multiprocess(x_norm, **kwargs)
        elif strategy == 'chunked':
            return compute_hermite_coeffs_chunked_multiprocess(x_norm, **kwargs)
        elif strategy == 'sequential':
            return compute_hermite_coeffs_ultra_safe_v2(x_norm, **kwargs)
        else:
            raise ValueError(f"未知策略: {strategy}")
            
    except Exception as e:
        print(f"策略 {strategy} 失败: {e}")
        
        # 回退策略
        if strategy != 'sequential':
            print("回退到顺序计算...")
            return compute_hermite_coeffs_ultra_safe_v2(x_norm, verbose=kwargs.get('verbose', True))
        else:
            raise

def compute_hermite_coeffs_ultra_safe_v2(x_norm, verbose=True, save_checkpoint=True):
    """
    改进的安全顺序计算，支持断点续算
    """
    N, T, D = x_norm.shape
    result_shape = (N, T-1, D*4)
    coeffs_result = np.zeros(result_shape, dtype=np.float32)
    
    if verbose:
        print(f"安全顺序计算: {N} 样本")
    
    # 断点续算支持
    checkpoint_file = None
    if save_checkpoint:
        checkpoint_file = tempfile.mktemp(suffix='_hermite_checkpoint.npy')
    
    failed_count = 0
    
    for i in tqdm(range(N), desc="Computing Hermite coeffs"):
        try:
            sample = x_norm[i:i+1]  # [1, T, D]
            x_tensor = torch.from_numpy(sample).float()
            
            sample_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_tensor)
            coeffs_result[i] = sample_coeffs.numpy()[0]
            
            # 清理
            del x_tensor, sample_coeffs, sample
            
            # 定期保存检查点
            if save_checkpoint and (i + 1) % 200 == 0:
                np.save(checkpoint_file, coeffs_result[:i+1])
            
            # 定期清理内存
            if (i + 1) % 100 == 0:
                gc.collect()
                
        except Exception as e:
            if verbose and failed_count < 10:  # 只打印前10个错误
                print(f"样本 {i} 失败: {e}")
            failed_count += 1
            continue
    
    if verbose:
        print(f"计算完成，失败样本数: {failed_count}")
    
    # 清理检查点文件
    if checkpoint_file and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    return coeffs_result



#################采用逐样本并行策略, 每个进程只处理一个样本#################
# ——— 1) 定义 ESN_Func ———
class ESN_Func(torch.nn.Module):
    def __init__(self, input_dim, n_reservoir, leaky, activation=torch.tanh):
        super().__init__()
        self.n_reservoir = n_reservoir
        self.leaky       = leaky
        self.activation  = activation

        # 先初始化占位，用后面覆盖
        W_in  = torch.zeros(input_dim, n_reservoir)
        W_res = torch.zeros(n_reservoir, n_reservoir)
        self.W_in  = torch.nn.Parameter(W_in,  requires_grad=False)
        self.W_res = torch.nn.Parameter(W_res, requires_grad=False)
        self.register_buffer('X_eval', None)

    def set_weights(self, W_in_np, W_res_np):
        """载入外部计算好的 numpy 权重"""
        self.W_in.data.copy_(torch.from_numpy(W_in_np))
        self.W_res.data.copy_(torch.from_numpy(W_res_np))

    def set_spline(self, cubic_spline):
        self.X_eval = cubic_spline

    def forward(self, t, z):
        X_t = self.X_eval.evaluate(t)            # [batch?, input_dim]
        pre = X_t @ self.W_in + z @ self.W_res   # [batch?, n_reservoir]
        return -self.leaky * z + self.activation(pre)

# ——— 2) worker：对单条样本一次性积分 + Ridge ———
def _worker_run_ode(args):
    """
    args: (coeff_np, x_miss_np, ts_np, W_in_np, W_res_np,
           leaky, method, rtol, atol, alpha)
    返回: W_out_np,   shape [n_reservoir, input_dim]
    """
    (coeff_np, x_miss_np, ts_np,
     W_in_np, W_res_np,
     leaky, method, rtol, atol, alpha, nForgetPoints) = args

    # 1) 重建 model.func
    input_dim   = W_in_np.shape[0]
    n_reservoir = W_res_np.shape[0]
    func = ESN_Func(input_dim, n_reservoir, leaky)
    func.set_weights(W_in_np, W_res_np)

    # 2) 准备数据
    coeff   = torch.from_numpy(coeff_np)       # [T, D, C]
    x_miss  = torch.from_numpy(x_miss_np)      # [T, D]
    ts      = torch.from_numpy(ts_np).float()  # [T]

    # 3) 样条 & 一次性积分
    spline = torchcde.CubicSpline(coeff)
    func.set_spline(spline)
    z0 = torch.zeros(n_reservoir)  # [n_reservoir]
    # odeint 得到 [T, n_reservoir]
    z = odeint(func, z0, ts, method=method, rtol=rtol, atol=atol)
    # 4) 统一 nonlinear
    # a = z.copy()
    # a_np = a.numpy()
    # print(a_np)
    Z = torch.tanh(z[1:])  # [T, R]

    # 5) 找出连续可预测时刻
    origin_idx, target_idx = find_continue_idx(ts, nForgetPoints)
    
    # 6) Ridge 回归
    Xf = Z[origin_idx]         # [L, R]
    data_idxs = ts[target_idx].long()   # [L]
    Y = x_miss[data_idxs]               # [L, D]
    R = Xf.size(1)
    # # ——— NaN 检测 ———
    # if torch.isnan(Xf).any() :
    #     raise RuntimeError(f"NaN in Xf :{Xf}")
    # if torch.isnan(Y).any():
    #     raise RuntimeError(f"NaN in Y :{Y}")
    
    # 1) 计算 Xf^T Xf  和  Xf^T Y
    XtX = Xf.T @ Xf        # [R, R]
    XtY = Xf.T @ Y         # [R, D]

    # 2) 构造 α I
    I = torch.eye(R, device=Xf.device) * alpha  # [R, R]

    # 3) 求解 (XtX + α I) W = XtY
    W_out = torch.linalg.solve(XtX + I, XtY)    # [R, D]

    return W_out.numpy()

# ——— 3) 并行推理主函数 ———
def run_neuralESN_parallel(
    W_in_np, W_res_np,
    coeffs, x_miss, ts,
    leaky, method, rtol, atol, alpha, nForgetPoints,
    num_workers=8
):
    """
    Inputs:
      - W_in_np:    numpy [D_in, R]
      - W_res_np:   numpy [R, R]
      - coeffs:     numpy array [N, T, D_in, C]
      - x_miss:     numpy array [N, T, D_in]
      - ts:         numpy array [N, T]
    Returns:
      - W_out_all:  numpy [N, R, D_in]
    """
    N = coeffs.shape[0]
    # prepare args list
    tasks = []
    for i in range(N):
        tasks.append((
            coeffs[i],        # [T, D, C]
            x_miss[i],        # [T, D]
            ts[i],            # [T]
            W_in_np, W_res_np,
            leaky, method, rtol, atol, alpha, nForgetPoints
        ))

    W_out_list = [None] * N
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = {exe.submit(_worker_run_ode, t): i for i, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=N, desc="Parallel ODE-ESN"):
            idx = futures[fut]
            W_out_list[idx] = fut.result()

    return np.stack(W_out_list, axis=0)  # [N, R, D]
