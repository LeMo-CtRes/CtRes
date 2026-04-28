import os
import csv
import time
import h5py
import torch
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline

import matplotlib.patches as mpatches

def seed_everything(seed):
    # （可选）固定 Python 的 hash 随机化
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Python 内置 random
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch 所有 GPU
    torch.cuda.manual_seed_all(seed)
    # cuDNN: 保证确定性，关闭 benchmark
    torch.backends.cudnn.deterministic    = True
    torch.backends.cudnn.benchmark        = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def normalize_meanmax(data):
    """
    使用最大值进行归一化: (data - mean) / max_abs_val
    
    参数:
        data: 张量，形状为 [num_of_data, time_len, input_size]
    
    返回:
        归一化后的数据
    """
    mean = torch.nanmean(data, dim=1, keepdim=True)
    max_val = torch.nanmax(torch.abs(data - mean), dim=1, keepdim=True)[0]  # 获取最大偏差
    normalized = (data - mean) / (max_val + 1e-8)
    return normalized


def normalize(data):
    """
    对数据进行 Z-score 归一化
    
    参数:
        data: 张量，形状为 [num_of_data, time_len, input_size]
    
    返回:
        归一化后的数据
    """
    if isinstance(data, np.ndarray):
        # 对numpy数组进行处理
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif isinstance(data, torch.Tensor):
        # 计算非NaN值的均值
        mask = ~torch.isnan(data)
        count = mask.sum(dim=0, keepdim=True).float()
        sum_val = torch.where(mask, data, torch.zeros_like(data)).sum(dim=0, keepdim=True)
        mean = sum_val / count
        
        # 计算非NaN值的标准差
        sum_squared_diff = torch.where(mask, (data - mean) ** 2, torch.zeros_like(data)).sum(dim=0, keepdim=True)
        std = torch.sqrt(sum_squared_diff / count)
        
        # 归一化数据，保留NaN
        normalized = torch.where(mask, (data - mean) / (std + 1e-8), data)
        
        return normalized
    else:
        raise TypeError("Input must be either numpy array or torch tensor")


def get_data(dataset):
    x_train = torch.from_numpy(dataset.x_train).to(torch.float32) # (time_series_num, series_len, sensors_num)
    x_test = torch.from_numpy(dataset.x_test).to(torch.float32)

    if dataset.y_train.ndim > 1:
        y_train = torch.from_numpy(dataset.y_train.argmax(axis=1)).to(torch.long)  # 10分类
        y_test = torch.from_numpy(dataset.y_test.argmax(axis=1)).to(torch.long) 
    else:
        y_train = torch.from_numpy(dataset.y_train).to(torch.long)
        y_test = torch.from_numpy(dataset.y_test).to(torch.long)

    time_dim_train = torch.arange(x_train.size(1)).reshape(1, -1, 1).expand(x_train.size(0), -1, 1).to(torch.float32)
    time_dim_test = torch.arange(x_test.size(1)).reshape(1, -1, 1).expand(x_test.size(0), -1, 1).to(torch.float32)
    assert(time_dim_train.size(1) == time_dim_test.size(1))

    # x_train = torch.cat((time_dim_train, x_train), 2)
    # x_test = torch.cat((time_dim_test, x_test), 2)
    time_stamp = time_dim_train[0].squeeze(-1)

    return x_train, y_train, x_test, y_test, time_stamp


def get_data_mat(data_path, input_scaling=1):
    time_series_data = []
    sampled_timestamps = []
    labels = []

    # Open the file to access each time series and label entry
    with h5py.File(data_path, 'r') as mat_data:
        # Loop through each entry to retrieve time series, timestamps, and labels
        for i in range(400):
            # Extracting each time series (1024x2 double) and time (1024x1 double)
            time_series = np.array(mat_data[mat_data['data'][0][i]])
            timestamp = np.array(mat_data[mat_data['time'][0][i]]).flatten() - 1
            label = int(mat_data['label'][0][i])  # Correcting label extraction

            # Determine full time range based on max timestamp
            max_time = int(np.max(timestamp))
            full_series = np.full((time_series.shape[0], max_time+1), np.nan)  # Initialize with NaN for shape (2, max_time + 1)
            
            # Populate available data into full series based on timestamps
            full_series[:, timestamp.astype(int)] = time_series

            # Append processed data
            time_series_data.append(full_series)
            sampled_timestamps.append(timestamp)  # Full timestamp
            labels.append(label)


    # Convert lists to numpy arrays for easier manipulation
    x_data = np.array(time_series_data).transpose(0, 2, 1)  # Shape: (400, max_time+1, 2)
    y_data = np.array(labels)
    time_stamp = np.array(sampled_timestamps)
    # Split dataset
    x_train, x_test, y_train, y_test, time_stamp_train, time_stamp_test = train_test_split(
        x_data, y_data, time_stamp, test_size=0.5, random_state=42, stratify=y_data
    )
    # 转成 torch.Tensor
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    time_stamp_train = torch.from_numpy(time_stamp_train)
    time_stamp_test = torch.from_numpy(time_stamp_test)  
    return x_train, y_train, x_test, y_test, time_stamp_train, time_stamp_test



def find_continue_idx(ts: torch.Tensor, nForgetPoints: int):
    """
    ts: 1D 张量，长度 M， 是“非 NaN”点对应的原始时间戳（已升序，无重复）。
    nForgetPoints: wash-out 个数。

    返回 origin_idx, target_idx：
      - origin_idx 中的 i 表示：第 i 个 ts 点（即 Z[i]）是用来做回归的隐状态；
      - target_idx 中的 j 表示：第 j 个 ts 点（即 x_miss[ts[j]]）是对应的“下一步”观测。
    """
    origin_idx = []
    target_idx = []
    
    # 找出所有相邻时间戳差为1的点
    for i in range(nForgetPoints, len(ts) - 1):
        if ts[i+1] - ts[i] == 1:
            # 对Z的索引计算（Z从第1个点开始，因此索引减1）
            origin_idx.append(i - 1)
            # 对应的目标索引
            target_idx.append(i)
    
    return origin_idx, target_idx




def sample_data_with_nans(data, retain_rate, generator=None):
    """
    对时间序列数据进行随机缺失采样，使用NaNs填充缺失部分，保留原始张量形状。
    
    参数：
        data : torch.Tensor
            时间序列数据，维度为 [series_num, length, dim]。
        retain_rate : float
            数据的保留百分比（0-1之间）。
    
    返回：
        sampled_data : torch.Tensor
            随机缺失后的数据，维度为 [series_num, length, dim]。
            缺失部分使用NaNs替换。
        sampled_data_without_NaN : torch.Tensor
            连续的不带NaN的裁剪数据，维度为 [series_num, retain_length, dim]。
        time_stamps : torch.Tensor
            每条序列保留的时间戳索引，维度为 [series_num, retain_length]。
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    if data.ndim < 3:
        data = data.unsqueeze(2)
        
    retain_rate = float(retain_rate)

    series_num, length, dim = data.shape
    retain_length = int(length * retain_rate)  # 计算保留的元素数量

    # 初始化输出数据为原数据的拷贝
    sampled_data = data.clone()
    time_stamps = []
    sampled_data_without_NaN = []
    
    # 如果没有传 generator，就自己 new 一个
    if generator is None:
        generator = torch.Generator().manual_seed(42)

    # 每一条数据独立进行随机缺失
    for i in range(series_num):
        if length <= 2:
            # 如果长度小于等于2，就没有可以随机选择的中间部分，直接返回原数据
            time_stamps.append(torch.arange(length, device=data.device))
            sampled_data_without_NaN.append(data[i])
            continue
            
        # 确保没有0和length-1
        indices = torch.arange(1, length-1, device=data.device)
        perm = torch.randperm(indices.size(0), generator=generator)  # torch.randperm(length) 生成一个从 0 到 length-1 的随机排列序列
        chosen = perm[: retain_length - 2]
        # 包含开头和结尾的索引
        chosen_indices = torch.cat([
            torch.tensor([0], device=data.device),
            indices[chosen],
            torch.tensor([length - 1], device=data.device),
        ])
        
        # 对选择的索引进行排序
        chosen_indices_sorted = chosen_indices.sort().values
        
        # 构造掩码，将未选择的索引位置设置为NaN
        mask = torch.ones(length, dtype=torch.bool, device=data.device)
        mask[chosen_indices] = False  # 被保留的设置为False
        sampled_data[i, mask, :] = float('nan')  # 将未保留的位置替换为NaN
        
        # 提取连续的不带NaN的数据
        continuous_data = data[i, chosen_indices_sorted, :]
        
        time_stamps.append(chosen_indices_sorted)
        sampled_data_without_NaN.append(continuous_data)

    # 将列表转换为张量
    time_stamps = torch.stack(time_stamps, dim=0)
    sampled_data_without_NaN = torch.stack(sampled_data_without_NaN, dim=0)
    
    return sampled_data, sampled_data_without_NaN, time_stamps

def evaluate_classifiers_and_tsne_from_npz(npz_path, save_dir, logger, seed=42):
    """
    从 .npz 文件中读取 W_all、y_all、Ntr，
    分别用 RandomForest、SVM、KNN 三种分类器做分类并输出评估结果到 CSV，
    并对全部样本做 t-SNE 降维，绘制 2D 散点图保存为 PNG。

    参数:
        npz_path: str or Path，包含 W_all, y_all, Ntr 的 .npz 文件路径
        save_dir: str or Path，结果保存目录
        seed: int, 随机种子
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取数据
    data = np.load(npz_path)
    W_all = data['W_all']    # shape [N, R, D]
    y_all = data['y_all']    # shape [N,]
    Ntr   = int(data['Ntr']) # 训练样本数量

    # 2) 展平并切分
    X = W_all.reshape(W_all.shape[0], -1)
    X_train, X_test = X[:Ntr], X[Ntr:]
    y_train, y_test = y_all[:Ntr], y_all[Ntr:]

    # 3) 定义分类器
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=seed),
        'SVM':          SVC(kernel='rbf', decision_function_shape='ovo', random_state=seed),
        'LinearSVM':    LinearSVC(random_state=seed),
        'KNN':          KNeighborsClassifier(n_neighbors=5),
    }

    # 4) 写 CSV header（若不存在）
    clf_res_file = save_dir / 'classification_results.csv'
    write_header = not clf_res_file.exists()
    with open(clf_res_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'classifier', 'train_acc', 'test_acc',
                'precision', 'recall', 'f1_score',
                'fit_time', 'pred_time'
            ])

        # 5) 逐个分类器训练、评估、写入
        for name, clf in classifiers.items():
            t0 = time.time()
            clf.fit(X_train, y_train)
            fit_time = time.time() - t0

            t1 = time.time()
            test_pred = clf.predict(X_test)
            pred_time = time.time() - t1

            train_pred   = clf.predict(X_train)
            train_acc    = accuracy_score(y_train, train_pred)
            test_acc     = accuracy_score(y_test,  test_pred)
            precision    = precision_score(y_test, test_pred, average='macro', zero_division=0)
            recall       = recall_score(y_test,    test_pred, average='macro', zero_division=0)
            f1           = f1_score(y_test,       test_pred, average='macro', zero_division=0)

            writer.writerow([
                name, train_acc, test_acc,
                precision, recall, f1,
                fit_time, pred_time
            ])
            logger.info(f"{name}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
                        f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")

    # 6) t-SNE 可视化并保存
    tsne = TSNE(n_components=2, random_state=seed, init='pca')
    X_emb = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for lbl in np.unique(y_all):
        idxs = (y_all == lbl)
        plt.scatter(X_emb[idxs, 0], X_emb[idxs, 1],
                    s=10, alpha=0.7, label=str(lbl))
    plt.legend(title='Label')
    plt.title('t-SNE of ESN representations')

    tsne_path = save_dir / 'tsne_plot.png'
    plt.savefig(tsne_path)
    plt.close()
    logger.info(f"Saved t-SNE plot to {tsne_path}")


def plot_from_csv_new(csv_path, rpm_name, smooth_window=5):
    # 全局字体设置
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    df = pd.read_csv(csv_path)
    epochs      = df['epoch']
    train_loss  = df['avg_loss']
    train_acc   = df['train_acc']
    ev          = df.dropna(subset=['test_acc'])
    eval_epochs = ev['epoch']
    test_acc    = ev['test_acc']

    # 平滑与标准差
    loss_smooth = train_loss.rolling(window=smooth_window, min_periods=1).mean()
    loss_std    = train_loss.rolling(window=smooth_window, min_periods=1).std().fillna(0)
    acc_smooth  = train_acc.rolling(window=smooth_window, min_periods=1).mean()
    acc_std     = train_acc.rolling(window=smooth_window, min_periods=1).std().fillna(0)

    # —— 主图（不含 legend）——
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Train Loss', fontsize=16, color='tab:red')
    l1, = ax1.plot(epochs, loss_smooth, color='tab:red', linewidth=2, label='Train Loss (smoothed)')
    ax1.fill_between(
        epochs,
        loss_smooth - loss_std,
        loss_smooth + loss_std,
        color='tab:red', alpha=0.2,
        label=f'Loss ±1 std (w={smooth_window})'
    )
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
    ax1.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', fontsize=16, color='tab:blue')
    l2, = ax2.plot(epochs, acc_smooth, color='tab:blue', linewidth=2, label='Train Acc (smoothed)')
    ax2.fill_between(
        epochs,
        acc_smooth - acc_std,
        acc_smooth + acc_std,
        color='tab:blue', alpha=0.2,
        label=f'Acc ±1 std (w={smooth_window})'
    )
    l3, = ax2.plot(
        eval_epochs, test_acc,
        linestyle='-', marker='o', markersize=6,
        color='tab:green', linewidth=2,
        label='Test Acc'
    )
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)

    # plt.title(f'', fontsize=18)

    # 保存无 legend 的主图
    save_dir = os.path.dirname(csv_path)
    main_path = os.path.join(save_dir, f'loss_acc_{rpm_name}_main.pdf')
    fig.savefig(main_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Main plot saved to: {main_path}")

    # —— 单独画 Legend ——
    # 收集 handles/labels
    patch_loss = mpatches.Patch(
        color='tab:red', alpha=0.2,
        label=f'Loss ±1 std (w={smooth_window})'
    )
    patch_acc = mpatches.Patch(
        color='tab:blue', alpha=0.2,
        label=f'Acc ±1 std (w={smooth_window})'
    )

    handles = [l1, l2, l3, patch_loss, patch_acc]
    labels  = [h.get_label() for h in handles]

    # 新 Figure 仅用于 legend
    fig_leg = plt.figure(figsize=(8, 2))
    leg = fig_leg.legend(
        handles, labels,
        loc='center', ncol=3,
        prop={'family':'Times New Roman', 'size':16},
        frameon=False
    )
    fig_leg.canvas.draw()
    fig_leg.patch.set_visible(False)
    plt.axis('off')

    legend_path = os.path.join(save_dir, f'loss_acc_{rpm_name}_legend.pdf')
    fig_leg.savefig(legend_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_leg)
    print(f"Legend saved to: {legend_path}")

def plot_sr(results, file_name="rho_vs_acc.png", dir_path=str):
    # 输出目录
    # dir_path = r"E:\weichy\ODECode\src\output\Plot\ode"
    os.makedirs(dir_path, exist_ok=True)

    # 将 results 解包为三个列表
    spectral_radii, train_accs, test_accs = zip(*results)

    # 绘图
    plt.figure()
    plt.plot(spectral_radii, train_accs, marker='o', label='Train Accuracy')
    plt.plot(spectral_radii, test_accs,  marker='s', label='Test Accuracy')
    plt.xlabel("spectral_radius")
    plt.ylabel("accuracy")
    plt.title("Spectral Radius vs Accuracy")
    plt.legend()              # 添加图例
    plt.tight_layout()

    # 保存并关闭
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path, dpi=300)
    plt.close()




def plot_sr_plus(results,
            filename="rho_vs_metrics.png",
            dir_path="./plots"):
    """
    绘制增强的结果图，包含多个指标：
      1. 准确率（训练 & 测试）
      2. 精确率 & F1 分数
      3. 平均 ODE 积分时间
      4. 分类器拟合 & 预测时间
    """
    if not results:
        print("No results to plot.")
        return

    # 确保输出目录存在
    os.makedirs(dir_path, exist_ok=True)

    # 解包数据
    srs, train_accs, test_accs, precisions, recalls, f1_scores, avg_ode_times, fit_times, pred_times = zip(*results)

    # 创建 2x2 子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 准确率
    ax1.plot(srs, train_accs, marker='o', label='Train Accuracy')
    ax1.plot(srs, test_accs,  marker='s', label='Test Accuracy')
    ax1.set_xlabel('Spectral Radius')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Spectral Radius')
    ax1.grid(True)
    ax1.legend()

    # 2. 精确率 & F1 分数
    ax2.plot(srs, precisions, marker='^', label='Precision')
    ax2.plot(srs, recalls,  marker='s', label='Recall')
    ax2.plot(srs, f1_scores,  marker='d', label='F1 Score')
    ax2.set_xlabel('Spectral Radius')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision, Recall & F1 Score vs Spectral Radius')
    ax2.grid(True)
    ax2.legend()

    # 3. 平均 ODE 积分时间
    ax3.plot(srs, avg_ode_times, marker='o', label='Avg ODE Time')
    ax3.set_xlabel('Spectral Radius')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Average ODE Integration Time vs Spectral Radius')
    ax3.grid(True)
    ax3.legend()

    # 4. 分类器时间
    ax4.plot(srs, fit_times,  marker='s', label='Fit Time')
    ax4.plot(srs, pred_times, marker='^', label='Prediction Time')
    ax4.set_xlabel('Spectral Radius')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Classifier Time vs Spectral Radius')
    ax4.grid(True)
    ax4.legend()

    # 布局与保存
    plt.tight_layout()
    out_path = os.path.join(dir_path, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


def plot_retain_rate_comparison(rpm_results, rpm_name, retain_rates, output_dir, logger):
    """
    绘制不同retain_rate的对比图
    """
    if not rpm_results:
        logger.warning(f"No results to plot for {rpm_name}")
        return
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(retain_rates)))
    
    for i, retain_rate in enumerate(retain_rates):
        if retain_rate in rpm_results and rpm_results[retain_rate]:
            srs = [r[0] for r in rpm_results[retain_rate]]
            test_accs = [r[2] for r in rpm_results[retain_rate]]
            plt.plot(srs, test_accs, marker='o', color=colors[i], 
                   label=f'retain_rate={retain_rate:.1f}', linewidth=2)
    
    plt.xlabel('Spectral Radius')
    plt.ylabel('Test Accuracy')
    plt.title(f'RPM {rpm_name}: Impact of Retain Rate on Performance')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = output_dir / f"retain_rate_comparison_{rpm_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Retain rate comparison plot saved: {plot_path}")

def plot_complete_overview(all_results, rpm_names, retain_rates, output_dir, logger):
    """
    创建完整的总览图
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for rpm_idx, rpm_name in enumerate(rpm_names):
        if rpm_idx >= len(axes):
            break
            
        ax = axes[rpm_idx]
        
        if rpm_name in all_results:
            for retain_rate in retain_rates:
                if retain_rate in all_results[rpm_name]:
                    results = all_results[rpm_name][retain_rate]
                    if results:
                        srs = [r[0] for r in results]
                        test_accs = [r[2] for r in results]
                        ax.plot(srs, test_accs, marker='o', label=f'{retain_rate:.1f}', linewidth=1.5)
        
        ax.set_title(f'RPM {rpm_name}')
        ax.set_xlabel('Spectral Radius')
        ax.set_ylabel('Test Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Retain Rate', fontsize=8)
    
    # 隐藏多余的子图
    for i in range(len(rpm_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    overview_path = output_dir / "complete_overview.png"
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Complete overview plot saved: {overview_path}")



def plot_loss_accuracy(file_path='loss.txt'):
    epochs = []
    losses = []
    accuracies = []

    # 从文件中读取数据
    with open(file_path, 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            epoch, time, loss, accuracy = line.strip().split(',') # 没用到time
            epochs.append(int(epoch))
            losses.append(float(loss))
            accuracies.append(float(accuracy))

    # 绘制loss曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid()

    # 绘制accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Training Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('/home/weichy/ODE_RES/src/Plot/ode/loss_accuracy.png')  # 保存图像
    # plt.show()  # 显示图像

def evaluate_feature(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    svm_model = SVC(kernel='rbf')  # 定义 SVM 模型，使用默认的 RBF kernel
    svm_model.fit(x_train, y_train)  # 训练 SVM 模型
    predictions = svm_model.predict(x_test)  # 预测并评估模型性能
    accuracy = accuracy_score(y_test, predictions)
    return accuracy



def scipy_cubicspline(sampled_data, sampled_time_stamp, time_stamp):
    """
    使用 scipy 的 CubicSpline 对 sampled_data 进行插值，返回 time_stamp 上的插值结果。

    Parameters:
    - sampled_data (Tensor): 输入的采样数据，形状为 [batch, sampled_len, input_dim]
    - sampled_time_stamp (Tensor): 输入的时间戳，形状为 [batch, sampled_len]
    - time_stamp (Tensor): 目标时间戳，形状为 [batch, time_stamp_len]

    Returns:
    - interpolated_data (Tensor): 输出在 time_stamp 上的插值结果，形状为 [batch, time_stamp_len, input_dim]
    """

    batch_size, sampled_len, input_dim = sampled_data.shape
    time_stamp_len = time_stamp.shape[0]
    
    # 初始化输出结果
    interpolated_data = torch.zeros((batch_size, time_stamp_len, input_dim), dtype=sampled_data.dtype, device=sampled_data.device)

    # 对每个批次进行插值
    for i in range(batch_size):
        for j in range(input_dim):
            # 对于每个 input_dim 通道，使用 CubicSpline 插值
            cs = CubicSpline(sampled_time_stamp[i].cpu().numpy(), sampled_data[i, :, j].cpu().numpy())  # 注意转换到 numpy
            interpolated_data[i, :, j] = torch.tensor(cs(time_stamp.cpu().numpy()), dtype=sampled_data.dtype, device=sampled_data.device)

    return interpolated_data
