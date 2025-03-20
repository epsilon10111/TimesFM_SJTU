from os import path  # 导入 os 模块的 path，用于文件路径操作
from typing import Optional, Tuple  # 导入可选参数类型 Optional 和元组 Tuple
import matplotlib.pyplot as plt
import numpy as np  # 导入 NumPy，用于数值计算
import pandas as pd  # 导入 Pandas，用于数据处理
import torch  # 导入 PyTorch
import torch.multiprocessing as mp  # 导入 PyTorch 多进程支持
import yfinance as yf  # 导入 yfinance，用于获取股票数据
from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner  # 导入微调配置和微调器
from huggingface_hub import snapshot_download  # 导入 Hugging Face 模型下载工具
from torch.utils.data import Dataset  # 导入 PyTorch 数据集基类

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams  # 导入 TimesFM 相关类
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder  # 导入 TimesFM 解码器
import os  # 导入 os 模块
# ==============================================
# 1. 设置 Hugging Face 镜像，加速模型下载
# ==============================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用 Hugging Face 国内镜像加速下载
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows 用户使用 SimHei（黑体）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
class TimeSeriesDataset(Dataset):
    """TimesFM 兼容的时间序列数据集"""

    def __init__(self, series: np.ndarray, context_length: int, horizon_length: int, freq_type: int = 0):
        """
        初始化数据集

        参数：
            series: 时间序列数据（NumPy 数组）
            context_length: 输入的历史时间步数
            horizon_length: 需要预测的未来时间步数
            freq_type: 频率类型（0、1 或 2）
        """
        if freq_type not in [0, 1, 2]:
            raise ValueError("freq_type 必须为 0、1 或 2")

        self.series = series
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.freq_type = freq_type
        self._prepare_samples()  # 预处理样本数据

    def _prepare_samples(self) -> None:
        """使用滑动窗口创建时间序列样本"""
        self.samples = []
        total_length = self.context_length + self.horizon_length

        for start_idx in range(0, len(self.series) - total_length + 1):
            end_idx = start_idx + self.context_length
            x_context = self.series[start_idx:end_idx]  # 过去的数据
            x_future = self.series[end_idx:end_idx + self.horizon_length]  # 未来数据
            self.samples.append((x_context, x_future))

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取索引对应的数据样本"""
        x_context, x_future = self.samples[index]

        x_context = torch.tensor(x_context, dtype=torch.float32)  # 转换为 PyTorch 张量
        x_future = torch.tensor(x_future, dtype=torch.float32)

        input_padding = torch.zeros_like(x_context)  # 创建与 x_context 形状相同的零填充
        freq = torch.tensor([self.freq_type], dtype=torch.long)  # 频率类型张量

        return x_context, input_padding, freq, x_future


def prepare_datasets(series: np.ndarray, context_length: int, horizon_length: int, freq_type: int = 0,
                     train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
    """
    根据时间序列数据准备训练集和验证集

    参数：
        series: 输入时间序列数据
        context_length: 过去时间步数
        horizon_length: 未来预测时间步数
        freq_type: 频率类型（0、1 或 2）
        train_split: 训练集所占比例

    返回：
        训练集和验证集数据集
    """
    train_size = int(len(series) * train_split)  # 计算训练集大小
    train_data = series[:train_size]
    val_data = series[train_size:]

    train_dataset = TimeSeriesDataset(train_data, context_length, horizon_length, freq_type)
    val_dataset = TimeSeriesDataset(val_data, context_length, horizon_length, freq_type)

    return train_dataset, val_dataset


def get_model(load_weights: bool = False):
    """获取 TimesFM 预训练模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 检测 GPU 可用性
    repo_id = "google/timesfm-2.0-500m-pytorch"  # Hugging Face 模型仓库 ID

    hparams = TimesFmHparams(
        backend=device,
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=50,
        use_positional_embedding=False,
        context_len=192,  # 设定上下文长度，最大可达 2048，步长为 32
    )

    tfm = TimesFm(hparams=hparams, checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))

    model = PatchedTimeSeriesDecoder(tfm._model_config)

    if load_weights:
        checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(loaded_checkpoint)

    return model, hparams, tfm._model_config


import matplotlib.pyplot as plt
import pandas as pd
import torch

def plot_predictions(model: TimesFm, val_dataset: Dataset, save_path: Optional[str] = "predictions.png", save_csv: Optional[str] = "predicted_stock_prices.csv") -> None:
    """
    可视化模型预测的股市走势，并保存为 CSV 文件。

    参数：
        model: 训练好的 TimesFM 模型
        val_dataset: 验证数据集
        save_path: 预测图像的保存路径
        save_csv: 预测结果的 CSV 保存路径
    """
    model.eval()

    # 获取验证数据中的第一个样本
    x_context, x_padding, freq, x_future = val_dataset[0]
    x_context = x_context.unsqueeze(0)  # 添加 batch 维度
    x_padding = x_padding.unsqueeze(0)
    freq = freq.unsqueeze(0)
    x_future = x_future.unsqueeze(0)

    device = next(model.parameters()).device
    x_context, x_padding, freq, x_future = map(lambda x: x.to(device), (x_context, x_padding, freq, x_future))

    with torch.no_grad():
        predictions = model(x_context, x_padding.float(), freq)
        predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
        last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

    # 提取 NumPy 数组
    context_vals = x_context[0].cpu().numpy()
    future_vals = x_future[0].cpu().numpy()
    pred_vals = last_patch_pred[0].cpu().numpy()

    # **可视化历史价格 + 真实未来价格 + 预测未来价格**
    context_len = len(context_vals)
    horizon_len = len(future_vals)

    plt.figure(figsize=(12, 6))

    # **历史价格（蓝色）**
    plt.plot(range(context_len), context_vals, label="历史数据", color="blue", linewidth=2)

    # **真实未来价格（绿色虚线）**
    plt.plot(range(context_len, context_len + horizon_len), future_vals, label="真实未来价格", color="green", linestyle="--", linewidth=2)

    # **预测未来价格（红色）**
    plt.plot(range(context_len, context_len + horizon_len), pred_vals, label="预测价格", color="red", linewidth=2)

    plt.xlabel("时间步")
    plt.ylabel("价格")
    plt.title("TimesFM 股市预测可视化")
    plt.legend()
    plt.grid(True)

    # **保存图像**
    plt.savefig(save_path)
    print(f"预测图像已保存至 {save_path}")
    plt.show()

    # **保存预测数据到 CSV**
    df = pd.DataFrame({
        "未来天数": list(range(1, len(future_vals) + 1)),
        "真实价格": future_vals,
        "预测价格": pred_vals
    })
    df.to_csv(save_csv, index=False)
    print(f"预测数据已保存至 {save_csv}")



def get_data(context_len: int, horizon_len: int, freq_type: int = 0, security_id: str = "201000000060") -> Tuple[Dataset, Dataset]:
    """从本地 CSV 文件加载数据，并筛选特定 SecurityID 的数据"""

    file_path = "TRD_StockTrend.csv"  # 确保文件路径正确

    # 读取 CSV，SecurityID 作为字符串读取，避免整数浮动问题
    df = pd.read_csv(file_path, dtype={"SecurityID": str}, parse_dates=["TradingDate"])

    # **筛选 SecurityID**
    df = df[df["SecurityID"] == security_id]

    if df.empty:
        raise ValueError(f"未找到 SecurityID={security_id} 的数据，请检查 ID 是否正确！")

    # 确保 TradingDate 是索引，并按时间排序
    df.set_index("TradingDate", inplace=True)
    df.sort_index(inplace=True)

    # 选择收盘价作为时间序列数据
    time_series = df["BwardClosePrice"].values

    # 调用 prepare_datasets 进行数据拆分
    train_dataset, val_dataset = prepare_datasets(
        series=time_series,
        context_length=context_len,
        horizon_length=horizon_len,
        freq_type=freq_type,
        train_split=0.6,
    )

    print(f"数据集创建完成（SecurityID: {security_id}）：")
    print(f"- 训练样本数: {len(train_dataset)}")
    print(f"- 验证样本数: {len(val_dataset)}")
    print(f"- 频率类型: {freq_type}")

    return train_dataset, val_dataset


def single_gpu_example():
    """使用单 GPU 进行 TimesFM 微调的示例"""
    model, hparams, tfm_config = get_model(load_weights=True)

    config = FinetuningConfig(
        batch_size=256,
        num_epochs=50,
        learning_rate=1e-4,
        use_wandb=False,
        freq_type=1,
        log_every_n_steps=10,
        val_check_interval=0.5,
        use_quantile_loss=True,
    )

    train_dataset, val_dataset = get_data(128, tfm_config.horizon_len, freq_type=config.freq_type)
    finetuner = TimesFMFinetuner(model, config)

    print("\n开始微调...")
    results = finetuner.finetune(train_dataset=train_dataset, val_dataset=val_dataset)
    print("\n微调完成！")
    print(f"训练历史: {len(results['history']['train_loss'])} 轮")

    plot_predictions(model=model, val_dataset=val_dataset, save_path="timesfm_predictions.png")

single_gpu_example()
