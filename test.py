# 导入必要的库
import os
import time
import pandas as pd
import numpy as np
from collections import defaultdict
import timesfm
import matplotlib.pyplot as plt

# ==============================================
# 1. 设置 Hugging Face 镜像，加速模型下载
# ==============================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用 Hugging Face 国内镜像加速下载

# ==============================================
# 2. 初始化 TimesFM（使用 PyTorch 版本）
# ==============================================
timesfm_backend = "gpu"  # 选择计算后端：可以是 "gpu" 或 "cpu"

tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",             # 使用 GPU 进行计算
          per_core_batch_size=32,     # 每个核心的批处理大小
          horizon_len=128,            # 预测窗口大小（未来 128 个时间步）
          num_layers=50,              # 模型层数
          context_len=2048,           # 上下文长度（过去 2048 个时间步）
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-pytorch"  # PyTorch 版本模型的 Hugging Face 路径
      ),
)

# ==============================================
# 3. 读取时间序列数据
# ==============================================
df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/EPF_FR_BE.csv')
df['ds'] = pd.to_datetime(df['ds'])  # 将时间列转换为标准的日期时间格式

# ==============================================
# 4. 数据预处理（生成批量数据）
# ==============================================
def get_batched_data_fn(
        batch_size: int = 128,  # 每批数据的大小
        context_len: int = 120,  # 历史窗口大小（过去 120 个时间步）
        horizon_len: int = 24,   # 预测窗口大小（未来 24 个时间步）
):
    examples = defaultdict(list)
    num_examples = 0

    for country in ("FR", "BE"):  # 遍历法国（FR）和比利时（BE）的数据
        sub_df = df[df["unique_id"] == country]  # 获取该国家的数据子集
        for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
            num_examples += 1
            context_end = start + context_len
            examples["country"].append(country)  # 存储国家信息
            examples["inputs"].append(sub_df["y"][start:context_end].tolist())  # 过去的 `y` 值
            examples["gen_forecast"].append(sub_df["gen_forecast"][start:context_end + horizon_len].tolist())  # 未来的预测变量
            examples["week_day"].append(sub_df["week_day"][start:context_end + horizon_len].tolist())  # 未来的星期信息
            examples["outputs"].append(sub_df["y"][context_end:(context_end + horizon_len)].tolist())  # 真实的目标值（y）

    # 数据生成器，每次返回一批数据
    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size): ((i + 1) * batch_size)] for k, v in examples.items()}

    return data_fn  # 返回数据批量生成器

# ==============================================
# 5. 定义误差计算（MSE 和 MAE）
# ==============================================
def mse(y_pred, y_true):
    """计算均方误差（Mean Squared Error, MSE）"""
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

def mae(y_pred, y_true):
    """计算平均绝对误差（Mean Absolute Error, MAE）"""
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

# ==============================================
# 6. 运行预测并评估模型
# ==============================================
batch_size = 128
context_len = 120
horizon_len = 24
input_data = get_batched_data_fn(batch_size=128)  # 生成批量数据
metrics = defaultdict(list)  # 存储误差指标

for i, example in enumerate(input_data()):  # 遍历所有批次数据
    # 使用 TimesFM 进行基本预测
    raw_forecast, _ = tfm.forecast(
        inputs=example["inputs"], freq=[0] * len(example["inputs"])
    )

    start_time = time.time()  # 记录开始时间

    # 预测带有协变量的结果
    cov_forecast, ols_forecast = tfm.forecast_with_covariates(
        inputs=example["inputs"],
        dynamic_numerical_covariates={
            "gen_forecast": example["gen_forecast"],  # 未来生成的预测变量
        },
        dynamic_categorical_covariates={
            "week_day": example["week_day"],  # 未来时间特征（如星期几）
        },
        static_numerical_covariates={},
        static_categorical_covariates={
            "country": example["country"]  # 静态分类特征，如国家
        },
        freq=[0] * len(example["inputs"]),
        xreg_mode="xreg + timesfm",  # 预测方式：使用外部回归变量（xreg）和 TimesFM
        ridge=0.0,
        force_on_cpu=False,
        normalize_xreg_target_per_input=True,  # 归一化 Xreg 变量
    )

    print(f"\rFinished batch {i} linear in {time.time() - start_time} seconds", end="")

    # 计算误差（MAE 和 MSE）
    metrics["eval_mae_timesfm"].extend(mae(raw_forecast[:, :horizon_len], example["outputs"]))
    metrics["eval_mae_xreg_timesfm"].extend(mae(cov_forecast, example["outputs"]))
    metrics["eval_mae_xreg"].extend(mae(ols_forecast, example["outputs"]))
    metrics["eval_mse_timesfm"].extend(mse(raw_forecast[:, :horizon_len], example["outputs"]))
    metrics["eval_mse_xreg_timesfm"].extend(mse(cov_forecast, example["outputs"]))
    metrics["eval_mse_xreg"].extend(mse(ols_forecast, example["outputs"]))

print()  # 换行

# ==============================================
# 7. 输出最终评估指标
# ==============================================
for k, v in metrics.items():
    print(f"{k}: {np.mean(v)}")  # 计算误差的均值并打印

# 选择一个批次的数据进行可视化
batch_index = 0  # 选择要可视化的批次
example = next(iter(input_data()))  # 获取批量数据

# 获取预测值
raw_forecast, _ = tfm.forecast(inputs=example["inputs"], freq=[0] * len(example["inputs"]))
cov_forecast, ols_forecast = tfm.forecast_with_covariates(
    inputs=example["inputs"],
    dynamic_numerical_covariates={"gen_forecast": example["gen_forecast"]},
    dynamic_categorical_covariates={"week_day": example["week_day"]},
    static_numerical_covariates={},
    static_categorical_covariates={"country": example["country"]},
    freq=[0] * len(example["inputs"]),
    xreg_mode="xreg + timesfm",
    ridge=0.0,
    force_on_cpu=False,
    normalize_xreg_target_per_input=True,
)

# 选择某个样本的数据
index = 0  # 选择第一个样本
history = example["inputs"][index]  # 过去时间步的真实数据
actual = example["outputs"][index]  # 未来时间步的真实数据
timesfm_pred = raw_forecast[index, :horizon_len]  # `TimesFM` 直接预测值
xreg_pred = cov_forecast[index]  # 结合 XReg 的预测值

# 生成时间轴
history_time = list(range(len(history)))  # 过去数据时间点
future_time = list(range(len(history), len(history) + len(actual)))  # 预测时间点

# 绘制真实数据 vs. 预测数据
plt.figure(figsize=(12, 6))
plt.plot(history_time, history, label="Historical Data", color="blue", linestyle="dashed")
plt.plot(future_time, actual, label="Actual Future Data", color="black", marker="o")
plt.plot(future_time, timesfm_pred, label="TimesFM Prediction", color="red", linestyle="dotted")
plt.plot(future_time, xreg_pred, label="TimesFM + XReg Prediction", color="green", linestyle="solid")

# 标注
plt.axvline(x=len(history)-1, color='gray', linestyle="--")  # 预测起点
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("TimesFM Forecast vs. Actual Data")
plt.show()
# My output:
# eval_mae_timesfm: 6.762283045916956
# eval_mae_xreg_timesfm: 5.39219617611074
# eval_mae_xreg: 37.15275842572484
# eval_mse_timesfm: 166.7771466306823
# eval_mse_xreg_timesfm: 120.64757721021306
# eval_mse_xreg: 1672.2116821201796