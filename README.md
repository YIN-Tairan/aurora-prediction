# Aurora Prediction

## Project Overview

This project aims to predict auroral activity using solar wind data from NASA's OMNI dataset. By analyzing time-series data of solar wind parameters (such as magnetic field components, plasma velocity, density, and geomagnetic indices), we can forecast aurora occurrences and intensity.

The prediction system uses machine learning and deep learning approaches to learn the complex relationships between solar wind parameters and aurora activity, with applications in space weather forecasting and geophysical research.

## Repository Structure

```
aurora-prediction/
├── get_data_from_nasa.sh       # Download high-resolution OMNI data from NASA
├── get_data_from_noaa.sh       # Download OMNI data from NASA OMNIWeb (1-minute resolution, 1981-2025)
├── clean_data.sh               # Data cleaning script (removes invalid entries)
├── omni_process.py             # Data processing and preparation for PyTorch
├── SimpleDataset.py            # PyTorch Dataset class for sequence data
├── train_mlp.py                # MLP baseline model training
├── train_autoencoder.py        # Convolutional autoencoder training
├── data_print.py               # Data inspection utility
└── README.md                   # This file
```

## Data Acquisition

### Step 1: Download OMNI Data

The project uses two data sources from NASA's OMNIWeb service:

**Option 1: NASA High-Resolution Data (HRO)**
```bash
bash get_data_from_nasa.sh
```
- Downloads 1-minute resolution data from 1981-2025
- Variables include magnetic field (BX, BY, BZ), plasma parameters (velocity, density, temperature), and geomagnetic indices (AE, AL, AU, SYM/D, SYM/H, etc.)
- Output: `hro_1981_2025_1min.txt`

**Option 2: NASA OMNIWeb Data**
```bash
bash get_data_from_noaa.sh
```
- Downloads 1-minute resolution data from NASA OMNIWeb (1981-2025)
- Note: Script downloads data from 1981-2025, though output filename indicates 1964-2025
- Output: `omni_1min_data_1964_2025.txt` (actual data range: 1981-2025)

**Key Features:**
- Automatic year-by-year downloading with rate limiting
- Header extraction and data concatenation
- Variables 4-45 (or 1-37 depending on the source) covering comprehensive solar wind and magnetosphere parameters

### Step 2: Data Cleaning

Remove invalid data entries using the cleaning script:

```bash
bash clean_data.sh
```

**Cleaning Process:**
- Checks columns 5 and 6 (spacecraft ID columns in OMNI format)
- Removes rows where both columns 5 and 6 contain missing value flags (e.g., 999, 9999)
- Preserves rows where at least one of these columns has valid data
- Non-data lines (HTML headers, footers) are kept in the output
- Output: `omni_1min_cleaned.txt`

### Step 3: Data Processing and Preparation

Process the cleaned data into PyTorch-ready format:

```bash
python omni_process.py
```

**Processing Pipeline:**
1. **Continuous Segment Identification**: Identifies continuous data chunks based on time gaps (max 15 minutes missing allowed)
2. **Missing Value Handling**: Replaces OMNI missing value flags (99, 999, 9999, etc.) with NaN
3. **Resampling**: Ensures uniform 1-minute time intervals
4. **Linear Interpolation**: Fills small gaps (up to 15 minutes) using linear interpolation
5. **Feature Selection**: Excludes time-related and spacecraft ID columns from feature set
6. **Parquet Export**: Saves processed segments with `Segment_ID` for training

**Note**: Sliding window creation (3 days input + 3 hours prediction) is handled by `SimpleDataset.py` during model training, not by this processing script.

**Configuration Parameters (in `omni_process.py`):**
- `INPUT_FILE`: Input data file path (default: `'hro_data_sample.txt'` - **change this to your cleaned data file**, e.g., `'omni_1min_cleaned.txt'`)
- `OUTPUT_PARQUET`: Output Parquet file (default: `omni_ready_for_pytorch.parquet`)
- `MAX_MISSING_MINUTES`: Maximum allowed gap for interpolation (default: 15 minutes)
- `WINDOW_DAYS`: Window size for segment validation (default: 3.125 days = 4500 minutes)
- `STRIDE_MINUTES`: Stride for segment validation (default: 10 minutes)

**Output:** 
- `omni_ready_for_pytorch.parquet`: Processed data ready for model training

## Model Training

### Baseline MLP Model

Train a multi-layer perceptron for sequence-to-sequence prediction:

```bash
python train_mlp.py
```

**Architecture:**
- Input: 3 days of solar wind data (4320 minutes)
- Output: 3 hours of predicted values (180 minutes)
- 6-layer fully connected network with batch normalization and dropout
- Features: ~30 parameters (excluding spacecraft IDs and time columns)

**Training Details:**
- Loss: MSE (Mean Squared Error)
- Optimizer: AdamW with weight decay
- Data normalization: Standardization (mean=0, std=1)
- Mixed precision training (automatic mixed precision)
- Gradient clipping to prevent instability

### Convolutional Autoencoder

Train a 1D-CNN autoencoder for dimensionality reduction and feature learning:

```bash
python train_autoencoder.py
```

**Architecture:**
- Encoder: 1D convolution layers with max pooling (4320 → 120 timesteps)
- Latent space: 128-dimensional compressed representation
- Decoder: Upsampling + 1D convolution layers (120 → 4320 timesteps)
- Reconstruction task: Input = Output (self-supervised learning)

**Use Cases:**
- Feature extraction for downstream tasks
- Anomaly detection in solar wind data
- Data compression for efficient storage

## Current Status

**Completed:**
- ✅ Data acquisition scripts for NASA OMNI dataset
- ✅ Data cleaning and preprocessing pipeline
- ✅ PyTorch Dataset implementation with sliding windows
- ✅ Baseline MLP model for time series prediction
- ✅ Convolutional autoencoder for feature learning

**In Progress:**
- 🚧 Transformer-based architecture (see roadmap below)
- 🚧 Advanced data processing methods
- 🚧 Missing data recovery using diffusion models

## Future Work Roadmap

### 1. Machine Learning Architecture for the Project
**Objective:** Establish a comprehensive ML framework with experiment tracking, model versioning, and deployment pipeline.

**Tasks:**
- Implement MLflow or Weights & Biases for experiment tracking
- Create modular model registry (transformers, CNNs, RNNs, hybrid models)
- Develop cross-validation and hyperparameter tuning framework
- Add model evaluation metrics (RMSE, MAE, correlation, skill scores)
- Create inference API for real-time predictions

**Deliverables:**
- `models/` directory with base classes and specific implementations
- `experiments/` directory for configuration files
- `utils/` directory for training, evaluation, and logging utilities

### 2. Transformer Architecture for Time Series Data
**Objective:** Build state-of-the-art transformer models tailored for multivariate time series forecasting.

**Tasks:**
- Implement positional encoding for temporal data
- Design attention mechanisms for long-range dependencies (up to 3 days)
- Create encoder-decoder transformer architecture
- Implement Informer or PatchTST variants for efficiency
- Add temporal fusion transformer (TFT) for interpretability
- Multi-scale attention for capturing different temporal patterns

**Deliverables:**
- `models/transformers/` with multiple transformer variants
- Attention visualization tools
- Comparative benchmarks vs. MLP/CNN baselines

### 3. Enhanced Data Cleaning and Processing Methods
**Objective:** Develop robust, scalable, and automated data preprocessing pipeline.

**Tasks:**
- Implement adaptive outlier detection (IQR, isolation forest, DBSCAN)
- Create automated quality control checks (sensor drift, calibration issues)
- Add data augmentation techniques (time warping, magnitude scaling)
- Develop real-time data streaming and processing pipeline
- Implement efficient data versioning with DVC (Data Version Control)
- Create data validation schemas and unit tests

**Deliverables:**
- `data_processing/` module with modular cleaning functions
- Automated quality reports and data health dashboards
- Data pipeline orchestration with Apache Airflow or Prefect

### 4. Missing Data Handling with Masks and Diffusion Models
**Objective:** Advanced imputation methods that preserve temporal patterns and uncertainty quantification.

**Tasks:**
- Implement attention masks for transformer models (BERT-style masking)
- Create binary mask tensors indicating missing vs. observed data
- Design diffusion-based imputation model (CSDI - Conditional Score-based Diffusion for Imputation)
- Implement variational autoencoders for probabilistic imputation
- Add Gaussian process regression for smooth interpolation
- Develop ensemble imputation with uncertainty estimates

**Deliverables:**
- `imputation/` module with multiple imputation strategies
- Comparison of imputation methods (diffusion, VAE, GP, linear)
- Uncertainty quantification for imputed values
- Validation framework for imputation quality

### 5. SLURM Job Submission for HPC Environments
**Objective:** Enable large-scale training on high-performance computing clusters.

**Tasks:**
- Create SLURM batch scripts for model training
- Implement distributed data parallel (DDP) training with PyTorch
- Design hyperparameter sweep jobs using SLURM job arrays
- Add checkpoint saving and resumption for long training jobs
- Create monitoring and logging for cluster jobs
- Implement resource allocation optimization (GPU hours, memory)

**Deliverables:**
- `slurm_scripts/` directory with training templates
- `sbatch_train.sh` - Single GPU training job
- `sbatch_distributed.sh` - Multi-GPU distributed training
- `sbatch_sweep.sh` - Hyperparameter search job array
- Documentation for cluster setup and job submission

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Bug fixes
- New model architectures
- Improved data processing methods
- Documentation improvements

## License

Copyright (c) Aurora Prediction contributors.

Permission is granted to use, copy, modify, and distribute this project and its documentation for non-commercial academic and research purposes only, provided that this copyright notice and this permission notice are included in all copies or substantial portions of the project.

Commercial use is not permitted without prior written permission from the copyright holders.

THE PROJECT IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE PROJECT OR THE USE OR OTHER DEALINGS IN THE PROJECT.

## References

- NASA OMNI Dataset: https://omniweb.gsfc.nasa.gov/
- Space Weather Prediction Center: https://www.swpc.noaa.gov/
- Aurora forecasting methodologies and geomagnetic indices

---

# 极光预测项目

## 项目概述

本项目旨在利用NASA OMNI数据集中的太阳风数据预测极光活动。通过分析太阳风参数（如磁场分量、等离子体速度、密度和地磁指数）的时间序列数据，我们可以预测极光的发生和强度。

该预测系统使用机器学习和深度学习方法来学习太阳风参数与极光活动之间的复杂关系，可应用于空间天气预报和地球物理研究。

## 仓库结构

```
aurora-prediction/
├── get_data_from_nasa.sh       # 从NASA下载高分辨率OMNI数据
├── get_data_from_noaa.sh       # 从NASA OMNIWeb下载OMNI数据（1分钟分辨率，实际下载1981-2025年，输出文件名为omni_1min_data_1964_2025.txt）
├── clean_data.sh               # 数据清洗脚本（删除无效条目）
├── omni_process.py             # 数据处理和PyTorch准备
├── SimpleDataset.py            # PyTorch序列数据集类
├── train_mlp.py                # MLP基线模型训练
├── train_autoencoder.py        # 卷积自编码器训练
├── data_print.py               # 数据检查工具
└── README.md                   # 本文件
```

## 数据获取

### 步骤1：下载OMNI数据

项目使用NASA OMNIWeb服务的两个数据源：

**选项1：NASA高分辨率数据（HRO）**
```bash
bash get_data_from_nasa.sh
```
- 下载1981-2025年的1分钟分辨率数据
- 变量包括磁场（BX、BY、BZ）、等离子体参数（速度、密度、温度）和地磁指数（AE、AL、AU、SYM/D、SYM/H等）
- 输出：`hro_1981_2025_1min.txt`

**选项2：NASA OMNIWeb数据**
```bash
bash get_data_from_noaa.sh
```
- 从NASA OMNIWeb下载1分钟分辨率数据（1981-2025年）
- 注意：脚本下载1981-2025年的数据，虽然输出文件名显示1964-2025
- 输出：`omni_1min_data_1964_2025.txt`（实际数据范围：1981-2025年）

**主要特点：**
- 按年份自动下载，带有速率限制
- 表头提取和数据拼接
- 变量4-45（或1-37，取决于数据源），涵盖全面的太阳风和磁层参数

### 步骤2：数据清洗

使用清洗脚本删除无效数据条目：

```bash
bash clean_data.sh
```

**清洗过程：**
- 检查第5列和第6列（OMNI格式中的航天器ID列）
- 当这两列都包含缺失值标志（如999、9999等）时删除该行
- 若第5列或第6列中至少有一列不是缺失值标志，则保留该行
- 非数据行（如原始下载中的HTML、表头或尾部内容）会被保留在输出中
- 输出：`omni_1min_cleaned.txt`

### 步骤3：数据处理和准备

将清洗后的数据处理成PyTorch就绪格式：

```bash
python omni_process.py
```

**处理流程：**
1. **连续段识别**：基于时间间隔识别连续数据块（允许最多15分钟缺失）
2. **缺失值处理**：将OMNI缺失值标志（99、999、9999等）替换为NaN
3. **重采样**：确保均匀的1分钟时间间隔
4. **线性插值**：使用线性插值填补小间隙（最多15分钟）
5. **特征选择**：从特征集中排除时间相关列和航天器ID列
6. **Parquet导出**：保存带有`Segment_ID`的处理后数据段供训练使用

**注意**：滑动窗口创建（3天输入 + 3小时预测）由模型训练时的`SimpleDataset.py`处理，而非此处理脚本。

**配置参数（在`omni_process.py`中）：**
- `INPUT_FILE`：输入数据文件路径（默认：`'hro_data_sample.txt'` - **请将此更改为您的清洗后数据文件**，例如`'omni_1min_cleaned.txt'`）
- `OUTPUT_PARQUET`：输出Parquet文件（默认：`omni_ready_for_pytorch.parquet`）
- `MAX_MISSING_MINUTES`：插值允许的最大间隙（默认：15分钟）
- `WINDOW_DAYS`：总窗口大小（天）（默认：3.125天 = 4500分钟）
- `STRIDE_MINUTES`：滑动窗口步长（默认：10分钟）

**输出：**
- `omni_ready_for_pytorch.parquet`：准备好用于模型训练的处理后数据

## 模型训练

### 基线MLP模型

训练多层感知机进行序列到序列预测：

```bash
python train_mlp.py
```

**架构：**
- 输入：3天太阳风数据（4320分钟）
- 输出：3小时预测值（180分钟）
- 6层全连接网络，带批归一化和dropout
- 特征：约30个参数（不包括航天器ID和时间列）

**训练细节：**
- 损失函数：MSE（均方误差）
- 优化器：AdamW带权重衰减
- 数据归一化：标准化（均值=0，标准差=1）
- 混合精度训练（自动混合精度）
- 梯度裁剪以防止不稳定

### 卷积自编码器

训练1D-CNN自编码器进行降维和特征学习：

```bash
python train_autoencoder.py
```

**架构：**
- 编码器：1D卷积层与最大池化（4320 → 120时间步）
- 潜在空间：128维压缩表示
- 解码器：上采样 + 1D卷积层（120 → 4320时间步）
- 重建任务：输入 = 输出（自监督学习）

**应用场景：**
- 下游任务的特征提取
- 太阳风数据的异常检测
- 高效存储的数据压缩

## 当前状态

**已完成：**
- ✅ NASA OMNI数据集的数据获取脚本
- ✅ 数据清洗和预处理流程
- ✅ 带滑动窗口的PyTorch数据集实现
- ✅ 时间序列预测的基线MLP模型
- ✅ 特征学习的卷积自编码器

**进行中：**
- 🚧 基于Transformer的架构（见下方路线图）
- 🚧 高级数据处理方法
- 🚧 使用扩散模型进行缺失数据恢复

## 未来工作路线图

### 1. 为项目搭建机器学习架构
**目标：** 建立全面的ML框架，包括实验跟踪、模型版本控制和部署流程。

**任务：**
- 实现MLflow或Weights & Biases进行实验跟踪
- 创建模块化模型注册表（transformers、CNNs、RNNs、混合模型）
- 开发交叉验证和超参数调优框架
- 添加模型评估指标（RMSE、MAE、相关性、技能评分）
- 创建实时预测的推理API

**交付成果：**
- `models/`目录，包含基类和具体实现
- `experiments/`目录用于配置文件
- `utils/`目录用于训练、评估和日志工具

### 2. 为数据搭建Transformer架构
**目标：** 为多变量时间序列预测构建最先进的transformer模型。

**任务：**
- 为时间数据实现位置编码
- 为长程依赖关系设计注意力机制（最长3天）
- 创建编码器-解码器transformer架构
- 实现Informer或PatchTST变体以提高效率
- 添加时间融合transformer（TFT）以提高可解释性
- 多尺度注意力以捕获不同的时间模式

**交付成果：**
- `models/transformers/`，包含多个transformer变体
- 注意力可视化工具
- 与MLP/CNN基线的比较基准

### 3. 更便捷优质的数据清理和数据处理方法
**目标：** 开发健壮、可扩展和自动化的数据预处理流程。

**任务：**
- 实现自适应异常值检测（IQR、孤立森林、DBSCAN）
- 创建自动化质量控制检查（传感器漂移、校准问题）
- 添加数据增强技术（时间扭曲、幅度缩放）
- 开发实时数据流和处理流程
- 使用DVC（数据版本控制）实现高效的数据版本控制
- 创建数据验证模式和单元测试

**交付成果：**
- `data_processing/`模块，包含模块化清洗函数
- 自动化质量报告和数据健康仪表板
- 使用Apache Airflow或Prefect进行数据流程编排

### 4. 对缺失数据添加mask和基于Diffusion Model的缺失数据恢复方法
**目标：** 高级插补方法，保留时间模式和不确定性量化。

**任务：**
- 为transformer模型实现注意力掩码（BERT风格掩码）
- 创建二进制掩码张量，指示缺失数据与观测数据
- 设计基于扩散的插补模型（CSDI - 基于条件分数的扩散插补）
- 实现变分自编码器进行概率插补
- 添加高斯过程回归进行平滑插值
- 开发带不确定性估计的集成插补

**交付成果：**
- `imputation/`模块，包含多种插补策略
- 插补方法比较（扩散、VAE、GP、线性）
- 插补值的不确定性量化
- 插补质量的验证框架

### 5. 在SLURM系统上提交作业的相关代码
**目标：** 在高性能计算集群上实现大规模训练。

**任务：**
- 为模型训练创建SLURM批处理脚本
- 使用PyTorch实现分布式数据并行（DDP）训练
- 使用SLURM作业数组设计超参数扫描作业
- 为长时间训练作业添加检查点保存和恢复
- 为集群作业创建监控和日志记录
- 实现资源分配优化（GPU小时数、内存）

**交付成果：**
- `slurm_scripts/`目录，包含训练模板
- `sbatch_train.sh` - 单GPU训练作业
- `sbatch_distributed.sh` - 多GPU分布式训练
- `sbatch_sweep.sh` - 超参数搜索作业数组
- 集群设置和作业提交文档

## 贡献

欢迎贡献！请随时提交以下方面的问题或拉取请求：
- Bug修复
- 新模型架构
- 改进的数据处理方法
- 文档改进

## 许可证

本项目可用于学术和研究目的。

## 参考文献

- NASA OMNI数据集：https://omniweb.gsfc.nasa.gov/
- 空间天气预报中心：https://www.swpc.noaa.gov/
- 极光预报方法和地磁指数
