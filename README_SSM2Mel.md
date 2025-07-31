# SSM2Mel EEG到音乐重建项目

本项目实现了从脑电信号(EEG)到Mel频谱的映射，用于音乐刺激的重建。

## 项目结构

```
CBD/
├── Generate/                    # 原始EEG数据目录
│   ├── *.vhdr                  # BrainVision头文件
│   ├── *.vmrk                  # 事件标记文件
│   └── *.eeg                   # EEG数据文件
├── BCMI/dataset/music_gen_wav_22050/  # 音频刺激文件
│   ├── All Of Me - All of Me (Karaoke Version).wav
│   ├── Richard Clayderman - 梦中的鸟.wav
│   └── ...
├── SSM2Mel/SSM2Mel/           # SSM2Mel模型代码
│   ├── train.py               # 训练脚本
│   ├── prepare_dataset.py     # 数据预处理脚本
│   ├── models/                # 模型定义
│   └── util/                  # 工具函数
├── data/                      # 预处理后的数据
│   ├── train/                 # 训练数据
│   ├── val/                   # 验证数据
│   └── test/                  # 测试数据
└── run_training.py            # 训练启动脚本
```

## 数据说明

### 输入数据
- **EEG数据**: 32通道，500Hz采样率，BrainVision格式
- **音频刺激**: 8段音乐，22050Hz采样率，每段约9.95秒
- **事件标记**: 每段音乐有7个标记，取前5个作为有效试验

### 音频映射关系
- s1: All Of Me - All of Me (Karaoke Version)
- s2: Richard Clayderman - 梦中的鸟
- s3: Robin Spielberg - Turn the Page
- s4: dylanf - 梦中的婚礼 (经典钢琴版)
- s5: 文武贝 - 夜的钢琴曲5
- s6: 昼夜 - 千与千寻 (钢琴版)
- s7: 演奏曲 - 【钢琴Piano】雨中漫步Stepping On The Rainy S
- s8: 郭宴 - 天空之城 (钢琴版)

## 数据预处理流程

### 1. EEG预处理
- 加载BrainVision格式数据
- 平均参考设置
- 1-45Hz带通滤波 + 50Hz陷波
- 下采样到64Hz
- 根据事件标记提取9.95秒片段
- 通道填充到64维（32通道 + 32个零通道）
- Z-score标准化

### 2. 音频处理
- 加载22050Hz音频
- 生成80维Mel频谱
- 调整到640帧（与EEG同步）
- 逐bin标准化

### 3. 数据分割
- 训练集：70%的试验
- 验证集：15%的试验
- 测试集：15%的试验

## 使用方法

### 1. 数据预处理

```bash
# 运行数据预处理脚本
python SSM2Mel/SSM2Mel/prepare_dataset.py

# 可选参数
python SSM2Mel/SSM2Mel/prepare_dataset.py \
    --eeg_dir Generate \
    --audio_dir BCMI/dataset/music_gen_wav_22050 \
    --output_dir data \
    --target_sr 64 \
    --stimulus_duration 9.95 \
    --max_trials 5
```

### 2. 模型训练

```bash
# 使用默认参数训练
python run_training.py

# 自定义训练参数
python run_training.py \
    --epochs 200 \
    --batch_size 16 \
    --gpu 0 \
    --learning_rate 0.0005 \
    --data_dir data
```

### 3. 直接运行训练脚本

```bash
cd SSM2Mel/SSM2Mel
python train.py \
    --epoch 200 \
    --batch_size 16 \
    --gpu 0 \
    --dataset_folder ../../data \
    --split_folder "" \
    --win_len 10 \
    --sample_rate 64 \
    --in_channel 64
```

## 模型配置

### 默认参数
- **输入长度**: 640帧 (10秒 × 64Hz)
- **EEG通道**: 64维 (32原始 + 32填充)
- **Mel频谱**: 80维 × 640帧
- **模型架构**: 
  - d_model: 64
  - n_head: 2
  - n_layers: 1
  - d_inner: 1024
- **损失函数**: Pearson损失 + L1损失
- **优化器**: Adam (lr=0.0005)

### 训练设置
- **批次大小**: 16 (可根据GPU内存调整)
- **学习率调度**: StepLR (step_size=50, gamma=0.9)
- **保存间隔**: 每10个epoch
- **日志间隔**: 每5个epoch

## 输出文件

### 数据文件格式
- **EEG文件**: `train_-_sub-{被试ID}_-_s{刺激编号}_trial{试验编号}_-_eeg.npy`
  - 形状: (64, 640)
  - 数据类型: float32
- **Mel文件**: `train_-_sub-{被试ID}_-_s{刺激编号}_trial{试验编号}_-_mel.npy`
  - 形状: (80, 640)
  - 数据类型: float32

### 模型输出
- **检查点**: `result_model_conformer/model_epoch{epoch}.pt`
- **日志**: TensorBoard格式，保存在`test_results/`目录

## 依赖包

```bash
pip install torch torchvision torchaudio
pip install mne
pip install librosa
pip install numpy
pip install tensorboard
```

## 注意事项

1. **GPU内存**: 如果GPU内存不足，可以减小batch_size
2. **数据路径**: 确保EEG和音频文件路径正确
3. **文件编码**: .vmrk文件使用UTF-8编码
4. **采样率**: 确保EEG和音频的采样率设置正确

## 故障排除

### 常见问题
1. **"could not convert string to float: 'BrainVision'"**: 已修复，使用手动解析.vmrk文件
2. **音频文件未找到**: 检查音频映射关系是否正确
3. **内存不足**: 减小batch_size或使用CPU训练
4. **数据长度不匹配**: 检查stimulus_duration参数

### 调试建议
- 检查数据文件数量和质量
- 验证EEG和音频的时间对齐
- 监控训练过程中的损失变化
- 使用TensorBoard查看训练曲线

## 扩展功能

### 支持的功能
- 多被试者训练 (--g_con True/False)
- 不同采样率支持 (修改sample_rate参数)
- 自定义模型架构 (修改d_model, n_head等参数)

### 未来扩展
- 支持128Hz EEG数据
- 集成WaveNet/HiFi-GAN解码器
- 支持更多音频格式
- 实时EEG到音频转换 