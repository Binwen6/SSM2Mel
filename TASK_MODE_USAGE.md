# SSM2Mel 任务模式使用说明

本项目现在支持两种任务模式：**音频包络重建**和**完整Mel频谱重建**。

## 任务模式概述

### 1. 音频包络重建模式 (envelope)
- **目标**：重建语音的包络信息
- **输出维度**：1个频带
- **特点**：计算简单，训练快速，适合语音活动检测等任务
- **数据格式**：将128频带Mel频谱聚合为单频带

### 2. 完整Mel频谱重建模式 (mel_spectrogram)
- **目标**：重建完整的多频带Mel频谱
- **输出维度**：80个频带
- **特点**：保留丰富的时频信息，重建质量更高
- **数据格式**：直接使用80频带Mel频谱

## 使用方法

### 方法1：使用配置切换脚本（推荐）

```bash
# 查看当前模式
python switch_task_mode.py status

# 切换到音频包络重建模式
python switch_task_mode.py envelope

# 切换到完整Mel频谱重建模式
python switch_task_mode.py mel_spectrogram
```

### 方法2：直接修改配置文件

编辑 `data_prep.py` 文件中的 `SSM2MelConfig` 类：

```python
class SSM2MelConfig:
    # ... 其他配置 ...
    
    # 修改这一行来切换任务模式
    TASK_MODE = "envelope"  # 或 "mel_spectrogram"
```

### 方法3：通过训练参数指定

```bash
# 音频包络重建模式
python SSM2Mel/train.py --task_mode envelope

# 完整Mel频谱重建模式
python SSM2Mel/train.py --task_mode mel_spectrogram
```

## 完整工作流程

### 音频包络重建模式

1. **切换模式**：
   ```bash
   python switch_task_mode.py envelope
   ```

2. **重新生成数据**：
   ```bash
   python data_prep.py
   ```

3. **训练模型**：
   ```bash
   python SSM2Mel/train.py --task_mode envelope
   ```

4. **推理和可视化**：
   ```bash
   python SSM2Mel/inference.py
   python SSM2Mel/visualize_results.py
   ```

### 完整Mel频谱重建模式

1. **切换模式**：
   ```bash
   python switch_task_mode.py mel_spectrogram
   ```

2. **重新生成数据**：
   ```bash
   python data_prep.py
   ```

3. **训练模型**：
   ```bash
   python SSM2Mel/train.py --task_mode mel_spectrogram
   ```

4. **推理和可视化**：
   ```bash
   python SSM2Mel/inference.py
   python SSM2Mel/visualize_results.py
   ```

## 技术细节

### 数据预处理差异

- **音频包络模式**：
  - 计算128频带Mel频谱
  - 对所有频带取平均值，得到单频带
  - 输出形状：(640, 1)

- **完整Mel频谱模式**：
  - 直接计算80频带Mel频谱
  - 保留所有频带信息
  - 输出形状：(640, 80)

### 模型架构差异

- **音频包络模式**：
  - 输出层：`nn.Linear(64, 1)`
  - 输出维度：1

- **完整Mel频谱模式**：
  - 输出层：`nn.Linear(64, 80)`
  - 输出维度：80

### 可视化差异

- **音频包络模式**：
  - 使用高斯分布"猜测"完整频谱
  - 重建质量有限

- **完整Mel频谱模式**：
  - 直接使用多频带信息
  - 重建质量更高

## 性能对比

| 模式 | 输出维度 | 训练速度 | 重建质量 | 适用场景 |
|------|----------|----------|----------|----------|
| 音频包络 | 1 | 快 | 中等 | 语音活动检测 |
| 完整Mel频谱 | 80 | 慢 | 高 | 高质量语音重建 |

## 注意事项

1. **数据重新生成**：切换模式后必须重新运行 `data_prep.py`
2. **模型重新训练**：不同模式需要重新训练模型
3. **存储空间**：完整Mel频谱模式需要更多存储空间
4. **计算资源**：完整Mel频谱模式需要更多计算资源

## 故障排除

### 常见问题

1. **配置错误**：
   ```bash
   python switch_task_mode.py status  # 检查当前模式
   ```

2. **数据格式错误**：
   - 确保重新生成了数据
   - 检查数据文件大小和格式

3. **模型维度不匹配**：
   - 确保模型参数与任务模式一致
   - 检查训练时的 `--task_mode` 参数

### 调试命令

```bash
# 检查数据文件
ls -la data/split_data/

# 检查模型输出维度
python -c "from SSM2Mel.models.SSM2Mel import Decoder; print(Decoder(task_mode='envelope').fc.out_features)"
python -c "from SSM2Mel.models.SSM2Mel import Decoder; print(Decoder(task_mode='mel_spectrogram').fc.out_features)"
``` 