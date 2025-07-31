import torch
import torch.nn as nn
import numpy as np
import os
from models.SSM2Mel import Decoder

def load_trained_model(model_path, device='cuda:0'):
    """加载训练好的模型"""
    # 模型参数（需要与训练时保持一致）
    model_args = {
        'in_channel': 64,
        'd_model': 64,
        'd_inner': 1024,
        'n_head': 2,
        'n_layers': 1,
        'fft_conv1d_kernel': (9, 1),
        'fft_conv1d_padding': (4, 0),
        'dropout': 0.5,
        'g_con': True
    }
    
    model = Decoder(**model_args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_mel_from_eeg(model, eeg_data, sub_id=0, device='cuda:0'):
    """
    从EEG数据预测mel频谱
    
    Args:
        model: 训练好的模型
        eeg_data: EEG数据，形状为 [T, C] 或 [B, T, C]
        sub_id: 受试者ID
        device: 计算设备
    
    Returns:
        mel_prediction: 预测的mel频谱
    """
    model.eval()
    
    with torch.no_grad():
        # 确保输入格式正确
        if len(eeg_data.shape) == 2:  # [T, C]
            eeg_data = eeg_data.unsqueeze(0)  # [1, T, C]
        
        eeg_data = eeg_data.to(device)
        sub_id = torch.tensor([sub_id]).to(device)
        
        # 进行推理
        output = model(eeg_data, sub_id)
        return output.squeeze().cpu().numpy()

def main():
    # 模型路径
    model_path = '/home/binwen6/code/CBD/SSM2Mel/result_model_conformer/model_epoch100.pt'
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载模型...")
    model = load_trained_model(model_path, device)
    print("模型加载成功!")
    
    # 创建示例EEG数据（640时间步，64通道）
    # 在实际使用中，这里应该是你的真实EEG数据
    sample_eeg = np.random.randn(640, 64).astype(np.float32)
    print(f"输入EEG数据形状: {sample_eeg.shape}")
    
    # 进行推理
    print("正在进行推理...")
    mel_prediction = predict_mel_from_eeg(model, sample_eeg, sub_id=0, device=device)
    print(f"预测的mel频谱形状: {mel_prediction.shape}")
    print(f"预测值范围: [{mel_prediction.min():.4f}, {mel_prediction.max():.4f}]")
    
    # 保存结果
    output_dir = '/home/binwen6/code/CBD/SSM2Mel/inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'sample_prediction.npy'), mel_prediction)
    print(f"结果已保存到: {output_dir}/sample_prediction.npy")

if __name__ == '__main__':
    main() 