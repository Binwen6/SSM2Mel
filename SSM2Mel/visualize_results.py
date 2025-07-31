import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import resample
import os
import argparse
from scipy.io import wavfile
import torch

def load_inference_results(results_path):
    """加载推理结果"""
    if results_path.endswith('.npz'):
        data = np.load(results_path)
        outputs = data['outputs']
        labels = data['labels']
    else:
        outputs = np.load(results_path)
        labels = None
    
    return outputs, labels

def mel_to_audio(mel_spec, sr=22050, hop_length=512, n_fft=2048):
    """
    将mel频谱转换回音频波形
    
    Args:
        mel_spec: mel频谱，形状为 (time, 1) 或 (1, time)
        sr: 目标采样率
        hop_length: hop长度
        n_fft: FFT窗口大小
    
    Returns:
        audio: 音频波形
    """
    print(f"    mel_to_audio 输入形状: {mel_spec.shape}")
    print(f"    mel_to_audio 输入范围: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    
    # 确保mel_spec是 (time, 1) 格式
    if len(mel_spec.shape) == 1:
        # 一维数组，重塑为 (time, 1)
        mel_spec = mel_spec.reshape(-1, 1)
        print(f"    重塑一维数组为: {mel_spec.shape}")
    elif len(mel_spec.shape) == 2 and mel_spec.shape[0] == 1:
        mel_spec = mel_spec.T  # 现在形状是 (time, 1)
        print(f"    转置后形状: {mel_spec.shape}")
    
    # 将分贝值转换回功率谱
    mel_power = librosa.db_to_power(mel_spec)
    print(f"    功率谱形状: {mel_power.shape}")
    print(f"    功率谱范围: [{mel_power.min():.4f}, {mel_power.max():.4f}]")
    
    # 由于我们只有单个mel频带，需要重建完整的mel频谱
    # 创建一个合理的mel频谱分布
    n_mels = 128  # 标准mel频带数
    time_steps = mel_power.shape[0]
    print(f"    时间步数: {time_steps}")
    
    # 创建一个更合理的mel频谱分布，而不是简单复制
    # 使用高斯分布来模拟不同频带的能量分布
    freqs = np.linspace(0, 1, n_mels)
    center_freq = 0.5  # 中心频率
    bandwidth = 0.3    # 带宽
    
    # 创建频带权重
    weights = np.exp(-((freqs - center_freq) ** 2) / (2 * bandwidth ** 2))
    weights = weights.reshape(-1, 1)  # (n_mels, 1)
    print(f"    权重形状: {weights.shape}")
    
    # 将单个mel值扩展到多个频带
    full_mel_spec = mel_power.T * weights  # (n_mels, time_steps)
    print(f"    完整mel频谱形状: {full_mel_spec.shape}")
    print(f"    完整mel频谱范围: [{full_mel_spec.min():.4f}, {full_mel_spec.max():.4f}]")
    
    # 使用griffin-lim重建
    try:
        print(f"    开始griffin-lim重建...")
        # 计算正确的hop_length以获得10秒音频
        # 640个时间步对应10秒，所以每个时间步是10/640秒
        time_per_step = 10.0 / 640  # 秒
        target_hop_length = int(sr * time_per_step)  # 采样点
        print(f"    目标hop_length: {target_hop_length} (对应10秒音频)")
        
        audio = librosa.feature.inverse.mel_to_audio(
            full_mel_spec, 
            sr=sr, 
            hop_length=target_hop_length, 
            n_fft=n_fft,
            n_iter=32
        )
        print(f"    griffin-lim重建成功，音频长度: {len(audio)} 采样点 ({len(audio)/sr:.2f}秒)")
        return audio
    except Exception as e:
        print(f"    音频重建失败: {e}")
        import traceback
        traceback.print_exc()
        # 如果重建失败，返回一个简单的正弦波
        duration = 10.0  # 固定10秒
        t = np.linspace(0, duration, int(sr * duration))
        # 使用mel值的平均值作为频率
        freq = 440 + np.mean(mel_power) * 100  # 基础频率440Hz + mel值的影响
        audio = np.sin(2 * np.pi * freq * t) * 0.1
        print(f"    生成备用正弦波，长度: {len(audio)} 采样点 ({len(audio)/sr:.2f}秒)")
        return audio

def visualize_mel_spectrogram(mel_spec, title="Mel Spectrogram", save_path=None):
    """可视化mel频谱"""
    plt.figure(figsize=(12, 6))
    
    # 处理不同形状的输入
    if len(mel_spec.shape) == 1:
        # 一维数组，需要重塑为 (1, time)
        mel_spec_viz = mel_spec.reshape(1, -1)
        print(f"    重塑一维数组为: {mel_spec_viz.shape}")
    elif len(mel_spec.shape) == 2 and mel_spec.shape[1] == 1:
        # (time, 1) 格式，转置为 (1, time)
        mel_spec_viz = mel_spec.T
        print(f"    转置二维数组为: {mel_spec_viz.shape}")
    else:
        mel_spec_viz = mel_spec
        print(f"    保持原始形状: {mel_spec_viz.shape}")
    
    # 由于我们只有单个mel频带，创建一个简单的热力图显示
    time_steps = mel_spec_viz.shape[1]
    time_axis = np.linspace(0, 10, time_steps)  # 10秒
    
    # 创建热力图
    plt.imshow(mel_spec_viz, aspect='auto', cmap='viridis', 
               extent=[0, 10, 0, 1], origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Band')
    plt.yticks([0.5], ['Single Mel Band'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    频谱图已保存到: {save_path}")
    
    plt.show()

def compare_predictions_vs_labels(outputs, labels, save_dir):
    """比较预测结果和真实标签"""
    if labels is None:
        print("没有标签数据，跳过比较")
        return
    
    # 选择前几个样本进行比较
    n_samples = min(5, len(outputs))
    
    for i in range(n_samples):
        pred = outputs[i].squeeze()
        true = labels[i].squeeze()
        
        # 创建时间轴（10秒，640个时间步）
        time_steps = len(pred)
        time_axis = np.linspace(0, 10, time_steps)  # 10秒
        
        plt.figure(figsize=(15, 5))
        
        # 预测结果
        plt.subplot(1, 2, 1)
        plt.plot(time_axis, pred, label='Predicted', color='red')
        plt.title(f'Sample {i+1}: Predicted Mel')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel Value (dB)')
        plt.legend()
        plt.grid(True)
        
        # 真实标签
        plt.subplot(1, 2, 2)
        plt.plot(time_axis, true, label='Ground Truth', color='blue')
        plt.title(f'Sample {i+1}: Ground Truth Mel')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel Value (dB)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'comparison_sample_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"比较图已保存到: {save_path}")
        plt.show()

def convert_to_audio_files(outputs, save_dir, sr=22050):
    """将预测结果转换为音频文件"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i, pred in enumerate(outputs):
        pred = pred.squeeze()
        print(f"\n处理样本 {i+1}:")
        print(f"  预测数据形状: {pred.shape}")
        print(f"  预测数据范围: [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"  预测数据均值: {pred.mean():.4f}")
        
        # 将预测的mel频谱转换为音频
        try:
            print(f"  开始音频转换...")
            audio = mel_to_audio(pred, sr=sr)
            print(f"  音频转换成功，音频长度: {len(audio)} 采样点")
            print(f"  音频范围: [{audio.min():.4f}, {audio.max():.4f}]")
            
            # 保存音频文件
            audio_path = os.path.join(save_dir, f'predicted_audio_{i+1}.wav')
            sf.write(audio_path, audio, sr)
            print(f"  音频文件已保存到: {audio_path}")
            
            # 可视化mel频谱
            mel_viz_path = os.path.join(save_dir, f'mel_spectrogram_{i+1}.png')
            visualize_mel_spectrogram(pred, f'Predicted Mel Spectrogram - Sample {i+1}', mel_viz_path)
            
        except Exception as e:
            print(f"  转换样本 {i+1} 时出错: {e}")
            import traceback
            print(f"  详细错误信息:")
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Visualize and convert inference results')
    parser.add_argument('--results_path', type=str, 
                       default='/home/binwen6/code/CBD/SSM2Mel/inference_results/inference_results.npz',
                       help='Path to inference results')
    parser.add_argument('--output_dir', type=str,
                       default='/home/binwen6/code/CBD/SSM2Mel/visualization_results',
                       help='Output directory for visualizations and audio files')
    parser.add_argument('--sample_rate', type=int, default=22050,
                       help='Target audio sample rate')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载推理结果
    print(f"加载推理结果: {args.results_path}")
    outputs, labels = load_inference_results(args.results_path)
    
    print(f"输出形状: {outputs.shape}")
    if labels is not None:
        print(f"标签形状: {labels.shape}")
    
    # 限制处理的样本数量
    outputs = outputs[:args.max_samples]
    if labels is not None:
        labels = labels[:args.max_samples]
    
    # 比较预测和真实标签
    if labels is not None:
        compare_predictions_vs_labels(outputs, labels, args.output_dir)
    
    # 转换为音频文件
    print("正在将预测结果转换为音频文件...")
    convert_to_audio_files(outputs, args.output_dir, sr=args.sample_rate)
    
    # 显示统计信息
    print("\n统计信息:")
    print(f"预测值范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"预测值均值: {outputs.mean():.4f}")
    print(f"预测值标准差: {outputs.std():.4f}")
    
    if labels is not None:
        print(f"标签值范围: [{labels.min():.4f}, {labels.max():.4f}]")
        print(f"标签值均值: {labels.mean():.4f}")
        print(f"标签值标准差: {labels.std():.4f}")
        
        # 计算相关系数
        from util.cal_pearson import pearson_metric
        correlation = pearson_metric(torch.tensor(outputs), torch.tensor(labels))
        print(f"Pearson相关系数: {correlation.mean().item():.4f}")

if __name__ == '__main__':
    main() 