#!/usr/bin/env python3
"""
音频对比测试脚本
从mel spectrogram重建音频波形，用于对比predicted和ground truth的结果
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
import argparse
from pathlib import Path
import json
from scipy.signal import resample

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

def find_unique_stimuli(labels, threshold=0.8):
    """
    找到真正不同的stimulus
    
    Args:
        labels: 真实标签数据 (n_samples, time_steps, n_mels)
        threshold: 相似度阈值，默认0.8（更宽松的阈值）
    
    Returns:
        unique_indices: 唯一stimulus的索引列表
    """
    print(f"分析 {len(labels)} 个样本，寻找唯一stimulus...")
    print(f"相似度阈值: {threshold}")
    
    unique_indices = [0]  # 第一个样本总是唯一的
    print(f"样本 0 是第一个stimulus")
    
    for i in range(1, len(labels)):
        is_unique = True
        max_similarity = 0
        most_similar_idx = -1
        
        for j in unique_indices:
            # 计算两个样本的相似度
            similarity = np.corrcoef(labels[i].flatten(), labels[j].flatten())[0, 1]
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_idx = j
            
            if similarity > threshold:
                is_unique = False
                print(f"样本 {i} 与样本 {j} 相似度: {similarity:.3f} (不唯一)")
                break
        
        if is_unique:
            unique_indices.append(i)
            print(f"样本 {i} 是唯一的stimulus (与最相似样本的相似度: {max_similarity:.3f})")
        else:
            print(f"样本 {i} 与样本 {most_similar_idx} 最相似，相似度: {max_similarity:.3f}")
    
    print(f"找到 {len(unique_indices)} 个唯一stimulus: {unique_indices}")
    return unique_indices

def mel_to_audio_griffin_lim(mel_spec, sr=22050, n_fft=2048):
    """
    使用Griffin-Lim算法将mel spectrogram转换为音频波形
    基于原有的visualize_results.py中的mel_to_audio函数逻辑
    
    Args:
        mel_spec: mel spectrogram (time_steps, n_mels)
        sr: 采样率
        n_fft: FFT窗口大小
    
    Returns:
        audio: 音频波形
    """
    print(f"mel_to_audio_griffin_lim 输入形状: {mel_spec.shape}")
    print(f"mel_to_audio_griffin_lim 输入范围: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    
    # 检测任务模式
    if len(mel_spec.shape) == 2 and mel_spec.shape[1] > 1:
        task_mode = "mel_spectrogram"
        print(f"检测到完整Mel频谱重建模式 (频带数: {mel_spec.shape[1]})")
    else:
        task_mode = "envelope"
        print(f"检测到音频包络重建模式")
    
    if task_mode == "mel_spectrogram":
        # 完整Mel频谱重建模式：直接使用多频带输入
        print(f"直接使用多频带Mel频谱输入")
        
        # 确保mel_spec是 (time, num_mel_bands) 格式
        if len(mel_spec.shape) == 2 and mel_spec.shape[1] > 1:
            # 已经是多频带格式，直接使用
            full_mel_spec = librosa.db_to_power(mel_spec.T)  # 转换为 (num_mel_bands, time)
            print(f"直接使用多频带Mel频谱，形状: {full_mel_spec.shape}")
        else:
            print(f"警告：期望多频带输入但收到单频带，使用高斯分布扩展...")
            # 如果意外收到单频带，使用高斯分布扩展
            if len(mel_spec.shape) == 1:
                mel_spec = mel_spec.reshape(-1, 1)
            elif len(mel_spec.shape) == 2 and mel_spec.shape[0] == 1:
                mel_spec = mel_spec.T
            
            mel_power = librosa.db_to_power(mel_spec)
            n_mels = 128
            freqs = np.linspace(0, 1, n_mels)
            center_freq = 0.5
            bandwidth = 0.3
            weights = np.exp(-((freqs - center_freq) ** 2) / (2 * bandwidth ** 2))
            weights = weights.reshape(-1, 1)
            full_mel_spec = mel_power.T * weights
    
    else:
        raise ValueError(f"不支持的任务模式: {task_mode}")
    
    # 使用griffin-lim重建
    try:
        print(f"开始griffin-lim重建...")
        # 计算正确的hop_length以获得10秒音频
        # 640个时间步对应10秒，所以每个时间步是10/640秒
        time_per_step = 10.0 / 640  # 秒
        target_hop_length = int(sr * time_per_step)  # 采样点
        print(f"目标hop_length: {target_hop_length} (对应10秒音频)")
        
        audio = librosa.feature.inverse.mel_to_audio(
            full_mel_spec, 
            sr=sr, 
            hop_length=target_hop_length, 
            n_fft=n_fft,
            n_iter=32
        )
        print(f"griffin-lim重建成功，音频长度: {len(audio)} 采样点 ({len(audio)/sr:.2f}秒)")
        return audio
    except Exception as e:
        print(f"音频重建失败: {e}")
        import traceback
        traceback.print_exc()
        # 如果重建失败，返回一个简单的正弦波
        duration = 10.0  # 固定10秒
        t = np.linspace(0, duration, int(sr * duration))
        freq = 440 + np.mean(full_mel_spec) * 100  # 基础频率440Hz + mel值的影响
        audio = np.sin(2 * np.pi * freq * t) * 0.1
        print(f"生成备用正弦波，长度: {len(audio)} 采样点 ({len(audio)/sr:.2f}秒)")
        return audio

def create_comparison_audio(outputs, labels, output_dir, sample_indices=None, similarity_threshold=0.8):
    """
    创建对比音频文件
    
    Args:
        outputs: 预测的mel spectrograms
        labels: 真实的mel spectrograms
        output_dir: 输出目录
        sample_indices: 要处理的样本索引列表
        similarity_threshold: 相似度阈值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if sample_indices is None:
        # 找到唯一stimulus
        sample_indices = find_unique_stimuli(labels, similarity_threshold)
    
    print(f"开始处理 {len(sample_indices)} 个唯一样本的对比音频...")
    
    for i, sample_idx in enumerate(sample_indices):
        print(f"\n处理样本 {i + 1}/{len(sample_indices)} (索引: {sample_idx})")
        
        # 获取预测和真实值
        pred_mel = outputs[sample_idx].squeeze()
        true_mel = labels[sample_idx].squeeze()
        
        print(f"预测mel形状: {pred_mel.shape}")
        print(f"真实mel形状: {true_mel.shape}")
        
        # 转换为音频
        print("转换预测mel为音频...")
        pred_audio = mel_to_audio_griffin_lim(pred_mel)
        
        print("转换真实mel为音频...")
        true_audio = mel_to_audio_griffin_lim(true_mel)
        
        # 保存音频文件
        pred_path = os.path.join(output_dir, f"stimulus_{sample_idx}_predicted.wav")
        true_path = os.path.join(output_dir, f"stimulus_{sample_idx}_ground_truth.wav")
        
        sf.write(pred_path, pred_audio, 22050)
        sf.write(true_path, true_audio, 22050)
        
        print(f"已保存音频文件:")
        print(f"  预测音频: {pred_path}")
        print(f"  真实音频: {true_path}")
        
        # 创建对比图
        create_comparison_plot(pred_mel, true_mel, sample_idx, output_dir)

def create_comparison_plot(pred_mel, true_mel, sample_idx, output_dir):
    """创建mel spectrogram对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 预测mel spectrogram
    im1 = axes[0].imshow(pred_mel.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f'Stimulus {sample_idx} - Predicted Mel Spectrogram')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Mel Bands')
    plt.colorbar(im1, ax=axes[0], label='Mel Value (dB)')
    
    # 真实mel spectrogram
    im2 = axes[1].imshow(true_mel.T, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f'Stimulus {sample_idx} - Ground Truth Mel Spectrogram')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Mel Bands')
    plt.colorbar(im2, ax=axes[1], label='Mel Value (dB)')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"stimulus_{sample_idx}_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='音频对比测试')
    parser.add_argument('--results_path', type=str, 
                       default='SSM2Mel/inference_results/inference_results.npz',
                       help='推理结果文件路径')
    parser.add_argument('--output_dir', type=str, 
                       default='SSM2Mel/comparison_audio_results',
                       help='输出目录')
    parser.add_argument('--sample_indices', type=str, default=None,
                       help='要处理的样本索引，用逗号分隔，如 "0,1,2"')
    parser.add_argument('--similarity_threshold', type=float, default=0.9,
                       help='相似度阈值，用于确定唯一stimulus (默认0.8，更宽松)')
    
    args = parser.parse_args()
    
    # 解析样本索引
    sample_indices = None
    if args.sample_indices:
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(',')]
    
    print(f"加载推理结果: {args.results_path}")
    outputs, labels = load_inference_results(args.results_path)
    
    print(f"输出形状: {outputs.shape}")
    if labels is not None:
        print(f"标签形状: {labels.shape}")
    
    # 创建对比音频
    create_comparison_audio(outputs, labels, args.output_dir, sample_indices, args.similarity_threshold)
    
    print(f"\n对比测试完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 