#!/usr/bin/env python3
"""
模型输出诊断脚本
分析模型输出的Mel频谱质量，找出问题所在
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from SSM2Mel.visualize_results import load_inference_results

def analyze_mel_spectrogram(mel_spec, title="Mel Spectrogram Analysis"):
    """分析Mel频谱的特征"""
    print(f"\n=== {title} ===")
    print(f"形状: {mel_spec.shape}")
    print(f"数据类型: {mel_spec.dtype}")
    print(f"数值范围: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    print(f"均值: {mel_spec.mean():.4f}")
    print(f"标准差: {mel_spec.std():.4f}")
    print(f"中位数: {np.median(mel_spec):.4f}")
    
    # 计算动态范围
    dynamic_range = mel_spec.max() - mel_spec.min()
    print(f"动态范围: {dynamic_range:.4f} dB")
    
    # 计算时间维度的变化
    if len(mel_spec.shape) == 2:
        time_variance = np.var(mel_spec, axis=0)  # 每个时间步的方差
        freq_variance = np.var(mel_spec, axis=1)  # 每个频带的方差
        
        print(f"时间维度方差 - 均值: {time_variance.mean():.4f}, 最大值: {time_variance.max():.4f}")
        print(f"频带维度方差 - 均值: {freq_variance.mean():.4f}, 最大值: {freq_variance.max():.4f}")
        
        # 检查是否有时间结构
        time_correlation = np.corrcoef(mel_spec.T)  # 时间步之间的相关性
        print(f"时间相关性 - 均值: {np.mean(time_correlation):.4f}")
        
        # 检查频带之间的相关性
        freq_correlation = np.corrcoef(mel_spec)  # 频带之间的相关性
        print(f"频带相关性 - 均值: {np.mean(freq_correlation):.4f}")
    
    return {
        'shape': mel_spec.shape,
        'range': (mel_spec.min(), mel_spec.max()),
        'mean': mel_spec.mean(),
        'std': mel_spec.std(),
        'dynamic_range': dynamic_range
    }

def visualize_analysis(pred, true, save_dir):
    """可视化分析结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 时间序列对比
    plt.figure(figsize=(15, 10))
    
    # 预测值的时间序列
    plt.subplot(2, 3, 1)
    if len(pred.shape) == 2:
        # 显示前5个频带
        for i in range(min(5, pred.shape[1])):
            plt.plot(pred[:, i], alpha=0.7, label=f'Band {i+1}')
        plt.title('Predicted - Time Series (First 5 bands)')
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Value (dB)')
        plt.legend()
    else:
        plt.plot(pred)
        plt.title('Predicted - Time Series')
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Value (dB)')
    plt.grid(True, alpha=0.3)
    
    # 真实值的时间序列
    plt.subplot(2, 3, 2)
    if len(true.shape) == 2:
        # 显示前5个频带
        for i in range(min(5, true.shape[1])):
            plt.plot(true[:, i], alpha=0.7, label=f'Band {i+1}')
        plt.title('Ground Truth - Time Series (First 5 bands)')
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Value (dB)')
        plt.legend()
    else:
        plt.plot(true)
        plt.title('Ground Truth - Time Series')
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Value (dB)')
    plt.grid(True, alpha=0.3)
    
    # 2. 频谱图对比
    plt.subplot(2, 3, 3)
    if len(pred.shape) == 2:
        plt.imshow(pred.T, aspect='auto', origin='lower', cmap='viridis')
        plt.title('Predicted - Spectrogram')
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Bands')
        plt.colorbar(label='dB')
    else:
        plt.imshow(pred.reshape(1, -1), aspect='auto', origin='lower', cmap='viridis')
        plt.title('Predicted - Single Band')
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Band')
        plt.colorbar(label='dB')
    
    plt.subplot(2, 3, 4)
    if len(true.shape) == 2:
        plt.imshow(true.T, aspect='auto', origin='lower', cmap='viridis')
        plt.title('Ground Truth - Spectrogram')
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Bands')
        plt.colorbar(label='dB')
    else:
        plt.imshow(true.reshape(1, -1), aspect='auto', origin='lower', cmap='viridis')
        plt.title('Ground Truth - Single Band')
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Band')
        plt.colorbar(label='dB')
    
    # 3. 分布对比
    plt.subplot(2, 3, 5)
    plt.hist(pred.flatten(), bins=50, alpha=0.7, label='Predicted', density=True)
    plt.hist(true.flatten(), bins=50, alpha=0.7, label='Ground Truth', density=True)
    plt.xlabel('Mel Value (dB)')
    plt.ylabel('Density')
    plt.title('Value Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 相关性分析
    plt.subplot(2, 3, 6)
    if len(pred.shape) == 2 and len(true.shape) == 2:
        # 计算每个频带的相关系数
        correlations = []
        for i in range(min(pred.shape[1], true.shape[1])):
            corr = np.corrcoef(pred[:, i], true[:, i])[0, 1]
            correlations.append(corr)
        
        plt.plot(correlations)
        plt.title('Correlation by Frequency Band')
        plt.xlabel('Mel Band')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
    else:
        # 单频带情况
        corr = np.corrcoef(pred.flatten(), true.flatten())[0, 1]
        plt.text(0.5, 0.5, f'Correlation: {corr:.4f}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Correlation')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'detailed_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"详细分析图已保存到: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Diagnose model output quality')
    parser.add_argument('--results_path', type=str, 
                       default='/home/binwen6/code/CBD/SSM2Mel/inference_results/inference_results.npz',
                       help='Path to inference results')
    parser.add_argument('--output_dir', type=str,
                       default='/home/binwen6/code/CBD/SSM2Mel/diagnosis_results',
                       help='Output directory for diagnosis results')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to analyze (0-based)')
    
    args = parser.parse_args()
    
    # 加载推理结果
    print(f"加载推理结果: {args.results_path}")
    outputs, labels = load_inference_results(args.results_path)
    
    if outputs is None:
        print("无法加载推理结果")
        return
    
    print(f"输出形状: {outputs.shape}")
    if labels is not None:
        print(f"标签形状: {labels.shape}")
    
    # 选择要分析的样本
    sample_idx = min(args.sample_idx, len(outputs) - 1)
    pred = outputs[sample_idx]
    true = labels[sample_idx] if labels is not None else None
    
    print(f"\n分析样本 {sample_idx + 1}:")
    
    # 分析预测结果
    pred_analysis = analyze_mel_spectrogram(pred, "预测结果分析")
    
    # 分析真实标签
    if true is not None:
        true_analysis = analyze_mel_spectrogram(true, "真实标签分析")
        
        # 比较分析
        print(f"\n=== 对比分析 ===")
        print(f"预测值动态范围: {pred_analysis['dynamic_range']:.4f} dB")
        print(f"真实值动态范围: {true_analysis['dynamic_range']:.4f} dB")
        print(f"动态范围比率: {pred_analysis['dynamic_range'] / true_analysis['dynamic_range']:.4f}")
        
        # 可视化分析
        visualize_analysis(pred, true, args.output_dir)
    
    # 问题诊断
    print(f"\n=== 问题诊断 ===")
    
    if pred_analysis['dynamic_range'] < 10:
        print("❌ 问题1: 动态范围过小，模型输出过于平滑")
        print("   建议: 检查损失函数，可能需要增加L1损失的权重")
    
    if pred_analysis['std'] < 5:
        print("❌ 问题2: 标准差过小，缺乏变化")
        print("   建议: 检查数据归一化，可能需要调整归一化方法")
    
    if len(pred.shape) == 2 and pred.shape[1] > 1:
        # 检查频带之间的相关性
        freq_corr = np.corrcoef(pred)[0, 1]  # 前两个频带的相关性
        if freq_corr > 0.9:
            print("❌ 问题3: 频带之间相关性过高，模型没有学会区分不同频带")
            print("   建议: 检查模型架构，可能需要增加频带特定的处理")
    
    print("✅ 诊断完成")

if __name__ == '__main__':
    main() 