import os
import glob
import numpy as np
from collections import defaultdict

def analyze_test_data_distribution():
    """分析测试数据的分布情况"""
    data_folder = '/home/binwen6/code/CBD/SSM2Mel/data/split_data'
    
    # 获取所有测试文件
    test_files = glob.glob(os.path.join(data_folder, "test_-_*"))
    
    print(f"找到 {len(test_files)} 个测试文件")
    
    # 分析被试分布
    subjects = defaultdict(list)
    for file_path in test_files:
        filename = os.path.basename(file_path)
        parts = filename.split("_-_")
        if len(parts) >= 2:
            subject_name = parts[1]
            feature_type = parts[2].split(".")[0]  # 去掉.npy
            subjects[subject_name].append(feature_type)
    
    print(f"\n被试分布:")
    for subject, features in subjects.items():
        print(f"  被试: {subject}, 特征: {features}")
    
    # 分析每个被试的数据长度
    print(f"\n每个被试的数据长度:")
    for subject, features in subjects.items():
        for feature in features:
            file_path = os.path.join(data_folder, f"test_-_{subject}_-_{feature}.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)
                print(f"  {subject}_{feature}: {data.shape}")
    
    # 分析刺激分布（通过文件名推断）
    print(f"\n可能的刺激分布:")
    print("注意：这里只能通过文件名推断，实际刺激信息需要查看原始数据")
    
    # 检查是否有多个刺激的数据
    print(f"\n检查是否有多个刺激的数据:")
    for subject, features in subjects.items():
        eeg_file = os.path.join(data_folder, f"test_-_{subject}_-_eeg.npy")
        if os.path.exists(eeg_file):
            data = np.load(eeg_file)
            print(f"  {subject}_eeg: {data.shape}")
            # 如果数据很长，可能包含多个刺激
            if data.shape[0] > 6400:  # 超过10个10秒片段
                print(f"    -> 可能包含多个刺激片段")
            else:
                print(f"    -> 可能只包含单个刺激")

def check_inference_results():
    """检查推理结果的分布"""
    results_path = '/home/binwen6/code/CBD/SSM2Mel/inference_results/inference_results.npz'
    
    if os.path.exists(results_path):
        data = np.load(results_path)
        outputs = data['outputs']
        labels = data['labels']
        
        print(f"\n推理结果分析:")
        print(f"  输出形状: {outputs.shape}")
        print(f"  标签形状: {labels.shape}")
        
        # 分析预测值的分布
        print(f"  预测值统计:")
        print(f"    范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
        print(f"    均值: {outputs.mean():.4f}")
        print(f"    标准差: {outputs.std():.4f}")
        
        # 分析标签值的分布
        print(f"  标签值统计:")
        print(f"    范围: [{labels.min():.4f}, {labels.max():.4f}]")
        print(f"    均值: {labels.mean():.4f}")
        print(f"    标准差: {labels.std():.4f}")
        
        # 检查是否有重复的预测
        unique_predictions = np.unique(outputs.round(2))
        print(f"  唯一预测值数量: {len(unique_predictions)}")
        if len(unique_predictions) < 10:
            print(f"    警告：预测值变化很小，可能所有样本都来自同一刺激")
    else:
        print(f"推理结果文件不存在: {results_path}")

if __name__ == '__main__':
    print("=== 测试数据分布分析 ===")
    analyze_test_data_distribution()
    
    print("\n=== 推理结果分析 ===")
    check_inference_results() 