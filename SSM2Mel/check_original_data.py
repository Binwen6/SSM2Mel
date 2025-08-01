import os
import glob
import numpy as np
from collections import defaultdict

def check_original_data():
    """检查原始数据中的刺激分布"""
    original_data_dir = "/home/binwen6/code/CBD/BCMI/preprocessed_data"
    
    if not os.path.exists(original_data_dir):
        print(f"原始数据目录不存在: {original_data_dir}")
        return
    
    print("=== 原始数据刺激分布分析 ===")
    
    # 遍历所有被试
    subjects = os.listdir(original_data_dir)
    subjects = [s for s in subjects if os.path.isdir(os.path.join(original_data_dir, s))]
    
    for subject in subjects:
        subject_path = os.path.join(original_data_dir, subject)
        print(f"\n被试: {subject}")
        
        # 检查该被试下的所有刺激
        stimuli = os.listdir(subject_path)
        stimuli = [s for s in stimuli if os.path.isdir(os.path.join(subject_path, s))]
        
        print(f"  刺激数量: {len(stimuli)}")
        print(f"  刺激列表: {stimuli}")
        
        # 检查每个刺激的epoch数量
        for stimulus in stimuli:
            stimulus_path = os.path.join(subject_path, stimulus)
            npz_files = glob.glob(os.path.join(stimulus_path, f"{subject}_{stimulus}_epoch*.npz"))
            print(f"    {stimulus}: {len(npz_files)} 个epoch")
            
            # 检查第一个epoch的数据形状
            if npz_files:
                try:
                    data = np.load(npz_files[0])
                    eeg_shape = data['eeg_data'].shape
                    print(f"      EEG形状: {eeg_shape}")
                except Exception as e:
                    print(f"      读取失败: {e}")

def check_processed_data():
    """检查处理后的数据"""
    processed_data_dir = "/home/binwen6/code/CBD/SSM2Mel/data/split_data"
    
    print("\n=== 处理后数据检查 ===")
    
    test_files = glob.glob(os.path.join(processed_data_dir, "test_-_*"))
    
    for file_path in test_files:
        filename = os.path.basename(file_path)
        print(f"\n文件: {filename}")
        
        try:
            data = np.load(file_path)
            print(f"  形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            
            # 检查数据的变化
            if len(data.shape) == 2:
                # 计算相邻时间步的差异
                diff = np.diff(data, axis=0)
                print(f"  相邻时间步差异统计:")
                print(f"    均值: {np.mean(diff):.6f}")
                print(f"    标准差: {np.std(diff):.6f}")
                print(f"    最大值: {np.max(diff):.6f}")
                print(f"    最小值: {np.min(diff):.6f}")
                
                # 检查是否有重复的行
                unique_rows = np.unique(data.round(3), axis=0)
                print(f"  唯一行数: {len(unique_rows)} / {len(data)}")
                if len(unique_rows) < len(data) * 0.1:
                    print(f"    警告: 数据重复率很高，可能所有片段都来自同一刺激")
                    
        except Exception as e:
            print(f"  读取失败: {e}")

if __name__ == '__main__':
    check_original_data()
    check_processed_data() 