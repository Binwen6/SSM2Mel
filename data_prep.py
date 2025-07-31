import os
import numpy as np
import librosa
import librosa.display
import torch
from sklearn.model_selection import train_test_split
from scipy.signal import resample
import glob
import warnings

warnings.filterwarnings('ignore') # Suppress librosa warnings

# --- 您的原始预处理数据配置 ---
class OriginalDataConfig:
    # 您的第一阶段预处理脚本的输出目录
    OUTPUT_DIR = "/home/binwen6/code/CBD/BCMI/preprocessed_data" 
    # 音频时长信息，来自您的第一阶段配置
    STIMULUS_DURATIONS = {
        's1': 9.95, 's2': 9.95, 's3': 8.96, 's4': 9.95,
        's5': 9.95, 's6': 9.95, 's7': 9.95, 's8': 9.95
    }
    # 原始EEG数据特性
    ORIGINAL_SFREQ_EEG = 500 # 原始EEG采样率
    ORIGINAL_CHANNELS_EEG = 32 # 原始EEG通道数

# --- SSM2MEL项目期望的数据配置 ---
class SSM2MelConfig:
    # SSM2MEL项目的数据集根目录 (来自main.py的args.dataset_folder)
    DATASET_FOLDER = "/home/binwen6/code/CBD/SSM2Mel/data" 
    # SSM2MEL项目的数据分割子目录 (来自main.py的args.split_folder)
    SPLIT_FOLDER = "split_data" 
    # 模型期望的EEG/Mel特征采样率 (来自main.py的args.sample_rate)
    TARGET_SFREQ_FEATURES = 64 
    # 模型期望的EEG输入特征维度 (来自main.py的args.in_channel)
    TARGET_CHANNELS_EEG_MODEL_INPUT = 64 
    # 模型期望的序列长度 (来自main.py的args.sample_rate * args.win_len)
    TARGET_SEQUENCE_LENGTH = 640 
    # 模型期望的Mel输出特征维度 (来自SSM2Mel.py的Decoder中fc层的输出)
    TARGET_MEL_BANDS = 1 

# --- 音频文件路径 (TODO: 请根据您的实际情况修改此路径) ---
# 示例: AUDIO_FILES_DIR = "/home/binwen6/code/CBD/Generate/audio_stimuli"
AUDIO_FILES_DIR = "/home/binwen6/code/CBD/BCMI/dataset/music_gen_wav_22050" 
# 音频文件名映射，确保与您的刺激名称对应
# 修复: 将集合 (set) 更改为字典 (dict)
AUDIO_FILE_MAP = {
    's1': 'All Of Me - All of Me (Karaoke Version).wav',
    's2': 'Richard Clayderman - 梦中的鸟.wav',
    's3': 'Robin Spielberg - Turn the Page.wav',
    's4': 'dylanf - 梦中的婚礼 (经典钢琴版).wav',
    's5': '文武贝 - 夜的钢琴曲5.wav',
    's6': '昼夜 - 千与千寻 (钢琴版).wav',
    's7': '演奏曲 - 【钢琴Piano】雨中漫步Stepping On The Rainy S.wav',
    's8': '郭宴 - 天空之城 (钢琴版).wav'
}

def generate_mel_spectrogram(audio_path, target_sfreq_features, target_sequence_length, target_mel_bands):
    """
    加载音频文件，计算其Mel频谱，并处理以匹配目标序列长度和特征维度。
    
    Args:
        audio_path (str): 音频文件完整路径。
        target_sfreq_features (int): 目标特征采样率 (例如 64Hz)。
        target_sequence_length (int): 目标序列长度 (例如 640)。
        target_mel_bands (int): 目标Mel频谱频带数 (例如 1)。
        
    Returns:
        np.ndarray: 处理后的Mel频谱数据，形状为 (target_sequence_length, target_mel_bands)。
    """
    try:
        y, sr = librosa.load(audio_path, sr=None) # 加载原始采样率音频

        # 计算hop_length以获得目标特征采样率 (例如 64帧/秒)
        # 假设我们希望每秒有 target_sfreq_features 帧
        hop_length = int(sr / target_sfreq_features)
        
        # 确保hop_length至少为1，避免除以0或过小
        if hop_length == 0:
            hop_length = 1 

        # 常用FFT窗口大小
        n_fft = 2048 
        
        # 计算Mel频谱
        # 默认n_mels通常是128或80，这里我们先计算一个合理的数量，后续再聚合到target_mel_bands
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
        
        # 转换为分贝尺度
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 确保Mel频谱的维度匹配目标 (target_mel_bands = 1)
        if target_mel_bands == 1:
            # 聚合所有频带到单个特征 (例如，取平均值)
            mel_feature = np.mean(mel_spec_db, axis=0, keepdims=True) # 形状 (1, n_frames)
        else:
            # 如果模型期望多个频带，则在这里进行切片或调整n_mels参数
            if mel_spec_db.shape[0] < target_mel_bands:
                # 如果计算出的频带数不足，进行填充
                padding = np.zeros((target_mel_bands - mel_spec_db.shape[0], mel_spec_db.shape[1]))
                mel_feature = np.concatenate((mel_spec_db, padding), axis=0)
            else:
                mel_feature = mel_spec_db[:target_mel_bands, :] # 切片到目标频带数
        
        # 确保Mel频谱的序列长度匹配目标 (target_sequence_length = 640)
        current_frames = mel_feature.shape[1]
        if current_frames > target_sequence_length:
            mel_feature = mel_feature[:, :target_sequence_length] # 截断
        elif current_frames < target_sequence_length:
            padding = np.zeros((mel_feature.shape[0], target_sequence_length - current_frames))
            mel_feature = np.concatenate((mel_feature, padding), axis=1) # 填充
        
        # 转置为 (sequence_length, target_mel_bands) 形状以匹配模型输入约定
        return mel_feature.T 
    except Exception as e:
        print(f"Error generating Mel spectrogram for {audio_path}: {e}")
        return None

def transform_eeg_channels(eeg_data_32_channels):
    """
    将32通道的EEG数据转换为模型期望的64特征维度。
    这是一个关键的特征工程步骤。此处提供一个简单的示例实现。
    
    Args:
        eeg_data_32_channels (np.ndarray): EEG数据，形状为 (32, 样本数)。
        
    Returns:
        np.ndarray: 转换后的EEG数据，形状为 (样本数, 64)。
    """
    n_samples = eeg_data_32_channels.shape[1]
    
    # 将 (32, n_samples) 转置为 (n_samples, 32)，以便于按时间点进行特征扩展
    eeg_data_transposed = eeg_data_32_channels.T # 形状 (n_samples, 32)
    
    # --- 简单的通道扩展策略 (请根据您的需求进行优化) ---
    # 策略1: 简单复制通道以达到64个特征
    # 假设我们简单地将原始32个通道复制一份，得到64个特征
    if OriginalDataConfig.ORIGINAL_CHANNELS_EEG * 2 == SSM2MelConfig.TARGET_CHANNELS_EEG_MODEL_INPUT:
        transformed_data = np.concatenate((eeg_data_transposed, eeg_data_transposed), axis=1)
    # 策略2: 如果不是简单的2倍关系，则进行零填充
    else:
        transformed_data = np.zeros((n_samples, SSM2MelConfig.TARGET_CHANNELS_EEG_MODEL_INPUT))
        transformed_data[:, :OriginalDataConfig.ORIGINAL_CHANNELS_EEG] = eeg_data_transposed
        print(f"警告: EEG通道从 {OriginalDataConfig.ORIGINAL_CHANNELS_EEG} 扩展到 {SSM2MelConfig.TARGET_CHANNELS_EEG_MODEL_INPUT} 正在使用简单的填充/复制策略。考虑进行更合适的特征工程。")
        
    # 返回形状为 (样本数, 64) 的数据
    return transformed_data

def convert_data_for_ssm2mel():
    # 1. 创建目标输出目录
    output_base_dir = os.path.join(SSM2MelConfig.DATASET_FOLDER, SSM2MelConfig.SPLIT_FOLDER)
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"✅ 已创建输出目录: {output_base_dir}")

    all_data_points_per_subject = {} # 存储每个被试的所有处理后的数据点

    # 2. 遍历加载预处理后的EEG数据并生成Mel频谱
    for subject_folder in os.listdir(OriginalDataConfig.OUTPUT_DIR):
        subject_path = os.path.join(OriginalDataConfig.OUTPUT_DIR, subject_folder)
        if not os.path.isdir(subject_path):
            continue
        
        subject_name_base = subject_folder # 被试名称，例如 "jia_haoxuan"
        print(f"\n正在处理被试: {subject_name_base}")
        
        # 初始化当前被试的数据列表
        if subject_name_base not in all_data_points_per_subject:
            all_data_points_per_subject[subject_name_base] = {
                'eeg_epochs': [], 
                'mel_epochs': [], 
                'subject_id': None # 稍后分配
            }

        # 存储当前被试已处理的刺激的Mel数据，避免重复计算
        mel_data_cache = {} 

        for stimulus_folder in os.listdir(subject_path):
            stimulus_path = os.path.join(subject_path, stimulus_folder)
            if not os.path.isdir(stimulus_path):
                continue
            
            stimulus_name = stimulus_folder # 刺激名称，例如 's1'

            # 检查Mel数据是否已缓存
            if stimulus_name not in mel_data_cache:
                audio_file_name = AUDIO_FILE_MAP.get(stimulus_name)
                if not audio_file_name:
                    print(f"⚠️ 找不到刺激 '{stimulus_name}' 对应的音频文件映射。跳过。")
                    continue
                
                audio_full_path = os.path.join(AUDIO_FILES_DIR, audio_file_name)
                if not os.path.exists(audio_full_path):
                    print(f"❌ 音频文件未找到: {audio_full_path}。跳过刺激 '{stimulus_name}'。")
                    continue

                # 生成Mel频谱 (每个刺激类型只需生成一次)
                print(f"正在为 '{stimulus_name}' 生成Mel频谱...")
                mel_data = generate_mel_spectrogram(
                    audio_full_path, 
                    SSM2MelConfig.TARGET_SFREQ_FEATURES,
                    SSM2MelConfig.TARGET_SEQUENCE_LENGTH,
                    SSM2MelConfig.TARGET_MEL_BANDS
                )
                if mel_data is None: # 如果Mel生成失败，跳过此刺激
                    continue
                mel_data_cache[stimulus_name] = mel_data
                print(f"  Mel数据形状: {mel_data.shape}")
            else:
                mel_data = mel_data_cache[stimulus_name]

            # 遍历当前刺激下的所有EEG分段文件
            npz_files = glob.glob(os.path.join(stimulus_path, f"{subject_name_base}_{stimulus_name}_epoch*.npz"))
            if not npz_files:
                print(f"  警告: 未找到 '{stimulus_name}' 的EEG分段文件。")

            for npz_file in npz_files:
                try:
                    data = np.load(npz_file)
                    eeg_data_original = data['eeg_data'] # 形状 (32, 原始样本数)
                    
                    # 重采样EEG数据 (从500Hz到64Hz，并统一长度为640)
                    # resample函数期望 (..., n_samples_original) -> (..., n_samples_target)
                    eeg_data_resampled = resample(eeg_data_original, SSM2MelConfig.TARGET_SEQUENCE_LENGTH, axis=1) # 形状 (32, 640)
                    
                    # 通道转换 (从32通道到64特征)
                    eeg_data_transformed = transform_eeg_channels(eeg_data_resampled) # 形状 (640, 64)

                    # 添加到当前被试的数据列表
                    all_data_points_per_subject[subject_name_base]['eeg_epochs'].append(eeg_data_transformed)
                    all_data_points_per_subject[subject_name_base]['mel_epochs'].append(mel_data) # Mel数据与每个epoch对应
                    
                    print(f"  已处理 {os.path.basename(npz_file)}。EEG形状: {eeg_data_transformed.shape}")

                except Exception as e:
                    print(f"❌ 处理 {npz_file} 时出错: {e}")

    # 3. 数据分割 (按被试进行分割)
    unique_subjects = list(all_data_points_per_subject.keys())
    # 训练集:验证集:测试集 比例约为 70%:15%:15%
    train_subjects, val_test_subjects = train_test_split(unique_subjects, test_size=0.3, random_state=42)
    val_subjects, test_subjects = train_test_split(val_test_subjects, test_size=0.5, random_state=42) 
    
    # 为每个被试分配一个唯一的ID (用于模型中的sub_id输入)
    subject_id_map = {subj: i for i, subj in enumerate(unique_subjects)}
    for subj_name in all_data_points_per_subject:
        all_data_points_per_subject[subj_name]['subject_id'] = subject_id_map[subj_name]

    print(f"\n数据分割摘要:")
    print(f"  训练集被试 ({len(train_subjects)}): {train_subjects}")
    print(f"  验证集被试 ({len(val_subjects)}): {val_subjects}")
    print(f"  测试集被试 ({len(test_subjects)}): {test_subjects}")

    # 4. 保存数据为 .npy 文件 (按被试和分割类型聚合)
    for split_type, subjects_list in [('train', train_subjects), ('val', val_subjects), ('test', test_subjects)]:
        for subj_name in subjects_list:
            if subj_name not in all_data_points_per_subject:
                print(f"警告: 被试 '{subj_name}' 在数据点中不存在，跳过保存。")
                continue

            subj_data = all_data_points_per_subject[subj_name]
            
            if not subj_data['eeg_epochs'] or not subj_data['mel_epochs']:
                print(f"警告: 被试 '{subj_name}' 在 '{split_type}' 集中没有有效的EEG或Mel数据，跳过保存。")
                continue

            # 拼接所有EEG分段 (形状: (总样本数, 64))
            concatenated_eeg = np.concatenate(subj_data['eeg_epochs'], axis=0) 
            # 拼接所有Mel分段 (形状: (总样本数, 1))
            concatenated_mel = np.concatenate(subj_data['mel_epochs'], axis=0) 

            # 保存EEG数据
            eeg_filename = f"{split_type}_-_{subj_name}_-_eeg.npy"
            eeg_filepath = os.path.join(output_base_dir, eeg_filename)
            np.save(eeg_filepath, concatenated_eeg)
            print(f"💾 已保存 {eeg_filepath} (形状: {concatenated_eeg.shape})")

            # 保存Mel数据
            mel_filename = f"{split_type}_-_{subj_name}_-_mel.npy"
            mel_filepath = os.path.join(output_base_dir, mel_filename)
            np.save(mel_filepath, concatenated_mel)
            print(f"💾 已保存 {mel_filepath} (形状: {concatenated_mel.shape})")

            # TODO: 如果需要保存 subject_id，可以在 RegressionDataset 中处理，
            # 或者将其作为单独的元数据文件保存。
            # SSM2MEL的main.py中，sub_id是作为DataLoader的第三个返回值，
            # 这意味着它应该在RegressionDataset的__getitem__中返回。
            # 通常，RegressionDataset会根据文件名解析出subject_id。
            # 在这里，我们只需确保数据文件本身是正确的。

    print("\n🎉 数据转换过程完成。请检查输出目录中的 .npy 文件。")
    print("\n下一步: 确保您的SSM2MEL训练脚本的 `args.dataset_folder` 和 `args.split_folder` 参数指向正确的位置。")

if __name__ == "__main__":

    convert_data_for_ssm2mel()
