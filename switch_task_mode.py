#!/usr/bin/env python3
"""
任务模式切换脚本
用于在音频包络重建和完整Mel频谱重建之间切换
"""

import os
import sys

def switch_task_mode(mode):
    """
    切换任务模式
    
    Args:
        mode (str): "envelope" 或 "mel_spectrogram"
    """
    if mode not in ["envelope", "mel_spectrogram"]:
        print(f"错误：未知的任务模式 '{mode}'")
        print("支持的模式：envelope, mel_spectrogram")
        return False
    
    # 读取当前配置
    config_file = "data_prep.py"
    if not os.path.exists(config_file):
        print(f"错误：找不到配置文件 {config_file}")
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新任务模式
    old_line = "    TASK_MODE = \"envelope\"  # 可选: \"envelope\" (音频包络重建) 或 \"mel_spectrogram\" (完整Mel频谱重建)"
    new_line = f"    TASK_MODE = \"{mode}\"  # 可选: \"envelope\" (音频包络重建) 或 \"mel_spectrogram\" (完整Mel频谱重建)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
    else:
        # 尝试其他可能的格式
        old_line = "    TASK_MODE = \"mel_spectrogram\""
        new_line = f"    TASK_MODE = \"{mode}\""
        if old_line in content:
            content = content.replace(old_line, new_line)
        else:
            print("错误：无法找到TASK_MODE配置行")
            return False
    
    # 写回文件
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 已成功切换到 {mode} 模式")
    
    # 显示当前配置信息
    if mode == "envelope":
        print("📊 当前配置：音频包络重建模式")
        print("   - 输出维度：1个频带")
        print("   - 目标：重建语音包络")
    else:
        print("📊 当前配置：完整Mel频谱重建模式")
        print("   - 输出维度：80个频带")
        print("   - 目标：重建完整的多频带Mel频谱")
    
    return True

def show_current_mode():
    """显示当前任务模式"""
    config_file = "data_prep.py"
    if not os.path.exists(config_file):
        print(f"错误：找不到配置文件 {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找TASK_MODE行
    for line in content.split('\n'):
        if 'TASK_MODE = "' in line:
            mode = line.split('"')[1]
            print(f"📋 当前任务模式：{mode}")
            if mode == "envelope":
                print("   - 音频包络重建模式")
            else:
                print("   - 完整Mel频谱重建模式")
            return
    
    print("❌ 无法找到TASK_MODE配置")

def main():
    if len(sys.argv) < 2:
        print("使用方法：")
        print("  python switch_task_mode.py envelope     # 切换到音频包络重建模式")
        print("  python switch_task_mode.py mel_spectrogram  # 切换到完整Mel频谱重建模式")
        print("  python switch_task_mode.py status       # 显示当前模式")
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        show_current_mode()
    elif command in ["envelope", "mel_spectrogram"]:
        switch_task_mode(command)
    else:
        print(f"错误：未知命令 '{command}'")
        print("支持的命令：envelope, mel_spectrogram, status")

if __name__ == "__main__":
    main() 