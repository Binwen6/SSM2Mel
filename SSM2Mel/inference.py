import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from models.SSM2Mel import Decoder
from util.dataset import RegressionDataset
import glob

def load_model(model_path, device, **model_args):
    """加载训练好的模型"""
    model = Decoder(**model_args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def inference_single_sample(model, eeg_data, sub_id, device):
    """对单个样本进行推理"""
    model.eval()
    with torch.no_grad():
        # 确保输入格式正确
        if len(eeg_data.shape) == 2:  # [T, C]
            eeg_data = eeg_data.unsqueeze(0)  # [1, T, C]
        
        eeg_data = eeg_data.to(device)
        sub_id = torch.tensor([sub_id]).to(device)
        
        # 进行推理
        output = model(eeg_data, sub_id)
        return output.squeeze()

def inference_on_dataset(model, data_loader, device):
    """对整个数据集进行推理"""
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, sub_id in data_loader:
            inputs = inputs.squeeze(0).to(device)
            labels = labels.squeeze(0).to(device)
            sub_id = sub_id.to(device)
            
            outputs = model(inputs, sub_id)
            
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return np.concatenate(all_outputs, axis=0), np.concatenate(all_labels, axis=0)

def main():
    parser = argparse.ArgumentParser(description='SSM2Mel Inference')
    parser.add_argument('--model_path', type=str, 
                       default='/home/binwen6/code/CBD/SSM2Mel/result_model_conformer/model_epoch100.pt',
                       help='Path to the trained model')
    parser.add_argument('--data_folder', type=str, 
                       default='/home/binwen6/code/CBD/SSM2Mel/data/split_data',
                       help='Path to the test data folder')
    parser.add_argument('--output_folder', type=str, 
                       default='/home/binwen6/code/CBD/SSM2Mel/inference_results',
                       help='Path to save inference results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    
    # 模型参数（需要与训练时保持一致）
    parser.add_argument('--win_len', type=int, default=10)
    parser.add_argument('--sample_rate', type=int, default=64)
    parser.add_argument('--in_channel', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_inner', type=int, default=1024)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--fft_conv1d_kernel', type=tuple, default=(9, 1))
    parser.add_argument('--fft_conv1d_padding', type=tuple, default=(4, 0))
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--g_con', default=True)
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出文件夹
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 准备模型参数
    model_args = {
        'in_channel': args.in_channel,
        'd_model': args.d_model,
        'd_inner': args.d_inner,
        'n_head': args.n_head,
        'n_layers': args.n_layers,
        'fft_conv1d_kernel': args.fft_conv1d_kernel,
        'fft_conv1d_padding': args.fft_conv1d_padding,
        'dropout': args.dropout,
        'g_con': args.g_con
    }
    
    # 加载模型
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device, **model_args)
    print("Model loaded successfully!")
    
    # 准备测试数据
    input_length = args.sample_rate * args.win_len
    features = ["eeg"] + ["mel"]
    
    # 获取测试文件
    test_files = [x for x in glob.glob(os.path.join(args.data_folder, "test_-_*")) 
                  if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    
    if not test_files:
        print("No test files found!")
        return
    
    print(f"Found {len(test_files)} test files")
    
    # 创建测试数据集
    test_set = RegressionDataset(test_files, input_length, args.in_channel, 'test', args.g_con)
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=None,
        drop_last=True,
        shuffle=False
    )
    
    # 进行推理
    print("Starting inference...")
    outputs, labels = inference_on_dataset(model, test_dataloader, device)
    
    # 保存结果
    output_file = os.path.join(args.output_folder, 'inference_results.npz')
    np.savez(output_file, outputs=outputs, labels=labels)
    print(f"Inference results saved to {output_file}")
    
    # 计算一些基本统计信息
    print(f"Output shape: {outputs.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"Labels range: [{labels.min():.4f}, {labels.max():.4f}]")
    
    # 计算相关系数
    from util.cal_pearson import pearson_metric
    correlation = pearson_metric(torch.tensor(outputs), torch.tensor(labels))
    print(f"Pearson correlation: {correlation.mean().item():.4f}")

if __name__ == '__main__':
    main() 