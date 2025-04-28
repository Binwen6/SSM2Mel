import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
import pdb
from models.SubLayers import MultiHeadAttention, PositionwiseFeedForward

from torch.nn.utils import weight_norm



from torch.nn import init


import einops
from einops import rearrange
from einops.layers.torch import Rearrange

from s4_block.s4_model import S4Model


from mamba_ssm import Mamba
import math

from models.bimamba import Bimamba_outer





class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
 
        # self.mk = KANLinear(d_model, S)
        self.mk = nn.Linear(d_model, S, bias=False)

        self.mv = nn.Linear(S, d_model, bias=False)
        # self.mv = KANLinear(S, d_model)

        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model

        return out


class UNetModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNetModule, self).__init__()

        self.s4_model1 = S4Model(d_input=64,
            d_output=160,
            d_model=64,
            n_layers=8,
            dropout=0.3,
            prenorm=False)


        self.s4_model2 = S4Model(d_input=64,
            d_output=320,
            d_model=64,
            n_layers=8,
            dropout=0.3,
            prenorm=False)

        self.s4_model3 = S4Model(d_input=64,
            d_output=640,
            d_model=64,
            n_layers=8,
            dropout=0.3,
            prenorm=False)
        
        # 定义编码器块
        self.convd1 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1) #640 - 320

        self.convd2 = nn.Conv1d(64, 64, kernel_size=2, stride=2) #320 - 160

        self.up_conv = nn.ConvTranspose1d(
            64,  # in_channels
            64,  # out_channels
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1  
        )                                     #160 - 320
        
        self.up_conv1 = nn.ConvTranspose1d(
            64,  # in_channels，输入特征图的通道数
            64,  # out_channels，输出特征图的通道数，保持不变
            kernel_size=3,  # 卷积核大小
            stride=2,  # 步长，每次卷积操作将特征图宽度增加2倍
            padding=1,  # 填充，通常设置为(kernel_size-1)/2
            output_padding=1,  # 输出填充，调整输出尺寸
            bias=False  # 根据需要决定是否使用偏置项
        )
        
       


    def forward(self, x):
        # 编码器

        enc0 = x

        enc1 = self.convd1(x)  #320
        enc10 = enc1

        
        enc2 = self.convd2(enc1)  #160
        enc20 = enc2

        enc2 = enc2.permute(0, 2, 1)
        
        enc21 = self.s4_model1(enc2)

        enc21 = enc21.permute(0, 2, 1)


        dec1 = enc20+enc21


        dec21 = self.up_conv(dec1)

        dec21 = dec21+enc10

        dec21 = dec21.permute(0, 2, 1)

        dec2 = self.s4_model2(dec21)

        dec2 = dec2.permute(0, 2, 1)

        dec31 = self.up_conv1(dec2)

        dec32 = dec31+enc0

        dec32 = dec32.permute(0, 2, 1)

        dec32 = self.s4_model3(dec32)

        dec32 = dec32.permute(0, 2, 1)

        dec3 = dec32

        return dec3


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(FeedForwardModule, self).__init__()
      
        self.linear1 = nn.Linear(64, 2048)
     
        self.linear2 = nn.Linear(2048, 64)
    
        self.layer_norm = nn.LayerNorm(d_model)
      
        self.swish = nn.SiLU()
      
        self.dropout1 = nn.Dropout(dropout_rate)
      
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
      
        normalized = self.layer_norm(x)
     
        x = self.linear1(normalized)
  
        x = self.swish(x)
    
        x = self.dropout1(x)
   
        x = self.linear2(x)
      
        x = self.dropout2(x)
        return x

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=32, padding=1, dropout_rate=0.1):
        super(ConvolutionBlock, self).__init__()
  
        self.pw_conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.group_norm = nn.GroupNorm(1, out_channels)  
        self.glu = nn.GLU()
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        
  
        self.depthwise_conv = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, groups=out_channels)
        self.group_norm_2 = nn.GroupNorm(1, out_channels) 
        
   
        self.pw_conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)


        self.s4_model = S4Model(d_input=32,
            d_output=640,
            d_model=64,
            n_layers=8,
            dropout=0.3,
            prenorm=False)


        self.conv_layer = nn.Conv1d(32, 640, kernel_size=1)

        self.conv_layer1 = nn.Conv1d(32, 64, kernel_size=1)

    def forward(self, x):

        
        x = self.pw_conv_1(x)
        x = self.group_norm(x)
        x = self.glu(x)



        x = self.swish(x)
        x = self.dropout(x)

  
        
        x = self.depthwise_conv(x)
        x = self.group_norm_2(x)

   
        x = self.pw_conv_2(x)



        x = self.s4_model(x)


        return x



class PreLNFFTBlock(torch.nn.Module):

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout,
                 **kwargs):

        super(PreLNFFTBlock, self).__init__()

        d_k = d_model // n_head
        d_v = d_model // n_head

        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel,fft_conv1d_padding, dropout=dropout)

    def forward(self, fft_input):

        # dec_input size: [B,T,C]
        fft_output, _= self.slf_attn(
            fft_input, fft_input, fft_input)

        fft_output = self.pos_ffn(fft_output)

        return fft_output



class CustomBlock(nn.Module):
    def __init__(self, in_channel, d_model, d_inner, n_head, n_layers, fft_conv1d_kernel, fft_conv1d_padding,
                 dropout, g_con, within_sub_num=71, **kwargs):
        super(CustomBlock, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head

        self.within_sub_num = within_sub_num
        self.g_con = g_con

        self.unet = UNetModule(in_channels=640, mid_channels=640, out_channels=640)


        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channel, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        self.ffm = FeedForwardModule(d_model, d_inner, dropout)



        self.conv_block = ConvolutionBlock(
            in_channels=640,  
            out_channels=640,  
            kernel_size=7,
            padding=3,
            dropout_rate=dropout
        )

        self.projection = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1)

        self.reduction = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1)

        self.layer_stack = nn.ModuleList([PreLNFFTBlock(
            d_model, d_inner, n_head, fft_conv1d_kernel, fft_conv1d_padding, dropout) for _ in range(n_layers)])


        self.sub_proj = nn.Linear(self.within_sub_num, d_model)


    def forward(self, dec_input, sub_id):

       
        dec_output = self.conv3(dec_input.transpose(1, 2))



        dec_output = self.unet(dec_output)




        dec_output = dec_output.transpose(1, 2)


        res1 = dec_output
        dec_output = self.ffm(dec_output)
        dec_output = dec_output * 0.5
        dec_output = dec_output+res1


        res = dec_output
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output)
        dec_output = dec_output+res
        

        res2 = dec_output

        dec_output = self.conv_block(dec_output)
        



        dec_output = dec_output+res2


        res1 = dec_output
        dec_output = self.ffm(dec_output)
        dec_output = dec_output * 0.5
        dec_output = dec_output+res1




        return dec_output


class mambaBlock(nn.Module):
    def __init__(self, in_channel, d_model, d_inner, n_head, n_layers, fft_conv1d_kernel, fft_conv1d_padding,
                 dropout, g_con, within_sub_num=71, **kwargs):
        super(mambaBlock, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head

        self.within_sub_num = within_sub_num
        self.g_con = g_con


        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channel, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        self.Bimamba_outer = Bimamba_outer(
            d_model=64,
            d_state=128,
            d_conv=4,
            expand=2
        ).to("cuda")




        self.ffm = FeedForwardModule(d_model, d_inner, dropout)

        self.unet = UNetModule(in_channels=640, mid_channels=640, out_channels=640)


   
        self.conv_block = ConvolutionBlock(
            in_channels=640, 
            out_channels=640,  
            kernel_size=7,
            padding=3,
            dropout_rate=dropout
        )

        self.projection = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1)

        self.reduction = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1)

    

        self.layer_stack = nn.ModuleList([PreLNFFTBlock(
            d_model, d_inner, n_head, fft_conv1d_kernel, fft_conv1d_padding, dropout) for _ in range(n_layers)])


        self.sub_proj = nn.Linear(self.within_sub_num, d_model)


    def forward(self, dec_input, sub_id):

       
        dec_output = self.conv3(dec_input.transpose(1, 2))



        dec_output = self.unet(dec_output)




        dec_output = dec_output.transpose(1, 2)


 

        dec_output = self.Bimamba_outer(dec_output)


        res1 = dec_output
        dec_output = self.ffm(dec_output)
        dec_output = dec_output * 0.5
        dec_output = dec_output+res1


        res = dec_output
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output)
            dec_output = dec_output+res
        

        res2 = dec_output


        dec_output = self.conv_block(dec_output)

   
        dec_output = dec_output+res2


        res1 = dec_output
        dec_output = self.ffm(dec_output)
        dec_output = dec_output * 0.5
        dec_output = dec_output+res1



        return dec_output



class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=640):
        super(ScaledPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))


        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        self.register_buffer('pe', pe)

    def forward(self, x):



        y = self.pe[:x.size(1), :].unsqueeze(0)  


        x = x + self.pe[:x.size(1), :].unsqueeze(0)  
        return x



class Decoder(nn.Module):
    def __init__(self, in_channel, d_model, d_inner, n_head, n_layers, fft_conv1d_kernel, fft_conv1d_padding,
                 dropout, g_con, within_sub_num=85, **kwargs):
        super(Decoder, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head
        self.g_con = g_con
        self.within_sub_num = within_sub_num


        self.fc = nn.Linear(64, 1)


        self.unet = UNetModule(in_channels=640, mid_channels=640, out_channels=640)

        self.mamba_block = mambaBlock(
            in_channel=64,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            n_layers=n_layers,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
            g_con=g_con,
            within_sub_num=within_sub_num)  
        
        self.custom_block = CustomBlock(
            in_channel=in_channel,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            n_layers=n_layers,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
            g_con=g_con,
            within_sub_num=within_sub_num)  

        self.block = ExternalAttention(d_model=64, S=8).cuda()

        self.positional_encoding = ScaledPositionalEncoding(d_model, max_len=640)

        self.sub_proj = nn.Linear(self.within_sub_num, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel,fft_conv1d_padding, dropout=dropout)




        
    def forward(self, dec_input, sub_id):



        qianru = self.positional_encoding(dec_input)



        sub_emb    = F.one_hot(sub_id, self.within_sub_num)
        sub_emb    = self.sub_proj(sub_emb.float())
        sub_emb =  sub_emb.unsqueeze(1)

        sub_emb = self.layer_norm(sub_emb)
        dec_input = dec_input + sub_emb



        fft_output, _= self.slf_attn(
            qianru, dec_input, dec_input)

        fft_output = self.layer_norm(fft_output)

        fft_output = self.pos_ffn(fft_output)+fft_output



        dec_input = dec_input+fft_output

        original_dec_input = dec_input


        dec_input = dec_input.unsqueeze(1)   

        dec_input = dec_input.permute(0, 1, 3, 2)



        dec_input = dec_input.permute(0, 1, 3, 2)

        dec_input = dec_input.squeeze(1)


        dec_input = dec_input.permute(0, 2, 1)

        dec_input = self.unet(dec_input)

        dec_input = dec_input.permute(0, 2, 1)


        dec_input = self.block(dec_input)




        for i in range(2):  
          
            dec_input = original_dec_input + dec_input


            dec_output = self.custom_block(dec_input, sub_id)


            
            dec_input = dec_output

        for i in range(1):  
          
            dec_input = original_dec_input + dec_input


            dec_output = self.mamba_block(dec_input, sub_id)


            
            dec_input = dec_output

        for i in range(2):  
          
            dec_input = original_dec_input + dec_input


            dec_output = self.custom_block(dec_input, sub_id)


            
            dec_input = dec_output


        
        for i in range(1):  
          
            dec_input = original_dec_input + dec_input


            dec_output = self.mamba_block(dec_input, sub_id)


            dec_input = dec_output



        output = self.fc(dec_input)

        return output