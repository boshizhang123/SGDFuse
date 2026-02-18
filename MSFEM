import torch
import torch.nn as nn

class MSFEM(nn.Module):
    """
    多尺度特征增强模块 (MSFEM)
    通过并行卷积分支捕捉不同感受野下的结构特征
    """
    def __init__(self, in_channels=1, mid_channels=32):
        super(MSFEM, self).__init__()
        
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3)

  
        self.feature_enhancement = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels * 3, kernel_size=3, padding=1, groups=mid_channels * 3), # DWConv
            nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=1), # Conv1x1 压缩
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels) # DWConv
        )

        self.channel_reduction = nn.Sequential(
            nn.Conv2d(mid_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv5x5(x)
        f4 = self.conv7x7(x)

        f_ms = torch.cat([f2, f3, f4], dim=1)
        
        f_enh = self.feature_enhancement(f_ms)

        f_cat = torch.cat([f_enh, f1], dim=1)
        
        weights = self.channel_reduction(f_cat)
        out = x + weights # 最终输出 F_out 
        
        return out
