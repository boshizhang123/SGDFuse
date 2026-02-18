import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class HFAH(nn.Module):
    def __init__(self, feature_channels=[64, 128, 256, 512], out_channels=3):
        super(HFAH, self).__init__()
        self.attention_layers = nn.ModuleList([ChannelAttention(ch) for ch in feature_channels])
        self.reduce_convs = nn.ModuleList([nn.Conv2d(ch, 64, kernel_size=1) for ch in feature_channels])
        total_channels = 64 * len(feature_channels)
        self.fusion_head = nn.Sequential(
            nn.Conv2d(total_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, hierarchical_features):
        aggregated_features = []
        target_size = hierarchical_features[0].shape[2:]
        for i, feat in enumerate(hierarchical_features):
            attn_feat = self.attention_layers[i](feat)
            reduced_feat = self.reduce_convs[i](attn_feat)
            if reduced_feat.shape[2:] != target_size:
                reduced_feat = F.interpolate(reduced_feat, size=target_size, mode='bilinear', align_corners=True)
            aggregated_features.append(reduced_feat)
        return self.fusion_head(torch.cat(aggregated_features, dim=1))
