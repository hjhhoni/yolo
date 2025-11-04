# 1. 导入必要库
from ultralytics import YOLO
import torch
import torch.nn as nn

# 2. 定义CBAM模块（直接放在脚本开头）
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        channel_out = self.channel_att(x) * x
        avg_out = torch.mean(channel_out, dim=1, keepdim=True)
        max_out, _ = torch.max(channel_out, dim=1, keepdim=True)
        spatial_out = self.spatial_att(torch.cat([avg_out, max_out], dim=1)) * channel_out
        return spatial_out

# 3. 加载YOLO模型
model = YOLO("./ultralytics/cfg/models/11/yolo11.yaml")  # 或 "yolov11n.yaml"

# 4. 正确方式添加CBAM模块
# YOLOv11模型结构与之前版本不同，需要直接访问模型层
# 首先打印模型结构以了解如何访问
print("模型结构:", model.model)

# 在模型的关键位置添加CBAM模块
# 方法1: 在模型加载后修改模型结构
in_channels = 1024  # 根据模型结构确定通道数
# 假设我们要在模型的第10层后添加CBAM (通常是SPPF层后)
if hasattr(model.model, 'model'):
    # 对于某些YOLO版本，模型结构可能是model.model.model
    target_layer_idx = 10  # 根据实际模型结构调整
    if len(model.model.model) > target_layer_idx:
        original_layer = model.model.model[target_layer_idx]
        model.model.model[target_layer_idx] = nn.Sequential(
            original_layer,
            CBAM(in_channels=in_channels)
        )
        print(f"CBAM模块已添加到模型的第{target_layer_idx}层后")
else:
    # 直接尝试访问模型的各个部分
    print("无法直接访问模型层，请检查模型结构并相应调整代码")

# 5. 开始训练
if __name__ == '__main__':
    results = model.train(
        data='./data.yaml',  # 你的数据集配置文件路径
        epochs=150,
        imgsz=640,
        batch=32,
        workers=16,  
        amp=False,
    )