import cv2
import numpy as np
import torch

from einops import rearrange
from .api import MiDaSInference


class MidasDetector:
    def __init__(self):
        self.model = MiDaSInference(model_type="dpt_hybrid").cuda()

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = np.ones_like(x) * a
            x[depth_pt < bg_th] = 0
            y[depth_pt < bg_th] = 0
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

            return depth_image, normal_image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .api import MiDaSInference


# class MidasDetector(nn.Module):
#     """
#     可微分的 MiDaS 深度/法线检测器，用于训练阶段。

#     返回:
#         depth_image  : (B,1,H,W)，范围 [0,1]（归一化深度图）
#         normal_image : (B,3,H,W)，范围 [0,1]（表面法线图）

#     关键修复：
#     - 覆盖 train()，防止 DDP/Lightning 将内部 BN 切回训练模式。
#     - 软背景掩码（sigmoid）替代硬阈值，保持梯度连续。
#     - depth 任务返回深度图而非法线图。
#     """

#     def __init__(self, a: float = 2 * torch.pi, bg_th: float = 0.1):
#         super().__init__()

#         self.model = MiDaSInference(model_type="dpt_hybrid").cuda()
#         self.model.eval()
#         for param in self.model.parameters():
#             param.requires_grad = False

#         self.a = a
#         self.bg_th = bg_th

#         # ── 可微 Sobel 核 ─────────────────────────────────────────
#         sobel_x = torch.tensor(
#             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
#         ).view(1, 1, 3, 3)
#         sobel_y = torch.tensor(
#             [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
#         ).view(1, 1, 3, 3)
#         self.register_buffer("sobel_x", sobel_x)
#         self.register_buffer("sobel_y", sobel_y)

#         # 初始化后固定 BN
#         self._freeze_bn()

#     # ── BN 冻结方法 ───────────────────────────────────────────────
#     def _freeze_bn(self):
#         for module in self.model.modules():
#             if isinstance(
#                 module,
#                 (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
#                  nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm2d),
#             ):
#                 module.eval()
#                 module.train = lambda mode=True: module

#     def train(self, mode: bool = True):
#         super().train(mode)
#         self._freeze_bn()
#         return self

#     # ── forward ─────────────────────────────────────────────────
#     def forward(self, input_image: torch.Tensor):
#         """
#         input_image : (3,H,W) | (H,W,3) | (B,3,H,W) | (B,H,W,3)
#                       像素值范围 [0,1] 或 [0,255]
#         返回:
#             depth_image  : (B,1,H,W)  [0,1]
#             normal_image : (B,3,H,W)  [0,1]
#         """
#         # ── 统一为 (B,3,H,W) ─────────────────────────────────────
#         if input_image.dim() == 3:
#             if input_image.shape[0] == 3:       # (C,H,W)
#                 input_image = input_image.unsqueeze(0)
#             else:                               # (H,W,C)
#                 input_image = input_image.permute(2, 0, 1).unsqueeze(0)
#         elif input_image.dim() == 4:
#             if input_image.shape[1] != 3:       # (B,H,W,C)
#                 input_image = input_image.permute(0, 3, 1, 2)
#         else:
#             raise ValueError(f"Unsupported input shape: {input_image.shape}")

#         # ── 归一化到 [-1,1] ───────────────────────────────────────
#         if input_image.max() > 1.0:
#             input_image = input_image / 255.0
#         input_norm = input_image * 2.0 - 1.0

#         # ── 深度估计（保持梯度）──────────────────────────────────
#         with torch.set_grad_enabled(True):
#             depth = self.model(input_norm)      # (B,H,W) 或 (B,1,H,W)
#             if depth.dim() == 3:
#                 depth = depth.unsqueeze(1)      # → (B,1,H,W)

#         # ── 深度归一化到 [0,1] ────────────────────────────────────
#         depth_min = depth.amin(dim=(2, 3), keepdim=True)
#         depth_max = depth.amax(dim=(2, 3), keepdim=True)
#         depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

#         # ── 可微 Sobel 梯度 ───────────────────────────────────────
#         grad_x = F.conv2d(depth, self.sobel_x.to(depth.device), padding=1)
#         grad_y = F.conv2d(depth, self.sobel_y.to(depth.device), padding=1)

#         # 软背景掩码：sigmoid 近似，保持梯度（硬阈值在边界处梯度为 0）
#         softmask = torch.sigmoid((depth_norm - self.bg_th) * 50.0)
#         grad_x = grad_x * softmask
#         grad_y = grad_y * softmask

#         # ── 法线构造 ──────────────────────────────────────────────
#         x = grad_x.squeeze(1)                           # (B,H,W)
#         y = grad_y.squeeze(1)
#         z = torch.ones_like(x) * self.a
#         normal = torch.stack([x, y, z], dim=1)          # (B,3,H,W)
#         norm_len = torch.norm(normal, dim=1, keepdim=True) + 1e-8
#         normal = normal / norm_len
#         normal_image = (normal * 0.5 + 0.5).clamp(0, 1)  # (B,3,H,W) [0,1]

#         depth_image = depth_norm                          # (B,1,H,W) [0,1]

#         return depth_image, normal_image