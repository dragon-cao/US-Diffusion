import os

from annotator.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from annotator.uniformer.mmseg.core.evaluation import get_palette
from annotator.util import annotator_ckpts_path


checkpoint_file = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth"


class UniformerDetector:
    def __init__(self):
        modelpath = os.path.join(annotator_ckpts_path, "upernet_global_small.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(checkpoint_file, model_dir=annotator_ckpts_path)
        config_file = os.path.join(os.path.dirname(annotator_ckpts_path), "uniformer", "exp", "upernet_global_small", "config.py")
        self.model = init_segmentor(config_file, modelpath).cuda()

    def __call__(self, img):
        result = inference_segmentor(self.model, img)
        res_img = show_result_pyplot(self.model, img, result, get_palette('ade'), opacity=1)
        return res_img

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# from annotator.util import annotator_ckpts_path
# from annotator.uniformer.mmseg.apis import init_segmentor
# from annotator.uniformer.mmseg.core.evaluation import get_palette

# checkpoint_file = (
#     "https://huggingface.co/lllyasviel/ControlNet/resolve/main/"
#     "annotator/ckpts/upernet_global_small.pth"
# )


# class UniformerDetector(nn.Module):
#     """
#     可微分的 Uniformer 分割检测器，用于训练阶段。
#     输出软分割 RGB 图像（范围 [0,1]），支持梯度反向传播。

#     关键修复：
#     - 覆盖 train() 方法，防止 DDP/Lightning 调用 model.train() 时
#       将内部 BN 层切回训练模式，避免 batch_size=1 时 BN 报错。
#     - 使用 extract_feat + decode_head.forward 获取真实 logits，
#       替代不返回 logits 的 forward_dummy。
#     """

#     def __init__(self):
#         super().__init__()

#         # ── 加载模型权重 ──────────────────────────────────────────
#         modelpath = os.path.join(annotator_ckpts_path, "upernet_global_small.pth")
#         if not os.path.exists(modelpath):
#             from basicsr.utils.download_util import load_file_from_url
#             load_file_from_url(checkpoint_file, model_dir=annotator_ckpts_path)

#         config_file = os.path.join(
#             os.path.dirname(annotator_ckpts_path),
#             "uniformer", "exp", "upernet_global_small", "config.py",
#         )
#         self.model = init_segmentor(config_file, modelpath).cuda()
#         self.model.eval()

#         # 冻结所有参数（只允许输入梯度回传，不更新 detector 权重）
#         for param in self.model.parameters():
#             param.requires_grad = False
            

#         # 兼容 mmseg 0.x / 1.x
#         self._has_forward_test = hasattr(self.model.decode_head, "forward_test")

#         # ── ADE20K 调色板（150 类）────────────────────────────────
#         palette = get_palette("ade")  # list of (R,G,B) int
#         palette_tensor = torch.tensor(palette, dtype=torch.float32) / 255.0  # (C,3)
#         self.register_buffer("palette", palette_tensor)

#         # ── ImageNet 预处理参数 ───────────────────────────────────
#         self.register_buffer(
#             "mean",
#             torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1),
#         )
#         self.register_buffer(
#             "std",
#             torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1),
#         )

#         # 初始化完成后固定所有 BN 为 eval 模式
#         self._freeze_bn()

#     # ── BN 冻结方法 ───────────────────────────────────────────────
#     def _freeze_bn(self):
#         """
#         将 self.model 内所有归一化层永久固定为 eval 模式。
#         通过覆盖各层的 .train() 方法，防止外部调用（DDP/Lightning
#         的 model.train()）将其切回训练模式导致 BN 对单样本报错。
#         """
#         for module in self.model.modules():
#             if isinstance(
#                 module,
#                 (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
#                  nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm2d),
#             ):
#                 module.eval()
#                 # 覆盖该层的 train()，使其永远保持 eval 状态
#                 module.train = lambda mode=True: module

#     def train(self, mode: bool = True):
#         """
#         覆盖 nn.Module.train()。
#         DDP/Lightning 调用 detector.train(True) 时，
#         UniformerDetector 本身可切换模式，但内部 BN 层始终保持 eval。
#         """
#         super().train(mode)
#         self._freeze_bn()   # 每次被拉入 train 模式后立即重新冻结 BN
#         return self

#     # ── 内部：获取分割 logits ────────────────────────────────────
#     def _get_seg_logits(self, img_norm: torch.Tensor) -> torch.Tensor:
#         """
#         img_norm : (B,3,H,W)，已做 ImageNet 归一化
#         返回     : (B, num_classes, H, W)  未经 softmax 的 logits
#         """
#         feats = self.model.extract_feat(img_norm)       # backbone + neck 特征

#         # decode_head.forward 在 mmseg 0.x / 1.x 中均直接返回 logits
#         logits = self.model.decode_head.forward(feats)

#         # 部分 head 返回 list（含 auxiliary outputs），取主输出
#         if isinstance(logits, (list, tuple)):
#             logits = logits[0]

#         # 若空间尺寸小于输入，上采样到输入分辨率
#         if logits.shape[-2:] != img_norm.shape[-2:]:
#             logits = F.interpolate(
#                 logits,
#                 size=img_norm.shape[-2:],
#                 mode="bilinear",
#                 align_corners=False,
#             )

#         return logits  # (B, num_classes, H, W)

#     # ── forward ─────────────────────────────────────────────────
#     def forward(self, img: torch.Tensor) -> torch.Tensor:
#         """
#         img : (3,H,W) | (H,W,3) | (B,3,H,W) | (B,H,W,3)
#               像素值范围 [0,1] 或 [0,255]
#         返回: (B,3,H,W)，范围 [0,1] 的软分割 RGB 图（可微）
#         """
#         # ── 统一为 (B,3,H,W) ─────────────────────────────────────
#         if img.dim() == 3:
#             if img.shape[-1] == 3:              # (H,W,C)
#                 img = img.permute(2, 0, 1).unsqueeze(0)
#             else:                               # (C,H,W)
#                 img = img.unsqueeze(0)
#         elif img.dim() == 4:
#             if img.shape[1] != 3:               # (B,H,W,C)
#                 img = img.permute(0, 3, 1, 2)
#         else:
#             raise ValueError(f"Unsupported input shape: {img.shape}")

#         # ── 归一化到 [0,1] ────────────────────────────────────────
#         if img.max() > 1.0:
#             img = img / 255.0

#         # ── ImageNet 预处理 ───────────────────────────────────────
#         img_norm = (img * 255.0 - self.mean.to(img.device)) / self.std.to(img.device)

#         # ── 获取 logits（保持梯度）───────────────────────────────
#         with torch.set_grad_enabled(True):
#             logits = self._get_seg_logits(img_norm)     # (B, num_classes, H, W)

#         # ── softmax → 软颜色加权 ──────────────────────────────────
#         probs = F.softmax(logits, dim=1)                # (B, num_classes, H, W)
#         seg_rgb = torch.einsum(
#             "bkhw, kc -> bchw", probs, self.palette.to(img.device)
#         )                                               # (B, 3, H, W)  [0,1]

#         return seg_rgb