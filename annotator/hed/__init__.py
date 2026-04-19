import numpy as np
import cv2
import os
import torch
from einops import rearrange
from annotator.util import annotator_ckpts_path


class Network(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(model_path).items()})

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))


class HEDdetector:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth"
        modelpath = os.path.join(annotator_ckpts_path, "network-bsds500.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        self.netNetwork = Network(modelpath).cuda().eval()

    def __call__(self, input_image):
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image).float().cuda()
            image_hed = image_hed / 255.0
            image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
            edge = self.netNetwork(image_hed)[0]
            edge = (edge.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            return edge[0]


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z

# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from annotator.util import annotator_ckpts_path


# # ── HED 骨干网络（保持不变）────────────────────────────────────────
# class Network(nn.Module):
#     def __init__(self, model_path):
#         super().__init__()

#         self.netVggOne = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=False),
#             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
#         )
#         self.netVggTwo = nn.Sequential(
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=False),
#             nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=False),
#         )
#         self.netVggThr = nn.Sequential(
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=False),
#             nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=False),
#             nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=False),
#         )
#         self.netVggFou = nn.Sequential(
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=False),
#             nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=False),
#             nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=False),
#         )
#         self.netVggFiv = nn.Sequential(
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=False),
#             nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=False),
#             nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=False),
#         )

#         self.netScoreOne = nn.Conv2d(64,  1, 1, 1, 0)
#         self.netScoreTwo = nn.Conv2d(128, 1, 1, 1, 0)
#         self.netScoreThr = nn.Conv2d(256, 1, 1, 1, 0)
#         self.netScoreFou = nn.Conv2d(512, 1, 1, 1, 0)
#         self.netScoreFiv = nn.Conv2d(512, 1, 1, 1, 0)

#         self.netCombine = nn.Sequential(
#             nn.Conv2d(5, 1, 1, 1, 0),
#             nn.Sigmoid(),
#         )

#         self.load_state_dict(
#             {k.replace("module", "net"): v
#              for k, v in torch.load(model_path).items()}
#         )

#     def forward(self, tenInput):
#         tenInput = tenInput * 255.0
#         tenInput = tenInput - torch.tensor(
#             [104.00698793, 116.66876762, 122.67891434],
#             dtype=tenInput.dtype, device=tenInput.device,
#         ).view(1, 3, 1, 1)

#         tenVggOne = self.netVggOne(tenInput)
#         tenVggTwo = self.netVggTwo(tenVggOne)
#         tenVggThr = self.netVggThr(tenVggTwo)
#         tenVggFou = self.netVggFou(tenVggThr)
#         tenVggFiv = self.netVggFiv(tenVggFou)

#         tenScoreOne = self.netScoreOne(tenVggOne)
#         tenScoreTwo = self.netScoreTwo(tenVggTwo)
#         tenScoreThr = self.netScoreThr(tenVggThr)
#         tenScoreFou = self.netScoreFou(tenVggFou)
#         tenScoreFiv = self.netScoreFiv(tenVggFiv)

#         h, w = tenInput.shape[2], tenInput.shape[3]
#         tenScoreOne = F.interpolate(tenScoreOne, size=(h, w), mode="bilinear", align_corners=False)
#         tenScoreTwo = F.interpolate(tenScoreTwo, size=(h, w), mode="bilinear", align_corners=False)
#         tenScoreThr = F.interpolate(tenScoreThr, size=(h, w), mode="bilinear", align_corners=False)
#         tenScoreFou = F.interpolate(tenScoreFou, size=(h, w), mode="bilinear", align_corners=False)
#         tenScoreFiv = F.interpolate(tenScoreFiv, size=(h, w), mode="bilinear", align_corners=False)

#         return self.netCombine(
#             torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1)
#         )


# # ── HED 检测器（可微分，训练用）────────────────────────────────────
# class HEDdetector(nn.Module):
#     """
#     可微分的 HED 边缘检测器，用于训练阶段。
#     返回边缘概率图 (B,1,H,W)，范围 [0,1]，支持梯度反向传播。

#     关键修复：
#     - 覆盖 train()，防止 DDP/Lightning 将内部 BN 切回训练模式。
#       （HED 的 Network 本身无 BN，但作为统一规范仍加入此保护。）
#     """

#     def __init__(self, model_path: str = None):
#         super().__init__()

#         if model_path is None:
#             remote_model_path = (
#                 "https://huggingface.co/lllyasviel/ControlNet/resolve/main/"
#                 "annotator/ckpts/network-bsds500.pth"
#             )
#             model_path = os.path.join(annotator_ckpts_path, "network-bsds500.pth")
#             if not os.path.exists(model_path):
#                 from basicsr.utils.download_util import load_file_from_url
#                 load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)

#         self.netNetwork = Network(model_path).cuda()
#         self.netNetwork.eval()
#         for param in self.netNetwork.parameters():
#             param.requires_grad = False

#         self._freeze_bn()

#     # ── BN 冻结方法 ───────────────────────────────────────────────
#     def _freeze_bn(self):
#         for module in self.netNetwork.modules():
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
#     def forward(self, input_image: torch.Tensor) -> torch.Tensor:
#         """
#         input_image : (H,W,3) | (B,C,H,W)
#                       像素值范围 [0,255] 或 [0,1]
#         返回        : (B,1,H,W)  边缘概率图 [0,1]（可微）
#         """
#         # ── 统一为 (B,3,H,W) ─────────────────────────────────────
#         if input_image.dim() == 3:
#             # (H,W,C) → (1,C,H,W)
#             input_image = input_image.unsqueeze(0).permute(0, 3, 1, 2)
#         elif input_image.dim() == 4:
#             pass  # 已是 (B,C,H,W)
#         else:
#             raise ValueError(f"Unsupported input shape: {input_image.shape}")

#         # ── 归一化到 [0,1] ────────────────────────────────────────
#         if input_image.max() > 1.0:
#             input_image = input_image / 255.0

#         # ── 前向推理（保持梯度）──────────────────────────────────
#         with torch.set_grad_enabled(True):
#             edge = self.netNetwork(input_image)     # (B,1,H,W)  [0,1]

#         return edge