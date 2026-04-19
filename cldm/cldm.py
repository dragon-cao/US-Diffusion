import einops
import torch
import torch as th
import torch.nn as nn
import numpy as np
import os
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm_multi import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default 
from ldm.models.diffusion.ddim_multi import DDIMSampler
from annotator.util import HWC3
import pdb
import copy
from utils import image_grid, label_transform, group_random_crop
import torchvision
from torchvision import transforms

from PIL import Image
from torchvision.transforms.functional import normalize

from annotator.hed import HEDdetector
from annotator.uniformer import UniformerDetector
from annotator.midas import MidasDetector



import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint  # 新增：gradient checkpointing
import gc

# class ControlledUnetModel(UNetModel):
#     def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
#         hs = []
#         with torch.no_grad():
#             t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#             emb = self.time_embed(t_emb)
#             h = x.type(self.dtype)
#             for module in self.input_blocks:
#                 h = module(h, emb, context)
#                 hs.append(h)
#             h = self.middle_block(h, emb, context)

#         if control is not None:
#             h += control.pop()

#         for i, module in enumerate(self.output_blocks):
#             if only_mid_control or control is None:
#                 h = torch.cat([h, hs.pop()], dim=1)
#             else:
#                 h = torch.cat([h, hs.pop() + control.pop()], dim=1)
#             h = module(h, emb, context)

#         h = h.type(x.dtype)
#         return self.out(h)

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            # # output_blocks 需要梯度，用 checkpoint 降低激活值显存
            # h = checkpoint(module, h, emb, context, use_reentrant=False)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            all_tasks_num = 6,           
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.all_tasks_num = all_tasks_num
        self.tasks_to_id = {"inv_seg":0, "seg":1, "inv_depth":2, "depth":3, "inv_hed":4,"hed":5}
        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
         
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
 
        

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)]) # ie, model_channels -> 320
        

        
        self.input_hint_block_list_moe_1 = nn.ModuleList([TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU()
        ) for _ in range( self.all_tasks_num)])
        

        self.input_hint_block_list_moe_2 = nn.ModuleList([TimestepEmbedSequential(
            conv_nd(dims, 3, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU()
        ) for _ in range( self.all_tasks_num)])

        self.input_hint_block_share_1 = TimestepEmbedSequential(
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )
        self.input_hint_block_share_2 = TimestepEmbedSequential(
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        ) 


        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                
                # self.task_id_layernet.append(linear(time_embed_dim, ch))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                
                # self.task_id_layernet.append(linear(time_embed_dim, ch))
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch
                

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
        # self.task_id_layernet = nn.ModuleList(self.task_id_layernet)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))
    
    def visualize_and_save(self, output, title, filename):
        
        output = output.detach().cpu()  # 将张量移到 CPU 并脱离计算图
        plt.figure(figsize=(10, 10))
        plt.imshow(output[0, 0], cmap='gray')  # 可视化第一个 batch 的第一个通道
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(self.save_visualization_dir, filename))  # 保存为图片
        plt.close()

    def forward(self, x, example_pair, query, timesteps, context, **kwargs):
        
        '''
        x -> 4,4,64,64
        hint -> 4, 3, 512, 512
        context - > 4, 77, 768
        '''
        BS = 1 # x.shape[0], one batch one task
        BS_Real = x.shape[0]
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if kwargs is not None:
            # print(kwargs)
            task_name = kwargs['task']
            if task_name in ["inv_seg", "seg", "inv_depth", "depth", "inv_hed","hed"]:   #训练任务
                task_id = self.tasks_to_id[task_name]
                example_pair_hint = self.input_hint_block_list_moe_1[task_id](example_pair, emb, context)
                query_hint = self.input_hint_block_list_moe_2[task_id](query, emb, context)
                example_pair_hint = self.input_hint_block_share_1(example_pair_hint, emb, context)
                query_hint = self.input_hint_block_share_2(query_hint, emb, context)
                guided_hint=example_pair_hint+query_hint
                
            else:  #简单平均的新任务泛化
                example_pair_hint = self.input_hint_block_list_moe_1[1](example_pair, emb, context)+self.input_hint_block_list_moe_1[3](example_pair, emb, context)+self.input_hint_block_list_moe_1[5](example_pair, emb, context)
                query_hint = self.input_hint_block_list_moe_2[1](query, emb, context)+self.input_hint_block_list_moe_2[3](query, emb, context)+self.input_hint_block_list_moe_2[5](query, emb, context)
                example_pair_hint=example_pair_hint/3
                query_hint=query_hint/3
                example_pair_hint = self.input_hint_block_share_1(example_pair_hint, emb, context)
                query_hint = self.input_hint_block_share_2(query_hint, emb, context)
                guided_hint=example_pair_hint+query_hint

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_tasks_num = 6
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.hed_annotator = HEDdetector()
        self.uniformer_annotator = UniformerDetector()
        self.midas_annotator = MidasDetector()
        # self.canny_annotator = CannyDetector()

        # # annotator 初始化时全部放在 CPU，按需加载到 GPU
        # device_cpu = torch.device('cpu')
        
        # self._annotators = {}  # 普通字典，不被 nn.Module 追踪
        
        # device_cpu = torch.device('cpu')
        # self._annotators['midas'] = MidasDetector(
        #     model_path='Intel/dpt-hybrid-midas',
        #     output_channels=3,
        #     normalize_per_batch=True,
        #     use_fp16=True
        # ).to(device_cpu)
        
        # self._annotators['uniformer'] = UniformerDetector(
        #     model_path='mmseg::upernet/upernet_r50_4xb4-160k_ade20k-512x512.py',
        #     dataset_name='ADE20K',
        #     use_soft_palette=True
        # ).to(device_cpu)
        
        # self._annotators['hed'] = HEDdetector(
        #     model_path='https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth',
        #     output_channels=3,
        #     use_sigmoid=True
        # ).to(device_cpu)
        
        # # 任务 → annotator_key 的映射
        # self.annotator_mapping = {
        #     'inv_depth': 'midas', 'depth': 'midas',
        #     'inv_hed': 'hed', 'hed': 'hed',
        #     'inv_seg': 'uniformer', 'seg': 'uniformer',
        # }
        
        # self._active_annotator_on_gpu = None  # 追踪当前在GPU上的annotator

        
        # # reward 计算频率控制
        # self._reward_step_counter = 0
        # self._reward_every_n_steps = 8
        
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, drop_rate=0.05, *args, **kwargs):
        # print(batch)
        task_name = batch['txt_log'][0] # one task for one batch
        BS = len(batch['txt_log'])
        

        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        # print(batch['txt'])
        # print(c.shape)
        xc = batch['query']
        example_pair = batch['example_pair']
        if bs is not None:
            xc, example_pair = xc[:bs], example_pair[:bs]
            # x, c, xc, example_pair = x[:bs], c[:bs], xc[:bs], example_pair[:bs]

        example_pair = example_pair.to(self.device)
        example_pair = einops.rearrange(example_pair, 'b h w c -> b c h w')
        example_pair = example_pair.to(memory_format=torch.contiguous_format).float()

        xc = xc.to(self.device)
        xc = einops.rearrange(xc, 'b h w c -> b c h w')
        xc = xc.to(memory_format=torch.contiguous_format).float()

        cond = {}
        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        random = torch.rand(x.size(0), device=x.device)
        prompt_mask = rearrange(random < 2 * drop_rate, "n -> n 1 1")
        example_pair_mask = 1 - rearrange((random >= drop_rate).float() * (random < 3 * drop_rate).float(), "n -> n 1 1 1")
        # Text Guidance
        null_prompt = self.get_learned_conditioning([""])
        if task_name=='outpainting':
            cond["c_crossattn"] = [torch.where(prompt_mask, null_prompt, c)]
            # Example Pair
            cond["example_pair"] = [example_pair * example_pair_mask]
            # Image Query
            cond["query"] = [xc]
            #Task Name
            cond["task"] = task_name
            
            cond['query_path'] = batch['query_path']
        else:
            cond["c_crossattn"] = [torch.where(prompt_mask, null_prompt, c)]
            # Example Pair
            cond["example_pair"] = [example_pair * example_pair_mask]
            # Image Query
            cond["query"] = [xc]
            #Task Name
            cond["task"] = task_name 
        return x, cond

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        # print(cond)
        # print('apply_model is computing')
        assert isinstance(cond, dict)

        task_name = cond['task']# dict['name', 'feature']
        diffusion_model = self.model.diffusion_model # -> ControlledUnetModel
        # print(task_name)
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        assert cond['example_pair'] is not None
        # print('apply_model_t',t)
        control = self.control_model(x=x_noisy, timesteps=t,
                                     example_pair=torch.cat(cond['example_pair'], 1),
                                     query=cond['query'][0], context=cond_txt, task=task_name)
        # print(control)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        # print(eps)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()

        z, c = self.get_input(batch, self.first_stage_key, bs=N, drop_rate=0.)

        task_name = c['task']
        # print(task_dic)
        c_cat, c_example_pair, c = c["query"][0][:N], c["example_pair"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["input"] = einops.rearrange(batch['query'], 'b h w c -> b c h w')
        log["output"] = einops.rearrange(batch['jpg'], 'b h w c -> b c h w')
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"example_pair": [c_example_pair], "query": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            cond = {"query": [c_cat], "example_pair": [c_example_pair], "c_crossattn": [c], "task": task_name}
            # print(cond["task"])
            uc_full = {"example_pair": [c_example_pair], "query": [c_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log
    
    @torch.no_grad()
    def log_images_infer(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()

        z, c = self.get_input(batch, self.first_stage_key, bs=N, drop_rate=0.)

        task_name = c['task']

        c_cat, c_example_pair, c = c["query"][0][:N], c["example_pair"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        # log["reconstruction"] = self.decode_first_stage(z)
        # log["input"] = einops.rearrange(batch['query'], 'b h w c -> b c h w')
        # log["output"] = einops.rearrange(batch['jpg'], 'b h w c -> b c h w')
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        uc_cross = self.get_unconditional_conditioning(N)
        uc_full = {"example_pair": [c_example_pair], "query": [c_cat], "c_crossattn": [uc_cross]}
        cond = {"query": [c_cat], "example_pair": [c_example_pair], "c_crossattn": [c], "task": task_name}
        samples_cfg, _ = self.sample_log(cond,
                                            batch_size=N, ddim=use_ddim,
                                            ddim_steps=ddim_steps, eta=ddim_eta,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc_full,
                                            )
        x_samples_cfg = self.decode_first_stage(samples_cfg)
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        # print('cond["example_pair"][0].shape',cond["example_pair"][0].shape)
        # print('cond["query"][0].shape',cond["query"][0].shape)
        # print(cond["task"])      
        b, c, h, w = cond["example_pair"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # print('t',t)
        model_output = self.apply_model(x_noisy, t, cond)
        # print(model_output)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])   
        
        #reward_loss
        task_name=cond['task']   
        pred_original_sample=[]
        # pred_original_sample, _ = self.get_pred_original_sample(x_noisy, t, cond)
        noise_pre = self.apply_model(x_noisy, t, cond)
        pred_original_sample = self.predict_start_from_noise(x_noisy, t, noise_pre)
        images_pre=self.decode_first_stage(pred_original_sample.detach()) # 16 3 256 256
        # images_pre = (images_pre / 2 + 0.5).clamp(0, 1)
        images_pre = (einops.rearrange(images_pre, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        images_ori =cond["query"][0]
        images_ori = (einops.rearrange(images_ori, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        # print('t',t.reshape(-1, 1))
        # print(task_name)
        # print('images_pre',images_pre.shape)
        image_pre_cond=[]
        image_ori_cond=[]
        image_pre_conds=[]
        image_ori_conds=[]
        reward_loss_sigle=[]
        reward_loss=[] 
        for j, (y, x) in enumerate(zip(images_ori, images_pre)):
            if task_name == 'inv_depth':
                image_pre_cond,_ = self.midas_annotator(x) #(256, 256)
                image_ori_cond = y / 255. #(256, 256, 3)
                image_pre_cond=image_pre_cond/ 255.
                image_pre_cond=np.stack((image_pre_cond ,)*3, axis=-1)
            elif task_name == 'inv_hed':         
                image_pre_cond = self.hed_annotator(x)/ 255. #(256, 256)
                image_ori_cond = y/ 255.  #(256, 256, 3)
                image_pre_cond=np.stack((image_pre_cond ,)*3, axis=-1)
            elif task_name == 'inv_seg':
                image_pre_cond = self.uniformer_annotator(x)/ 255. #(256, 256, 3)
                image_ori_cond = y/ 255.  #(256, 256, 3)
            elif task_name == 'inv_canny':
                image_pre_cond = self.canny_annotator(x,100,200)/ 255. #(256, 256, 3)
                image_ori_cond = y/ 255.  #(256, 256, 3)
                image_pre_cond=np.stack((image_pre_cond ,)*3, axis=-1)                  
            elif task_name == 'depth':
                image_pre_cond = x/ 255.   #(256, 256, 3)
                image_ori_cond,_ = self.midas_annotator(y)#(256, 256)
                image_ori_cond=image_ori_cond/ 255.
                image_ori_cond=np.stack((image_ori_cond ,)*3, axis=-1)
            elif task_name == 'hed':
                image_pre_cond = x/ 255.  #(256, 256, 3)
                image_ori_cond = self.hed_annotator(y)/ 255. #(256, 256)
                image_ori_cond=np.stack((image_ori_cond,)*3, axis=-1) 
            elif task_name == 'seg' :
                image_pre_cond = x/ 255. #(256, 256, 3)
                image_ori_cond = self.uniformer_annotator(y)/ 255.#(256, 256, 3)
            elif task_name == 'canny' :
                image_pre_cond = x/ 255. #(256, 256, 3)
                image_ori_cond = self.canny_annotator(y,100,200)/ 255.#(256, 256)
                image_ori_cond=np.stack((image_ori_cond,)*3, axis=-1)
            else:
                image_pre_cond = x
                image_ori_cond = Image.open(cond['query_path'][j])
                reize_res = torch.randint(256, 257, ()).item()
                image_ori_cond = image_ori_cond.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
                image_ori_cond = 2 * torch.tensor(HWC3(np.array(image_ori_cond))).float() / 255. - 1
            if task_name=='outpainting':
                image_pre_cond=torch.from_numpy(image_pre_cond).float() 
                image_pre_cond=image_pre_cond.permute(2,1,0)
                image_ori_cond=image_ori_cond.permute(2,1,0)
                reward_loss_sigle = nn.functional.mse_loss(image_pre_cond, image_ori_cond)
                reward_loss.append(reward_loss_sigle)                
            else: 
                image_pre_cond=torch.from_numpy(image_pre_cond).float() 
                image_ori_cond=torch.from_numpy(image_ori_cond).float() 
                image_pre_cond=image_pre_cond.permute(2,1,0)
                image_ori_cond=image_ori_cond.permute(2,1,0)
                reward_loss_sigle = nn.functional.mse_loss(image_pre_cond, image_ori_cond)
                reward_loss.append(reward_loss_sigle)
        reward_loss=torch.stack(reward_loss,dim=0).to(self.device)
        
        min_timestep_rewarding=0
        max_timestep_rewarding=200
        
        # timestep-based filtering
        timestep_mask = (min_timestep_rewarding <= t.reshape(-1, 1)) & (t.reshape(-1, 1) <= max_timestep_rewarding)

        reward_loss = reward_loss.reshape_as(timestep_mask)
        reward_loss = (timestep_mask * reward_loss).sum() / (timestep_mask.sum() + 1e-10)
        grad_scale=1
        loss_simple = loss_simple+ reward_loss*grad_scale      
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
        return loss , loss_dict

        
    # def _move_annotator_to_gpu(self, task_name):
    #     if task_name not in self.annotator_mapping:
    #         return None, False

    #     annotator_key = self.annotator_mapping[task_name]
    #     annotator = self._annotators[annotator_key]
    #     target_device = self.device

    #     if self._active_annotator_on_gpu == annotator_key:
    #         return annotator, False

    #     # 将旧的移回 CPU
    #     if self._active_annotator_on_gpu is not None:
    #         prev_annotator = self._annotators[self._active_annotator_on_gpu]
    #         prev_annotator.to(torch.device('cpu'))
    #         self._active_annotator_on_gpu = None
    #         torch.cuda.empty_cache()
    #         gc.collect()

    #     # 移动新 annotator 到 GPU
    #     annotator.to(target_device)
    #     # 确保所有参数都在目标设备上
    #     for p in annotator.parameters():
    #         assert p.device == target_device, f"Parameter device mismatch: {p.device} vs {target_device}"
    #     self._active_annotator_on_gpu = annotator_key

    #     return annotator, True

    # def _release_annotator_to_cpu(self):
    #     """backward 完成后释放 annotator 显存。幂等，可重复调用。"""
    #     if self._active_annotator_on_gpu is not None:
    #         annotator = self._annotators[self._active_annotator_on_gpu]
    #         annotator.to(torch.device('cpu'))
    #         self._active_annotator_on_gpu = None
    #         torch.cuda.empty_cache()
    #         gc.collect()

    # def _safe_annotator_forward(self, annotator, x, requires_grad=False, max_retries=2):
    #     try:
    #         annotator_device = next(annotator.parameters()).device
    #     except StopIteration:
    #         annotator_device = self.device

    #     # 确保输入张量与 annotator 设备一致
    #     x = x.to(device=annotator_device).contiguous()

    #     if requires_grad:
    #         annotator.train()
    #         for p in annotator.parameters():
    #             p.requires_grad = False
    #     else:
    #         annotator.eval()

    #     def _run(use_cudnn):
    #         ctx = torch.backends.cudnn.flags(enabled=use_cudnn)
    #         if requires_grad:
    #             with ctx:
    #                 return annotator(x)
    #         else:
    #             with ctx, torch.no_grad():
    #                 return annotator(x)

    #     for attempt in range(max_retries + 1):
    #         try:
    #             return _run(use_cudnn=(attempt == 0))
    #         except RuntimeError as e:
    #             err_str = str(e)
    #             if ("cuDNN" in err_str or "CUDA out of memory" in err_str) and attempt < max_retries:
    #                 torch.cuda.empty_cache()
    #                 gc.collect()
    #                 continue
    #             raise e

             
    # def _get_annotator(self, task_name):
    #     """
    #     根据任务名返回对应的 annotator 实例（不处理设备移动）。
    #     仅用于查询映射关系，设备管理请交给 _move_annotator_to_gpu。
    #     """
    #     mapping = {
    #         'inv_depth': self.midas_annotator,
    #         'depth':     self.midas_annotator,
    #         'inv_hed':   self.hed_annotator,
    #         'hed':       self.hed_annotator,
    #         'inv_seg':   self.uniformer_annotator,
    #         'seg':       self.uniformer_annotator,
    #     }
    #     return mapping.get(task_name, None)


    
    # def _decode_latent_chunked(self, z, chunk=1):
    #     results = []
    #     for i in range(0, z.shape[0], chunk):
    #         z_sub = z[i:i + chunk]
    #         # 先过 post_quant_conv（轻量，不 checkpoint）
    #         if hasattr(self.first_stage_model, 'post_quant_conv'):
    #             z_sub = self.first_stage_model.post_quant_conv(z_sub)
    #         # decoder 是显存大头，单独 checkpoint
    #         img = checkpoint(
    #             self.first_stage_model.decoder,
    #             z_sub,
    #             use_reentrant=False
    #         )
    #         results.append(img)
    #     return torch.cat(results, dim=0)


    # def p_losses(self, x_start, cond, t, noise=None):
    #     """
    #     扩散模型训练损失计算，集成：
    #     1. 基础 diffusion loss (eps/x0/v 参数化)
    #     2. 任务特定的 reward loss（通过 annotator 计算）
    #     3. 动态 annotator 设备管理 + cuDNN 兼容性处理
    #     4. backward 完成后才释放 annotator（通过 register_hook）
    #     """
    #     noise = default(noise, lambda: torch.randn_like(x_start))
    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #     model_output = self.apply_model(x_noisy, t, cond)

    #     loss_dict = {}
    #     prefix = 'train' if self.training else 'val'

    #     # ── 目标选择（根据参数化方式）──
    #     if self.parameterization == "x0":
    #         target = x_start
    #     elif self.parameterization == "eps":
    #         target = noise
    #     elif self.parameterization == "v":
    #         target = self.get_v(x_start, noise, t)
    #     else:
    #         raise NotImplementedError(f"Unknown parameterization {self.parameterization}")

    #     # ── 基础 diffusion loss ──
    #     loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

    #     # ── Reward Loss 配置 ──
    #     task_name = cond['task']
    #     t_mask = (t >= 0) & (t <= 200)  # 仅在早期 timestep 计算 reward

    #     grad_scale_map = {
    #         'inv_depth': 1.0, 'depth': 1.0,
    #         'inv_hed':   1.0, 'hed':   1.0,
    #         'inv_seg':   1.0, 'seg':   1.0,
    #     }

    #     self._reward_step_counter += 1
    #     should_compute_reward = (
    #         self._reward_step_counter % self._reward_every_n_steps == 0
    #     )

    #     # ── 动态加载 annotator 到 GPU ──
    #     annotator, just_moved = self._move_annotator_to_gpu(task_name)
    #     if just_moved:
    #         torch.cuda.empty_cache()

    #     # ── 计算 reward loss ──
    #     if t_mask.any() and should_compute_reward and annotator is not None:
    #         # 预测 x0 用于 annotator 输入
    #         pred_x0 = self.predict_start_from_noise(x_noisy, t, model_output)
    #         valid_idx = t_mask.nonzero(as_tuple=True)[0]

    #         if len(valid_idx) == 0:
    #             reward_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
    #             # 没有有效 idx，annotator 不会参与 backward，直接释放
    #             self._release_annotator_to_cpu()
    #         else:
    #             sub_latent = (1. / self.scale_factor) * pred_x0[valid_idx]
    #             sub_query  = cond["query"][0][valid_idx]

    #             # 释放不再需要的张量
    #             del pred_x0
    #             torch.cuda.empty_cache()

    #             is_inv = task_name.startswith('inv_')

    #             # ── 解码 latent → 图像，并统一做值域/dtype 修正 ──
    #             sub_images = self._decode_latent_chunked(sub_latent, chunk=1)
    #             # 解码器输出在 [-1, 1]，统一转为 fp32 + [0, 1]
    #             sub_images = (sub_images.float().clamp(-1, 1) + 1) / 2.0
    #             sub_images = sub_images.contiguous()
    #             torch.cuda.empty_cache()

    #             ann_out = None  # 提前初始化，供后续通道对齐判断使用

    #             if is_inv:

    #                 try:
    #                     annotator_device = next(annotator.parameters()).device
    #                 except StopIteration:
    #                     annotator_device = self.device
    #                 sub_images = sub_images.to(annotator_device)

    #                 # inverse 任务：模型输出图像 → annotator → 与 query 比较
    #                 ann_out = self._safe_annotator_forward(
    #                     annotator, sub_images, requires_grad=True
    #                 )

    #                 # query 作为 target，归一化到 [0, 1]
    #                 target_for_mse = (
    #                     sub_query.float().clamp(-1, 1) + 1
    #                 ) / 2.0
    #                 target_for_mse = target_for_mse.detach().contiguous()

    #             else:
    #                 # forward 任务：query → annotator → 与模型输出图像比较
    #                 sub_query_norm = (
    #                     sub_query.float().clamp(-1, 1) + 1
    #                 ) / 2.0
    #                 sub_query_norm = sub_query_norm.contiguous()


    #                 try:
    #                     annotator_device = next(annotator.parameters()).device
    #                 except StopIteration:
    #                     annotator_device = self.device
    #                 sub_query_norm = sub_query_norm.to(annotator_device)

    #                 # requires_grad=False：forward 任务 annotator 输出作为固定 target
    #                 ann_out = self._safe_annotator_forward(
    #                     annotator, sub_query_norm, requires_grad=False
    #                 )

    #                 ann_out = ann_out.detach().to(self.device)
    #                 target_for_mse = sub_images  # 模型输出图像，需要梯度

    #             # ── 通道数对齐 ──
    #             if ann_out is not None:
    #                 if ann_out.shape[1] != target_for_mse.shape[1]:
    #                     if ann_out.shape[1] == 1 and target_for_mse.shape[1] == 3:
    #                         ann_out = ann_out.expand_as(target_for_mse)
    #                     elif ann_out.shape[1] == 3 and target_for_mse.shape[1] == 1:
    #                         target_for_mse = target_for_mse.expand_as(ann_out)
    #                     else:
    #                         # 无法对齐，跳过 reward
    #                         ann_out = None

    #                 if ann_out is not None:
    #                     ann_out = ann_out.to(self.device)
    #                     target_for_mse = target_for_mse.to(self.device)
    #                     reward_loss = F.mse_loss(ann_out, target_for_mse, reduction='mean')
    #                     reward_loss = reward_loss * grad_scale_map.get(task_name, 1.0)
                        

    #                     # annotator 留在 GPU，等 on_after_backward 来释放
                        
    #                 else:
    #                     reward_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
    #                     self._release_annotator_to_cpu()  # ann_out 为 None，annotator 没参与图，可立即释放

    #             else:
    #                 reward_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
    #                 self._release_annotator_to_cpu()  # 没有计算 reward，立即释放

    #             # ── 清理中间变量 ──
    #             del sub_latent, sub_query
    #             if is_inv:
    #                 del sub_images
    #             else:
    #                 del sub_query_norm
    #             torch.cuda.empty_cache()

    #     else:
    #         reward_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
    #         # 没有计算 reward，若 annotator 因上一步遗留在 GPU，也一并清理
    #         if self._active_annotator_on_gpu is not None:
    #             self._release_annotator_to_cpu()

    #     # ── 合并 loss ──
    #     grad_scale = 0.1 # reward loss 的权重系数
    #     loss_simple = loss_simple + reward_loss * grad_scale

    #     loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
    #     loss_dict.update({f'{prefix}/reward_loss': reward_loss.detach()})

    #     # ── logvar 加权（如果启用）──
    #     logvar_t = self.logvar[t].to(self.device)
    #     loss = loss_simple / torch.exp(logvar_t) + logvar_t

    #     if self.learn_logvar:
    #         loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
    #         loss_dict.update({'logvar': self.logvar.data.mean()})

    #     # ── VLB loss（可选）──
    #     loss = self.l_simple_weight * loss.mean()

    #     if self.original_elbo_weight > 0:
    #         loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
    #         loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
    #         loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
    #         loss += (self.original_elbo_weight * loss_vlb)

    #     loss_dict.update({f'{prefix}/loss': loss})

    #     return loss, loss_dict