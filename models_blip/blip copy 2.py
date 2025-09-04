import warnings
warnings.filterwarnings("ignore")
try:
    from models_blip.vit import VisionTransformer, interpolate_pos_embed
    from models_blip.med import BertConfig, BertModel, BertLMHeadModel
    from models_blip.vit import Block
except:
    from vit import VisionTransformer, interpolate_pos_embed
    from med import BertConfig, BertModel, BertLMHeadModel
    from vit import Block
from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
import ImageReward
import timm
import torch
import time
import random
import numpy as np
from torch.fft import fft2, fftshift
class BLIP_Base(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 fix_rate=0.5
                 ):
        super().__init__()
        self.fix_rate = fix_rate
        #self.reward = ImageReward.load("ImageReward-v1.0")
        self.reward = torch.load("/home/data1/lq/SC-AGIQA-main/models_blip/ImageReward-v1.0/full_model.pth")
        self.visual_encoder=timm.create_model("vit_base_patch16_224", pretrained=False)
        self.visual_encoder.load_state_dict(
            torch.load("/home/data1/lq/SC-AGIQA-main/models_blip/vit_base_patch16_224.pth")
        )
    def forward(self, image, captions, mode):
        if isinstance(captions, str):
            captions = [captions]
        prompts = []
        answers = []
        for caption in captions:
            separator = "|||"
            if separator in caption:
                prompt, answer = caption.split(separator, 1)
                prompts.append(prompt.strip())
                answers.append(answer.strip())
            else:
                prompts.append(caption.strip())
                answers.append("")
        self.reward.to(image.device)
        resized_images = image[:, 3:, :, :]
        image = image[:, :3, :, :]
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        image_embeds = self.reward.blip.visual_encoder(resized_images)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(resized_images.device)
        text_input = self.reward.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(image.device)
        text_input_answer = self.reward.blip.tokenizer(answers, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(image.device)
        if mode=='image':
            pass
        elif mode=='text':
            pass
        elif mode=='multimodal':
            image_features = self.visual_encoder(image)
            text_output = self.reward.blip.text_encoder(text_input.input_ids,
                                                     attention_mask = text_input.attention_mask,
                                                     encoder_hidden_states = image_embeds,
                                                     encoder_attention_mask = image_atts,
                                                     return_dict = True,
                                                    )
            text_output_answer = self.reward.blip.text_encoder(text_input_answer.input_ids,
                                                     attention_mask = text_input_answer.attention_mask,
                                                     encoder_hidden_states = image_embeds,
                                                     encoder_attention_mask = image_atts,
                                                     return_dict = True,
                                                    )
            text_features = text_output.last_hidden_state
            text_features_answer=text_output_answer.last_hidden_state
            return text_features,text_features_answer,image_features


def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    return model



class IQAPooling(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv_avg = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_adjust = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
    def weighted_adaptive_pool2d(self,feature, weight):
        weight_sum = torch.sum(weight, dim=(2, 3), keepdim=True)
        weight_norm = weight / weight_sum
        weight_norm = weight_norm.expand_as(feature)
        weighted_feature = feature * weight_norm +0.4*feature
        output = torch.sum(weighted_feature, dim=(2, 3), keepdim=True)
        return output
    def forward(self, x,x1_csf):
        x1_csf.to(x.device)
        B, C, h,w = x.shape
        x_spatial = x.view(B, C, h, w).contiguous()
        x=x.view(B, C, h*w).contiguous()
        x1_csf_resized = F.interpolate(x1_csf, size=(x_spatial.shape[-1], x_spatial.shape[-1]), mode='bilinear', align_corners=False)
        quality_weights = self.conv_avg(x_spatial)
        channel_weights = self.channel_adjust(x.mean(dim=-1)).sigmoid()
        weighted = x_spatial * quality_weights
        weighted = weighted * channel_weights.view(B, C, 1, 1).contiguous()
        output=self.weighted_adaptive_pool2d(weighted,x1_csf_resized)
        return output.view(B, C, 1).contiguous()
def get_vit_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[8][:, 1:, :],
            save_output.outputs[9][:, 1:, :],
            save_output.outputs[10][:, 1:, :],
            save_output.outputs[11][:, 1:, :]
        ),
        dim=2
    )
    return feat
import torch
import torch.nn as nn
class SaveOutput:
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    def clear(self):
        self.outputs = []
from torch import einsum
from einops import rearrange, repeat
from inspect import isfunction
def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, 1)
        )
    def forward(self, x):
        return self.project(x.unsqueeze(0)).squeeze(0)
class MoERegressor(nn.Module):
    def __init__(self, input_dim=1024, select_dim=512, num_experts=6, top_k=3, shared_dim=32):
        super().__init__()
        assert input_dim == 1024
        self.select_dim = select_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_dim = shared_dim

        self.mask_logits = nn.Parameter(torch.randn(num_experts, input_dim))

        self.expert_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(select_dim + shared_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(input_dim, num_experts)

        self.cross_attn = CrossAttention(query_dim=512, context_dim=512, heads=4, dim_head=16)
        self.shared_proj = nn.Linear(512, shared_dim)

    def forward(self, x):
        B = x.size(0)
        device = x.device

        # 拆分语义和视觉特征
        x_sem = x[:, :512]   # [B, 512]
        x_vis = x[:, 512:]   # [B, 512]

        # 交叉注意力生成共享信息
        sem_token = x_sem.unsqueeze(1)  # [B, 1, 512]
        vis_token = x_vis.unsqueeze(1)  # [B, 1, 512]
        shared_token = self.cross_attn(sem_token, context=vis_token)  # [B, 1, 512]
        shared_info = self.shared_proj(shared_token.squeeze(1))       # [B, shared_dim]

        # 门控 top-k 选择
        gate_logits = self.gate(x)  # [B, num_experts]
        topk_scores, topk_indices = torch.topk(gate_logits, self.top_k, dim=1)  # [B, top_k]
        topk_softmax = torch.softmax(topk_scores, dim=1)  # [B, top_k]

        output = torch.zeros(B, 1, device=device)

        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  # [B]
            weight = topk_softmax[:, i].unsqueeze(1)  # [B, 1]
            out = torch.zeros(B, 1, device=device)

            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)[0]  # [n]
                    x_sel = x[idx]
                    shared_sel = shared_info[idx]

                    # 选择 expert_id 掩码下的特征维度
                    mask_prob = torch.sigmoid(self.mask_logits[expert_id])
                    _, selected_indices = torch.topk(mask_prob, self.select_dim)
                    x_selected = x_sel[:, selected_indices]  # [n, select_dim]

                    expert_input = torch.cat([x_selected, shared_sel], dim=1)  # [n, select_dim + shared_dim]
                    out_part = self.expert_fcs[expert_id](expert_input)       # [n, 1]

                    out[idx] = out_part.view(-1, 1).to(out.dtype)


            output += out * weight  # 按权重加权专家输出

        return output  # [B, 1]
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Tuple

class TokenAligner:
    def __init__(self):
        pass

    def _align_single_item(self, a_item: torch.Tensor, b_item: torch.Tensor) -> torch.Tensor:
        a_norm = F.normalize(a_item, p=2, dim=1)
        b_norm = F.normalize(b_item, p=2, dim=1)

        similarity_matrix = torch.matmul(a_norm, b_norm.T)
        
        cost_matrix = -similarity_matrix.detach().cpu().numpy()
        
        _, col_ind = linear_sum_assignment(cost_matrix)
        
        b_item_aligned = b_item[col_ind]
        
        return b_item_aligned

    def align(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor_a.shape != tensor_b.shape:
            raise ValueError("Input tensors must have the same shape.")
        if tensor_a.dim() != 3:
            raise ValueError("Input tensors must be 3-dimensional [batch, num_tokens, dim].")

        batch_size = tensor_a.shape[0]
        
        aligned_b_list = []

        for i in range(batch_size):
            a_item = tensor_a[i]
            b_item = tensor_b[i]
            
            b_item_aligned = self._align_single_item(a_item, b_item)
            aligned_b_list.append(b_item_aligned)
            
        tensor_b_aligned = torch.stack(aligned_b_list, dim=0)
        
        return tensor_a, tensor_b_aligned

    def __call__(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.align(tensor_a, tensor_b)
# class TokenAligner:
#     def __init__(self, bidirectional=True):
#         """
#         初始化对齐器
#         :param bidirectional: 是否启用双向对齐（同时优化两个序列的顺序）
#         """
#         self.bidirectional = bidirectional

#     def _align_single_item(self, a_item: torch.Tensor, b_item: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         对齐单个样本的token序列
#         :param a_item: [num_tokens, dim]
#         :param b_item: [num_tokens, dim]
#         :return: (aligned_a, aligned_b)
#         """
#         # 计算归一化后的余弦相似度矩阵
#         a_norm = F.normalize(a_item, p=2, dim=1)
#         b_norm = F.normalize(b_item, p=2, dim=1)
#         similarity_matrix = torch.matmul(a_norm, b_norm.T)  # [num_tokens, num_tokens]
        
#         # 使用匈牙利算法找到最优匹配
#         cost_matrix = -similarity_matrix.detach().cpu().numpy()
#         row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
#         if self.bidirectional:
#             # 双向对齐：同时对两个序列重新排序
#             # 按照相似度从高到低排序匹配对
#             matched_scores = similarity_matrix[row_ind, col_ind]
#             sorted_indices = torch.argsort(matched_scores, descending=True)
            
#             # 按照匹配质量重新排列两个序列
#             a_aligned = a_item[row_ind][sorted_indices]
#             b_aligned = b_item[col_ind][sorted_indices]
#             return a_aligned, b_aligned
#         else:
#             # 原始单向对齐：只调整b的顺序
#             b_aligned = b_item[col_ind]
#             return a_item, b_aligned

#     def align(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         对齐批量token序列
#         :param tensor_a: [batch, num_tokens, dim]
#         :param tensor_b: [batch, num_tokens, dim]
#         :return: (aligned_a, aligned_b)
#         """
#         if tensor_a.shape != tensor_b.shape:
#             raise ValueError("Input tensors must have the same shape.")
#         if tensor_a.dim() != 3:
#             raise ValueError("Input tensors must be 3-dimensional [batch, num_tokens, dim].")

#         batch_size = tensor_a.shape[0]
#         aligned_a_list = []
#         aligned_b_list = []

#         for i in range(batch_size):
#             a_item = tensor_a[i]  # [num_tokens, dim]
#             b_item = tensor_b[i]  # [num_tokens, dim]
            
#             a_aligned, b_aligned = self._align_single_item(a_item, b_item)
#             aligned_a_list.append(a_aligned)
#             aligned_b_list.append(b_aligned)
            
#         tensor_a_aligned = torch.stack(aligned_a_list, dim=0)
#         tensor_b_aligned = torch.stack(aligned_b_list, dim=0)
        
#         return tensor_a_aligned, tensor_b_aligned

#     def __call__(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         return self.align(tensor_a, tensor_b)
import torch
import torch.nn as nn

class GeminiFusionAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int = None, dropout: float = 0.0):
        super().__init__()
        if dim_head is None:
            dim_head = dim
            
        self.dim = dim
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim_head, bias=False)
        self.to_k = nn.Linear(dim, dim_head, bias=False)
        self.to_v = nn.Linear(dim, dim_head, bias=False)

        self.noise = nn.Parameter(torch.randn(1, 1, dim))

        self.relation_discriminator = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.to_out = nn.Sequential(
            #nn.Linear(dim_head, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = x1.device
        self.to(device)

        # --- Fused update for x1 (y1) ---
        q1 = self.to_q(x1)
        k1_self = self.to_k(x1 + self.noise)
        v1_self = self.to_v(x1)
        k2_cross = self.to_k(x2)
        v2_cross = self.to_v(x2)

        x_cat_12 = torch.cat((x1, x2), dim=-1)
        relation_score_12 = self.relation_discriminator(x_cat_12)
        k2_cross_modulated = k2_cross * relation_score_12

        k1_combined = torch.stack([k1_self, k2_cross_modulated], dim=-2)
        v1_combined = torch.stack([v1_self, v2_cross], dim=-2)
        
        q1_reshaped = q1.unsqueeze(-2)
        dots1 = torch.matmul(q1_reshaped, k1_combined.transpose(-1, -2)) * self.scale
        attn_weights1 = self.softmax(dots1.squeeze(-2))
        fused_output_1 = torch.matmul(attn_weights1.unsqueeze(-2), v1_combined).squeeze(-2)
        
        y1 = x1 + self.to_out(fused_output_1)
        
        # --- Fused update for x2 (y2) ---
        q2 = self.to_q(x2)
        k2_self = self.to_k(x2 + self.noise)
        v2_self = self.to_v(x2)
        k1_cross = self.to_k(x1)
        v1_cross = self.to_v(x1)

        x_cat_21 = torch.cat((x2, x1), dim=-1)
        relation_score_21 = self.relation_discriminator(x_cat_21)
        k1_cross_modulated = k1_cross * relation_score_21

        k2_combined = torch.stack([k2_self, k1_cross_modulated], dim=-2)
        v2_combined = torch.stack([v2_self, v1_cross], dim=-2)
        
        q2_reshaped = q2.unsqueeze(-2)
        dots2 = torch.matmul(q2_reshaped, k2_combined.transpose(-1, -2)) * self.scale
        attn_weights2 = self.softmax(dots2.squeeze(-2))
        fused_output_2 = torch.matmul(attn_weights2.unsqueeze(-2), v2_combined).squeeze(-2)

        y2 = x2 + self.to_out(fused_output_2)
        
        return y1, y2

# import torch
# from torch import nn, einsum
# from einops import rearrange, repeat

# # 假设这是一个你项目中已有的辅助函数
# def exists(val):
#     return val is not None

# def default(val, d):
#     return val if exists(val) else d

# # --- 修改后的 GeminiFusionAttention ---

# class GeminiFusionAttention(nn.Module):
#     """
#     一个融合了多头注意力和关系判别器的特殊注意力模块。
#     它接收两个输入张量 x1 和 x2，并为每个张量生成一个更新后的版本 y1 和 y2。
#     更新过程结合了自注意力和经过关系分数调制的交叉注意力。
#     """
#     def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, dropout: float = 0.0):
#         """
#         初始化函数

#         Args:
#             dim (int): 输入和输出张量的维度。
#             dim_head (int, optional): 每个注意力头的维度。默认为 64。
#             heads (int, optional): 注意力头的数量。默认为 8。
#             dropout (float, optional): to_out层后的dropout率。默认为 0.0。
#         """
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.dim = dim
#         self.dim_head = dim_head
#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         # 投影层现在输出到 inner_dim (dim_head * heads)
#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(dim, inner_dim, bias=False)

#         # 噪声和关系判别器保持不变，它们在原始维度上操作
#         self.noise = nn.Parameter(torch.randn(1, 1, dim))
#         self.relation_discriminator = nn.Sequential(
#             nn.Linear(dim * 2, dim // 4),
#             nn.GELU(),
#             nn.Linear(dim // 4, 1),
#             nn.Sigmoid()
#         )
        
#         self.softmax = nn.Softmax(dim=-1)
        
#         # to_out 层现在从 inner_dim 投影回 dim
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         h = self.heads
#         b1, n1, _ = x1.shape
#         b2, n2, _ = x2.shape # 假设x1和x2的序列长度和批大小相同

#         q1 = self.to_q(x1)
#         k1_self = self.to_k(x1 + self.noise)
#         v1_self = self.to_v(x1)
#         k2_cross = self.to_k(x2)
#         v2_cross = self.to_v(x2)

#         q2 = self.to_q(x2)
#         k2_self = self.to_k(x2 + self.noise)
#         v2_self = self.to_v(x2)
#         k1_cross = self.to_k(x1)
#         v1_cross = self.to_v(x1)

#         q1, k1_self, v1_self, k2_cross, v2_cross = map(
#             lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), 
#             (q1, k1_self, v1_self, k2_cross, v2_cross)
#         )
#         q2, k2_self, v2_self, k1_cross, v1_cross = map(
#             lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), 
#             (q2, k2_self, v2_self, k1_cross, v1_cross)
#         )

#         # --- Fused update for x1 (y1) ---

#         x_cat_12 = torch.cat((x1, x2), dim=-1)
#         relation_score_12 = self.relation_discriminator(x_cat_12) # shape: (b, n, 1)
        
#         relation_score_12_h = repeat(relation_score_12, 'b n 1 -> (b h) n 1', h=h)
        
#         k2_cross_modulated = k2_cross * relation_score_12_h

#         k1_combined = torch.stack([k1_self, k2_cross_modulated], dim=-2) # shape: [(b*h), n, 2, dim_head]
#         v1_combined = torch.stack([v1_self, v2_cross], dim=-2)           # shape: [(b*h), n, 2, dim_head]
        
#         # q1: [(b*h), n, dim_head] -> [(b*h), n, 1, dim_head]
#         q1_reshaped = q1.unsqueeze(-2) 
#         # dots1: [(b*h), n, 1, 2]
#         dots1 = torch.matmul(q1_reshaped, k1_combined.transpose(-1, -2)) * self.scale
#         # attn_weights1: [(b*h), n, 2]
#         attn_weights1 = self.softmax(dots1.squeeze(-2))
        
#         # fused_output_1: [(b*h), n, dim_head]
#         fused_output_1 = torch.matmul(attn_weights1.unsqueeze(-2), v1_combined).squeeze(-2)
        
#         # '(b h) n d' -> 'b n (h d)'
#         fused_output_1 = rearrange(fused_output_1, '(b h) n d -> b n (h d)', h=h, b=b1)
        
#         y1 = x1 + self.to_out(fused_output_1)
        
#         # --- Fused update for x2 (y2) ---
        
#         x_cat_21 = torch.cat((x2, x1), dim=-1)
#         relation_score_21 = self.relation_discriminator(x_cat_21)
#         relation_score_21_h = repeat(relation_score_21, 'b n 1 -> (b h) n 1', h=h)
#         k1_cross_modulated = k1_cross * relation_score_21_h

#         k2_combined = torch.stack([k2_self, k1_cross_modulated], dim=-2)
#         v2_combined = torch.stack([v2_self, v1_cross], dim=-2)
        
#         q2_reshaped = q2.unsqueeze(-2)
#         dots2 = torch.matmul(q2_reshaped, k2_combined.transpose(-1, -2)) * self.scale
#         attn_weights2 = self.softmax(dots2.squeeze(-2))
#         fused_output_2 = torch.matmul(attn_weights2.unsqueeze(-2), v2_combined).squeeze(-2)

#         fused_output_2 = rearrange(fused_output_2, '(b h) n d -> b n (h d)', h=h, b=b2)

#         y2 = x2 + self.to_out(fused_output_2)
        
#         return y1, y2
class IQARegression(nn.Module):
    def __init__(self, inchannels=768, outchannels=512):
        super().__init__()
        self.down_channel= nn.Conv2d(inchannels*4 , 768, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        scale = inchannels ** -0.5
        self.cross_attention = CrossAttention(outchannels)
        self.cross_attention_text = CrossAttention(outchannels)
        self.norm1 = nn.LayerNorm(outchannels)
        self.norm2 = nn.LayerNorm(outchannels)
        self.norm3 = nn.LayerNorm(outchannels)
        self.proj = nn.Parameter(scale * torch.randn(inchannels, outchannels))
        self.proj_pool=nn.Linear(35, 1)
        self.gating_network = nn.Sequential(
            nn.Linear(outchannels*2, 4),
            nn.Softmax(dim=1)
        )
        proj_experts = []
        self.proj_nums = 4
        for i in range(self.proj_nums):
            proj_experts.append(Projection(2*outchannels, outchannels))
        self.proj_experts = nn.Sequential(*proj_experts)
        self.iqapool=IQAPooling(512)
        self.experts_new= MoERegressor(num_experts=4,select_dim=512,top_k=3)
        self.aligner = TokenAligner()
        self.gemini_fusion_layer = GeminiFusionAttention(dim=512)
    
    def forward(self, x, text_features,text_features_answer,crop_image_csf):
        f_dis = self.down_channel(x)
        f_dis = self.conv(f_dis)
        B, C, W, H = f_dis.shape
        L = W*H
        f_dis = f_dis.view(B, C, L).permute(0, 2, 1).contiguous()
        text_features = text_features @ self.proj
        text_features_answer =text_features_answer @ self.proj
        f_dis = self.norm1(f_dis)
        f_dis = f_dis + self.cross_attention(f_dis, self.norm2(text_features))
        #consistency_text_features = self.cross_attention_text(text_features_answer,text_features)
        text_features_answer,text_features=self.aligner(text_features_answer,text_features)
        #consistency_text_features=text_features_answer-text_features
        _,consistency_text_features=self.gemini_fusion_layer(text_features_answer,text_features)
        consistency_text_features=self.proj_pool(consistency_text_features.permute(0,2,1).contiguous()).squeeze(-1)
        f_dis = f_dis.permute(0, 2, 1).view(B, C, W, H).contiguous()
        f_dis=self.iqapool(f_dis,crop_image_csf).unsqueeze(-1)
        f_dis = f_dis.view(f_dis.size(0), -1).contiguous()
        f_dis=torch.cat((f_dis,self.norm3(consistency_text_features)),1)  #[batch,1024]
        #混合专家回归

        pred=self.experts_new(f_dis)
        return pred   #[batch,1]
class BLIPRegressionModel(nn.Module):
    def __init__(self, pretrained='', **kwargs):
        super().__init__()
        self.blip_encoder = blip_feature_extractor(med_config='', image_size=224, vit='base')
        self.init_saveoutput()
        self.cross_attention = CrossAttention(512)
        self.down_channel= nn.Conv2d(768*4 , 768, kernel_size=1)
        self.regressor = IQARegression()
    def compute_csf_weight(self, x_spatial):
        B, C, H, W = x_spatial.shape
        device = x_spatial.device
        patch_size = 16
        x_gray = x_spatial.mean(dim=1, keepdim=True)
        x_gray = (x_gray - x_gray.min()) / (x_gray.max() - x_gray.min() + 1e-6)
        h_patches = H // patch_size
        w_patches = W // patch_size
        temp_output = torch.zeros(B, 1, h_patches, w_patches, device=device)
        for i in range(h_patches):
            for j in range(w_patches):
                patch = x_gray[:, :,
                            i*patch_size:(i+1)*patch_size,
                            j*patch_size:(j+1)*patch_size]
                fft_patch = fftshift(fft2(patch, dim=(-2, -1)), dim=(-2, -1))
                magnitude = torch.abs(fft_patch)
                u = torch.fft.fftshift(torch.fft.fftfreq(patch_size, d=0.02, device=device))
                v = torch.fft.fftshift(torch.fft.fftfreq(patch_size, d=0.02, device=device))
                uu, vv = torch.meshgrid(u, v, indexing='ij')
                freq = torch.sqrt(uu**2 + vv**2)
                csf_weights = 2.6 * (0.0192 + 0.114 * freq) * torch.exp(-(0.114 * freq)**1.1)
                temp_output[:, :, i, j] = torch.sum(magnitude * csf_weights)
        sensitivity_map = F.interpolate(temp_output,
                                    size=(H, W),
                                    mode='bilinear',
                                    align_corners=False)
        sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min() + 1e-6)
        return sensitivity_map
    def init_saveoutput(self):
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.blip_encoder.visual_encoder.blocks:
            handle = layer.register_forward_hook(self.save_output)
            hook_handles.append(handle)
    def forward(self, image, caption):
        crop_image=image[:, :3, :, :]
        crop_image_csf=self.compute_csf_weight(crop_image)
        text_features_answer,text_features,_=self.blip_encoder(image, caption, mode='multimodal')
        vit_dis = get_vit_feature(self.save_output)
        self.save_output.outputs.clear()
        B = vit_dis.shape[0]
        feat = vit_dis.transpose(1, 2).contiguous()
        feat = feat.view(B, 3072, 14, 14).contiguous()
        scores = self.regressor(feat, text_features,text_features_answer,crop_image_csf)
        return scores
def build_agiqa_model(config,device):
    model=BLIPRegressionModel().to(device)
    return model