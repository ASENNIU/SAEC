import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config
import os
import logging

logger = logging.getLogger(__name__)

class CoMultiHeadedAttention(nn.Module):
    """
        @brief: Cross MultiHead Attention Transformer
    """
    def __init__(self, config: Config):
        super(CoMultiHeadedAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        logger.info(f"MultiHeadAttention num: {self.num_heads}")


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
            prior_attention: num_texts x num_vids
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q


        attention_logits = attention_logits / math.sqrt(self.embed_dim)
        attention_weights = F.softmax(attention_logits, dim=2)



        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)

        return o, attention_weights


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.cross_attn = CoMultiHeadedAttention(config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds, is_save_attention_weights=False):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
            cos_sim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out, attention_weights = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)
        if is_save_attention_weights == True:
            return out, attention_weights
        return out

class MLP(nn.Module):
    def __init__(self, n_dim):
        super(MLP, self).__init__()
        self.n_dim = n_dim
        self.linear1 = nn.Linear(self.n_dim, self.n_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(self.n_dim, self.n_dim)

        self._init_parameters()
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)

        return x


class SelfMultiHeadedAttention(nn.Module):
    """
        @brief: Cross MultiHead Attention Transformer
    """

    def __init__(self, config: Config):
        super(SelfMultiHeadedAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        logger.info(f"MultiHeadAttention num: {self.num_heads}")

    def forward(self, video_embeds):
        """
        Input

            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_frames x embed_dim
        """
        num_vids, num_frames, n_dim = video_embeds.shape
        # num_vids x n_um_frames x embed_dim
        q = self.q_proj(video_embeds)
        q = q.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        q = q.permute(0, 2, 3, 1)


        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0, 2, 1, 3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0, 2, 3, 1)

        # num_vids x num_heads x num_frames x num_frames
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.embed_dim)
        attention_weights = F.softmax(attention_logits, dim=3)

        # num_vids x num_heads x head_dim x num_frames
        attention = v @ attention_weights
        # num_vids x num_frames x num_heads x head_dim
        attention = attention.permute(0, 3, 1, 2)
        attention = attention.reshape(num_vids, num_frames, self.embed_dim)

        # num_vids x n_frames x embed_dim
        o = self.out_proj(attention)

        return o, attention_weights


class FusionTransformer(nn.Module):
    def __init__(self, config: Config):
        super(FusionTransformer, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.fusion_attention = SelfMultiHeadedAttention(config)

        # self.mlp = MLP(self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, video_embeds):
        """
        Input
            video_embeds: num_vids x num_frames x embed_dim
            cos_smi: n_text, n_vids
        Output
            out: num_vids x num_frames x embed_dim
        """
        video_embeds_ln = self.layer_norm1(video_embeds)

        # num_vids x num_frames x embed_dim
        attn_out, attention_weighs = self.fusion_attention(video_embeds_ln)
        attn_out = attn_out + video_embeds

        attn_out_ln = self.layer_norm2(attn_out)
        out = self.ffn_proj(attn_out_ln)
        out = self.dropout(out) + attn_out

        return out

class CrossAttetionLayer(nn.Module):
    def __init__(self, config: Config):
        super(CrossAttetionLayer, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        # self.num_heads = config.num_mha_heads
        self.num_heads = 8
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds, is_save_attention_weights=False):
        """
        Input
            text_embeds: num_vids x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
            cos_sim
        Output
            out: num_vids  x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        num_texts, _ = text_embeds.shape
        # num_vids x embed_dim
        q = self.q_proj(text_embeds)
        # num_vids x num_heads x 1 x head_dim
        q = q.reshape(num_texts, self.num_heads, 1, self.head_dim)


        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        k = k.permute(0, 2, 3, 1)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        v = v.permute(0, 2, 1, 3)

        # num_vids x num_heads x num_frames
        attention_logits = q @ k

        # attention_logits = attention_logits.squeeze(2)

        attention_logits = attention_logits / math.sqrt(self.embed_dim)
        attention_weights = F.softmax(attention_logits, dim=3)

        # num_vids x num_heads x head_dim
        attention = attention_weights @ v

        # logging.info(f"attention shape: {attention.shape}")
        attention = attention.squeeze(2).reshape(num_vids, self.embed_dim)

        # num_vids x embed_dim
        attn_out = self.out_proj(attention)

        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)
        if is_save_attention_weights == True:
            return out, attention_weights
        return out
