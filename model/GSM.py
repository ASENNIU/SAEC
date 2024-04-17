import logging

import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer, FusionTransformer, MLP, CrossAttetionLayer
import torch.nn.functional as F
from modules.differential_topk import TextTokenSelection
import logging

logger = logging.getLogger(__name__)

class GSE(nn.Module):
    def __init__(self, config: Config):
        super(GSE, self).__init__()
        self.config = config

        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch, config.device)

        if self.config.frozen_clip == 1:
            for name, parameter in self.clip.named_parameters():
                parameter.requires_grad = False

        config.pooling_type = 'transformer'
        self.pool_texts = CrossAttetionLayer(config)
        # self.text_rounting = nn.Linear(config.embed_dim * 2, config.embed_dim, bias=False)
        self.caption_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)


    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        titles_data = data['titles']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        text_features = self.clip.encode_text(text_data)
        video_features = self.clip.encode_image(video_data)

        titles_features = self.clip.encode_text(titles_data, return_hidden=False)

        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled_by_cap = self.pool_texts(titles_features, video_features)

        if self.config.is_routing_feature == 1:
            fusion_features = self.text_rounting(torch.cat([video_features_pooled_by_cap, self.caption_proj(titles_features)], dim=-1))
        elif self.config.is_routing_feature == 2:
            fusion_features = (video_features_pooled_by_cap + titles_features) / 2.0
        else:
            fusion_features = video_features_pooled_by_cap

        return text_features, fusion_features




