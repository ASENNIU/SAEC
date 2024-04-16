import torch.nn as nn
import torch
import torch.nn.functional as F
from config.base_config import Config
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CLIPLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        logger.info("use clip loss")
    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """

        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0


class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.clip_loss = CLIPLoss(config)
        self.lambda_mse = nn.Parameter(torch.tensor([0.1]), requires_grad=True).to(self.device)
        logger.info("use reg loss")

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """

        clip_loss = self.clip_loss(sims, logit_scale)
        positive = torch.diag(sims)
        mse_loss = F.mse_loss(positive, torch.ones(positive.shape[0], requires_grad=False).to(self.device)) / 2

        return mse_loss * self.lambda_mse + clip_loss


class DualSoftmaxLoss(nn.Module):
    def __init__(self, config):
        super(DualSoftmaxLoss, self).__init__()
        self.config = config

    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix / temp, dim=0) * len(
            sim_matrix)  # With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -torch.mean(logpt)
        return loss

class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == 'clip':
            return CLIPLoss(config)
        elif config.loss == 'reg_loss':
            return Loss(config)
        elif config.loss == 'dsl':
            return DualSoftmaxLoss(config)
