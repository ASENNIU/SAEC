from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id, \
    sim_pert2v_matrix_training, t2v_metrics, v2t_metrics, v2t_metrics_dsl, t2v_metrics_dsl, _compute_dsl_metrics, _compute_metrics, \
    sim_matrix
from tqdm import tqdm
import os
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0

        self.config = config
        self.caption_weight = config.caption_weight

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.cuda(self.device)
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps - 1, self.evals_per_epoch + 1, dtype=int)[1:]

        for batch_idx, data in enumerate(self.train_data_loader):
            # then assume we must tokenize the input, e.g. its a string
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)

                data['titles'], data['titles_mask'] = self.tokenizer(data['titles'], return_tensors='pt', padding=True,
                                                 truncation=True, return_mask=True)


            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
                data['titles'] = data['titles'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                data['titles'] = {key: val.to(self.device) for key, val in data['titles'].items()}

            data['video'] = data['video'].to(self.device)

            text_embeds, global_video_embeds = self.model(data)

            output_global_sim = sim_matrix_training(text_embeds, global_video_embeds, "avg")
            loss = self.loss(output_global_sim, self.model.clip.logit_scale)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                logger.info('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps - 1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps - 1)
                self.model.train()

                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']
                    self._save_checkpoint(epoch, save_best=True)
                logger.info("\t--------------------------- Best Modeling ----------------------------------")
                logger.info(" Current Best Window Average R@1 is {}".format(self.best_window))
                logger.info(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train': total_loss / num_steps
        }

        return res

    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        # self.model.cuda(self.device)
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    data['titles'], data['titles_mask'] = self.tokenizer(data['titles'], return_tensors='pt',
                                                                         padding=True,
                                                                         truncation=True, return_mask=True)

                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                    data['titles'] = data['titles'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['titles'] = {key: val.to(self.device) for key, val in data['titles'].items()}

                data['video'] = data['video'].to(self.device)

                text_embed,  global_vid_embed = self.model(data)
                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(global_vid_embed.cpu())
                sims_batch_global = sim_matrix_training(text_embed, global_vid_embed, "avg")

                curr_loss = self.loss(sims_batch_global, self.model.clip.logit_scale)

                total_val_loss += curr_loss.item()

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)

            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)

            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            # vid_embeds_per_video_id = {}
            # for idx, v_id in enumerate(all_vid_ids):
            #     if v_id not in vid_embeds_per_video_id:
            #         vid_embeds_per_video_id[v_id] = vid_embeds[idx]
            #
            # vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])


            sims = sim_matrix(text_embeds, vid_embeds)

            logger.info(f"sims shape: {sims.shape}")


            sim_matrix_state = {
                "sims_global": sims,
            }

            matrix_path = os.path.join(self.checkpoint_dir, 'matrix.pth')
            torch.save(sim_matrix_state, matrix_path)
            logger.info(f"Saving matrix state: {matrix_path} ...")



            total_val_loss = total_val_loss / len(self.valid_data_loader)

            res = _compute_metrics(sims)
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            logger.info(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}, loss: {total_val_loss}-----\n")
            logger.info(f"\t>>> Text -> Video:")
            logger.info(f"\t>>> R@1: {res['R1']}, R@5: {res['R5']}, R@10: {res['R10']}, R@50: {res['R50']}, MedR: {res['MedR']}, MeanR: {res['MeanR']}")
            logger.info(f"\t>>> window:")
            logger.info(f"\t>>> R@1: {res['R1-window']}, R@5: {res['R5-window']}, R@10: {res['R10-window']}, R@50: {res['R50-window']}, MedR: {res['MedR-window']}, MeanR: {res['MeanR-window']}")

            # res_v2t = v2t_metrics(sims)
            res_v2t = _compute_metrics(sims.T)
            logger.info(f"\t>>> Video -> Text:")
            logger.info(
                f"\t>>> R@1: {res_v2t['R1']}, R@5: {res_v2t['R5']}, R@10: {res_v2t['R10']}, R@50: {res_v2t['R50']}, MedR: {res_v2t['MedR']}, MeanR: {res_v2t['MeanR']}\n")
            logger.info("\t--------------------------- Metrics after DSL ----------------------------------")
            # res_v2t_dsl = v2t_metrics_dsl(sims)
            res_v2t_dsl = _compute_dsl_metrics(sims.T)
            # res_t2v_dsl = t2v_metrics_dsl(sims)
            res_t2v_dsl = _compute_dsl_metrics(sims)
            logger.info(f"\t>>> Text -> Video DSL:")
            logger.info(
                f"\t>>> R@1: {res_t2v_dsl['R1']}, R@5: {res_t2v_dsl['R5']}, R@10: {res_t2v_dsl['R10']}, R@50: {res_t2v_dsl['R50']}, MedR: {res_t2v_dsl['MedR']}, MeanR: {res_t2v_dsl['MeanR']}")
            logger.info(f"\t>>> Video -> Text DSL:")
            logger.info(
                f"\t>>> R@1: {res_v2t_dsl['R1']}, R@5: {res_v2t_dsl['R5']}, R@10: {res_v2t_dsl['R10']}, R@50: {res_v2t_dsl['R50']}, MedR: {res_v2t_dsl['MedR']}, MeanR: {res_v2t_dsl['MeanR']}\n")


            res['loss_val'] = total_val_loss

            logger.info(f"\t>>> Cur Evaluation Loss: {total_val_loss}")

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res
