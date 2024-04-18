python train.py \
  --exp_name=MSRVTT_GSM_CoAttention_routing0 \
  --videos_dir=/home/leon/workspace/multimodal/dataset/MSR-VTT-CLIP4clip/MSRVTT_Videos \
  --arch=GSE \
  --batch_size=64 \
  --num_mha_heads=1 \
  --loss=clip \
  --caption_weight=0.1 \
  --is_routing_feature=0 \
  --frozen_clip=0 \
  --device=cuda:1 \
  --seed=8 \
  --noclip_lr=3e-5 \
  --transformer_dropout=0.3 \
  --text_patch=12 \
  --dataset_name=MSRVTT \
  --msrvtt_train_file=9k \
  --evals_per_epoch=5 \
  --num_epochs=5







