python test.py \
  --exp_name=MSRVTT_GSM_CoAttention_routing0 \
  --videos_dir=/home/leon/workspace/multimodal/dataset/MSR-VTT-CLIP4clip/MSRVTT_Videos \
  --arch=GSE \
  --batch_size=64 \
  --num_mha_heads=1 \
  --loss=clip \
  --caption_weight=0.1 \
  --is_routing_feature=0 \
  --text_patch=12 \
  --frozen_clip=0 \
  --device=cuda:1 \
  --noclip_lr=3e-4 \
  --transformer_dropout=0.3 \
  --dataset_name=MSRVTT \
  --msrvtt_train_file=9k \
  --load_epoch=-1










