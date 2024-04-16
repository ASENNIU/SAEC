python test.py \
  --exp_name=MSRVTT_plus_caption_fusion_0.1 \
  --videos_dir=/home/leon/workspace/multimodal/dataset/MSR-VTT-CLIP4clip/MSRVTT_Videos \
  --batch_size=64 \
  --num_mha_heads=1 \
  --loss=clip \
  --caption_weight=0.1 \
  --is_routing_feature=1 \
  --text_patch=12 \
  --frozen_clip=0 \
  --device=cuda:0 \
  --noclip_lr=3e-4 \
  --transformer_dropout=0.3 \
  --dataset_name=MSRVTT \
  --msrvtt_train_file=9k \
  --load_epoch=-1










