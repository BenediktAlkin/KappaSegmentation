# architecture changes for segmentation
- learnable attn bias per block (initialized with 0.0)
- layerscale is added (initialized with 1.0)
- last norm is removed

# notes
- as far as i understand the img_scale=(2048, 512) is useless in mmseg; it always resizes the shortest edge to 512

# TODO
- weights of last attn/mlp proj are rescaled based on layer index?? (fix_init_weight)

# differences to original implementation
- attn bias is excluded from weight decay
- conv layers are initialized with xavier_uniform and zero bias
- replace apex with torch.amp
- schedule is warmup cosine instead of warmup linear (polynomial with power=1 is linear)
  - warmup_ratio=1e-06 is ignored
- warmup duration is slightly different (original has 1500 iterations)
- when rescaling an image, the new size is calculated with round(height * scale) instead of int(height * scale + 0.5)
- colorjitter is slightly different
- no learnable absolute position for cls token
