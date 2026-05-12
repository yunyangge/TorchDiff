import torch
from torchdiff.modules.osp_next import OSPNextModel

orig_weights_path = 'output/hif8/osp_next_14b_81f720p_sparse2d2_ssp4_from_full/iter_000006000/ema_model_state_dict.pt'
save_path = 'osp_next/hif8_osp_next_14b_81f720p_sparse2d2'

config = {
  'dim': 5120,
  'ffn_dim': 13824,
  'freq_dim': 256,
  'in_dim': 16,
  'num_heads': 40,
  'num_layers': 40,
  'out_dim': 16,
  'text_len': 512,
  'skiparse_model_type': "dual_end",
  'sparse_ratio': 2,
  'num_full_blocks': 8,
  'num_register_tokens': 0,
  'skiparse_1d': False,
  'skiparse_2d': True,
  'quant': "hif8",
  'quant_attn': "hif8",
  'scale_max_forward': 15.0,    # forward activation scale clamp (HIF8 max ≈ 15)
  'scale_max_backward': 224.0  # backward gradient scale clamp  (HIF8 max ≈ 224)
}

state_dict = torch.load(orig_weights_path, map_location='cpu')
model = OSPNextModel(**config)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"missing_keys: {missing_keys} \nunexpected_keys: {unexpected_keys}")
model.save_pretrained(save_path)