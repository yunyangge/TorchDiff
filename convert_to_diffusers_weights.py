import torch
from torchvideo.modules.flashi2v import FlashI2VModel

orig_weights_path = '/work/share1/checkpoint/gyy/flashi2v_14b/iter_000040000/ema_model_state_dict.pt'
save_path = '/work/share1/checkpoint/gyy/flashi2v_14b/diffusers_weights'
state_dict = torch.load(orig_weights_path, map_location='cpu')
config = {
  'dim': 5120,
  'ffn_dim': 13824,
  'freq_dim': 256,
  'in_dim': 16,
  'num_heads': 40,
  'num_layers': 40,
  'out_dim': 16,
  'text_len': 512,
  'low_freq_energy_ratio': 0.1,
  'fft_return_abs': True,
  'conv3x3x3_proj': False,
}
model = FlashI2VModel(**config)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"missing_keys: {missing_keys} \nunexpected_keys: {unexpected_keys}")
model.save_pretrained(save_path)