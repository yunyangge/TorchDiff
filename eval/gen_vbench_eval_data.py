import json
import glob
import shutil
import os
from tqdm import tqdm
from einops import rearrange

from torchdiff.data.utils.utils import LMDBReader, LMDBWriter
from torchdiff.utils.infer_utils import load_prompts

vbench_eval_data_save_dir = "/home/ma-user/work/gyy/TorchDiff/eval/moviegen"

os.makedirs(vbench_eval_data_save_dir, exist_ok=True)

prompt_txt = "/home/ma-user/work/gyy/TorchDiff/assets/t2v/moviegen_for_osp.txt"
short_prompt_txt = "/home/ma-user/work/gyy/TorchDiff/assets/t2v/moviegen_for_osp.txt"
idx2prompt_txt = os.path.join(vbench_eval_data_save_dir, 'idx2prompt_moviegen_for_osp.txt')
prompts = load_prompts(prompt_txt)
short_prompts = load_prompts(short_prompt_txt)

with open(idx2prompt_txt, 'w') as f:
    for idx, prompt in enumerate(short_prompts):
        f.write(f"{idx}\t{prompt}\n")

prompts = [{"cap": item} for item in prompts]

writer = LMDBWriter()
writer.save_filtered_data_samples(prompts, vbench_eval_data_save_dir)
test_reader = LMDBReader(vbench_eval_data_save_dir) 
print(test_reader.getitem(0))