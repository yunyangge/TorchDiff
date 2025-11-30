import json
import glob
import shutil
import os
from tqdm import tqdm
from einops import rearrange

from torchvideo.data.utils.utils import LMDBReader, LMDBWriter

sample_height = 480
sample_width = 832
vbench_image_dir = "/work/share/projects/gyy/resi2v/ResI2V_MM/vbench_26-15"
vbench_json_path = "/work/share/projects/gyy/resi2v/ResI2V_MM/vbench2_i2v_full_info_refine_2.json"
with open(vbench_json_path, 'r') as f:
    org_meta_info_list = json.load(f)

new_meta_info_list = []

for item in org_meta_info_list:
    if "camera_motion" in item["dimension"]:
        new_meta_info_list.append(item)

name2idx = {}
for index, item in enumerate(new_meta_info_list):
    name2idx[item['image_name']] = index

vbench_eval_data_save_dir = "vbench_eval_data_camera_motion"
os.makedirs(vbench_eval_data_save_dir, exist_ok=True)

save_meta_info_list = []
for index, item in enumerate(new_meta_info_list):
    meta_info = {"sample_height": sample_height, "sample_width": sample_width}
    meta_info["prompt_en"] = item["prompt_en"]
    meta_info["cap"] = item["refined_caption"]
    meta_info["old_name"] = item["image_name"]
    save_path = os.path.join(vbench_eval_data_save_dir, f'{index:06d}.jpg')
    meta_info["path"] = save_path
    shutil.copy(os.path.join(vbench_image_dir, item["image_name"]), save_path)
    save_meta_info_list.append(meta_info)

with open(os.path.join(vbench_eval_data_save_dir, 'meta_info.json'), 'w') as f:
    json.dump(save_meta_info_list, f, indent=4)

writer = LMDBWriter()
writer.save_filtered_data_samples(save_meta_info_list, vbench_eval_data_save_dir)
reader = LMDBReader(vbench_eval_data_save_dir) 
print(reader.getitem(0))

save_txt_path = 'vbench_idx_to_name_camera_motion.txt'

with open(save_txt_path, 'w') as f:
    for i in range(len(reader)):
        f.write(f'{i}\t{reader.getitem(i)["prompt_en"]}\n')