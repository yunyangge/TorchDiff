import os
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

sample_videos_dir = "/home/ma-user/work/gyy/TorchDiff/samples/vbench/wan_t2v_baseline"
videos = glob.glob(f"{sample_videos_dir}/*.mp4")

idx_to_name_txt = "eval/idx2prompt.txt"
save_dir = "/home/ma-user/work/gyy/TorchDiff/samples/vbench_renamed/wan_t2v_baseline"
os.makedirs(save_dir, exist_ok=True)

with open(idx_to_name_txt, "r") as f:
    idx_to_name = {int(line.split('\t')[0]): line.split('\t')[1].strip() for line in f}

# 定义单个视频的处理任务
def process_video(video):
    video_name = os.path.basename(video)
    try:
        idx = int(video_name.split("_")[1])
        local_idx = int(video_name.split("_")[-1].removesuffix(".mp4"))
    except (IndexError, ValueError) as e:
        return {"status": "error", "video": video_name, "msg": f"Error parsing index: {e}"}

    if idx in idx_to_name:
        # 完全保留你原始的命名逻辑，不做任何更改
        safe_name = idx_to_name[idx].replace("/", "_")
        new_name = safe_name + f"-{local_idx}.mp4"
        dest_path = os.path.join(save_dir, new_name)
        
        try:
            # 执行复制（单独对 I/O 加上 try-except 以捕获复制失败）
            shutil.copy(video, dest_path)
            return {"status": "success", "video": video_name, "msg": f"Success: Renamed/Copied {video_name} to {new_name}"}
        except Exception as e:
            # 如果复制失败（如文件名过长），在此捕获并返回
            return {"status": "error", "video": video_name, "msg": f"Copy Failed [{type(e).__name__}]: {e}"}
    else:
        return {"status": "error", "video": video_name, "msg": f"Index {idx} not found in idx_to_name mapping."}

# 设置线程池的工作线程数
max_workers = 64

print(f"Starting multi-thread processing with {max_workers} workers...")
print(f"Total videos to process: {len(videos)}")

# 用于记录失败的文件
failed_logs = []

# 使用 ThreadPoolExecutor 并发执行
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 使用字典来保存 future 和对应的 video 路径，防止由于未知严重异常导致未来丢失文件名信息
    futures = {executor.submit(process_video, video): video for video in videos}
    
    for future in as_completed(futures):
        try:
            result = future.result()
            if result["status"] == "error":
                failed_logs.append(result)
                print(f"❌ FAILED: {result['video']} -> {result['msg']}")
            else:
                # 成功也可以打印，如果你嫌太刷屏可以把下面这行注释掉
                print(f"✅ {result['msg']}")
        except Exception as exc:
            # 捕获到极其罕见的线程崩溃异常
            video_path = futures[future]
            video_name = os.path.basename(video_path)
            error_dict = {"status": "error", "video": video_name, "msg": f"Thread crashed: {exc}"}
            failed_logs.append(error_dict)
            print(f"❌ FAILED (Thread Error): {video_name} -> {exc}")

# ======================== 结果总结与 Log 写入 ========================
print("\n" + "="*50)
print("All tasks completed.")
print(f"Total files : {len(videos)}")
print(f"Successful  : {len(videos) - len(failed_logs)}")
print(f"Failed      : {len(failed_logs)}")

# 将所有失败记录写入一个独立的 log 文件中
if failed_logs:
    log_filename = "copy_failed.log"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        for item in failed_logs:
            log_file.write(f"Original Video: {item['video']} | Error Detail: {item['msg']}\n")
    print(f"⚠️ [ATTENTION] The list of failed files has been saved to '{log_filename}'. Please check it!")
print("="*50)