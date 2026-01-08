from multiprocessing.dummy import freeze_support
import os
# 이 줄을 반드시 'import lerobot' 보다 위에 적으세요!
os.environ["HF_HUB_OFFLINE"] = "1"
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import openpi.training.config as _config
import openpi.training.data_loader as _data
import torch
def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    # dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    # print("dataset_meta:", dataset_meta)
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()

def print_batch_structure(batch, prefix=""):
    if isinstance(batch, dict):
        for k, v in batch.items():
            print(f"{prefix}{k}: {type(v)}")
            print_batch_structure(v, prefix + "  ")
    elif hasattr(batch, "shape"):
        print(f"{prefix}shape = {batch.shape}, dtype = {batch.dtype}")
    else:
        print(f"{prefix}{batch}")


import os
# os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


from huggingface_hub import HfApi
# hub_api = HfApi()
# hub_api.create_tag("whwjdqls99/libero_hdfr_lerobot_dataset_depth", tag="v2.0", repo_type="dataset")
# # # 1. Do NOT set HF_HUB_OFFLINE to 1, or the download will fail.
# # # (If you previously set it in this script, delete that line or set it to '0')

dataset = lerobot_dataset.LeRobotDataset(
    # repo_id는 메타데이터 식별용으로 남겨둡니다 (경로 탐색엔 안 쓰임)
    repo_id="whwjdqls99/libero_hdfr_lerobot_dataset_depth_latents",
    
    # [핵심] root를 'meta 폴더가 들어있는' 데이터셋 폴더 전체 경로로 지정하세요!
    root="/scratch2/whwjdqls99/libero/whwjdqls99/libero_hdfr_lerobot_dataset_depth_latents",
    
    delta_timestamps={
        "actions": [t / 10 for t in range(50)], 
        "depth_latent": [(t) / 10 for t in range(51)]
    },
)

print("✅ 데이터셋 로드 성공!")
# print(dataset.root)
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=2,     # 먼저 1~2로 확인하는 게 좋음
    shuffle=False,
    num_workers=8,    # 디버깅할 땐 0 추천
)
total_points = []
# print(len(loader))
print(loader.dataset.meta.stats)
for i, batch in enumerate(loader):
    print("depth_image shape:", batch["depth_latent"].shape)
    print("actions shape:", batch["actions"].shape)
exit()
for i, batch in enumerate(loader):
    if i == 100:
        break
    if i % 10 == 0:
        print(f"Processing batch {i} / {len(loader)}")
    BS, T, N, D = batch["point_cloud"].shape
    points = batch["point_cloud"].reshape(BS * T * N, D)
    total_points.append(points.cpu())
    
pc_stacked = torch.cat(total_points, dim=0)
mean = pc_stacked.mean(dim=0) # (D,)
std = pc_stacked.std(dim=0)   # (D,)
print(f"Point Cloud Mean: {mean}, Std: {std}")
# observation = next(iter(loader))
# print(observation)
# print("Observation keys:", observation.keys())WWW
# print("Actions keys:", actions.keys())

# print_batch_structure(batch)
# # print data
# for i in range(3):
#     data = dataset[i]
#     print(f"Data sample {i}:")
#     for key, value in data.items():
#         if isinstance(value, bytes):
#             print(f"  {key}: <bytes data of length {len(value)}>")
#         else:
#             print(f"  {key}: {value}")
# from huggingface_hub import HfApi
# import json

# # 1. Inspect your local info.json to check the version (usually "v2.0")
# # If you don't have local access, assume "v2.0" based on your error log.
# version_tag = "v2.0" 

# repo_id = "whwjdqls99/libero_hdfr_lerobot_track"

# api = HfApi()
# # This creates the missing tag on your remote repository
# api.create_tag(repo_id, tag=version_tag, repo_type="dataset")

# print(f"Successfully tagged {repo_id} with {version_tag}")
