from multiprocessing.dummy import freeze_support
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import openpi.training.config as _config
import openpi.training.data_loader as _data

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



if __name__ == "__main__":
    import torch
    freeze_support()
    config = _config.cli()
    loader, data_config = build_datasets(config)
        # self._data_loader = data_loader
    print(loader._data_loader._data_loader.dataset._dataset._dataset.meta.stats)
    print("data_config:", data_config)

    image_tensors = {}
    pc_tensors = []
    num_batches = 100

    for i, (obs, _) in enumerate(loader):
        if i >= num_batches:
            break
        print(f"Processing batch {i + 1} / {num_batches}")
        # Access images directly
        for key, val in obs.images.items():
            if key not in image_tensors: image_tensors[key] = []
            image_tensors[key].append(val.cpu())# (Bs, 3, 256, 256)

        if obs.point_cloud is None:
            continue
        B, T, N, D = obs.point_cloud.shape
        points = obs.point_cloud.reshape(B * T * N, D)
        pc_tensors.append(points.cpu())

    print("\n--- Statistics ---")
    for key, data in image_tensors.items():
        stacked = torch.cat(data, dim=0) 
        mean = stacked.mean(dim=(0, 2, 3))  # (3,)
        std = stacked.std(dim=(0, 2, 3))    # (3,)
        print(f"Image [{key}] Mean: {mean}, Std: {std}")
        # print(f"Image [{key}] Mean: {stacked.mean():.4f}, Std: {stacked.std():.4f}")

    pc_stacked = torch.cat(pc_tensors, dim=0)
    mean = pc_stacked.mean(dim=0) # (D,)
    std = pc_stacked.std(dim=0)   # (D,)
    print(f"Point Cloud Mean: {mean}, Std: {std}")
# # # 1. Do NOT set HF_HUB_OFFLINE to 1, or the download will fail.
# # # (If you previously set it in this script, delete that line or set it to '0')

# dataset = lerobot_dataset.LeRobotDataset(
#     # 2. Use the full Hugging Face repository ID (User/Dataset)
#     repo_id="whwjdqls99/libero_hdfr_lerobot_track_datasets_w_pt",
#     root="/scratch2/whwjdqls99/libero/libero_hdfr_lerobot_track_datasets_w_pt",
#     delta_timestamps={"actions": [t / 10 for t in range(50)]},
# )
# print(dataset.root)
# from torch.utils.data import DataLoader

# loader = DataLoader(
#     dataset,
#     batch_size=1,     # 먼저 1~2로 확인하는 게 좋음
#     shuffle=False,
#     num_workers=0,    # 디버깅할 땐 0 추천
# )

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