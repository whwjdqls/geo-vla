#!/usr/bin/env python3
"""
Real World 데이터 준비 스크립트

원본 LeRobot v3.0 데이터셋을 LeRobot v2.1 호환 형식으로 변환하고,
3D point track 데이터를 통합합니다.

Usage:
    # Track head 학습용 (point_cloud 포함)
    python scripts/prepare_real_world_data.py \
        --input_dir /weka/jisookim/dataset/real_world/omy_f3m_pick_hat_depth_0128 \
        --track_dir /weka/.../pointrack/results \
        --output_dir /weka/jisookim/dataset/real_world_lerobot/omy_f3m_pick_hat_depth_0128_pt \
        --mode track \
        --num_points 1024

    # Base model 학습용 (point_cloud 없음)
    python scripts/prepare_real_world_data.py \
        --input_dir /weka/jisookim/dataset/real_world/omy_f3m_pick_hat_depth_0128 \
        --output_dir /weka/jisookim/dataset/real_world_lerobot/omy_f3m_pick_hat_depth_0128_base \
        --mode base

Input (LeRobot v3.0):
    input_dir/
    ├── meta/
    │   ├── info.json          # codebase_version: v3.0
    │   ├── stats.json
    │   ├── tasks.parquet
    │   └── episodes/chunk-XXX/file-XXX.parquet
    ├── data/
    │   └── chunk-XXX/file-000.parquet  # all episodes in one file
    └── videos/
        └── {video_key}/chunk-XXX/file-XXX.mp4

Output (LeRobot v2.1):
    output_dir/
    ├── meta/
    │   ├── info.json          # codebase_version: v2.1
    │   ├── stats.json
    │   ├── tasks.jsonl
    │   ├── episodes.jsonl
    │   └── episodes_stats.jsonl
    ├── data/
    │   └── chunk-XXX/episode_XXXXXX.parquet  # one file per episode
    └── videos/
        └── {video_key}/chunk-XXX/episode_XXXXXX.mp4
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_lerobot_metadata(input_dir: Path) -> Tuple[Dict, Dict[int, str]]:
    """Load LeRobot dataset metadata."""
    with open(input_dir / "meta" / "info.json", "r") as f:
        info = json.load(f)

    tasks = {}
    tasks_parquet_path = input_dir / "meta" / "tasks.parquet"
    if tasks_parquet_path.exists():
        tasks_df = pd.read_parquet(tasks_parquet_path)
        for task_desc, row in tasks_df.iterrows():
            tasks[int(row["task_index"])] = str(task_desc)

    return info, tasks


def load_all_parquet_data(input_dir: Path, info: Dict) -> pd.DataFrame:
    """Load all parquet data from chunks."""
    all_dfs = []
    chunk_idx = 0
    while True:
        parquet_path = input_dir / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet"
        if not parquet_path.exists():
            break
        df = pd.read_parquet(parquet_path)
        all_dfs.append(df)
        chunk_idx += 1

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def load_tracks_3d(track_dir: Path, episode_idx: int, num_points: int = 1024) -> Optional[np.ndarray]:
    """Load 3D point tracks for a specific episode."""
    track_file = track_dir / f"file-{episode_idx:03d}" / "tracks_3d.npy"

    if not track_file.exists():
        return None

    tracks = np.load(track_file)  # (T, N, 3)

    # Sample or pad points
    if tracks.shape[1] > num_points:
        indices = np.linspace(0, tracks.shape[1] - 1, num_points, dtype=int)
        tracks = tracks[:, indices, :]
    elif tracks.shape[1] < num_points:
        padding = np.zeros((tracks.shape[0], num_points - tracks.shape[1], 3), dtype=tracks.dtype)
        tracks = np.concatenate([tracks, padding], axis=1)

    return tracks.astype(np.float32)


def compute_point_cloud_stats(
    track_dir: Path,
    total_episodes: int,
    num_points: int = 1024
) -> Dict:
    """Compute normalization statistics for point cloud data."""
    print("  Computing point cloud statistics...")

    all_points = []
    all_deltas = []

    for episode_idx in tqdm(range(total_episodes), desc="  Loading tracks"):
        tracks = load_tracks_3d(track_dir, episode_idx, num_points)
        if tracks is None:
            continue

        all_points.append(tracks.reshape(-1, 3))
        if tracks.shape[0] > 1:
            deltas = tracks[1:] - tracks[:-1]
            all_deltas.append(deltas.reshape(-1, 3))

    if not all_points:
        raise ValueError("No valid track data found!")

    all_points = np.concatenate(all_points, axis=0)
    all_deltas = np.concatenate(all_deltas, axis=0) if all_deltas else np.zeros((1, 3))

    return {
        "mean": all_points.mean(axis=0).tolist(),
        "std": all_points.std(axis=0).tolist(),
        "min": all_points.min(axis=0).tolist(),
        "max": all_points.max(axis=0).tolist(),
        "count": [len(all_points)],  # Required by LeRobot
        "q01": np.percentile(all_points, 1, axis=0).tolist(),
        "q10": np.percentile(all_points, 10, axis=0).tolist(),
        "q50": np.percentile(all_points, 50, axis=0).tolist(),
        "q90": np.percentile(all_points, 90, axis=0).tolist(),
        "q99": np.percentile(all_points, 99, axis=0).tolist(),
        "delta_mean": all_deltas.mean(axis=0).tolist(),
        "delta_std": all_deltas.std(axis=0).tolist(),
        "delta_q01": np.percentile(all_deltas, 1, axis=0).tolist(),
        "delta_q99": np.percentile(all_deltas, 99, axis=0).tolist(),
    }


def convert_info_json(
    input_dir: Path,
    output_dir: Path,
    include_point_cloud: bool = True,
    num_points: int = 1024
) -> Dict:
    """Convert info.json from v3.0 to v2.1 compatible format."""
    with open(input_dir / "meta" / "info.json", "r") as f:
        info = json.load(f)

    # Use v2.0 format to avoid LeRobot's aggregate_stats dropping delta_mean/delta_std
    # v2.1+ aggregates from episodes_stats.jsonl and loses extra keys
    # v2.0 loads directly from stats.json which preserves all keys
    info["codebase_version"] = "v2.0"

    # Fix data_path and video_path format strings
    # v3.0: chunk-{chunk_index:03d}/file-{file_index:03d}
    # v2.1: chunk-{episode_chunk:03d}/episode_{episode_index:06d}
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"

    # Rename 'action' feature to 'actions' (v3.0 -> v2.1 naming convention)
    if "action" in info.get("features", {}) and "actions" not in info.get("features", {}):
        info["features"]["actions"] = info["features"].pop("action")

    # Add point_cloud feature if needed
    if include_point_cloud:
        info["features"]["point_cloud"] = {
            "dtype": "float32",
            "shape": [num_points, 3],
            "names": ["point_index", "xyz"],
        }

    output_meta_dir = output_dir / "meta"
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    with open(output_meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    return info


def convert_tasks(input_dir: Path, output_dir: Path, tasks: Dict[int, str]) -> None:
    """Create tasks.jsonl from tasks dict."""
    output_meta = output_dir / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)

    # Copy original parquet
    tasks_parquet = input_dir / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        shutil.copy2(tasks_parquet, output_meta / "tasks.parquet")

    # Create tasks.jsonl
    with open(output_meta / "tasks.jsonl", "w") as f:
        for task_idx, task_desc in sorted(tasks.items()):
            f.write(json.dumps({"task_index": task_idx, "task": task_desc}) + "\n")


def convert_episodes(
    input_dir: Path,
    output_dir: Path,
    all_data: pd.DataFrame,
    info: Dict
) -> None:
    """Create episodes.jsonl from data."""
    output_meta = output_dir / "meta"

    # Group by episode
    episodes_info = []
    for ep_idx in sorted(all_data["episode_index"].unique()):
        ep_data = all_data[all_data["episode_index"] == ep_idx]
        ep_info = {
            "episode_index": int(ep_idx),
            "tasks": [{"task_index": int(ep_data["task_index"].iloc[0])}],
            "length": len(ep_data),
        }
        episodes_info.append(ep_info)

    # Write episodes.jsonl
    with open(output_meta / "episodes.jsonl", "w") as f:
        for ep_info in episodes_info:
            f.write(json.dumps(ep_info) + "\n")


def convert_episodes_stats(
    input_dir: Path,
    output_dir: Path,
    all_data: pd.DataFrame,
    point_cloud_stats: Optional[Dict] = None
) -> None:
    """Create episodes_stats.jsonl."""
    output_meta = output_dir / "meta"

    # Load global stats
    stats_file = input_dir / "meta" / "stats.json"
    if stats_file.exists():
        with open(stats_file, "r") as f:
            global_stats = json.load(f)
    else:
        global_stats = {}

    # Add point_cloud stats if provided
    if point_cloud_stats is not None:
        global_stats["point_cloud"] = point_cloud_stats

    # Write episodes_stats.jsonl (same stats for all episodes)
    with open(output_meta / "episodes_stats.jsonl", "w") as f:
        for ep_idx in sorted(all_data["episode_index"].unique()):
            ep_stats = {"episode_index": int(ep_idx), "stats": global_stats}
            f.write(json.dumps(ep_stats) + "\n")


def convert_parquet_data(
    input_dir: Path,
    output_dir: Path,
    all_data: pd.DataFrame,
    track_dir: Optional[Path],
    info: Dict,
    num_points: int = 1024,
    include_point_cloud: bool = True,
) -> None:
    """Convert parquet data to episode-based format."""
    print("  Converting parquet data...")

    chunks_size = info.get("chunks_size", 1000)

    for ep_idx in tqdm(sorted(all_data["episode_index"].unique()), desc="  Episodes"):
        ep_data = all_data[all_data["episode_index"] == ep_idx].copy()
        ep_data = ep_data.sort_values("frame_index").reset_index(drop=True)

        # Rename 'action' to 'actions' (v3.0 -> v2.1 naming convention)
        if "action" in ep_data.columns and "actions" not in ep_data.columns:
            ep_data = ep_data.rename(columns={"action": "actions"})

        # Add point_cloud column if needed
        if include_point_cloud and track_dir is not None:
            tracks = load_tracks_3d(track_dir, ep_idx, num_points)
            if tracks is not None:
                # Match tracks to frames
                point_cloud_list = []
                for frame_idx in range(len(ep_data)):
                    if frame_idx < len(tracks):
                        point_cloud_list.append(tracks[frame_idx].tolist())
                    else:
                        point_cloud_list.append(np.zeros((num_points, 3)).tolist())
                ep_data["point_cloud"] = point_cloud_list
            else:
                ep_data["point_cloud"] = [np.zeros((num_points, 3)).tolist()] * len(ep_data)

        # Determine output path
        ep_chunk = ep_idx // chunks_size
        output_chunk_dir = output_dir / "data" / f"chunk-{ep_chunk:03d}"
        output_chunk_dir.mkdir(parents=True, exist_ok=True)
        output_parquet = output_chunk_dir / f"episode_{ep_idx:06d}.parquet"

        ep_data.to_parquet(output_parquet, index=False)


def convert_videos(input_dir: Path, output_dir: Path, info: Dict) -> None:
    """Convert video files to episode-based naming."""
    print("  Converting video files...")

    chunks_size = info.get("chunks_size", 1000)
    total_episodes = info.get("total_episodes", 0)

    input_videos = input_dir / "videos"
    if not input_videos.exists():
        print("    No videos directory found, skipping...")
        return

    # Get video keys from info
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]

    for video_key in video_keys:
        print(f"    Processing {video_key}...")
        input_video_dir = input_videos / video_key

        for ep_idx in range(total_episodes):
            ep_chunk = ep_idx // chunks_size
            input_chunk = ep_idx // chunks_size  # assuming same chunking

            # v3.0 format: chunk-XXX/file-XXX.mp4
            input_video = input_video_dir / f"chunk-{input_chunk:03d}" / f"file-{ep_idx:03d}.mp4"

            if not input_video.exists():
                continue

            # v2.1 format: chunk-XXX/episode_XXXXXX.mp4
            output_video_dir = output_dir / "videos" / video_key / f"chunk-{ep_chunk:03d}"
            output_video_dir.mkdir(parents=True, exist_ok=True)
            output_video = output_video_dir / f"episode_{ep_idx:06d}.mp4"

            shutil.copy2(input_video, output_video)


def update_stats_json(
    input_dir: Path,
    output_dir: Path,
    point_cloud_stats: Optional[Dict] = None,
) -> None:
    """Update stats.json with point cloud statistics."""
    input_stats = input_dir / "meta" / "stats.json"
    output_stats = output_dir / "meta" / "stats.json"

    if input_stats.exists():
        with open(input_stats, "r") as f:
            stats = json.load(f)
    else:
        stats = {}

    # Rename 'action' to 'actions' (v3.0 -> v2.1 naming convention)
    if "action" in stats and "actions" not in stats:
        stats["actions"] = stats.pop("action")

    if point_cloud_stats is not None:
        stats["point_cloud"] = point_cloud_stats

    with open(output_stats, "w") as f:
        json.dump(stats, f, indent=2)


def compute_norm_stats(all_data: pd.DataFrame, point_cloud_stats: Optional[Dict] = None) -> Dict:
    """Compute normalization statistics for state, actions, point_cloud, and point_cloud_delta."""
    print("  Computing norm_stats for state, actions, point_cloud, and delta...")

    # Get state data (observation.state)
    state_data = np.stack(all_data["observation.state"].values)

    # Handle both v3.0 ('action') and v2.1 ('actions') column names
    action_col = "action" if "action" in all_data.columns else "actions"
    actions_data = np.stack(all_data[action_col].values)

    def compute_feature_stats(data: np.ndarray) -> Dict:
        return {
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "q01": np.percentile(data, 1, axis=0).tolist(),
            "q99": np.percentile(data, 99, axis=0).tolist(),
        }

    norm_stats = {
        "state": compute_feature_stats(state_data),
        "actions": compute_feature_stats(actions_data),
    }

    # Add point_cloud and point_cloud_delta stats if provided
    if point_cloud_stats is not None:
        # point_cloud stats (mean, std for absolute positions)
        norm_stats["point_cloud"] = {
            "mean": point_cloud_stats["mean"],
            "std": point_cloud_stats["std"],
            "q01": point_cloud_stats["q01"],
            "q99": point_cloud_stats["q99"],
            "delta_mean": point_cloud_stats["delta_mean"],
            "delta_std": point_cloud_stats["delta_std"],
        }
        # point_cloud_delta stats (for delta normalization)
        norm_stats["point_cloud_delta"] = {
            "mean": point_cloud_stats["delta_mean"],
            "std": point_cloud_stats["delta_std"],
            "q01": point_cloud_stats["delta_q01"],
            "q99": point_cloud_stats["delta_q99"],
        }

    return {"norm_stats": norm_stats}


def save_norm_stats(
    norm_stats: Dict,
    output_dir: Path,
) -> None:
    """Save norm_stats.json to dataset meta directory."""
    output_path = output_dir / "meta" / "norm_stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"  Saved norm_stats to: {output_path}")


def prepare_real_world_data(
    input_dir: Path,
    output_dir: Path,
    track_dir: Optional[Path] = None,
    mode: str = "track",
    num_points: int = 1024,
    assets_dir: Optional[Path] = None,
    config_name: str = "pi05_real_world_pt_v3_new_head",
    repo_id: str = "real_world_pt",
) -> None:
    """Main function to prepare real world data for training."""
    print(f"\n{'='*60}")
    print(f"Real World Data Preparation (v3.0 -> v2.1)")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode:   {mode}")
    if mode == "track":
        print(f"Track:  {track_dir}")
        print(f"Points: {num_points}")
    if assets_dir:
        print(f"Assets: {assets_dir}")

    include_point_cloud = mode == "track"

    if include_point_cloud and track_dir is None:
        raise ValueError("track_dir is required when mode='track'")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("\n[1/8] Loading metadata...")
    info, tasks = load_lerobot_metadata(input_dir)
    print(f"  Total episodes: {info['total_episodes']}")
    print(f"  Total frames: {info['total_frames']}")
    print(f"  FPS: {info['fps']}")
    print(f"  Original version: {info.get('codebase_version', 'unknown')}")

    # Load all data
    print("\n[2/8] Loading parquet data...")
    all_data = load_all_parquet_data(input_dir, info)
    print(f"  Loaded {len(all_data)} frames")

    # Compute point cloud statistics if needed
    point_cloud_stats = None
    if include_point_cloud:
        print("\n[3/8] Computing point cloud statistics...")
        point_cloud_stats = compute_point_cloud_stats(
            track_dir, info["total_episodes"], num_points
        )
        print(f"  Mean: {point_cloud_stats['mean']}")
        print(f"  Std: {point_cloud_stats['std']}")
    else:
        print("\n[3/8] Skipping point cloud statistics (base mode)...")

    # Convert info.json
    print("\n[4/8] Converting info.json (v3.0 -> v2.1)...")
    info = convert_info_json(input_dir, output_dir, include_point_cloud, num_points)

    # Convert tasks
    print("\n[5/8] Creating tasks.jsonl...")
    convert_tasks(input_dir, output_dir, tasks)

    # Convert episodes
    print("\n[6/8] Creating episodes.jsonl and episodes_stats.jsonl...")
    convert_episodes(input_dir, output_dir, all_data, info)
    convert_episodes_stats(input_dir, output_dir, all_data, point_cloud_stats)

    # Convert parquet data
    print("\n[7/8] Converting parquet data...")
    convert_parquet_data(
        input_dir, output_dir, all_data, track_dir, info, num_points, include_point_cloud
    )

    # Update stats.json
    update_stats_json(input_dir, output_dir, point_cloud_stats)

    # Convert videos
    print("\n[8/9] Converting videos...")
    convert_videos(input_dir, output_dir, info)

    # Generate norm_stats.json for training (saved in dataset directory)
    print("\n[9/9] Generating norm_stats.json...")
    norm_stats = compute_norm_stats(all_data, point_cloud_stats)
    save_norm_stats(norm_stats, output_dir)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - meta/info.json (v2.0 format)")
    print(f"  - meta/tasks.jsonl")
    print(f"  - meta/episodes.jsonl")
    print(f"  - meta/episodes_stats.jsonl")
    print(f"  - meta/stats.json")
    print(f"  - meta/norm_stats.json")  # Now in dataset directory!
    print(f"  - data/chunk-XXX/episode_XXXXXX.parquet")
    print(f"  - videos/*/chunk-XXX/episode_XXXXXX.mp4")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare real world data for OpenVLA OFT training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to original LeRobot v3.0 dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output converted dataset",
    )
    parser.add_argument(
        "--track_dir",
        type=str,
        default=None,
        help="Path to track results directory (required for mode='track')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["track", "base"],
        default="track",
        help="Conversion mode: 'track' (with point_cloud) or 'base' (without)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1024,
        help="Number of points to use (1024 for v3, 256 for v4)",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=None,
        help="Path to assets directory for norm_stats.json (default: project assets/)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_real_world_pt_v3_new_head",
        help="Config name for norm_stats directory structure",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="real_world_pt",
        help="Repo ID for norm_stats directory structure",
    )

    args = parser.parse_args()

    prepare_real_world_data(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        track_dir=Path(args.track_dir) if args.track_dir else None,
        mode=args.mode,
        num_points=args.num_points,
        assets_dir=Path(args.assets_dir) if args.assets_dir else None,
        config_name=args.config_name,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
