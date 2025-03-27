import argparse
import io
import json
from copy import deepcopy
from pathlib import Path

import evo
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib.pyplot as plt
import numpy as np
import torch
from decord import VideoReader
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D
from evo.tools.plot import PlotMode
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms as TF
from collections import defaultdict
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def eval_trajectory(poses_est, poses_gt, frame_ids, align=False):

    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_gt[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids)
    traj_est = PoseTrajectory3D(
        positions_xyz=poses_est[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_est[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids)

    ate_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align)
    ate = ate_result.stats["rmse"]

    are_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align)
    are = are_result.stats["rmse"]

    # RPE rotation and translation
    rpe_rots_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True)
    rpe_rot = rpe_rots_result.stats["rmse"]

    rpe_transs_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True)
    rpe_trans = rpe_transs_result.stats["rmse"]

    plot_mode = PlotMode.xz
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE: {round(ate, 3)}, ARE: {round(are, 3)}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est,
        ate_result.np_arrays["error_array"],
        plot_mode,
        min_map=ate_result.stats["min"],
        max_map=ate_result.stats["max"],
    )
    ax.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=90)
    buffer.seek(0)

    pillow_image = Image.open(buffer)
    pillow_image.load()
    buffer.close()
    plt.close(fig)

    return {
        "ate": ate,
        "are": are,
        "rpe_rot": rpe_rot,
        "rpe_trans": rpe_trans
    }, pillow_image


def load_poses(path, max_length, stride):
    c2ws = np.load(path)["poses"]
    c2ws = c2ws[:max_length]  # Match video length
    inf_ids = np.where(np.isinf(c2ws).any(axis=(1, 2)))[0]
    if inf_ids.size > 0:
        c2ws = c2ws[:inf_ids.min()]
    c2ws = np.linalg.inv(c2ws[0]) @ c2ws
    c2ws = c2ws[::stride]
    return c2ws


def generate_tail_overlapping_intervals(n, k, stride):
    intervals = []
    current = 0

    while True:
        interval = [current + i * stride for i in range(k)]
        if interval[-1] >= n:
            break
        intervals.append(interval)
        current = interval[-1]  # next interval starts from the last element

    return intervals


def get_vgg_input_imgs(images: np.ndarray):
    to_tensor = TF.ToTensor()
    vgg_input_images = []
    for image in images:
        img = Image.fromarray(image, mode="RGB")
        width, height = img.size
        new_width = 518
        new_height = round(height * (new_width / width) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        if new_height > 518:
            start_y = (new_height - 518) // 2
            img = img[:, start_y: start_y + 518, :]

        vgg_input_images.append(img)
    vgg_input_images = torch.stack(vgg_input_images)
    return vgg_input_images


def to_homogeneous(extrinsics):
    n = extrinsics.shape[0]
    homogeneous_extrinsics = np.eye(4)[None, :, :].repeat(n, axis=0)  # Create identity matrices
    homogeneous_extrinsics[:, :3, :4] = extrinsics  # Copy [R | t]
    return homogeneous_extrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default="datasets/ScanNetVideos")
    parser.add_argument('--split_file', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--plot', action="store_true")
    args = parser.parse_args()

    print("\nRunning with config...")
    torch.manual_seed(1234)

    # Load scene names from the split file
    with open(args.split_file, 'r') as f:
        scannet_scenes = sorted([line.strip() for line in f.readlines()])

    results = {}
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    for scene in scannet_scenes[:2]:
        print("Processing", scene)
        scene_dir = args.data_dir / f"{scene}"
        output_scene_dir = args.output_path / scene
        output_scene_dir.mkdir(parents=True, exist_ok=True)  # Create scene-specific folder

        if (output_scene_dir / "stats.json").exists():
            continue

        pose_path = scene_dir / "poses.npz"
        video_path = scene_dir / "video.mp4"
        pose_path = scene_dir / "poses.npz"
        intrinsics_path = scene_dir / "intrinsics.npz"

        c2ws = load_poses(pose_path, -1, 1)
        video_reader = VideoReader(str(video_path))

        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        model = model.cuda().eval()

        all_cam_to_world_mat = []
        stride = 3
        intervals = generate_tail_overlapping_intervals(c2ws.shape[0], 24, stride)  # take batches of 10 images to fit into GPU
        for frame_ids in intervals:
            images = video_reader.get_batch(frame_ids).asnumpy()  # (N, H, W, 3) [0, 255]
            vgg_input_images = get_vgg_input_imgs(images).cuda()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(vgg_input_images)
                    extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                    extrinsic = extrinsic.cpu().squeeze(0).numpy()  # remove batch dimension and convert to numpy
                    extrinsic = to_homogeneous(extrinsic)
                    if not all_cam_to_world_mat:
                        all_cam_to_world_mat.extend(extrinsic)
                    else:
                        last_c2w = all_cam_to_world_mat[-1]
                        for pr_pose in extrinsic[1:]:  # the intervals intersect at the first frame
                            all_cam_to_world_mat.append(last_c2w @ pr_pose)

        traj_est_poses = np.array(all_cam_to_world_mat)
        n = traj_est_poses.shape[0]

        frame_ids = sorted(set([idx for interval in intervals for idx in interval]))
        w2cs = np.linalg.inv(c2ws[frame_ids])[:n]

        timestamps = list(range(n))
        stats, traj_plot = eval_trajectory(traj_est_poses, w2cs, timestamps, align=False)
        stats_aligned, _ = eval_trajectory(traj_est_poses, w2cs, timestamps, align=True)

        all_metrics = deepcopy(stats)
        for metric_name, metric_value in stats_aligned.items():
            all_metrics[f"aligned_{metric_name}"] = metric_value

        with open(output_scene_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=4)

        if args.plot:
            traj_plot.save(output_scene_dir / "plot.png")

    average_results = defaultdict(list)
    for scene_path in sorted(args.output_path.glob("*")):
        if scene_path.is_file():
            continue
        with open(scene_path / "metrics.json", "r") as file:
            scene_stats = json.load(file)
        for metric_name, metric_value in scene_stats.items():
            average_results[metric_name].append(metric_value)

    for stat_name, stat_vals in average_results.items():
        average_results[stat_name] = np.mean(stat_vals)

    with open(args.output_path / "average_metrics.json", "w") as f:
        json.dump(average_results, f, indent=4)
