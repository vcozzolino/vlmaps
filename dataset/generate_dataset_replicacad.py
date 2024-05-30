import sys

sys.path.insert(0, "/home/v00609018/vlmaps")

import os
from pathlib import Path
from typing import Union

import habitat_sim
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from vlmaps.utils.habitat_utils import make_cfg, save_obs_no_sem


def generate_replicacad_scene_data(
    save_dir: Union[Path, str], config: DictConfig, scene_path: Path, scene_config_file: Path, poses: np.ndarray
) -> None:
    """
    config: config for the sensors of the collected data
    scene_path: path to the Matterport3D scene file *.glb
    poses: (N, 7), each line has (px, py, pz, qx, qy, qz, qw)
    """
    sim_setting = {
        "scene": scene_path,
        "default_agent": 0,
        "scene_dataset_config_file": scene_config_file,
        "sensor_height": config.camera_height,
        "color_sensor": config.rgb,
        "depth_sensor": config.depth,
        "semantic_sensor": config.semantic,
        "use_default_lighting": True,
        "move_forward": 0.1,
        "turn_left": 5,
        "turn_right": 5,
        "width": config.resolution.w,
        "height": config.resolution.h,
        "enable_physics": False,
        "seed": 42,
    }
    cfg = make_cfg(sim_setting)
    sim = habitat_sim.Simulator(cfg)

    # get the dict mapping object id to semantic id in this scene
    # obj2cls = get_obj2cls_dict(sim)

    # initialize the agent in sim
    _ = sim.initialize_agent(sim_setting["default_agent"])
    pbar = tqdm(poses, leave=False)
    for pose_i, pose in enumerate(pbar):
        pbar.set_description(desc=f"Frame {pose_i:06}")
        agent_state = habitat_sim.AgentState()
        agent_state.position = pose[:3]
        agent_state.rotation = pose[3:]
        sim.get_agent(0).set_state(agent_state)
        obs = sim.get_sensor_observations(0)
        save_obs_no_sem(save_dir, sim_setting, obs, pose_i)

    sim.close()


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="generate_dataset.yaml",
)
def main(config: DictConfig) -> None:
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    # if config.scene_names:
    #     data_dirs = sorted([dataset_dir / x for x in config.scene_names])
    # pbar = tqdm(data_dirs)
    # for data_dir_i, data_dir in enumerate(pbar):
    #     pbar.set_description(desc=f"Scene {data_dir.name:14}")
    #     scene_name = data_dir.name.split("_")[0]
    #     scene_path = Path(config.data_paths.habitat_scene_dir) / scene_name / (scene_name + ".glb")

    data_dir = "/home/v00609018/habitat-sim/muc/"
    pose_path = data_dir + "poses.txt"
    poses = np.loadtxt(pose_path, delimiter=",")  # (N, 7), each line has (px, py, pz, qx, qy, qz, qw)
    scene_config = "/home/v00609018/habitat-sim/data/replica_cad/replicaCAD.scene_dataset_config.json"
    scene_path = "apt_1"
    save_dir = "/home/v00609018/vlmaps/data/vlmaps_dataset/replicacad_apt_1/"
    generate_replicacad_scene_data(save_dir, config.data_cfg, scene_path, scene_config, poses)


if __name__ == "__main__":
    main()
