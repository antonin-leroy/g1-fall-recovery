"""Turn the poses saved by ``scripts/rsl_rl/record_poses.py`` into an mp4, with MuJoCo.

This is a kinematic replay: MuJoCo simulates nothing, it just places the robot in each
recorded pose and takes a picture. The motion is exactly what Isaac Lab computed. It runs on a
Mac (no NVIDIA GPU, no Isaac Sim).

Setup (once, on the Mac):

    pip install mujoco "imageio[ffmpeg]" numpy
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie

Example:

    python scripts/replay_mujoco.py poses.npz \
        --mjcf ~/mujoco_menagerie/unitree_g1/scene.xml --env 0 --out media/getup_env0.mp4

Note: Isaac Lab's G1 is the older 37-joint model (one ``torso_joint``, elbow pitch + elbow
roll, 7 finger joints per hand). Menagerie's G1 is the current 29-joint one (3 waist joints,
elbow + 3 wrist joints, no fingers). ALIASES below maps the old names onto the new ones;
fingers are skipped. The two bodies differ slightly, which is fine for a video.
"""

import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

# Isaac Lab (old G1) joint name -> menagerie (G1 29-dof) joint name, when they differ
ALIASES = {
    "torso_joint": "waist_yaw_joint",
    "left_elbow_pitch_joint": "left_elbow_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "left_elbow_roll_joint": "left_wrist_roll_joint",
    "right_elbow_roll_joint": "right_wrist_roll_joint",
}


def build_joint_map(model: mujoco.MjModel, isaac_names: list[str]) -> list[tuple[int, int]]:
    """Return (column in the recorded joint_pos, qpos address in MuJoCo) for each joint both models share."""
    mapping, skipped = [], []
    for col, name in enumerate(isaac_names):
        mj_name = ALIASES.get(name, name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
        if joint_id < 0:
            skipped.append(name)
            continue
        mapping.append((col, int(model.jnt_qposadr[joint_id])))
    print(f"[INFO] mapped {len(mapping)}/{len(isaac_names)} joints")
    if skipped:
        print(f"[INFO] not in the MuJoCo model, left at 0: {skipped}")
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("poses", help=".npz written by record_poses.py")
    parser.add_argument("--mjcf", required=True, help="MuJoCo scene, e.g. mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--env", type=int, default=0, help="Which recorded environment to replay.")
    parser.add_argument("--out", default=None, help="Output mp4 (default: next to the .npz).")
    parser.add_argument("--full", action="store_true", help="Keep going after the first episode ends.")
    parser.add_argument("--start", type=float, default=0.0, help="Start the video at this time (s).")
    parser.add_argument("--end", type=float, default=None, help="Stop the video at this time (s).")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--distance", type=float, default=3.0, help="Camera distance to the robot (m).")
    parser.add_argument("--azimuth", type=float, default=135.0, help="Camera angle around the robot (deg).")
    parser.add_argument("--elevation", type=float, default=-15.0, help="Camera angle above the floor (deg).")
    args = parser.parse_args()

    data_npz = np.load(args.poses)
    root_pos = data_npz["root_pos"][:, args.env]  # (T, 3)
    root_quat = data_npz["root_quat"][:, args.env]  # (T, 4) w, x, y, z
    joint_pos = data_npz["joint_pos"][:, args.env]  # (T, J)
    dones = data_npz["dones"][:, args.env]  # (T,)
    joint_names = [str(n) for n in data_npz["joint_names"]]
    dt = float(data_npz["dt"])

    # stop at the end of the first episode: the frame flagged done is already the next episode's start
    num_frames = len(root_pos)
    if not args.full and dones.any():
        num_frames = int(np.argmax(dones))
    if args.end is not None:
        num_frames = min(num_frames, int(round(args.end / dt)))
    first_frame = int(round(args.start / dt))
    print(f"[INFO] replaying frames {first_frame}-{num_frames} ({(num_frames - first_frame) * dt:.1f} s) of env {args.env}")

    model = mujoco.MjModel.from_xml_path(os.path.expanduser(args.mjcf))
    # the offscreen buffer defaults to 640x480; it must be at least as big as the video
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)

    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    free_adr = int(model.jnt_qposadr[free_id])
    joint_map = build_joint_map(model, joint_names)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    out_path = args.out or os.path.splitext(args.poses)[0] + f"_env{args.env}.mp4"
    fps = round(1.0 / dt)
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer, imageio.get_writer(
        out_path, fps=fps, macro_block_size=1
    ) as writer:
        for t in range(first_frame, num_frames):
            data.qpos[:] = model.qpos0
            data.qpos[free_adr : free_adr + 3] = root_pos[t]
            data.qpos[free_adr + 3 : free_adr + 7] = root_quat[t]
            for col, adr in joint_map:
                data.qpos[adr] = joint_pos[t, col]
            # forward kinematics only: computes where every body is, no dynamics
            mujoco.mj_forward(model, data)

            # follow the robot along the floor, keep the camera height fixed so it does not bob
            camera.lookat[:] = [root_pos[t, 0], root_pos[t, 1], 0.5]
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())

    print(f"[INFO] wrote {out_path} at {fps} fps")


if __name__ == "__main__":
    main()
