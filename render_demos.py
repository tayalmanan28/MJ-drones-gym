"""Generate demo GIFs for MJ-drones-gym features."""
import os
import numpy as np
from PIL import Image

os.makedirs("/root/multi_drone_mujoco/demo_gifs", exist_ok=True)

# Use BaseAviary with RPM action type (PID outputs raw RPMs)
from multi_drone_mujoco.envs.base_aviary import BaseAviary, _generate_aviary_xml
from multi_drone_mujoco.control.pid_control import PIDControl
from multi_drone_mujoco.utils.enums import Physics, ActionType, ObservationType, DroneModel


def save_gif(frames, path, fps=20):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000/fps), loop=0)
    print(f"  Saved: {path} ({len(frames)} frames)")


def make_env(num_drones=1, initial_xyzs=None):
    """Create a BaseAviary with RPM action type for PID control."""
    if initial_xyzs is None:
        initial_xyzs = np.array([[0.0, 0.0, 0.5]])
    return BaseAviary(
        num_drones=num_drones, ctrl_freq=48, sim_freq=240,
        physics=Physics.MJC, act_type=ActionType.RPM,
        obs_type=ObservationType.KIN, initial_xyzs=initial_xyzs,
        render_mode="rgb_array",
    )


def stabilize(env, ctrl, target, drone_id=0, steps=300):
    for _ in range(steps):
        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[drone_id], env.quat[drone_id],
            env.vel[drone_id], env.ang_v[drone_id], target)
        env.step(rpm.flatten())


def stabilize_multi(env, ctrls, targets, steps=300):
    for _ in range(steps):
        all_rpm = []
        for d in range(len(ctrls)):
            rpm, _, _ = ctrls[d].computeControl(
                env.CTRL_TIMESTEP, env.pos[d], env.quat[d],
                env.vel[d], env.ang_v[d], targets[d])
            all_rpm.append(rpm.flatten())
        env.step(np.concatenate(all_rpm))


# ============================================================
def demo_hover_track():
    print("Rendering: Hover (tracking camera)...")
    env = make_env(initial_xyzs=np.array([[0.0, 0.0, 0.5]]))
    ctrl = PIDControl(env)
    env.reset()
    target = np.array([0.0, 0.0, 1.0])
    stabilize(env, ctrl, target, steps=500)

    frames = []
    for i in range(200):
        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0], env.vel[0], env.ang_v[0], target)
        env.step(rpm.flatten())
        if i % 3 == 0:
            frames.append(env.render(camera_mode="track"))
    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/hover_track.gif")


def demo_hover_fixed():
    print("Rendering: Hover (fixed camera)...")
    env = make_env(initial_xyzs=np.array([[0.0, 0.0, 0.5]]))
    ctrl = PIDControl(env)
    env.reset()
    target = np.array([0.0, 0.0, 1.0])
    stabilize(env, ctrl, target, steps=500)

    frames = []
    for i in range(200):
        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0], env.vel[0], env.ang_v[0], target)
        env.step(rpm.flatten())
        if i % 3 == 0:
            frames.append(env.render(camera_mode="fixed"))
    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/hover_fixed.gif")


def demo_hover_fpv():
    print("Rendering: Hover (FPV camera)...")
    env = make_env(initial_xyzs=np.array([[0.0, 0.0, 0.5]]))
    ctrl = PIDControl(env)
    env.reset()
    target = np.array([0.0, 0.0, 1.0])
    stabilize(env, ctrl, target, steps=500)

    frames = []
    for i in range(200):
        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0], env.vel[0], env.ang_v[0], target)
        env.step(rpm.flatten())
        if i % 3 == 0:
            frames.append(env.render(camera_mode="fpv"))
    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/hover_fpv.gif")


def demo_multi_drone():
    print("Rendering: Multi-drone...")
    initial_xyzs = np.array([[-0.4, -0.4, 0.5], [0.4, -0.4, 0.5], [0.0, 0.4, 0.5]])
    env = BaseAviary(num_drones=3, ctrl_freq=48, sim_freq=240, physics=Physics.MJC,
                     act_type=ActionType.RPM, initial_xyzs=initial_xyzs, render_mode="rgb_array")
    ctrls = [PIDControl(env) for _ in range(3)]
    env.reset()
    targets = np.array([[-0.4, -0.4, 0.8], [0.4, -0.4, 1.0], [0.0, 0.4, 1.2]])
    stabilize_multi(env, ctrls, targets, steps=500)

    frames = []
    for i in range(200):
        all_rpm = []
        for d in range(3):
            rpm, _, _ = ctrls[d].computeControl(
                env.CTRL_TIMESTEP, env.pos[d], env.quat[d], env.vel[d], env.ang_v[d], targets[d])
            all_rpm.append(rpm.flatten())
        env.step(np.concatenate(all_rpm))
        if i % 3 == 0:
            frames.append(env.render(camera_mode="track"))
    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/multi_drone.gif")


def demo_formation():
    print("Rendering: Formation...")
    from multi_drone_mujoco.envs.formation_aviary import FormationAviary
    env = FormationAviary(num_drones=4, ctrl_freq=48, render_mode="rgb_array")
    # FormationAviary uses normalized actions — need to handle differently
    # Use BaseAviary instead for PID
    initial_xyzs = np.array([[-0.3, -0.3, 0.5], [0.3, -0.3, 0.5], [0.3, 0.3, 0.5], [-0.3, 0.3, 0.5]])
    env = BaseAviary(num_drones=4, ctrl_freq=48, sim_freq=240, physics=Physics.MJC,
                     act_type=ActionType.RPM, initial_xyzs=initial_xyzs, render_mode="rgb_array")
    ctrls = [PIDControl(env) for _ in range(4)]
    env.reset()

    # Formation offsets (square)
    offsets = np.array([[-0.3, -0.3, 0], [0.3, -0.3, 0], [0.3, 0.3, 0], [-0.3, 0.3, 0]])
    init_targets = [np.array([0, 0, 1.0]) + offsets[d] for d in range(4)]
    stabilize_multi(env, ctrls, init_targets, steps=500)

    frames = []
    for i in range(360):
        t = i / 48.0
        cx = 0.3 * np.cos(0.5 * t)
        cy = 0.3 * np.sin(0.5 * t)
        all_rpm = []
        for d in range(4):
            target = np.array([cx, cy, 1.0]) + offsets[d]
            rpm, _, _ = ctrls[d].computeControl(
                env.CTRL_TIMESTEP, env.pos[d], env.quat[d], env.vel[d], env.ang_v[d], target)
            all_rpm.append(rpm.flatten())
        env.step(np.concatenate(all_rpm))
        if i % 4 == 0:
            frames.append(env.render(camera_mode="track"))
    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/formation.gif")


def demo_wind():
    print("Rendering: Wind...")
    from multi_drone_mujoco.wrappers.wind import WindConfig, WindModel
    env = make_env(initial_xyzs=np.array([[0.0, 0.0, 0.5]]))
    ctrl = PIDControl(env)
    env.reset()
    target = np.array([0.0, 0.0, 1.0])
    stabilize(env, ctrl, target, steps=500)

    # Enable wind after stabilization
    wind_cfg = WindConfig(model=WindModel.COMBINED, constant_wind=np.array([0.015, 0.008, 0.0]),
                          gust_intensity=0.03, turbulence_intensity=3.0)
    env.set_wind(wind_cfg)

    frames = []
    for i in range(300):
        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0], env.vel[0], env.ang_v[0], target)
        env.step(rpm.flatten())
        if i % 3 == 0:
            frames.append(env.render(camera_mode="track"))
    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/wind.gif")


def demo_obstacles():
    print("Rendering: Obstacles (cluttered)...")
    import mujoco

    init_xyzs = np.array([[0.0, 0.0, 0.5]])
    init_rpys = np.array([[0.0, 0.0, 0.0]])
    xml = _generate_aviary_xml(1, DroneModel.CF2X, init_xyzs, init_rpys, False, False, 1/240)

    # Inject 25 colorful obstacles
    np.random.seed(42)
    obstacle_xml = ""
    colors = ["0.9 0.2 0.2 0.9", "0.2 0.8 0.2 0.9", "0.2 0.3 0.9 0.9",
              "0.9 0.6 0.1 0.9", "0.7 0.2 0.9 0.9", "0.1 0.8 0.8 0.9"]
    for i in range(25):
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(-1.5, 1.5)
        if abs(x) < 0.3 and abs(y) < 0.3:
            x += 0.5 * np.sign(x + 0.01)
        h = np.random.uniform(0.4, 1.5)
        r = np.random.uniform(0.03, 0.07)
        color = colors[i % len(colors)]
        obstacle_xml += f'    <body name="obs_{i}" pos="{x:.2f} {y:.2f} {h/2:.2f}">\n'
        obstacle_xml += f'      <geom type="cylinder" size="{r:.3f} {h/2:.2f}" rgba="{color}"/>\n'
        obstacle_xml += f'    </body>\n'
    xml = xml.replace("</worldbody>", obstacle_xml + "  </worldbody>")

    # Create env then swap model
    env = make_env(initial_xyzs=init_xyzs)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    env.model = model
    env.data = data
    env._renderer = None  # Reset renderer for new model
    mujoco.mj_forward(env.model, env.data)
    env._updateAndStoreKinematicInformation()

    ctrl = PIDControl(env)
    target = np.array([0.0, 0.0, 0.8])
    stabilize(env, ctrl, target, steps=500)

    waypoints = [np.array([0.0, 0.0, 0.8]), np.array([0.7, 0.5, 0.9]),
                 np.array([-0.5, 0.8, 0.7]), np.array([-0.8, -0.5, 1.0])]
    frames = []
    wp_idx = 0
    for i in range(480):
        target = waypoints[wp_idx]
        if np.linalg.norm(env.pos[0] - target) < 0.15 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0], env.vel[0], env.ang_v[0], target)
        env.step(rpm.flatten())
        if i % 3 == 0:
            frames.append(env.render(camera_mode="track"))
    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/obstacles.gif")


def demo_domain_randomization():
    print("Rendering: Domain Randomization...")
    from multi_drone_mujoco.wrappers import DomainRandomizationWrapper, DomainRandomizationConfig
    base_env = make_env(initial_xyzs=np.array([[0.0, 0.0, 0.5]]))
    dr_cfg = DomainRandomizationConfig(
        mass_range=(0.6, 1.4), inertia_range=(0.6, 1.4),
        kf_range=(0.7, 1.3), km_range=(0.7, 1.3),
        action_delay_steps=2, motor_time_constant=0.015)
    env = DomainRandomizationWrapper(base_env, dr_cfg)
    ctrl = PIDControl(base_env)
    target = np.array([0.0, 0.0, 1.0])

    all_frames = []
    for ep in range(3):
        env.reset(seed=ep * 7)
        for _ in range(400):
            rpm, _, _ = ctrl.computeControl(
                base_env.CTRL_TIMESTEP, base_env.pos[0], base_env.quat[0],
                base_env.vel[0], base_env.ang_v[0], target)
            env.step(rpm.flatten())
        for i in range(120):
            rpm, _, _ = ctrl.computeControl(
                base_env.CTRL_TIMESTEP, base_env.pos[0], base_env.quat[0],
                base_env.vel[0], base_env.ang_v[0], target)
            env.step(rpm.flatten())
            if i % 3 == 0:
                all_frames.append(base_env.render(camera_mode="track"))
    env.close()
    save_gif(all_frames, "/root/multi_drone_mujoco/demo_gifs/domain_randomization.gif")


def demo_vertical_circle():
    print("Rendering: Vertical circular track...")
    env = make_env(initial_xyzs=np.array([[0.0, 0.0, 0.5]]))
    ctrl = PIDControl(env)
    env.reset()

    # Stabilize at starting point of circle (radius=0.5, center at y=0, z=1.0)
    radius = 0.5
    center = np.array([0.0, 0.0, 1.0])
    start_target = center + np.array([0.0, radius, 0.0])
    stabilize(env, ctrl, start_target, steps=600)

    frames = []
    total_steps = 600  # ~2 full loops
    for i in range(total_steps):
        theta = 2 * np.pi * (i / 300)  # one loop per 300 steps
        # Circle in Y-Z plane
        target = center + np.array([0.0, radius * np.cos(theta), radius * np.sin(theta)])
        rpm, _, _ = ctrl.computeControl(
            env.CTRL_TIMESTEP, env.pos[0], env.quat[0], env.vel[0], env.ang_v[0], target)
        env.step(rpm.flatten())
        if i % 3 == 0:
            frames.append(env.render(camera_mode="track"))
    env.close()
    save_gif(frames, "/root/multi_drone_mujoco/demo_gifs/vertical_circle.gif")


if __name__ == "__main__":
    print("=" * 50)
    print("Generating MJ-drones-gym demo GIFs")
    print("=" * 50)
    demos = [demo_hover_track, demo_hover_fixed, demo_hover_fpv,
             demo_multi_drone, demo_formation, demo_wind,
             demo_obstacles, demo_domain_randomization, demo_vertical_circle]
    for demo in demos:
        try:
            demo()
        except Exception as e:
            import traceback
            print(f"  SKIP {demo.__name__}: {e}")
            traceback.print_exc()
    print("\nDone!")
