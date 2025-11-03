import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from sim_data_mj import SimEnv, angle_list, save_rgb_depth

# Test with FR3
env = SimEnv(
    model_path="../menagerie/franka_fr3/scene.xml",
    dof=7,
    radius=1.5,
    num_cameras=4,
    width=640,
    height=480,
)

print(f"Joints: {env.joint_names}")
print(f"Controlling: {env.dof_names}")

# Generate 3 random poses
angles = angle_list(num_step=3, dof=7, joint_limits=env.joint_limits, seed_i=0)

# Render
for step_id in range(3):
    env.set_joint_positions(angles[step_id])
    for cam_id in range(4):
        rgb, depth, cam = env.render_camera(cam_id)
        save_rgb_depth("fr3_test", step_id, cam_id, rgb, depth)

print(f"\nDone! Check fr3_test/ directory")

# Show first image
rgb = iio.imread("fr3_test/0000_cam00_rgb.png")
depth = iio.imread("fr3_test/0000_cam00_depth.png")

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(rgb)
plt.title("RGB")
plt.subplot(122)
plt.imshow(depth, cmap='turbo')
plt.title("Depth")
plt.tight_layout()
plt.savefig("fr3_test/preview.png")
print("Saved preview.png")