from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import quaternion  # pip install numpy-quaternion
import random

# -------------------------- CONFIG --------------------------
ROOT_PATH = Path('/home/saksham/samsad/mtech-project/datasets/SeeingThroghFog')
LIDAR_FOLDER = ROOT_PATH / 'lidar_hdl64_strongest/lidar_hdl64_strongest'
CALIB_FILE = ROOT_PATH / 'pcdet/calibs/calib_tf_tree_full.json'
IMAGE_SHAPE = (1080, 1920)  # (height, width)

MIN_FOV_POINTS = 3000
TRAIN_RATIO = 0.9  # 90% train, 10% val
RANDOM_SEED = 42

OUTPUT_DIR = ROOT_PATH / 'ImageSets'
OUTPUT_DIR.mkdir(exist_ok=True)
# -----------------------------------------------------------

def load_calibration():
    """Load extrinsics from JSON and compute LiDAR -> Camera transform"""
    with open(CALIB_FILE, 'r') as f:
        transforms = json.load(f)

    def find_transform(child_id):
        for t in transforms:
            if t['child_frame_id'] == child_id:
                trans = t['transform']['translation']
                rot = t['transform']['rotation']
                q = np.quaternion(rot['w'], rot['x'], rot['y'], rot['z'])
                R = quaternion.as_rotation_matrix(q)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [trans['x'], trans['y'], trans['z']]
                return T
        return None

    # Adjust these names if your JSON uses slightly different ones
    lidar_to_body = find_transform('lidar_hdl64_s3_roof')
    body_to_cam = find_transform('cam_stereo_left_optical')

    if lidar_to_body is None or body_to_cam is None:
        raise RuntimeError("Could not find 'lidar_hdl64_s3_roof' or 'cam_stereo_left_optical' in calibration JSON")

    # LiDAR → Camera transform
    lidar_to_body_inv = np.linalg.inv(lidar_to_body)
    V2C = body_to_cam @ lidar_to_body_inv

    class Calib:
        def __init__(self):
            self.V2C = V2C[:3, :]  # 3x4

        def lidar_to_rect(self, pts_lidar):
            pts_hom = np.hstack((pts_lidar, np.ones((pts_lidar.shape[0], 1))))
            return pts_hom @ self.V2C.T

        def rect_to_img(self, pts_rect):
            pts_img = pts_rect[:, :2] / pts_rect[:, 2:3]  # x/z, y/z
            pts_depth = pts_rect[:, 2]
            return pts_img, pts_depth

    return Calib()


def get_fov_flag(pts_rect, img_shape):
    pts_img, pts_depth = calib.rect_to_img(pts_rect)
    h, w = img_shape
    in_img_x = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < w)
    in_img_y = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < h)
    in_front = pts_depth > 0
    return np.logical_and(np.logical_and(in_img_x, in_img_y), in_front)


# Load calibration once
calib = load_calibration()

# Step 1: Get all frame IDs from LiDAR files
frame_ids = sorted([f.stem for f in LIDAR_FOLDER.glob('*.bin')])
total = len(frame_ids)
print(f"Found {total} total frames")

if total == 0:
    raise RuntimeError("No .bin files found in lidar_hdl64_strongest/")

# Step 2: Create initial train/val split
random.seed(RANDOM_SEED)
random.shuffle(frame_ids)
train_count = int(total * TRAIN_RATIO)
train_ids = frame_ids[:train_count]
val_ids = frame_ids[train_count:]

print(f"Initial split: {len(train_ids)} train, {len(val_ids)} val")

# Save initial splits
with open(OUTPUT_DIR / 'train.txt', 'w') as f:
    for fid in train_ids:
        f.write(fid + '\n')
with open(OUTPUT_DIR / 'val.txt', 'w') as f:
    for fid in val_ids:
        f.write(fid + '\n')

print("Saved initial train.txt and val.txt")

# Step 3: Filter each split by FOV points
for split_name, ids in [('train', train_ids), ('val', val_ids)]:
    print(f"\nFiltering {split_name.upper()} split ({len(ids)} frames)...")
    good_ids = []
    removed = 0

    for frame_id in tqdm(ids):
        bin_file = LIDAR_FOLDER / f'{frame_id}.bin'
        if not bin_file.exists():
            removed += 1
            continue

        pc = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
        pts_rect = calib.lidar_to_rect(pc[:, :3])
        fov_mask = get_fov_flag(pts_rect, IMAGE_SHAPE)
        fov_count = np.sum(fov_mask)

        if fov_count >= MIN_FOV_POINTS:
            good_ids.append(frame_id)
        else:
            removed += 1

    print(f"→ Kept {len(good_ids)} / {len(ids)} frames (>= {MIN_FOV_POINTS} FOV points)")
    print(f"→ Removed {removed} frames")

    # Save filtered split
    out_file = OUTPUT_DIR / f'{split_name}.txt'
    with open(out_file, 'w') as f:
        for fid in good_ids:
            f.write(fid + '\n')

    print(f"Saved: {out_file}")
