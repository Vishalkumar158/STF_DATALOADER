import logging
import numpy as np
from tqdm import tqdm
import copy
import pickle
from pathlib import Path
import json
import sys

from skimage import io
from multiprocessing import cpu_count
import quaternion  

from dataset import DatasetTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti


class STF(DatasetTemplate):

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, **kwargs):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.logger = logger
        # self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        if self.mode == 'test' and 'test' in self.dataset_cfg.DATA_SPLIT:
            self.split = self.dataset_cfg.DATA_SPLIT['test']
        else:
            self.split = self.dataset_cfg.DATA_SPLIT['val' if not self.training else 'train']

        split_dir = self.root_path / 'ImageSets' / f'{self.split}.txt'
        self.sample_id_list = [line.strip() for line in open(split_dir).readlines()] if split_dir.exists() else None

        self.stf_infos = []
        self.include_stf_data(self.mode)

        self.lidar_folder = 'lidar_hdl64_strongest/lidar_hdl64_strongest'      
        self.image_folder = 'cam_stereo_left'
        self.label_folder = 'labels/cam_left_labels_TMP'    
        self.calib_file = self.root_path / 'pcdet/calibs/calib_tf_tree_full.json' 

    def include_stf_data(self, mode):
        self.logger.info('Loading Seeing Through Fog (STF) dataset')
        self.stf_infos = []  # In this version, we generate on-the-fly, so no pre-loaded infos

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg,
            class_names=self.class_names,
            root_path=self.root_path,
            training=self.training,
            logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / f'{self.split}.txt'
        self.sample_id_list = [line.strip() for line in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_path / self.lidar_folder / f'{idx}.bin'
        assert lidar_file.exists(), f'{lidar_file} not found'
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)

    def get_image_shape(self, idx):
        img_file = self.root_path / self.image_folder / f'{idx}.png'  # Adjust extension if needed (.jpg)
        assert img_file.exists(), f'{img_file} not found'
        return np.array(io.imread(str(img_file)).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_path / self.label_folder / f'{idx}.txt'
        assert label_file.exists(), f'{label_file} not found'
        return self.parse_stf_label(label_file)

    @staticmethod
    def parse_stf_label(label_path):
        objects = []
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            cls_type = parts[0]
            if '_is_group' in cls_type:
                cls_type = cls_type.replace('_is_group', '')
            if cls_type == 'PassengerCar':
                cls_type = 'Car'
            elif cls_type not in ['Car', 'Pedestrian', 'LargeVehicle']:
                continue  # Skip unsupported classes

            truncation = float(parts[1])
            occlusion = int(parts[2])
            alpha = -10.0  # Not provided
            bbox = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float32)
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            level = int(parts[3]) if parts[3].isdigit() else 0
            score = 1.0

            # Build fake KITTI line string
            fake_line = f"{cls_type} {truncation:.2f} {occlusion} {alpha:.2f} " \
                        f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} " \
                        f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f} {score:.2f}"
            obj = object3d_kitti.Object3d(fake_line)
            obj.box2d = bbox
            obj.truncation = truncation
            obj.occlusion = occlusion
            obj.alpha = alpha
            obj.level = level
            obj.score = score

            objects.append(obj)
        return objects

    def get_calib(self):
        # Hardcoded intrinsics for Seeing Through Fog (cam_stereo_left)
        # Source: Official STF dataset uses fx ≈ 2076, fy ≈ 2077, cx ≈ 990, cy ≈ 540 for 1920x1080 images
        img_width, img_height = 1920, 1080
        fx, fy = 2076.0, 2077.0
        cx, cy = img_width / 2, img_height / 2
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)
        P2 = np.zeros((3, 4), dtype=np.float32)
        P2[:, :3] = K

        # Extrinsics: from calibration.json (lidar_hdl64_s3_roof -> body -> cam_stereo_left_optical)
        with open(self.calib_file, 'r') as f:
            transforms = json.load(f)

        def find_transform(child_id):
            for t in transforms:
                if t['child_frame_id'] == child_id:
                    trans = t['transform']['translation']
                    rot = t['transform']['rotation']
                    # q = np.quaternion(rot['w'], rot['x'], rot['y'], rot['z'])
                    q = quaternion.quaternion(rot['w'], rot['x'], rot['y'], rot['z'])
                    R = quaternion.as_rotation_matrix(q)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = [trans['x'], trans['y'], trans['z']]
                    return T
            return None

        lidar_to_body = find_transform('lidar_hdl64_s3_roof')
        body_to_cam = find_transform('cam_stereo_left_optical')

        if lidar_to_body is None or body_to_cam is None:
            raise RuntimeError("Required transforms not found in calibration.json")

        # Tr_velo_to_cam = body_to_cam @ inv(lidar_to_body)
        lidar_to_body_inv = np.linalg.inv(lidar_to_body)
        V2C = body_to_cam @ lidar_to_body_inv

        # calib = calibration_kitti.Calibration(None)
        # Create a fake calibration dict that mimics what get_calib_from_file() would return
        fake_calib_dict = {
            'P2': P2,                              # 3x4 projection matrix
            'R0': np.eye(3, dtype=np.float32),
            'Tr_velo2cam': V2C[:3, :],          # 3x4 extrinsic
        }

        # Use the internal helper that accepts a dict (bypasses file reading)
        calib = calibration_kitti.Calibration(fake_calib_dict)
        calib.P2 = P2
        calib.R0 = np.eye(3, dtype=np.float32)
        calib.V2C = V2C[:3, :]  # 3x4
        return calib

    def get_infos(self, logger, num_workers=cpu_count(), has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        calib = self.get_calib()

        def process_single_scene(sample_idx):
            info = {}
            info['point_cloud'] = {'lidar_idx': sample_idx, 'num_features': 5}

            try:
                img_shape = self.get_image_shape(sample_idx)
            except Exception:
                img_shape = np.array([1080, 1920], dtype=np.int32)  # fallback
            info['image'] = {'image_idx': sample_idx, 'image_shape': img_shape}

            # P2_4x4 = np.concatenate([calib.P2, np.zeros((3,1))], axis=1)
            # P2_4x4 = np.vstack([P2_4x4, [0,0,0,1]])
            # R0_4x4 = np.eye(4)
            # V2C_4x4 = np.vstack([calib.V2C, [0,0,0,1]])
            P2_4x4 = np.eye(4, dtype=np.float32)
            P2_4x4[:3, :4] = calib.P2  # P2 is 3x4 → embed in top-left

            R0_4x4 = np.eye(4, dtype=np.float32)
            R0_4x4[:3, :3] = calib.R0  # R0 is 3x3

            V2C_4x4 = np.eye(4, dtype=np.float32)
            V2C_4x4[:3, :4] = calib.V2C  # V2C is 3x4
            info['calib'] = {'P2': P2_4x4, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            if has_label:
                try:
                    obj_list = self.get_label(sample_idx)
                    obj_list = [obj for obj in obj_list if obj.cls_type in self.class_names]
                    if len(obj_list) == 0:
                        return None

                    annotations = {}
                    annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                    annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                    annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                    annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                    annotations['bbox'] = np.concatenate([obj.box2d.reshape(1,4) for obj in obj_list], axis=0)
                    annotations['dimensions'] = np.array([[obj.h, obj.w, obj.l] for obj in obj_list])  # h,w,l
                    annotations['location'] = np.array([[obj.loc[0], obj.loc[1], obj.loc[2]] for obj in obj_list])
                    annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                    annotations['score'] = np.array([obj.score for obj in obj_list])
                    annotations['difficulty'] = np.array([obj.level for obj in obj_list], dtype=np.int32)

                    # Convert camera to lidar coordinates
                    loc = annotations['location']
                    dims = annotations['dimensions']
                    rots = annotations['rotation_y']
                    loc_lidar = calib.rect_to_lidar(loc)
                    l, w, h = dims[:, 2:3], dims[:, 1:2], dims[:, 0:1]
                    loc_lidar[:, 2] += h[:, 0] / 2
                    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar

                    info['annos'] = annotations

                    if count_inside_pts:
                        points = self.get_lidar(sample_idx)
                        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                        num_points_in_gt = np.zeros(len(obj_list), dtype=np.int32)
                        for i in range(len(obj_list)):
                            flag = box_utils.in_hull(points[:, :3], corners_lidar[i])
                            num_points_in_gt[i] = flag.sum()
                        annotations['num_points_in_gt'] = num_points_in_gt

                except Exception as e:
                    logger.warning(f'{sample_idx} failed to load labels: {e}')
                    return None

            return info

        sample_id_list = sample_id_list or self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = list(tqdm(executor.map(process_single_scene, sample_id_list), total=len(sample_id_list)))

        valid_infos = [info for info in infos if info is not None]
        logger.info(f'Loaded {len(valid_infos)} valid samples out of {len(sample_id_list)}')

        return valid_infos

    def create_groundtruth_database(self, logger, info_path=None, used_classes=None, split='train'):
        # Same as original, just adapted paths
        import torch

        database_save_path = Path(self.root_path) / f'gt_database_{split}'
        db_info_save_path = Path(self.root_path) / f'stf_dbinfos_{split}.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for info in tqdm(infos):
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, :3]), torch.from_numpy(gt_boxes)
            ).numpy()

            for i in range(len(names)):
                filename = f'{sample_idx}_{names[i]}_{i}.bin'
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]
                gt_points.tofile(filepath)

                if used_classes is None or names[i] in used_classes:
                    db_info = {
                        'name': names[i],
                        'path': str(filepath.relative_to(self.root_path)),
                        'image_idx': sample_idx,
                        'gt_idx': i,
                        'box3d_lidar': gt_boxes[i],
                        'num_points_in_gt': int(gt_points.shape[0]),
                        'difficulty': annos['difficulty'][i],
                        'bbox': annos['bbox'][i],
                        'score': annos['score'][i]
                    }
                    all_db_infos.setdefault(names[i], []).append(db_info)

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        logger.info('Ground truth database created.')

    def __len__(self):
        return len(self.stf_infos) if hasattr(self, 'stf_infos') and self.stf_infos else len(self.sample_id_list or [])

    def __getitem__(self, index):
        # Simplified version for training (full version similar to original)
        info = copy.deepcopy(self.stf_infos[index]) if self.stf_infos else None
        if info is None:
            return self.__getitem__(np.random.randint(self.__len__()))

        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        calib = self.get_calib()

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['gt_boxes_lidar']
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['image_shape'] = info['image']['image_shape']
        return data_dict


def create_stf_infos(dataset_cfg, class_names, data_path, save_path, logger, workers=cpu_count()):
    dataset = STF(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False, logger=logger)

    train_split = dataset_cfg.DATA_SPLIT['train']
    val_split = dataset_cfg.DATA_SPLIT['val']
    test_split = dataset_cfg.DATA_SPLIT.get('test', 'test')

    print('--------------- Start to generate STF data infos ---------------')

    # Train
    dataset.set_split(train_split)
    train_infos = dataset.get_infos(logger=logger, num_workers=workers, has_label=True, count_inside_pts=True)
    train_pkl = save_path / f'stf_infos_{train_split}.pkl'
    with open(train_pkl, 'wb') as f:
        pickle.dump(train_infos, f)
    print(f'STF train infos saved to {train_pkl}')

    # Val
    dataset.set_split(val_split)
    val_infos = dataset.get_infos(logger=logger, num_workers=workers, has_label=True, count_inside_pts=True)
    val_pkl = save_path / f'stf_infos_{val_split}.pkl'
    with open(val_pkl, 'wb') as f:
        pickle.dump(val_infos, f)
    print(f'STF val infos saved to {val_pkl}')

    # Test (optional)
    if hasattr(dataset_cfg.DATA_SPLIT, 'test'):
        dataset.set_split(test_split)
        test_infos = dataset.get_infos(logger=logger, num_workers=workers, has_label=True, count_inside_pts=False)
        test_pkl = save_path / f'stf_infos_{test_split}.pkl'
        with open(test_pkl, 'wb') as f:
            pickle.dump(test_infos, f)
        print(f'STF test infos saved to {test_pkl}')

    # Create GT database
    print('--------------- Creating groundtruth database ---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(logger=logger, info_path=train_pkl, used_classes=class_names, split=train_split)

    print('--------------- STF Data preparation completed! ---------------')


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from easydict import EasyDict

    if len(sys.argv) < 3 or sys.argv[1] != 'create_stf_infos':
        print("Usage: python STF_Dataloader.py create_stf_infos /path/to/stf_config.yaml")
        sys.exit(1)

    config_path = sys.argv[2]
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    ROOT_DIR = Path(__file__).resolve().parents[3]  # Adjust based on your folder structure
    print('path:', ROOT_DIR)
    print(ROOT_DIR)
    log_file = ROOT_DIR / 'datasets/SeeingThroghFog/pcdet/create_stf_infos.log'
    log = common_utils.create_logger(str(log_file), log_level=logging.INFO)

    dataset_cfg = EasyDict(yaml.safe_load(open(config_path)))

    class_names = ['Car', 'Pedestrian', 'LargeVehicle']  # Adjust as needed: ['Car', 'Pedestrian', 'Cyclist']

    data_path = ROOT_DIR / 'datasets' / 'SeeingThroghFog'
    save_path = ROOT_DIR / 'datasets' / 'SeeingThroghFog' / 'OUTPUT'

    create_stf_infos(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        data_path=data_path,
        save_path=save_path,
        logger=log,
        workers=cpu_count()
    )
  