import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
from random import randrange


import numpy
import torch
from datasets.random_cuboid import RandomCuboid

import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml
# from yaml import CLoader as Loader
from torch.utils.data import Dataset
from datasets.scannet200.scannet200_constants import SCANNET_COLOR_MAP_200, SCANNET_COLOR_MAP_20

logger = logging.getLogger(__name__)


class SemanticSegmentationDataset(Dataset):
    """Docstring for SemanticSegmentationDataset. """

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling=0,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        add_unlabeled_pc=False,
        task="instance_segmentation",
        cropping=False,
        cropping_args=None,
        is_tta=False,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        area=-1,
        on_crops=False,
        eval_inner_core=-1,
        filter_out_classes=[],
        label_offset=0,
        add_clip=False,
        is_elastic_distortion=True,
        color_drop=0.0,
        part2human=False,
        is_mirroring=True,
        broken_mirror_version=False,
        use_color=False
    ):
        assert task in ["instance_segmentation", "semantic_segmentation"], "unknown task"

        self.use_color = use_color
        self.broken_mirror_version = broken_mirror_version
        self.is_mirroring = is_mirroring
        self.part2human = part2human
        self.add_clip = add_clip
        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop

        if self.dataset_name == "scannet":
            self.color_map = SCANNET_COLOR_MAP_20
            self.color_map[255] = (255, 255, 255)
        elif self.dataset_name == "stpls3d":
            self.color_map = {
                0:[0, 255, 0],  # Ground
                1:[0, 0, 255],  # Build
                2:[0, 255, 255],  # LowVeg
                3:[255, 255, 0],  # MediumVeg
                4:[255, 0, 255],  # HiVeg
                5:[100, 100, 255],  # Vehicle
                6:[200, 200, 100],  # Truck
                7:[170, 120, 200],  # Aircraft
                8:[255, 0, 0],  # MilitaryVec
                9:[200, 100, 100],  # Bike
                10:[10, 200, 100],  # Motorcycle
                11:[200, 200, 200],  # LightPole
                12:[50, 50, 50],  # StreetSign
                13:[60, 130, 60],  # Clutter
                14:[130, 30, 60]}  # Fence
        elif self.dataset_name == "scannet200":
            self.color_map = SCANNET_COLOR_MAP_200
        elif self.dataset_name == "s3dis":
            self.color_map = {
            0: [0, 255, 0],        # ceiling
            1: [0, 0, 255],        # floor
            2: [0, 255, 255],      # wall
            3: [255, 255, 0],      # beam
            4: [255, 0, 255],      # column
            5: [100, 100, 255],    # window
            6: [200, 200, 100],    # door
            7: [170, 120, 200],    # table
            8: [255, 0, 0],        # chair
            9: [200, 100, 100],    # sofa
            10: [10, 200, 100],     # bookcase
            11: [200, 200, 200],    # board
            12: [50, 50, 50]      # clutter
            }
        elif self.dataset_name == "human_segmentation":
            self.human_color_map = [
                [226., 226., 226.],
                [158.0, 143.0, 20.0],  # rightHand
                [243.0, 115.0, 68.0],  # rightUpLeg
                [228.0, 162.0, 227.0],  # leftArm
                [210.0, 78.0, 142.0],  # head
                [152.0, 78.0, 163.0],  # leftLeg
                [76.0, 134.0, 26.0],  # leftFoot
                [100.0, 143.0, 255.0],  # torso
                [129.0, 0.0, 50.0],  # rightFoot
                [255., 176., 0.],  # rightArm
                [192.0, 100.0, 119.0],  # leftHand
                [149.0, 192.0, 228.0],  # rightLeg
                [243.0, 232.0, 88.0],  # leftForeArm
                [90., 64., 210.],  # rightForeArm
                [152.0, 200.0, 156.0],  # leftUpLeg
                [129.0, 103.0, 106.0],  # hips
            ]

            '''
            self.part_color_map = [
                [0.0, 0.0, 0.0],
                [153.0, 170.0, 17.0],
                [73.0, 196.0, 37.0],
                [121.0, 25.0, 252.0],
                [1.0, 147.0, 72.0],
                [165.0, 38.0, 65.0],
                [27.0, 255.0, 80.0],
                [174.0, 22.0, 133.0],
                [47.0, 210.0, 21.0],
                [111.0, 140.0, 56.0],
                [222.0, 25.0, 45.0],
                [98.0, 255.0, 160.0],
                [54.0, 82.0, 122.0],
                [10.0, 19.0, 94.0],
                [98.0, 58.0, 83.0],
                [75.0, 233.0, 65.0]
            ]
            '''

            self.part_color_map = [
                [226., 226., 226.],
                [158.0, 143.0, 20.0],  # rightHand
                [243.0, 115.0, 68.0],  # rightUpLeg
                [228.0, 162.0, 227.0],  # leftArm
                [210.0, 78.0, 142.0],  # head
                [152.0, 78.0, 163.0],  # leftLeg
                [76.0, 134.0, 26.0],  # leftFoot
                [100.0, 143.0, 255.0],  # torso
                [129.0, 0.0, 50.0],  # rightFoot
                [255., 176., 0.],  # rightArm
                [192.0, 100.0, 119.0],  # leftHand
                [149.0, 192.0, 228.0],  # rightLeg
                [243.0, 232.0, 88.0],  # leftForeArm
                [90., 64., 210.],  # rightForeArm
                [152.0, 200.0, 156.0],  # leftUpLeg
                [129.0, 103.0, 106.0],  # hips
            ]

            if self.part2human:
                self.color_map = self.human_color_map
            else:
                self.color_map = self.part_color_map

            self.label_map = [
                "background",
                "rightHand",
                "rightUpLeg",
                "leftArm",
                "head",
                "leftLeg",
                "leftFoot",
                "torso",
                "rightFoot",
                "rightArm",
                "leftHand",
                "rightLeg",
                "leftForeArm",
                "rightForeArm",
                "leftUpLeg",
                "hips"
            ]

            self.chiral_classes = {class_name: class_id
                                   for class_id, class_name in enumerate(self.label_map)
                                   if "right" in class_name or "left" in class_name}

            # swap class labels for {right,left}XXX classes
            self.chiral_swap = {
                class_id: self.chiral_classes[class_name.replace("right", "left")
                if "right" in class_name else class_name.replace("left", "right")]
                for class_name, class_id in self.chiral_classes.items()}

            # add class which need no swapping
            self.chiral_swap.update(
                {class_id: class_id for class_id, class_name in enumerate(self.label_map)
                 if "right" not in class_name and "left" not in class_name})

        else:
            assert False, "dataset not known"

        self.task = task

        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        self.area = area
        self.eval_inner_core = eval_inner_core

        self.reps_per_epoch = reps_per_epoch

        self.cropping = cropping
        self.cropping_args = cropping_args
        self.is_tta = is_tta
        self.on_crops = on_crops

        self.crop_min_size = crop_min_size
        self.crop_length = crop_length

        self.version1 = cropping_v1

        self.random_cuboid = RandomCuboid(self.crop_min_size,
                                          crop_length=self.crop_length,
                                          version1=self.version1)

        self.mode = mode
        self.data_dir = data_dir
        self.add_unlabeled_pc = add_unlabeled_pc
        if add_unlabeled_pc:
            self.other_database = self._load_yaml(
                Path(data_dir).parent / "matterport" / "train_database.yaml"
            )
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.instance_oversampling = instance_oversampling
        self.place_around_existing = place_around_existing
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if self.dataset_name != "s3dis":
                if not (database_path / f"{mode}_database.yaml").exists():
                    print(f"generate {database_path}/{mode}_database.yaml first")
                    exit()
                # self._data.extend(self._load_yaml(database_path / f"{mode}_database_save2.yaml"))
                self._data.extend(self._load_yaml(database_path / f"{mode}_database.yaml"))
            else:
                mode_s3dis = f"Area_{self.area}"
                if self.mode == "train":
                    mode_s3dis = "train_" + mode_s3dis
                if not (database_path / f"{mode_s3dis}_database.yaml").exists():
                    print(f"generate {database_path}/{mode_s3dis}_database.yaml first")
                    exit()
                self._data.extend(self._load_yaml(database_path / f"{mode_s3dis}_database.yaml"))
        if data_percent < 1.0:
            self._data = sample(self._data, int(len(self._data) * data_percent))

        #if part2human:
        #    label_db_filepath = label_db_filepath.replace("part_database.yaml", "label_database.yaml")

        labels = self._load_yaml(Path(label_db_filepath))

        # if working only on classes for validation - discard others
        self._labels = self._select_correct_labels(labels, num_labels)

        if instance_oversampling > 0:
            self.instance_data = self._load_yaml(
                Path(label_db_filepath).parent / "instance_database.yaml"
            )

        # normalize color channels
        if self.dataset_name == "s3dis":
            color_mean_std = color_mean_std.replace("color_mean_std.yaml", f"Area_{self.area}_color_mean_std.yaml")

        if Path(str(color_mean_std)).exists():
            color_mean_std = self._load_yaml(color_mean_std)
            color_mean, color_std = (
                tuple(color_mean_std["mean"]),
                tuple(color_mean_std["std"]),
            )
        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]
        else:
            logger.error("pass mean and std as tuple of tuples, or as an .yaml file")
            color_mean, color_std = np.array([1.,1.,1.]), np.array([1.,1.,1.])

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (
            volume_augmentations_path != "none"
        ):
            self.volume_augmentations = V.load(
                Path(volume_augmentations_path), data_format="yaml"
            )
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (
            image_augmentations_path != "none"
        ):
            self.image_augmentations = A.load(
                Path(image_augmentations_path), data_format="yaml"
            )
        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        self.cache_data = cache_data
        # new_data = []
        if self.cache_data:
            new_data = []
            for i in range(len(self._data)):
                self._data[i]['data'] = np.load(self.data[i]["filepath"].replace("../../", ""))
                if self.on_crops:
                    if self.eval_inner_core == -1:
                        for block_id, block in enumerate(self.splitPointCloud(self._data[i]['data'])):
                            if len(block) > 10000:
                                new_data.append({
                                    'instance_gt_filepath': self._data[i]['instance_gt_filepath'][block_id] \
                                        if len(self._data[i]['instance_gt_filepath']) > 0 else list(),
                                    'scene': f"{self._data[i]['scene'].replace('.txt', '')}_{block_id}.txt",
                                    'raw_filepath': f"{self.data[i]['filepath'].replace('.npy', '')}_{block_id}",
                                    'data': block
                                })
                            else:
                                assert False
                    else:
                        conds_inner, blocks_outer = self.splitPointCloud(self._data[i]['data'],
                                                                         size=self.crop_length,
                                                                         inner_core=self.eval_inner_core)

                        for block_id in range(len(conds_inner)):
                            cond_inner = conds_inner[block_id]
                            block_outer = blocks_outer[block_id]

                            if cond_inner.sum() > 10000:
                                new_data.append({
                                    'instance_gt_filepath': self._data[i]['instance_gt_filepath'][block_id] \
                                        if len(self._data[i]['instance_gt_filepath']) > 0 else list(),
                                    'scene': f"{self._data[i]['scene'].replace('.txt', '')}_{block_id}.txt",
                                    'raw_filepath': f"{self.data[i]['filepath'].replace('.npy', '')}_{block_id}",
                                    'data': block_outer,
                                    'cond_inner': cond_inner
                                })
                            else:
                                assert False

            if self.on_crops:
                self._data = new_data
                #new_data.append(np.load(self.data[i]["filepath"].replace("../../", "")))
            #self._data = new_data

    def splitPointCloud(self, cloud, size=50.0, stride=50, inner_core=-1):
        if inner_core == -1:
            limitMax = np.amax(cloud[:, 0:3], axis=0)
            width = int(np.ceil((limitMax[0] - size) / stride)) + 1
            depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
            cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
            blocks = []
            for (x, y) in cells:
                xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
                ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
                cond = xcond & ycond
                block = cloud[cond, :]
                blocks.append(block)
            return blocks
        else:
            limitMax = np.amax(cloud[:, 0:3], axis=0)
            width = int(np.ceil((limitMax[0] - inner_core) / stride)) + 1
            depth = int(np.ceil((limitMax[1] - inner_core) / stride)) + 1
            cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
            blocks_outer = []
            conds_inner = []
            for (x, y) in cells:
                xcond_outer = (cloud[:, 0] <= x + inner_core / 2. + size / 2) & (cloud[:, 0] >= x + inner_core / 2. - size / 2)
                ycond_outer = (cloud[:, 1] <= y + inner_core / 2. + size / 2) & (cloud[:, 1] >= y + inner_core / 2. - size / 2)

                cond_outer = xcond_outer & ycond_outer
                block_outer = cloud[cond_outer, :]

                xcond_inner = (block_outer[:, 0] <= x + inner_core) & (block_outer[:, 0] >= x)
                ycond_inner = (block_outer[:, 1] <= y + inner_core) & (block_outer[:, 1] >= y)

                cond_inner = xcond_inner & ycond_inner

                conds_inner.append(cond_inner)
                blocks_outer.append(block_outer)
            return conds_inner, blocks_outer

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)

    def __len__(self):
        if self.is_tta:
            return 5*len(self.data)
        else:
            return self.reps_per_epoch*len(self.data)

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        if self.is_tta:
            idx = idx % len(self.data)

        if self.cache_data:
            points = self.data[idx]['data']
        else:
            assert not self.on_crops, "you need caching if on crops"
            points = np.load(self.data[idx]["filepath"].replace("../../", ""))

        if "train" in self.mode and self.dataset_name in ["s3dis", "stpls3d"]:
            inds = self.random_cuboid(points)
            points = points[inds]

        if self.dataset_name == "human_segmentation":
            coordinates, color, labels = (
                points[:, :3],
                points[:, 3:6],
                points[:, 6:8]
            )

            '''
            if self.part2human:
                labels[:, 0] = (labels[:, 0] > 0.0).astype(np.float32)
            else:
                # treat each human part as a separate instance (not attached to a human instance!)
                labels[:, 1] = labels[:, 0] * 100 + labels[:, 1]
            '''

            # labels[:, 1] = labels[:, 0] * 100 + labels[:, 1]

            normals = np.ones((color.shape[0], 3))
            segments = np.ones(color.shape[0])
        else:
            coordinates, color, normals, segments, labels = (
                points[:, :3],
                points[:, 3:6],
                points[:, 6:9],
                points[:, 9],
                points[:, 10:12],
            )

        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        if not self.add_colors:
            color = np.ones((len(color), 3))

        # volume and image augmentations for train
        if "train" in self.mode or self.is_tta:
            if self.cropping:
                new_idx = self.random_cuboid(coordinates, labels[:,1],
                                             self._remap_from_zero(labels[:, 0].copy()))

                coordinates = coordinates[new_idx]
                color = color[new_idx]
                labels = labels[new_idx]
                segments = segments[new_idx]
                raw_color = raw_color[new_idx]
                raw_normals = raw_normals[new_idx]
                normals = normals[new_idx]
                points = points[new_idx]


            coordinates -= coordinates.mean(0)

            try:
                coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
            except OverflowError as err:
                print(coordinates)
                print(coordinates.shape)
                raise err

            if self.instance_oversampling > 0.0:
                coordinates, color, normals, labels = self.augment_individual_instance(
                    coordinates, color, normals, labels, self.instance_oversampling
                )

            if self.flip_in_center:
                coordinates = flip_in_center(coordinates)

            if self.is_mirroring:
                for i in (0, 1):
                    if random() < 0.5:
                        coord_max = np.max(points[:, i])
                        coordinates[:, i] = coord_max - coordinates[:, i]

                        '''
                        if not self.part2human:
                            for point_id in range(labels.shape[0]):
                                # swap class ids for {left, right}XXX classes
                                labels[point_id, 0] = self.chiral_swap[labels[point_id, 0]]
                        '''

                        if not self.broken_mirror_version:
                            for point_id in range(labels.shape[0]):
                                # swap class ids for {left, right}XXX classes
                                labels[point_id, 0] = self.chiral_swap[labels[point_id, 0]]

            if random() < 0.95:
                if self.is_elastic_distortion:
                    for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                        coordinates = elastic_distortion(
                            coordinates, granularity, magnitude
                        )
            aug = self.volume_augmentations(
                points=coordinates, normals=normals, features=color, labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(self.image_augmentations(image=pseudo_image)["image"])

            if False: #self.point_per_cut != 0:
                number_of_cuts = int(len(coordinates) / self.point_per_cut)
                for _ in range(number_of_cuts):
                    size_of_cut = np.random.uniform(0.05, self.max_cut_region)
                    # not wall, floor or empty
                    point = choice(coordinates)
                    x_min = point[0] - size_of_cut
                    x_max = x_min + size_of_cut
                    y_min = point[1] - size_of_cut
                    y_max = y_min + size_of_cut
                    z_min = point[2] - size_of_cut
                    z_max = z_min + size_of_cut
                    indexes = crop(
                        coordinates, x_min, y_min, z_min, x_max, y_max, z_max
                    )
                    coordinates, normals, color, labels = (
                        coordinates[~indexes],
                        normals[~indexes],
                        color[~indexes],
                        labels[~indexes],
                    )

            # if self.noise_rate > 0:
            #     coordinates, color, normals, labels = random_points(
            #         coordinates,
            #         color,
            #         normals,
            #         labels,
            #         self.noise_rate,
            #         self.ignore_label,
            #     )

            if (self.resample_points > 0) or (self.noise_rate > 0):
                coordinates, color, normals, labels = random_around_points(
                    coordinates,
                    color,
                    normals,
                    labels,
                    self.resample_points,
                    self.noise_rate,
                    self.ignore_label,
                )

            if self.add_unlabeled_pc:
                if random() < 0.8:
                    new_points = np.load(
                        self.other_database[
                            np.random.randint(0, len(self.other_database) - 1)
                        ]["filepath"]
                    )
                    (
                        unlabeled_coords,
                        unlabeled_color,
                        unlabeled_normals,
                        unlabeled_labels,
                    ) = (
                        new_points[:, :3],
                        new_points[:, 3:6],
                        new_points[:, 6:9],
                        new_points[:, 9:],
                    )
                    unlabeled_coords -= unlabeled_coords.mean(0)
                    unlabeled_coords += (
                        np.random.uniform(
                            unlabeled_coords.min(0), unlabeled_coords.max(0)
                        )
                        / 2
                    )

                    aug = self.volume_augmentations(
                        points=unlabeled_coords,
                        normals=unlabeled_normals,
                        features=unlabeled_color,
                        labels=unlabeled_labels,
                    )
                    (
                        unlabeled_coords,
                        unlabeled_color,
                        unlabeled_normals,
                        unlabeled_labels,
                    ) = (
                        aug["points"],
                        aug["features"],
                        aug["normals"],
                        aug["labels"],
                    )
                    pseudo_image = unlabeled_color.astype(np.uint8)[np.newaxis, :, :]
                    unlabeled_color = np.squeeze(
                        self.image_augmentations(image=pseudo_image)["image"]
                    )

                    coordinates = np.concatenate((coordinates, unlabeled_coords))
                    color = np.concatenate((color, unlabeled_color))
                    normals = np.concatenate((normals, unlabeled_normals))
                    labels = np.concatenate(
                        (labels, np.full_like(unlabeled_labels, self.ignore_label))
                    )

            if random() < self.color_drop:
                color[:] = 255

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        labels[:, 1] = labels[:, 0] * 1000 + labels[:, 1]

        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            if not self.add_instance:
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()[..., None]

        labels = np.hstack((labels, segments[..., None].astype(np.int32)))

        features = color
        if self.add_normals:
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))

        #if self.task != "semantic_segmentation":
        if self.data[idx]['raw_filepath'].split("/")[-2] in ['scene0636_00', 'scene0154_00']:
            return self.__getitem__(0)

        if self.dataset_name == "s3dis":
            return coordinates, features, labels, self.data[idx]['area'] + "_" + self.data[idx]['scene'],\
                   raw_color, raw_normals, raw_coordinates, idx, None, None
        if self.dataset_name == "stpls3d":
            if labels.shape[1] != 1: # only segments --> test set!
                if np.unique(labels[:, -2]).shape[0] < 2:
                    print("NO INSTANCES")
                    return self.__getitem__(0)
            return coordinates, features, labels, self.data[idx]['scene'], \
                   raw_color, raw_normals, raw_coordinates, idx, None, None
        if self.dataset_name == "human_segmentation":
            if "train" in self.mode and np.unique(labels[:, -2]).shape[0] < 2:
                print("NO INSTANCES")
                return self.__getitem__(randrange(len(self.data)))
            if not self.use_color:
                features[:, :3] = 1.  # make sure no color information is leaked
            return coordinates, features, labels, self.data[idx]['scene'], \
                   raw_color, raw_normals, raw_coordinates, idx, None, None
        else:
            if self.add_clip:
                clip_basepath = "/nodes/dom/work/nekrasov/data/scannet/scannet/scans"
                clip_scene_path = f'{clip_basepath}/scene{self.data[idx]["filepath"].split("/")[-1].replace(".npy", "")}'
                clip_feature = np.load(f'{clip_scene_path}/feature_sparse.npy').astype(np.float32)
                clip_indx = np.load(f'{clip_scene_path}/feature_index.npy')
                clip_center_point = coordinates[clip_indx]

                return coordinates, features, labels, self.data[idx]['raw_filepath'].split("/")[-2], \
                       raw_color, raw_normals, raw_coordinates, idx, clip_center_point, clip_feature

            else:
                return coordinates, features, labels, self.data[idx]['raw_filepath'].split("/")[-2], \
                       raw_color, raw_normals, raw_coordinates, idx, None, None

    @property
    def data(self):
        """ database file containing information about preproscessed dataset """
        return self._data

    @property
    def label_info(self):
        """ database file containing information labels used by dataset """
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f)
        return file

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for k, v, in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for k, v, in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _remap_from_zero(self, labels):
        labels[~np.isin(labels, list(self.label_info.keys()))] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped

    def augment_individual_instance(
        self, coordinates, color, normals, labels, oversampling=1.0
    ):
        max_instance = int(len(np.unique(labels[:, 1])))
        # randomly selecting half of non-zero instances
        for instance in range(0, int(max_instance * oversampling)):
            if self.place_around_existing:
                center = choice(
                    coordinates[labels[:, 1] == choice(np.unique(labels[:, 1]))]
                )
            else:
                center = np.array([uniform(-5, 5), uniform(-5, 5), uniform(-0.5, 2)])
            instance = choice(choice(self.instance_data))
            instance = np.load(instance["instance_filepath"])
            # centering two objects
            instance[:, :3] = instance[:, :3] - instance[:, :3].mean(axis=0) + center
            max_instance = max_instance + 1
            instance[:, -1] = max_instance
            aug = V.Compose(
                [
                    V.Scale3d(),
                    V.RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(1, 0, 0)),
                    V.RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(0, 1, 0)),
                    V.RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 0, 1)),
                ]
            )(
                points=instance[:, :3],
                features=instance[:, 3:6],
                normals=instance[:, 6:9],
                labels=instance[:, 9:],
            )
            coordinates = np.concatenate((coordinates, aug["points"]))
            color = np.concatenate((color, aug["features"]))
            normals = np.concatenate((normals, aug["normals"]))
            labels = np.concatenate((labels, aug["labels"]))

        return coordinates, color, normals, labels


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
  """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def flip_in_center(coordinates):
    # moving coordinates to center
    coordinates -= coordinates.mean(0)
    aug = V.Compose(
        [
            V.Flip3d(axis=(0, 1, 0), always_apply=True),
            V.Flip3d(axis=(1, 0, 0), always_apply=True),
        ]
    )

    first_crop = coordinates[:, 0] > 0
    first_crop &= coordinates[:, 1] > 0
    # x -y
    second_crop = coordinates[:, 0] > 0
    second_crop &= coordinates[:, 1] < 0
    # -x y
    third_crop = coordinates[:, 0] < 0
    third_crop &= coordinates[:, 1] > 0
    # -x -y
    fourth_crop = coordinates[:, 0] < 0
    fourth_crop &= coordinates[:, 1] < 0

    if first_crop.size > 1:
        coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        minimum = coordinates[second_crop].min(0)
        minimum[2] = 0
        minimum[0] = 0
        coordinates[second_crop] = aug(points=coordinates[second_crop])["points"]
        coordinates[second_crop] += minimum
    if third_crop.size > 1:
        minimum = coordinates[third_crop].min(0)
        minimum[2] = 0
        minimum[1] = 0
        coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
        coordinates[third_crop] += minimum
    if fourth_crop.size > 1:
        minimum = coordinates[fourth_crop].min(0)
        minimum[2] = 0
        coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])["points"]
        coordinates[fourth_crop] += minimum

    return coordinates


def random_around_points(
    coordinates, color, normals, labels, rate=0.2, noise_rate=0, ignore_label=255
):
    coord_indexes = sample(
        list(range(len(coordinates))), k=int(len(coordinates) * rate)
    )
    noisy_coordinates = deepcopy(coordinates[coord_indexes])
    noisy_coordinates += np.random.uniform(
        -0.2 - noise_rate, 0.2 + noise_rate, size=noisy_coordinates.shape
    )

    if noise_rate > 0:
        noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
        noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
        noisy_labels = np.full(labels[coord_indexes].shape, ignore_label)

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))
    else:
        noisy_color = deepcopy(color[coord_indexes])
        noisy_normals = deepcopy(normals[coord_indexes])
        noisy_labels = deepcopy(labels[coord_indexes])

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))

    return coordinates, color, normals, labels


def random_points(
    coordinates, color, normals, labels, noise_rate=0.6, ignore_label=255
):
    max_boundary = coordinates.max(0) + 0.1
    min_boundary = coordinates.min(0) - 0.1

    noisy_coordinates = int((max(max_boundary) - min(min_boundary)) / noise_rate)

    noisy_coordinates = np.array(
        list(
            product(
                np.linspace(min_boundary[0], max_boundary[0], noisy_coordinates),
                np.linspace(min_boundary[1], max_boundary[1], noisy_coordinates),
                np.linspace(min_boundary[2], max_boundary[2], noisy_coordinates),
            )
        )
    )
    noisy_coordinates += np.random.uniform(
        -noise_rate, noise_rate, size=noisy_coordinates.shape
    )

    noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
    noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
    noisy_labels = np.full((noisy_coordinates.shape[0], labels.shape[1]), ignore_label)

    coordinates = np.vstack((coordinates, noisy_coordinates))
    color = np.vstack((color, noisy_color))
    normals = np.vstack((normals, noisy_normals))
    labels = np.vstack((labels, noisy_labels))
    return coordinates, color, normals, labels
