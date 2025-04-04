import numpy as np
from fire import Fire
from natsort import natsorted
from base_preprocessing import BasePreprocessing
from plyfile import PlyData
import pandas as pd

class HumanSegmentationDataset(BasePreprocessing):
    def __init__(
            self,
            data_dir: str = "../../data/raw/egobody",
            save_dir: str = "../../data/processed/egobody_color",
            modes: tuple = ("train",),
            min_points: int = 0,
            min_instances: int = 0,
            n_jobs: int = -1
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.min_points = min_points
        self.min_instances = min_instances

        self.class_map = {
            'background': 0,
            'human': 1,
        }

        self.color_map = [
            [0, 0, 255],        # background
            [0, 255, 0]]        # human

        self.COLOR_MAP_W_BODY_PARTS = {
            -1: (255., 255., 255.),
            0: (0., 0., 0.),
            1: (174., 199., 232.),
            2: (152., 223., 138.),
            3: (31., 119., 180.),
            4: (255., 187., 120.),
            5: (188., 189., 34.),
            6: (140., 86., 75.),
            7: (255., 152., 150.),
            8: (214., 39., 40.),
            9: (197., 176., 213.),
            10: (148., 103., 189.),
            11: (196., 156., 148.),
            12: (23., 190., 207.),
            14: (247., 182., 210.),
            15: (66., 188., 102.),
            16: (219., 219., 141.),
            17: (140., 57., 197.),
            18: (202., 185., 52.),
            19: (51., 176., 203.),
            20: (200., 54., 131.),
            21: (92., 193., 61.),
            22: (78., 71., 183.),
            23: (172., 114., 82.),
            24: (255., 127., 14.),
            25: (91., 163., 138.),
            26: (153., 98., 156.),
            27: (140., 153., 101.),
            28: (158., 218., 229.),
            29: (100., 125., 154.),
            30: (178., 127., 135.),
            31: (120., 185., 128.),
            32: (146., 111., 194.),
            33: (44., 160., 44.),
            34: (112., 128., 144.),
            35: (96., 207., 209.),
            36: (227., 119., 194.),
            37: (213., 92., 176.),
            38: (94., 106., 211.),
            39: (82., 84., 163.),
            40: (100., 85., 144.),
            41: (0., 0., 255.),  # artificial human
            # body parts
            100: (35., 69., 100.),  # rightHand
            101: (73., 196., 37.),  # rightUpLeg
            102: (121., 25., 252.),  # leftArm
            103: (96., 237., 31.),  # head
            104: (55., 40., 93.),  # leftEye
            105: (75., 180., 125.),  # rightEye
            106: (165., 38., 65.),  # leftLeg
            107: (63., 75., 77.),  # leftToeBase
            108: (27., 255., 80.),  # leftFoot
            109: (82., 110., 90.),  # spine1
            110: (87., 54., 10.),  # spine2
            111: (210., 200., 110.),  # leftShoulder
            112: (217., 212., 76.),  # rightShoulder
            113: (254., 176., 234.),  # rightFoot
            114: (111., 140., 56.),  # rightArm
            115: (83., 15., 157.),  # leftHandIndex1
            116: (98., 255., 160.),  # rightLeg
            117: (153., 170., 17.),  # rightHandIndex1
            118: (54., 82., 122.),  # leftForeArm
            119: (10., 19., 94.),  # rightForeArm
            120: (1., 147., 72.),  # neck
            121: (47., 210., 21.),  # rightToeBase
            122: (174., 22., 133.),  # spine
            123: (98., 58., 83.),  # leftUpLeg
            124: (222., 25., 45.),  # leftHand
            125: (75., 233., 65.),  # hips
        }

        # in each scene there are at most 10 human instances
        self.COLOR_MAP_INSTANCES = {
            0: (0., 0., 0.),
            1: (255., 0., 0.),
            2: (0., 255., 0.),
            3: (0., 0., 255.),
            4: (255., 255., 0.),
            5: (255., 0., 255.),
            6: (0., 255., 255.),
            7: (255., 204., 153.),
            8: (255., 102., 0.),
            9: (0., 128., 128.),
            10: (153., 153., 255.),
        }

        self.ORIG_BODY_PART_IDS = set(range(100, 126))

        #self.LABEL_LIST = ["background", "rightHand", "rightUpLeg", "leftArm", "head", "leftEye", "rightEye", "leftLeg",
        #                   "leftToeBase", "leftFoot", "spine1", "spine2", "leftShoulder", "rightShoulder",
        #                   "rightFoot", "rightArm", "leftHandIndex1", "rightLeg", "rightHandIndex1",
        #                   "leftForeArm", "rightForeArm", "neck", "rightToeBase", "spine", "leftUpLeg",
        #                   "leftHand", "hips"]

        self.LABEL_MAP = {
            0: "background",
            1: "rightHand",
            2: "rightUpLeg",
            3: "leftArm",
            4: "head",
            5: "leftLeg",
            6: "leftFoot",
            7: "torso",
            8: "rightFoot",
            9: "rightArm",
            10: "leftHand",
            11: "rightLeg",
            12: "leftForeArm",
            13: "rightForeArm",
            14: "leftUpLeg",
            15: "hips"
        }

        self.LABEL_MAPPER_FOR_BODY_PART_SEGM = {
            -1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
            11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0,
            22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0,
            33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0,  # background
            100: 1,  # rightHand
            101: 2,  # rightUpLeg
            102: 3,  # leftArm
            103: 4,  # head
            104: 4,  # head
            105: 4,  # head
            106: 5,  # leftLeg
            107: 6,  # leftFoot
            108: 6,  # leftFoot
            109: 7,  # torso
            110: 7,  # torso
            111: 7,  # torso
            112: 7,  # torso
            113: 8,  # rightFoot
            114: 9,  # rightArm
            115: 10,  # leftHand
            116: 11,  # rightLeg
            117: 1,  # rightHand
            118: 12,  # leftForeArm
            119: 13,  # rightForeArm
            120: 4,  # head
            121: 8,  # rightFoot
            122: 7,  # torso
            123: 14,  # leftUpLeg
            124: 10,  # leftHand
            125: 15,  # hips
        }

        self.create_label_database()

        for mode in self.modes:
            with open(f"{data_dir}/{mode}_list.txt") as file:
                # self.files[mode] = natsorted([f"{self.data_dir}/EgoBodyTestWBodyPart/{line.strip()}" for line in file])
                # self.files[mode] = natsorted([f"{self.data_dir}/EgoBodyTestWBodyPartTestCorrected/{line.strip()}" for line in file])
                self.files[mode] = natsorted([f"{self.data_dir}/scenes/{line.strip()}" for line in file])
                # self.files[mode] = natsorted([f"{self.data_dir}/{line.strip()}" for line in file])  # behave

    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                'color': self.color_map[class_id],
                'name': class_name,
                'validation': True
            }
        self._save_yaml(self.save_dir / "label_database.yaml", label_database)

        part_database = dict()
        #part_map = {i: part_name for i, part_name in enumerate(self.LABEL_LIST)}
        inv_label_map = {v: k for k,v in self.LABEL_MAPPER_FOR_BODY_PART_SEGM.items()}
        inv_label_map[0] = 0
        for part_id, part_name in self.LABEL_MAP.items():
            part_database[part_id] = {
                'color': self.COLOR_MAP_W_BODY_PARTS[inv_label_map[part_id]],
                'name': part_name,
                'validation': True
            }
        self._save_yaml(self.save_dir / "part_database.yaml", part_database)

    def read_plyfile(self, file_path):
        """Read ply file and return it as numpy array. Returns None if emtpy."""
        with open(file_path, 'rb') as f:
            plydata = PlyData.read(f)
        if plydata.elements:
            return pd.DataFrame(plydata.elements[0].data).values

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        # scene_name = "_".join(filepath.split("/")[-4:]).replace(".ply", "")  # behave
        # scene_name = "_".join(filepath.split("/")[-2:]).replace(".ply", "") # synthetic
        scene_name = "_".join(filepath.split("/")[-3:]).replace(".ply", "") # egobody

        filebase = {
            "filepath": filepath,
            "scene": scene_name,
            "raw_filepath": str(filepath),
        }

        # reading both files and checking that they are fitting
        pcd = self.read_plyfile(filepath)
        coords = pcd[:, :3]

        # fix rotation bug
        coords = coords[:, [0, 2, 1]]
        coords[:, 2] = -coords[:, 2]

        rgb = pcd[:, 3:6]
        instance_id = pcd[:, 6][..., None]

        if coords.shape[0] < self.min_points or np.unique(instance_id[:, 0]).shape[0] <= self.min_instances:
            return scene_name

        part_id = pcd[:, 7][..., None]

        part_id = np.array([self.LABEL_MAPPER_FOR_BODY_PART_SEGM[int(part_id[i, 0])]
                            for i in range(part_id.shape[0])])[..., None].astype(np.float32)

        # semantic_id = (instance_id > 0.).astype(np.float32)

        # no normals in the dataset
        # instance_normals = np.ones((pcd.shape[0], 3))
        # segment_id = np.ones((pcd.shape[0], 1))

        points = np.hstack((coords,
                            rgb,
                            # instance_normals,
                            # segment_id,
                            part_id,
                            instance_id))

        if np.isinf(points).sum() > 0:
            # some scenes (scene0573_01_frame_04) got nans
            return scene_name

        gt_part = part_id * 1000 + instance_id
        gt_human = (part_id > 0.) * 1000 + instance_id

        processed_filepath = self.save_dir / mode / f"{scene_name}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = self.save_dir / "gt_human" / mode / f"{scene_name}.txt"
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_human.astype(np.int32), fmt="%d")
        filebase["gt_human_filepath"] = str(processed_gt_filepath)

        processed_gt_filepath = self.save_dir / "gt_part" / mode / f"{scene_name}.txt"
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_part.astype(np.int32), fmt="%d")
        filebase["gt_part_filepath"] = str(processed_gt_filepath)

        filebase["color_mean"] = [
            float((rgb[:, 0] / 255).mean()),
            float((rgb[:, 1] / 255).mean()),
            float((rgb[:, 2] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((rgb[:, 0] / 255) ** 2).mean()),
            float(((rgb[:, 1] / 255) ** 2).mean()),
            float(((rgb[:, 2] / 255) ** 2).mean()),
        ]

        return filebase

    def joint_database(self, train_modes=("train",)):
        for mode in train_modes:
            joint_db = []
            for let_out in train_modes:
                if mode == let_out:
                    continue
                joint_db.extend(self._load_yaml(self.save_dir / (let_out + "_database.yaml")))
            self._save_yaml(self.save_dir / f"train_{mode}_database.yaml", joint_db)

    def compute_color_mean_std(
            self, train_database_path: str = "./data/processed/scannet/train_database.yaml"
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean ** 2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

if __name__ == "__main__":
    Fire(HumanSegmentationDataset)
