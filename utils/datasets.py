import torch
from torchvision import transforms
from pathlib import Path
from abc import abstractmethod
import affine
import numpy as np
from utils import augmentations, geofiles


class AbstractDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.root_path = Path(cfg.PATHS.DATASET)

        self.years = cfg.DATALOADER.YEARS
        self.label = cfg.DATALOADER.LABEL
        self.sensor = cfg.DATALOADER.SENSOR
        self.pos_class = cfg.DATALOADER.POS_CLASS
        available_s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        self.s2_band_indices = self._get_indices(available_s2_bands, cfg.DATALOADER.S2_BANDS)
        available_l8_bands = ['B2', 'B3', 'B4', 'B5']
        self.l8_band_indices = self._get_indices(available_l8_bands, cfg.DATALOADER.L8_BANDS)
        self.patch_size = cfg.AUGMENTATION.CROP_SIZE
        if self.sensor == 'sentinel2':
            self.n_features = len(self.s2_band_indices)
        elif self.sensor == 'landsat8':
            self.n_features = len(self.l8_band_indices)
        else:
            raise Exception('Unkown sensor.')

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def load_sentinel2_data(self, site: str, year: int, patch_id: str):
        file = self.root_path / site / 'sentinel2' / f'sentinel2_{site}{year}_{patch_id}.tif'
        img, *_ = geofiles.read_tif(file)
        img = img[:, :, self.s2_band_indices]
        return img.astype(np.float32)

    def load_landsat8_data(self, site: str, year: int, patch_id: str):
        file = self.root_path / site / 'landsat8' / f'landsat8_{site}{year}_{patch_id}.tif'
        img, *_ = geofiles.read_tif(file)
        img = img[:, :, self.l8_band_indices]
        return img.astype(np.float32)

    def load_satellite_data(self, site: str, year: int, patch_id: str):
        if self.sensor == 'sentinel2':
            img = self.load_sentinel2_data(site, year, patch_id)
        elif self.sensor == 'landsat8':
            img = self.load_landsat8_data(site, year, patch_id)
        else:
            raise Exception('Unkown sensor')
        return img

    def load_label(self, site, year, patch_id):
        raw_label_file = self.root_path / site / f'{self.label}{year}' / f'{self.label}_{site}{year}_{patch_id}.tif'
        raw_label, *_ = geofiles.read_tif(raw_label_file)
        label = np.isin(raw_label, self.pos_class)
        return label.astype(np.float32)

    def load_change_label(self, site, patch_id):
        label_file = self.root_path / site / f'{self.label}' / f'{self.label}_{site}_{patch_id}.tif'
        label, *_ = geofiles.read_tif(label_file)
        return label.astype(np.float32)

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]


# dataset for urban extraction with building footprints
class TrainDataset(AbstractDataset):

    def __init__(self, cfg, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg)

        self.run_type = run_type
        if run_type == 'train':
            self.sites = list(cfg.DATASET.TRAIN)
        elif run_type == 'val':
            self.sites = list(cfg.DATASET.VAL)
        else:
            raise Exception('unkown run type!')

        self.no_augmentations = no_augmentations
        if no_augmentations:
            self.transform = transforms.Compose([augmentations.Numpy2Torch()])
        else:
            self.transform = augmentations.compose_transformations(cfg)

        self.samples = []
        for site in self.sites:
            site_samples_file = self.root_path / site / f'{site}_samples.json'
            site_samples = geofiles.load_json(site_samples_file)
            for sample in site_samples:
                for year in self.years:
                    sample['year'] = year
                    self.samples.append(dict(sample))

        self.length = len(self.samples)
        self.crop_size = cfg.AUGMENTATION.CROP_SIZE

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        patch_id = sample['patch_id']
        site = sample['site']
        year = sample['year']

        img = self.load_satellite_data(site, 2013 if year == 2012 else year, patch_id)
        label = self.load_label(site, year, patch_id)

        img, label = self.transform((img, label))
        item = {
            'x': img,
            'y': label,
            'site': site,
            'year': year,
            'patch_id': patch_id,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'


# dataset for classifying a scene
class TilesInferenceDataset(AbstractDataset):

    def __init__(self, cfg, site: str, year: int, include_label: bool = False):
        super().__init__(cfg)

        self.site = site
        self.year = year
        self.include_label = include_label

        self.transform = transforms.Compose([augmentations.Numpy2Torch()])

        # getting all files
        samples_file = self.root_path / site / f'{site}_samples.json'
        self.samples = geofiles.load_json(samples_file)
        self.length = len(self.samples)

        # computing extent
        patch_ids = [s['patch_id'] for s in self.samples]
        self.coords = [[int(c) for c in patch_id.split('-')] for patch_id in patch_ids]
        self.max_y = max([c[0] for c in self.coords])
        self.max_x = max([c[1] for c in self.coords])

    def __getitem__(self, index):

        sample = self.samples[index]
        patch_id_center = sample['patch_id']

        y_center, x_center = patch_id_center.split('-')
        y_center, x_center = int(y_center), int(x_center)

        extended_patch = np.zeros((3 * self.patch_size, 3 * self.patch_size, self.n_features), dtype=np.float32)

        for i in range(3):
            for j in range(3):
                y = y_center + (i - 1) * self.patch_size
                x = x_center + (j - 1) * self.patch_size
                patch_id = f'{y:010d}-{x:010d}'
                if self._is_valid_patch_id(patch_id):
                    patch = self.load_satellite_data(self.site, 2013 if self.year == 2012 else self.year, patch_id)
                else:
                    patch = np.zeros((self.patch_size, self.patch_size, self.n_features), dtype=np.float32)
                i_start = i * self.patch_size
                i_end = (i + 1) * self.patch_size
                j_start = j * self.patch_size
                j_end = (j + 1) * self.patch_size
                extended_patch[i_start:i_end, j_start:j_end, :] = patch

        if self.include_label:
            label = self.load_label(self.site, self.year, patch_id_center)
        else:
            label = np.zeros((self.patch_size, self.patch_size))
        extended_patch, label = self.transform((extended_patch, label))

        item = {
            'x': extended_patch,
            'y': label,
            'i': y_center,
            'j': x_center,
            'site': self.site,
            'patch_id': patch_id_center,
        }

        return item

    def _is_valid_patch_id(self, patch_id):
        patch_ids = [s['patch_id'] for s in self.samples]
        return True if patch_id in patch_ids else False

    def get_arr(self, dtype=np.uint8):
        height = self.max_y + self.patch_size
        width = self.max_x + self.patch_size
        return np.zeros((height, width, 1), dtype=dtype)

    def get_geo(self):
        patch_id = f'{0:010d}-{0:010d}'
        # in training and validation set patches with no BUA were not downloaded -> top left patch may not be available
        assert(self._is_valid_patch_id(patch_id))
        file = self.root_path / self.site / 'sentinel2' / f'sentinel2_{self.site}2018_{patch_id}.tif'
        _, transform, crs = geofiles.read_tif(file)
        return transform, crs

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'


class TrainChangeDataset(AbstractDataset):
    def __init__(self, cfg, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg)

        self.run_type = run_type
        if run_type == 'train':
            self.sites = list(cfg.DATASET.TRAIN)
        elif run_type == 'val':
            self.sites = list(cfg.DATASET.VAL)
        else:
            raise Exception('unkown run type!')

        self.no_augmentations = no_augmentations
        if no_augmentations:
            self.transform = transforms.Compose([augmentations.Numpy2Torch()])
        else:
            self.transform = augmentations.compose_transformations(cfg)

        self.samples = []
        for site in self.sites:
            site_samples_file = self.root_path / site / f'uac1218_{site}_samples.json'
            site_samples = geofiles.load_json(site_samples_file)
            self.samples.extend(site_samples)

        self.length = len(self.samples)
        self.crop_size = cfg.AUGMENTATION.CROP_SIZE

    def __getitem__(self, index):
        # loading metadata of sample
        sample = self.samples[index]
        patch_id = sample['patch_id']
        site = sample['site']

        img_t1 = self.load_landsat8_data(site, 2013, patch_id)
        img_t2 = self.load_sentinel2_data(site, 2018, patch_id)
        label = self.load_change_label(site, patch_id)

        imgs = np.concatenate((img_t1, img_t2), axis=-1)
        imgs, label = self.transform((imgs, label))

        img_t1, img_t2 = imgs[:len(self.cfg.DATALOADER.S2_BANDS)], imgs[len(self.cfg.DATALOADER.S2_BANDS):]

        item = {
            'x_t1': img_t1,
            'x_t2': img_t2,
            'y': label,
            'site': site,
            'patch_id': patch_id,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'