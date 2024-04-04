import os
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class TrafficSignDataset(Dataset):
    def __init__(self, img_path: str, test_dataset: bool = False, transform=None):
        self.img_path = img_path
        self.test_dataset = test_dataset
        
        if not self.test_dataset:
            ranges = {}
            directories = [str(path) for path in Path(self.img_path).glob("*")]
            directories = sorted(directories, key=lambda x: int(x.split("/")[-1]))
            counter = 0
            for dir in directories:
                end = len(list(Path(dir).glob("*"))) + counter
                ranges[dir.split("/")[-1]] = (counter, end - 1)
                counter = end

            self.ranges = ranges
        else:
            self.ranges = None

        self.length = len([i for i in Path(self.img_path).glob("*")]) if self.test_dataset else len([i for i in Path(self.img_path).rglob("*") if i.is_file()])
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.test_dataset:
            image_path = list(Path(self.img_path).glob("*"))[idx]
            return Image.open(image_path) if self.transform is None else self.transform(Image.open(image_path)), None
        else:
            for k, v in self.ranges.items():
                if idx in range(v[0], v[1] + 1):
                    lst = list(Path(self.img_path + "/" + k).glob("*"))
                    image_path = lst[len(lst) - (v[1] - idx) - 1]
                    return Image.open(image_path) if self.transform is None else self.transform(Image.open(image_path)), int(k)


class AttackedDataset(Dataset):
    def __init__(self, img_path: str, annotation_file: str, transform=None):
        self.img_path = img_path
        self.labels = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.img_path, self.labels.iloc[idx]["image_name"])
        image = Image.open(image_path) if self.transform is None else self.transform(Image.open(image_path))
        return image, self.labels.iloc[idx]["label"]
