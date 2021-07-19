import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToTensor


class MyDataset:
    def __init__(self, images, row_splits, col_splits):
        self.images = images
        self.row_splits = row_splits
        self.col_splits = col_splits
        self.transforms = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            lambda x: x[:3] - 0.5,
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # )
        ])

    @staticmethod
    def read_csv(info_path):
        df = pd.read_csv(info_path)
        images = df['img_file'].tolist()
        label_files = df['row_header_split_file'].tolist()
        row_splits, col_splits = [], []
        for file in label_files:
            with open(file, 'r') as f:
                row_split, col_split = f.readline().split(',')
                row_splits.append(int(row_split))
                col_splits.append(int(col_split))
        return MyDataset(images, row_splits, col_splits)

    def train_test_split(self, val_frac: float = 0.2):
        pos = int(len(self) * (1 - val_frac))
        return (
            MyDataset(
                self.images[:pos], self.row_splits[:pos], self.col_splits[:pos]
            ),
            MyDataset(
                self.images[pos:], self.row_splits[pos:], self.col_splits[pos:]
            )
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        width, height = image.size
        return {
            'image': self.transforms(image),
            'row_split': self.row_splits[item] / height,
            'col_split': self.col_splits[item] / width
        }
