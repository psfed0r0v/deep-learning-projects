from torch.utils import data
from torchvision import transforms
from PIL import Image

import os
from os.path import join


class LoadDataset(data.Dataset):
    def __init__(self, dataroot, phase):
        self.dir_AB = join(dataroot, phase)  # get the image directory
        self.data = sorted([join(self.dir_AB, item)
                            for item in os.listdir(self.dir_AB)])  # get image paths

        self.transform_image = transforms.Compose([
            transforms.Resize(80),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        AB_path = self.data[index]
        AB = Image.open(AB_path).convert('RGB')

        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        return self.transform_image(A), self.transform_image(B)

    def __len__(self):
        return len(self.data)