from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os


class ImageData(Dataset):
    def __init__(self, data, path, is_test=False):
        super().__init__()
        self.data = data.values
        self.path = path
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        name, label = self.data[index]
        image = self._augmentation(os.path.join(self.path, name))

        return image, label

    def _augmentation(self, img):
        img = Image.open(img).convert('RGB')
        if self.is_test:
            transform_image = transforms.Compose([
                transforms.Resize(226),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5624, 0.5389, 0.4882], std=[0.1867, 0.1848, 0.1896]),
            ])
        else:
            transform_image = transforms.Compose([
                transforms.Resize(226),
                transforms.RandomApply([
                    transforms.RandomRotation(degrees=30, expand=False),
                    transforms.RandomRotation(degrees=15, expand=False),
                    transforms.RandomRotation(degrees=10, expand=False)],
                    p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5624, 0.5389, 0.4882], std=[0.1867, 0.1848, 0.1896]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False),

            ])

        return transform_image(img)
