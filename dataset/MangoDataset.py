import torch
from PIL import Image
import os
from torchvision import transforms

class MangoDataset(torch.utils.data.Dataset):
    def __init__(self, directory, dataframe, input_size):
        self.directory = directory
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size), Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[-1.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        for i in range(item, len(self)):
            line = self.dataframe.iloc[i]
            try:
                img = Image.open(os.path.join(self.directory, line['image_id']), mode='r')
                inputTensor = self.transform(img)
                outputTensor = {
                    'A': 0,
                    'B': 1,
                    'C': 2
                }.get(line['label'])
                return (inputTensor, outputTensor)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        raise IndexError()

    def __len__(self):
        return len(self.dataframe)