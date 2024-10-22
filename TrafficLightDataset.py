from torch.utils.data import Dataset
import torch
from PIL import Image


class TrafficLightDataset(Dataset):
    def __init__(self, label_data, root_dir, transforms = None, label_map = None):
        #super().__init__()
        self.label_data = label_data
        self.root_dir = root_dir
        self.transforms = transforms
        self.label_map = label_map

    def __len__(self):
        return len(self.label_data)
    
    def __getitem__(self, index: int):
        img_path = self.label_data[index]['image']
        label = self.label_data[index]['label']
        
        image = Image.open(img_path).convert("RGB")

        # if img_info['boxes']:
        #     for box in img_info['boxes']:
        #         x_min = box['x_min']
        #         y_min = box['y_min']
        #         x_max = box['x_max']
        #         y_max = box['y_max']
                
        #         boxes.append([x_min, y_min, x_max, y_max])
        #         labels.append(self.label_map[box['label']])

        # image = torch.tensor(image, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        sample = {
            'image': image,
            'label': label
        }

        return sample