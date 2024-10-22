"""
Custom dataset class that handles loading and preprocessing
"""
import os
import yaml
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TrafficLightDataset(Dataset):
    """
    Custom dataset class that handles loading and preprocessing
    """
    def __init__(self, yaml_file, transform=None):
        self.yaml_file = yaml_file
        self.transform = transform
        self.base_path = os.path.dirname(yaml_file)
        
        # Load YAML data
        with open(yaml_file, 'r') as f:
            self.data = yaml.safe_load(f)
            
        # Create label mapping
        self.label_map = LABEL_MAP

    def __len__(self):
        return len(self.data)
    
    # def validate_box(self, box, w, h):
    #     """Validate and fix bounding box coordinates."""
    #     x_min = max(0, min(box['x_min'], box['x_max']))
    #     x_max = min(w, max(box['x_min'], box['x_max']))
    #     y_min = max(0, min(box['y_min'], box['y_max']))
    #     y_max = min(h, max(box['y_min'], box['y_max']))
        
    #     # Ensure minimum size (1 pixel)
    #     if x_max - x_min < 1:
    #         x_max = x_min + 1
    #     if y_max - y_min < 1:
    #         y_max = y_min + 1
            
    #     return x_min, y_min, x_max, y_max
    
    def __getitem__(self, idx, clip = True):
        # Load image
        img_data = self.data[idx]
        img_path = os.path.join(self.base_path, img_data['path'].lstrip('./'))
        # import pdb; pdb.set_trace()
        image = Image.open(img_path).convert('RGB')
        
        for j, box in enumerate(img_data['boxes']):
            if box['x_min'] > box['x_max']:
                img_data['boxes'][j]['x_min'], img_data['boxes'][j]['x_max'] = (
                    img_data['boxes'][j]['x_max'], img_data['boxes'][j]['x_min'])
            if box['y_min'] > box['y_max']:
                img_data['boxes'][j]['y_min'], img_data['boxes'][j]['y_max'] = (
                    img_data['boxes'][j]['y_max'], img_data['boxes'][j]['y_min'])
        
        # Get original dimensions
        w, h = image.size
        
        if clip:
            for j, box in enumerate(img_data['boxes']):
                img_data['boxes'][j]['x_min'] = max(min(box['x_min'], w - 1), 0)
                img_data['boxes'][j]['x_max'] = max(min(box['x_max'], w - 1), 0)
                img_data['boxes'][j]['y_min'] = max(min(box['y_min'], h - 1), 0)
                img_data['boxes'][j]['y_max'] = max(min(box['y_max'], h - 1), 0)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        # Process boxes
        boxes = []
        labels = []
        
        if 'boxes' in img_data and img_data['boxes']:
            for box in img_data['boxes']:
                # Validate and fix box coordinates
                # x_min, y_min, x_max, y_max = self.validate_box(box, w, h)
                
                # Skip invalid boxes
                # if x_max <= x_min or y_max <= y_min:
                    # continue

                # Normalize coordinates to [0, 1]
                x_min = box['x_min'] / w
                y_min = box['y_min'] / h
                x_max = box['x_max'] / w
                y_max = box['y_max'] / h
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(self.label_map[SIMPLIFIED_CLASSES[box['label']]])
        
        # Ensure we have at least one box (add dummy box if needed)
        if not boxes:
            boxes = [[0, 0, 1, 1]]  # Dummy box covering whole image
            labels = [0]  # Background class
        
        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        return image, target
    
SIMPLIFIED_CLASSES = {
    'Green': 'Green',
    'GreenLeft': 'Green',
    'GreenRight': 'Green',
    'GreenStraight': 'Green',
    'GreenStraightRight': 'Green',
    'GreenStraightLeft': 'Green',
    'Yellow': 'Yellow',
    'Red': 'Red',
    'RedLeft': 'Red',
    'RedRight': 'Red',
    'RedStraight': 'Red',
    'RedStraightLeft': 'Red',
    'Off': 'Off',
    'off': 'Off',
}

LABEL_MAP = {
    'Green': 1,
    'Yellow': 2,
    'Red': 3,
    'Off': 4}