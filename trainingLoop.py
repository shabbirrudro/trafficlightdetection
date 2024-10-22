from TrafficLightDataset import TrafficLightDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from label_reader import get_all_labels


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

yaml_data = get_all_labels("data/dataset_train_rgb/train.yaml")
root_dir = "data/dataset_train_rgb"

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

label_map = {
    'Green': 1,
    'Yellow': 2,
    'Red' : 3,
    'Off': 4
}

for item in yaml_data:
    for box in item['boxes']:
        box['label'] = SIMPLIFIED_CLASSES[box['label']]

# print(yaml_data[7])

label_data = []

for item in yaml_data:
    if item['boxes'] is not []:
        path = item['path']
        for box in item['boxes']:
            label_data.append({'image': path, 'label': box['label']})

# print(label_data[0:9])
# import pdb; pdb.set_trace()

trafficLightDataset = TrafficLightDataset(
    label_data = label_data,
    root_dir = root_dir,
    transforms = train_transforms,
    label_map = label_map
)

train_loader = DataLoader(trafficLightDataset, batch_size=32, shuffle=True, collate_fn=None)

WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(weights = WEIGHTS)

NUM_CLASSES = 5
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Starting in ", 'cuda' if torch.cuda.is_available() else 'cpu', "mode")
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        # import pdb; pdb.set_trace()
        images = list(image.to(device) for image in batch['image'])
        targets = []
        for i in range(len(images)):
            targets.append({
                'image' : batch['image'][i],
                'label': batch['label'][i],
            })

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        running_loss += losses.item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss: .4f}")

