# Traffic Light Detection

A PyTorch-based deep learning project for detecting and classifying traffic lights in images from [BOSCH Small Traffic Light Database](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset). The model uses Faster R-CNN with a ResNet-50 backbone to identify traffic lights and classify them as Red, Yellow, or Green.

## Features

- Traffic light detection and classification
- Support for RGB images
- Custom dataset handling for traffic light annotations
- Data augmentation pipeline
- Training and evaluation scripts
- Support for both CPU and GPU training

## Project Structure

```
traffic_light_detection/
├── data/
│   ├── dataset_train_rgb/
│   │   ├── train.yaml
│   │   └── rgb/
│   │       └── train/
│   │           └── [image files]
│   └── dataset_test_rgb/
│       ├── test.yaml
│       └── rgb/
│           └── test/
│               └── [image files]
├── dataset.py
├── model.py
├── train.py
├── transforms.py
└── main.py
```

## Requirements

- Python 3.9+
- PyTorch 1.9+
- torchvision
- PIL (Pillow)
- PyYAML
- numpy<2

Install requirements using:
```bash
pip install torch torchvision pillow pyyaml numpy<2
```

## Dataset Format

The dataset should be organized in YAML format with the following structure:

```yaml
- boxes:
  - {label: Green, occluded: false, x_max: 1041.87, x_min: 1014.18,
     y_max: 243.92, y_min: 177.83}
  - {label: Green, occluded: false, x_max: 604.33, x_min: 573.58,
     y_max: 96.71, y_min: 27.45}
  path: ./rgb/train/image1.png
- boxes: []
  path: ./rgb/train/image2.png
```

- Each entry contains an image path and associated bounding boxes
- Boxes can have labels: Red, Yellow, Green, or Off
- Coordinates are in pixel values
- Images should be in RGB format

## Usage

1. **Prepare Your Data**
   - Organize your data following the project structure above
   - Ensure your YAML files follow the required format
   - Place images in the appropriate directories

2. **Training the Model**
   ```python
   python main.py
   ```
   This will:
   - Load the training and test datasets
   - Initialize the Faster R-CNN model
   - Train for the specified number of epochs
   - Save checkpoints after each epoch

3. **Configuration**
   You can modify the following parameters in `main.py`:
   ```python
   num_classes = 5  # background + 4 traffic light states
   num_epochs = 10
   batch_size = 4
   learning_rate = 0.005
   ```

## Model Architecture

- Base: Faster R-CNN
- Backbone: ResNet-50 with FPN
- Classes: Background, Red, Yellow, Green
- Input size: Flexible (default image size: 1280x720)

## Data Augmentation

The following augmentations are applied during training:
- Random horizontal flip (50% probability)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization using ImageNet statistics

## Model Outputs

The model outputs:
- Bounding box coordinates (normalized)
- Class labels
- Confidence scores for each detection

## Training Process

- The model saves checkpoints after each epoch
- Checkpoints include:
  - Model state
  - Optimizer state
  - Training and validation losses
  - Current epoch

## Performance Monitoring

During training, the following metrics are displayed:
```
Epoch [1/10]
Train Loss: 1.2345
Test Loss: 1.3456
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is released under the MIT License.

## Acknowledgments

- PyTorch team for the base Faster R-CNN implementation
- torchvision for the pre-trained models