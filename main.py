import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os

from dataset import TrafficLightDataset
from model import get_model
from train import train_one_epoch, evaluate

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    num_classes = 5  # background + 4 traffic light states
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.005
    
    # Data transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TrafficLightDataset(
        yaml_file='./data/dataset_train_rgb/train.yaml',
        transform=transform
    )
    
    test_dataset = TrafficLightDataset(
        yaml_file='./data/dataset_test_rgb/test.yaml',
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize the model
    model = get_model(num_classes)
    model = model.to(device)
    
    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        test_loss = evaluate(model, test_loader, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss
        }
        
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()