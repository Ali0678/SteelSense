import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import SteelCNN

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model,loader,criterion,optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images,labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct/total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct/total

def main():
    print(f'Starting training on {DEVICE}')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'NEU_Clean')
    train_loader, val_loader, class_names = get_data_loaders(data_path, BATCH_SIZE)
    model = SteelCNN(num_classes = len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)
    best_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        val_loss, val_acc = validate(model,val_loader,criterion)
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(script_dir, '..', 'models', 'best_model.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved! ({best_acc:.2f}%)')
            print(f'Training Completed!')

if __name__ == '__main__':
    main()

