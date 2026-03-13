import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size = 32, train_split = 0.8):
    transform = transforms.Compose([
        transforms.Resize((200,200)),
        transforms.ToTensor(),
    ])
    full_dataset = datasets.ImageFolder(root = data_dir, transform = transform)
    class_names = full_dataset.classes

    print(f'Loaded {len(full_dataset)} images from {len(class_names)} classes.')
    print(f'Classes: {class_names}')

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f'Training Set: {len(train_dataset)} images')
    print(f'Validation Set: {len(val_dataset)} images')

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset,batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, class_names

if __name__ == "__main__":
    data_path = '../data/NEU_Clean'
    if os.path.exists(data_path):
        tr,val,classes = get_data_loaders(data_path)
        images,labels = next(iter(tr))
        print(f"Batch shape: {images.shape}")
    else:
        print(f'Error: Path {data_path} not found.')

