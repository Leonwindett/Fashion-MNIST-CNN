
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_MNIST_data(batch_size, num_workers = 2, random_seed = torch.Generator().manual_seed(18)):
    
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Load the training and test datasets
    train_dataset = datasets.FashionMNIST(root='../data', train=True,
                                          download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='../data', train=False,
                                         download=True, transform=transform)
    
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, lengths = [0.8, 0.2], generator = random_seed)


    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=random_seed)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=random_seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=random_seed)

    return train_loader, validation_loader, test_loader