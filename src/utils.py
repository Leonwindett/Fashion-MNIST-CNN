
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset



class AugmentedFashionMNIST(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __len__(self):
        return len(self.dataset) * 6  # each image will generate 6 variants

    def __getitem__(self, idx):
        # original image index and augmentation index
        orig_idx = idx // 6
        aug_idx = idx % 6

        img, label = self.dataset[orig_idx]

        if aug_idx == 0:
            img = img  # original
        elif aug_idx == 1:
            img = img.rotate(90)
        elif aug_idx == 2:
            img = img.rotate(180)
        elif aug_idx == 3:
            img = img.rotate(270)
        elif aug_idx == 4:
            img = transforms.functional.hflip(img)
        elif aug_idx == 5:
            img = transforms.functional.vflip(img)

        # convert to tensor and normalize
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.5,), (0.5,))(img)

        return img, label



def load_MNIST_data(batch_size, num_workers = 2, random_seed = torch.Generator().manual_seed(18)):
    
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    

    # Load the training and test datasets
    train_dataset = datasets.FashionMNIST(root='../data', train=True,
                                          download=True)
    test_dataset = datasets.FashionMNIST(root='../data', train=False,
                                         download=True, transform=transform)
    
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, lengths = [0.8, 0.2], generator = random_seed)

    train_dataset = AugmentedFashionMNIST(train_dataset)

    
    validation_dataset = [(transform(img), label) for img, label in validation_dataset]


    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=random_seed)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=random_seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=random_seed)

    return train_loader, validation_loader, test_loader