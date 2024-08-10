import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.feature_extractor = nn.Sequential(
            # Convolutional block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 112x112x64
            
            # Convolutional block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 56x56x128
            
            # Convolutional block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 28x28x256
            
            # Convolutional block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 14x14x512
            
            # Convolutional block 4
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 7x7x1024
            
            # Convolutional block 5
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2048),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 3x3x2048
            
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048 * 3 * 3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        # self.fc1 = nn.Linear(2048 * 3 * 3, 4096)
        # self.fc2 = nn.Linear(4096, 2048)
        # self.fc3 = nn.Linear(2048, 512)
        # self.fc4 = nn.Linear(512, num_classes)
        
        # Fully connected layers - V2
        # self.fc1 = nn.Linear(2048 * 3 * 3, 6144)
        # self.fc2 = nn.Linear(6144, 2048)
        # self.fc3 = nn.Linear(2048, 512)
        # self.fc4 = nn.Linear(512, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        # Pass the input through the feature extractor
        
        x = self.feature_extractor(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Pass through the fully connected layers with dropout
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # FC - V2
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))
        # x = self.fc4(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter) # dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
