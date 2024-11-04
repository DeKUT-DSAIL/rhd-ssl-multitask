import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class SimCLR(nn.Module):
    def __init__(self, output_dim=512, projection_dim=128):
        super(SimCLR, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification layer
        self.fc = nn.Linear(512, output_dim)
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        h = self.fc(h)
        z = self.projector(h)
        return h, z

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        # If features is a tuple (from SimCLR), take only the first element (h)
        if isinstance(features, tuple):
            features = features[0]
        # Ensure features are flattened
        features = features.view(features.size(0), -1)
        return self.fc(features)