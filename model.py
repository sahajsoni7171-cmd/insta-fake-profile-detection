import torch.nn as nn

class InstaFakeDetector(nn.Module):
    def __init__(self, input_dim):
        super(InstaFakeDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 2 output classes: fake (1) or real (0)
        )

    def forward(self, x):
        return self.model(x)
