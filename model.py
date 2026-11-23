import torch
import torch.nn as nn

class FraudDetectionMLP(nn.Module):
    def __init__(self, input_size=30, dropout_rate=0.3):
        super(FraudDetectionMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    model = FraudDetectionMLP()
    sample_input = torch.randn(5, 30)
    output = model(sample_input)

    print("Model architecture:")
    print(model)
    print("\nOutput shape:", output.shape)
    print("Number of parameters:",
          sum(p.numel() for p in model.parameters()))