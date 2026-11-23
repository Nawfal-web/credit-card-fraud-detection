
import torch
import torch.nn as nn


class FraudDetectionCNN(nn.Module):

    def __init__(self, input_size=30, dropout_rate=0.3):
        super(FraudDetectionCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            # First convolutional block
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Second convolutional block
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Convert (batch_size, 30) to (batch_size, 1, 30)
        x = x.unsqueeze(1)

        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = FraudDetectionCNN()

    sample_input = torch.randn(5, 30)
    output = model(sample_input)

    print("CNN architecture:")
    print(model)

    print("\nInput shape:", sample_input.shape)
    print("Output shape:", output.shape)

    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters())
    )
