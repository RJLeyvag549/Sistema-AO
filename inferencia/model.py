import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(BaselineCNN, self).__init__()
        
        # Convolución 1: Entrada in_channels (PSF gris), 32 filtros 5x5, padding=2
        # Entrada: (B, in_channels, 96, 96) -> Salida: (B, 32, 96, 96)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        # Reducción: MaxPool 2x2 -> (B, 32, 48, 48)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolución 2: 32 -> 64 filtros 3x3, padding=1
        # Entrada: (B, 32, 48, 48) -> Salida: (B, 64, 48, 48)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        # Reducción: MaxPool 2x2 -> (B, 64, 24, 24)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolución 3: 64 -> 128 filtros 3x3, padding=1
        # Entrada: (B, 64, 24, 24) -> Salida: (B, 128, 24, 24)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        # Reducción: MaxPool 2x2 -> (B, 128, 12, 12)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Capas densas completamente conectadas (FC)
        # Dimensión aplanada = 12 * 12 * 128 = 18432 neuronas
        self.fc1 = nn.Linear(in_features=12 * 12 * 128, out_features=256)
        self.relu_fc = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(p=0.2)
        
        # Capa de salida: 11 neuronas correspondientes a los 11 modos de Zernike (Z1-Z11)
        self.fc2 = nn.Linear(in_features=256, out_features=11)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Aplanar tensor
        x = x.view(x.size(0), -1)
        
        # Capa lineal 1 con Dropout
        x = self.dropout(self.relu_fc(self.fc1(x)))
        
        # Predicción final
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Prueba rápida de paso directo (Forward pass)
    model = BaselineCNN()
    dummy_input = torch.randn(2, 1, 96, 96) # Batch size=2, 1 canal (gris), 96x96
    output = model(dummy_input)
    print(f"Dimensiones de entrada: {dummy_input.shape}")
    print(f"Dimensiones de salida de la CNN: {output.shape} (Esperado: [2, 11])")
