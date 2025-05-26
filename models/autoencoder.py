import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, input_channels=1, input_size=28, dropout_p=0.1):
        super(ConvAutoencoder, self).__init__()
        
        # Calcular dimensiones
        self.input_channels = input_channels
        self.input_size = input_size
        
        # Calcular tamaño de feature map después de convolutiones y pooling
        fm_size = input_size // 4  # Después de dos max pooling 2x2
        
        # Encoder
        self.encoder = nn.Sequential(
            # Conv1: input_channels x input_size x input_size -> 32 x input_size x input_size
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # MaxPooling: 32 x input_size x input_size -> 32 x (input_size/2) x (input_size/2)
            nn.MaxPool2d(2, stride=2),

            # Conv2: 32 x (input_size/2) x (input_size/2) -> 64 x (input_size/2) x (input_size/2)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # MaxPooling: 64 x (input_size/2) x (input_size/2) -> 64 x (input_size/4) x (input_size/4)
            nn.MaxPool2d(2, stride=2),

            # Conv3: 64 x (input_size/4) x (input_size/4) -> 128 x (input_size/4) x (input_size/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Calcular el tamaño plano del feature map para la capa fully connected
        self.flatten_size = 128 * fm_size * fm_size
        
        # Para aplanar y proyectar a latente
        self.fc_encoder = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(self.flatten_size, latent_dim)
        )
        
        # Para proyectar desde latente
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_size),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # ConvTranspose1: 128 x (input_size/4) x (input_size/4) -> 64 x (input_size/2) x (input_size/2)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # ConvTranspose2: 64 x (input_size/2) x (input_size/2) -> 32 x input_size x input_size
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Conv final: 32 x input_size x input_size -> input_channels x input_size x input_size
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Codificar
        encoded = self.encoder(x)
        # Aplanar
        encoded = encoded.view(encoded.size(0), -1)
        # Proyectar a latente
        z = self.fc_encoder(encoded)
        
        # Decodificar
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 128, self.input_size // 4, self.input_size // 4)
        reconstructed = self.decoder(x)
        
        return reconstructed, z
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_encoder(x)  # produce vector latente
        return z
    
    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 128, self.input_size // 4, self.input_size // 4)
        out = self.decoder(x)
        return out


def ae_loss_function(reconstructed, x, normalize_output=False):
    """
    Función de pérdida para el autoencoder.
    
    Para imágenes en [0,1] (MNIST, Fashion-MNIST): BCE estándar
    Para imágenes en [-1,1] (CIFAR10): MSE o BCE con ajuste
    
    Args:
        reconstructed: Imágenes reconstruidas por el modelo
        x: Imágenes originales
        normalize_output: Si True, los datos están normalizados en [-1,1]
        
    Returns:
        Pérdida total
    """
    
    if normalize_output:
        # Para datos normalizados a [-1,1], usamos MSE que funciona mejor
        return F.mse_loss(reconstructed, x, reduction='sum')
    else:
        # Para datos en [0,1], usamos BCE tradicional
        return F.binary_cross_entropy(reconstructed, x, reduction='sum')
