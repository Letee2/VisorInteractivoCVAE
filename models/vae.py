import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=16, input_channels=1, input_size=28, dropout_p=0.1):
        super(ConvVAE, self).__init__()
        
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
        
        # FC para mu y logvar
        self.fc_mu = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(self.flatten_size, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(self.flatten_size, latent_dim)
        )
        
        # FC para decodificar desde el espacio latente
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
        
    def encode(self, x):
        h = self.encoder(x)             # [batch_size, 128, input_size/4, input_size/4]
        h = h.view(h.size(0), -1)       # [batch_size, flatten_size]
        mu = self.fc_mu(h)              # [batch_size, latent_dim]
        logvar = self.fc_logvar(h)      # [batch_size, latent_dim]
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.fc_decoder(z)          # [batch_size, flatten_size]
        x = x.view(x.size(0), 128, self.input_size // 4, self.input_size // 4)
        out = self.decoder(x)
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z


def vae_loss_function(reconstructed, x, mu, logvar, normalize_output=False):
    """
    Función de pérdida para VAE: Reconstrucción + KL Divergence.
    
    Para imágenes en [0,1] (MNIST, Fashion-MNIST): BCE + KLD
    Para imágenes en [-1,1] (CIFAR10): MSE + KLD
    
    Args:
        reconstructed: Imágenes reconstruidas
        x: Imágenes originales
        mu: Vector de medias del espacio latente
        logvar: Vector de log-varianzas del espacio latente
        normalize_output: Si True, los datos están normalizados en [-1,1]
        
    Returns:
        Tuple: (Pérdida Total, Pérdida Reconstrucción, Pérdida KL)
    """

    
    # Reconstrucción
    if normalize_output:
        # MSE para datos normalizados a [-1,1]
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='sum')
    else:
        # BCE para datos en [0,1]
        reconstruction_loss = F.binary_cross_entropy(reconstructed, x, reduction='sum')
    
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Pérdida total = reconstrucción + KL
    total_loss = reconstruction_loss + kl_loss
    
    return total_loss, reconstruction_loss, kl_loss
