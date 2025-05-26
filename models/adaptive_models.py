import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, input_channels=1, input_size=28, dropout_p=0.1, complex_architecture=False):
        super(ConvAutoencoder, self).__init__()
        
        # Calcular dimensiones
        self.input_channels = input_channels
        self.input_size = input_size
        self.complex_architecture = complex_architecture
        
        # El parámetro complex_architecture determina si usar la arquitectura simple o compleja
        if complex_architecture:
            # Arquitectura compleja para datasets como CIFAR10
            
            # Calcular tamaño de feature map después de convolutiones y pooling
            self.fm_size = input_size // 8  # Después de tres max pooling 2x2
            self.output_channels = 512  # Número de canales de salida del encoder
            
            # Encoder con más filtros y una capa adicional
            self.encoder = nn.Sequential(
                # Conv1: input_channels x input_size x input_size -> 64 x input_size x input_size
                nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                # MaxPooling: 64 x input_size x input_size -> 64 x (input_size/2) x (input_size/2)
                nn.MaxPool2d(2, stride=2),

                # Conv2: 64 x (input_size/2) x (input_size/2) -> 128 x (input_size/2) x (input_size/2)
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # MaxPooling: 128 x (input_size/2) x (input_size/2) -> 128 x (input_size/4) x (input_size/4)
                nn.MaxPool2d(2, stride=2),

                # Conv3: 128 x (input_size/4) x (input_size/4) -> 256 x (input_size/4) x (input_size/4)
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # MaxPooling: 256 x (input_size/4) x (input_size/4) -> 256 x (input_size/8) x (input_size/8)
                nn.MaxPool2d(2, stride=2),
                
                # Conv4: 256 x (input_size/8) x (input_size/8) -> 512 x (input_size/8) x (input_size/8)
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )
            
            # Calcular el tamaño plano del feature map para la capa fully connected
            self.flatten_size = 512 * self.fm_size * self.fm_size
            
            # Para aplanar y proyectar a latente
            self.fc_encoder = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(self.flatten_size, latent_dim)
            )
            
            # Para proyectar desde latente
            self.fc_decoder = nn.Sequential(
                nn.Linear(latent_dim, self.flatten_size),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
            # Decoder con más filtros y una capa adicional
            self.decoder = nn.Sequential(
                # Conv inicial: 512 x (input_size/8) x (input_size/8) -> 256 x (input_size/8) x (input_size/8)
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                
                # ConvTranspose1: 256 x (input_size/8) x (input_size/8) -> 128 x (input_size/4) x (input_size/4)
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                # ConvTranspose2: 128 x (input_size/4) x (input_size/4) -> 64 x (input_size/2) x (input_size/2)
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),

                # ConvTranspose3: 64 x (input_size/2) x (input_size/2) -> 32 x input_size x input_size
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),

                # Conv final: 32 x input_size x input_size -> input_channels x input_size x input_size
                nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh()  # Tanh para rango [-1, 1] o Sigmoid para [0, 1]
            )
            
        else:
            # Arquitectura original/simple para datasets como MNIST
            
            # Calcular tamaño de feature map después de convolutiones y pooling
            self.fm_size = input_size // 4  # Después de dos max pooling 2x2
            self.output_channels = 128  # Número de canales de salida del encoder
            
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
            self.flatten_size = 128 * self.fm_size * self.fm_size
            
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
                nn.Sigmoid()  # Sigmoid para datasets como MNIST [0,1]
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
        # Usar las dimensiones correctas según la arquitectura
        x = x.view(x.size(0), self.output_channels, self.fm_size, self.fm_size)
        reconstructed = self.decoder(x)
        
        return reconstructed, z
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_encoder(x)  # produce vector latente
        return z
    
    def decode(self, z):
        x = self.fc_decoder(z)
        # Usar las dimensiones correctas según la arquitectura
        x = x.view(x.size(0), self.output_channels, self.fm_size, self.fm_size)
        out = self.decoder(x)
        return out


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=16, input_channels=1, input_size=28, dropout_p=0.1, complex_architecture=False):
        super(ConvVAE, self).__init__()
        
        # Calcular dimensiones
        self.input_channels = input_channels
        self.input_size = input_size
        self.complex_architecture = complex_architecture
        
        # El parámetro complex_architecture determina si usar la arquitectura simple o compleja
        if complex_architecture:
            # Arquitectura compleja para datasets como CIFAR10
            
            # Calcular tamaño de feature map después de convolutiones y pooling
            self.fm_size = input_size // 8  # Después de tres max pooling 2x2
            self.output_channels = 512  # Número de canales de salida del encoder
            
            # Encoder con más filtros y una capa adicional
            self.encoder = nn.Sequential(
                # Conv1: input_channels x input_size x input_size -> 64 x input_size x input_size
                nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                # MaxPooling: 64 x input_size x input_size -> 64 x (input_size/2) x (input_size/2)
                nn.MaxPool2d(2, stride=2),

                # Conv2: 64 x (input_size/2) x (input_size/2) -> 128 x (input_size/2) x (input_size/2)
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # MaxPooling: 128 x (input_size/2) x (input_size/2) -> 128 x (input_size/4) x (input_size/4)
                nn.MaxPool2d(2, stride=2),

                # Conv3: 128 x (input_size/4) x (input_size/4) -> 256 x (input_size/4) x (input_size/4)
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # MaxPooling: 256 x (input_size/4) x (input_size/4) -> 256 x (input_size/8) x (input_size/8)
                nn.MaxPool2d(2, stride=2),
                
                # Conv4: 256 x (input_size/8) x (input_size/8) -> 512 x (input_size/8) x (input_size/8)
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )
            
            # Calcular el tamaño plano del feature map para la capa fully connected
            self.flatten_size = 512 * self.fm_size * self.fm_size
            
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
                nn.LeakyReLU(0.2, inplace=True)
            )
            
            # Decoder con más filtros y una capa adicional
            self.decoder = nn.Sequential(
                # Conv inicial: 512 x (input_size/8) x (input_size/8) -> 256 x (input_size/8) x (input_size/8)
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                
                # ConvTranspose1: 256 x (input_size/8) x (input_size/8) -> 128 x (input_size/4) x (input_size/4)
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                # ConvTranspose2: 128 x (input_size/4) x (input_size/4) -> 64 x (input_size/2) x (input_size/2)
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),

                # ConvTranspose3: 64 x (input_size/2) x (input_size/2) -> 32 x input_size x input_size
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),

                # Conv final: 32 x input_size x input_size -> input_channels x input_size x input_size
                nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh()  # Tanh para rango [-1, 1]
            )
            
        else:
            # Arquitectura original/simple para datasets como MNIST
            
            # Calcular tamaño de feature map después de convolutiones y pooling
            self.fm_size = input_size // 4  # Después de dos max pooling 2x2
            self.output_channels = 128  # Número de canales de salida del encoder
            
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
            self.flatten_size = 128 * self.fm_size * self.fm_size
            
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
                nn.Sigmoid()  # Sigmoid para datasets como MNIST [0,1]
            )
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.fc_decoder(z)
        # Usar las dimensiones correctas según la arquitectura
        x = x.view(x.size(0), self.output_channels, self.fm_size, self.fm_size)
        out = self.decoder(x)
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z


# Función auxiliar para determinar si usar arquitectura compleja
def should_use_complex_architecture(dataset_name, input_channels, input_size):
    """
    Determina si se debe usar la arquitectura compleja
    
    Args:
        dataset_name: Nombre del dataset
        input_channels: Número de canales de la imagen
        input_size: Tamaño de la imagen
        
    Returns:
        bool: True si se debe usar arquitectura compleja, False en caso contrario
    """
    return (dataset_name.lower() == "cifar10" or
            (input_channels == 3 and input_size >= 32))


# Función para crear un autoencoder adaptado al dataset
def create_autoencoder(latent_dim=16, input_channels=1, input_size=28, dropout_p=0.1, dataset_name="mnist"):
    """
    Crea un autoencoder adaptado al dataset
    
    Args:
        latent_dim: Dimensión del espacio latente
        input_channels: Número de canales de entrada
        input_size: Tamaño de la imagen
        dropout_p: Probabilidad de dropout
        dataset_name: Nombre del dataset
        
    Returns:
        ConvAutoencoder: Modelo adaptado al dataset
    """
    complex_arch = should_use_complex_architecture(dataset_name, input_channels, input_size)
    
    return ConvAutoencoder(
        latent_dim=latent_dim,
        input_channels=input_channels,
        input_size=input_size,
        dropout_p=dropout_p,
        complex_architecture=complex_arch
    )


# Función para crear un VAE adaptado al dataset
def create_vae(latent_dim=16, input_channels=1, input_size=28, dropout_p=0.1, dataset_name="mnist"):
    """
    Crea un VAE adaptado al dataset
    
    Args:
        latent_dim: Dimensión del espacio latente
        input_channels: Número de canales de entrada
        input_size: Tamaño de la imagen
        dropout_p: Probabilidad de dropout
        dataset_name: Nombre del dataset
        
    Returns:
        ConvVAE: Modelo adaptado al dataset
    """
    complex_arch = should_use_complex_architecture(dataset_name, input_channels, input_size)
    
    return ConvVAE(
        latent_dim=latent_dim,
        input_channels=input_channels,
        input_size=input_size,
        dropout_p=dropout_p,
        complex_architecture=complex_arch
    )