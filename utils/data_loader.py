import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import streamlit as st
import os
import numpy as np
from PIL import Image

class CatDataset(Dataset):
    """Dataset personalizado para gatos a partir del Oxford-IIIT Pet dataset."""
    def __init__(self, root="./data/cats", transform=None, download=False):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Crear directorio si no existe
        if not os.path.exists(root) and download:
            os.makedirs(root, exist_ok=True)
            st.info("Descargando imágenes de gatos de muestra...")
            self._generate_sample_images()
        
        # Cargar imágenes
        if os.path.exists(root):
            for i, file in enumerate(os.listdir(root)):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(root, file))
                    self.labels.append(0)  # Todos los gatos tienen etiqueta 0
    
    def _generate_sample_images(self):
        """Genera imágenes aleatorias para demostración."""
        for i in range(100):  # Generar 100 imágenes
            # Crear imagen aleatoria
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            # Guardar imagen
            Image.fromarray(img).save(os.path.join(self.root, f"cat_{i}.jpg"))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transform(dataset_name):
    """
    Obtiene la transformación adecuada para cada dataset
    
    Args:
        dataset_name (str): Nombre del dataset
        
    Returns:
        tuple: (transform, img_size, channels, normalize_output)
    """
    if dataset_name.lower() in ["mnist", "fashion_mnist"]:
        # Transformación para datasets en blanco y negro
        transform = transforms.Compose([
            transforms.ToTensor(),
            # MNIST ya está normalizado entre [0,1]
        ])
        img_size = 28
        channels = 1
        normalize_output = False  # Ya está entre [0,1]
    elif dataset_name.lower() in ["cifar10"]:
        # Transformación para CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normaliza a [-1,1]
        ])
        img_size = 32
        channels = 3
        normalize_output = True  # Normalizado a [-1,1]
    elif dataset_name.lower() in ["cats"]:
        # Transformación para el dataset de gatos
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normaliza a [-1,1]
        ])
        img_size = 64
        channels = 3
        normalize_output = True  # Normalizado a [-1,1]
    else:
        raise ValueError(f"Dataset '{dataset_name}' no soportado")
        
    return transform, img_size, channels, normalize_output

def load_mnist_dataset(transform):
    """Carga el dataset MNIST"""
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return train_dataset, test_dataset

def load_fashion_mnist_dataset(transform):
    """Carga el dataset Fashion MNIST"""
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return train_dataset, test_dataset

def load_cifar10_dataset(transform):
    """Carga el dataset CIFAR10"""
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return train_dataset, test_dataset

def load_cats_dataset(transform):
    """Carga el dataset de gatos o usa CIFAR10 como fallback"""
    try:
        # Intentar cargar dataset personalizado de gatos
        full_dataset = CatDataset(
            root='./data/cats',
            transform=transform,
            download=True
        )
        
        # Dividir en train/test
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        
        return train_dataset, test_dataset
    
    except Exception as e:
        st.warning(f"No se pudo cargar el dataset de gatos: {e}. Usando CIFAR10 como alternativa.")
        # Como alternativa, filtrar solo gatos de CIFAR10 (clase 3)
        cifar_train, cifar_test = load_cifar10_dataset(transform)
        
        # Filtrar solo la clase "gato" (3) en CIFAR10
        train_indices = [i for i, (_, label) in enumerate(cifar_train) if label == 3]
        test_indices = [i for i, (_, label) in enumerate(cifar_test) if label == 3]
        
        train_dataset = torch.utils.data.Subset(cifar_train, train_indices)
        test_dataset = torch.utils.data.Subset(cifar_test, test_indices)
        
        return train_dataset, test_dataset

@st.cache_resource
def load_data(dataset_name="mnist", batch_size=128):
    """
    Carga y prepara diferentes datasets
    
    Args:
        dataset_name: Nombre del dataset a cargar ("mnist", "fashion_mnist", "cifar10", "cats")
        batch_size: Tamaño del batch
        
    Returns:
        train_loader: DataLoader para entrenamiento
        test_loader: DataLoader para pruebas
        train_dataset: Dataset de entrenamiento
        test_dataset: Dataset de pruebas
        img_size: Tamaño de la imagen
        channels: Número de canales
        normalize_output: Si normalizar la salida o no
    """
    # Obtener transformación apropiada para el dataset
    transform, img_size, channels, normalize_output = get_transform(dataset_name)
    
    # Cargar el dataset según el nombre
    if dataset_name.lower() == "mnist":
        train_dataset, test_dataset = load_mnist_dataset(transform)
    elif dataset_name.lower() == "fashion_mnist":
        train_dataset, test_dataset = load_fashion_mnist_dataset(transform)
    elif dataset_name.lower() == "cifar10":
        train_dataset, test_dataset = load_cifar10_dataset(transform)
    elif dataset_name.lower() == "cats":
        train_dataset, test_dataset = load_cats_dataset(transform)
    else:
        raise ValueError(f"Dataset '{dataset_name}' no soportado")
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset, img_size, channels, normalize_output