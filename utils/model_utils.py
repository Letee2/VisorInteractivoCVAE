import os
import torch
import pickle
import numpy as np
import streamlit as st

def get_model_directory(dataset_name, models_dir="./saved_models"):
    """
    Obtiene el directorio para guardar modelos de un dataset específico
    
    Args:
        dataset_name: Nombre del dataset
        models_dir: Directorio base para modelos
        
    Returns:
        str: Ruta al directorio del dataset
    """
    dataset_dir = os.path.join(models_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir

def get_model_paths(dataset_dir, epochs):
    """
    Obtiene las rutas para los archivos de modelos
    
    Args:
        dataset_dir: Directorio del dataset
        epochs: Número de épocas
        
    Returns:
        dict: Diccionario con rutas para cada archivo
    """
    return {
        'ae_model': f"{dataset_dir}/ae_model_{epochs}_epochs.pt",
        'vae_model': f"{dataset_dir}/vae_model_{epochs}_epochs.pt",
        'ae_losses': f"{dataset_dir}/ae_losses_{epochs}_epochs.pkl",
        'vae_losses': f"{dataset_dir}/vae_losses_{epochs}_epochs.pkl",
        'metadata': f"{dataset_dir}/metadata_{epochs}_epochs.pkl"
    }

def save_models(ae_model, vae_model, ae_losses, vae_losses, epochs, 
               dataset_name="mnist", input_channels=1, input_size=28, models_dir="./saved_models"):
    """
    Guarda los modelos entrenados y sus históricos de pérdidas
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        ae_losses: Histórico de pérdidas del Autoencoder
        vae_losses: Histórico de pérdidas del VAE
        epochs: Número de épocas
        dataset_name: Nombre del dataset usado
        input_channels: Número de canales de entrada
        input_size: Tamaño de la imagen
        models_dir: Directorio donde guardar los modelos
    """
    # Obtener el directorio para este dataset
    dataset_dir = get_model_directory(dataset_name, models_dir)
    
    # Obtener rutas de archivos
    paths = get_model_paths(dataset_dir, epochs)
    
    # Guardar modelos
    torch.save(ae_model.state_dict(), paths['ae_model'])
    torch.save(vae_model.state_dict(), paths['vae_model'])
    
    # Guardar histórico de pérdidas
    with open(paths['ae_losses'], 'wb') as f:
        pickle.dump(ae_losses, f)
    
    with open(paths['vae_losses'], 'wb') as f:
        pickle.dump(vae_losses, f)
    
    # Guardar metadatos
    metadata = {
        'input_channels': input_channels,
        'input_size': input_size,
        'epochs': epochs,
        'dataset': dataset_name
    }
    
    with open(paths['metadata'], 'wb') as f:
        pickle.dump(metadata, f)

def models_exist(epochs, dataset_name="mnist", models_dir="./saved_models"):
    """
    Verifica si existen modelos entrenados para un número específico de épocas y dataset
    
    Args:
        epochs: Número de épocas a verificar
        dataset_name: Nombre del dataset
        models_dir: Directorio donde deberían estar los modelos
        
    Returns:
        bool: True si existen todos los archivos necesarios
    """
    # Determinar la ruta del dataset específico
    dataset_dir = os.path.join(models_dir, dataset_name)
    
    # Si el directorio no existe, los modelos no existen
    if not os.path.exists(dataset_dir):
        return False
    
    # Obtener rutas de archivos
    paths = get_model_paths(dataset_dir, epochs)
    
    # Verificar si todos los archivos existen
    return (os.path.exists(paths['ae_model']) and 
            os.path.exists(paths['vae_model']) and
            os.path.exists(paths['ae_losses']) and
            os.path.exists(paths['vae_losses']))

def load_metadata(paths):
    """
    Carga metadatos del modelo guardado
    
    Args:
        paths: Diccionario con rutas de archivos
        
    Returns:
        dict: Metadatos del modelo o None si no existe
    """
    try:
        if not os.path.exists(paths['metadata']):
            return None
            
        with open(paths['metadata'], 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error al cargar metadatos: {e}")
        return None

def create_model_instances(latent_dim, device, input_channels, input_size, dataset_name):
    """
    Crea instancias de los modelos usando la arquitectura adaptativa
    
    Args:
        latent_dim: Dimensión del espacio latente
        device: Dispositivo (CPU/GPU)
        input_channels: Número de canales de entrada
        input_size: Tamaño de la imagen
        dataset_name: Nombre del dataset
        
    Returns:
        tuple: (ae_model, vae_model) - Instancias de los modelos
    """
    try:
        # Importar de manera dinámica para evitar circular imports
        from models.adaptive_models import create_autoencoder, create_vae
        
        ae_model = create_autoencoder(
            latent_dim=latent_dim,
            input_channels=input_channels,
            input_size=input_size,
            dataset_name=dataset_name
        ).to(device)
        
        vae_model = create_vae(
            latent_dim=latent_dim,
            input_channels=input_channels,
            input_size=input_size,
            dataset_name=dataset_name
        ).to(device)
        
        return ae_model, vae_model
    except Exception as e:
        st.error(f"Error al crear modelos: {e}")
        return None, None

def load_model_weights(model, weights_path, device):
    """
    Carga los pesos de un modelo desde un archivo
    
    Args:
        model: Instancia del modelo
        weights_path: Ruta al archivo de pesos
        device: Dispositivo (CPU/GPU)
        
    Returns:
        bool: True si la carga fue exitosa
    """
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        return True
    except Exception as e:
        st.error(f"Error al cargar pesos del modelo: {e}")
        return False

def load_losses(paths):
    """
    Carga los históricos de pérdidas
    
    Args:
        paths: Diccionario con rutas de archivos
        
    Returns:
        tuple: (ae_losses, vae_losses) - Históricos de pérdidas
    """
    try:
        with open(paths['ae_losses'], 'rb') as f:
            ae_losses = pickle.load(f)
        
        with open(paths['vae_losses'], 'rb') as f:
            vae_losses = pickle.load(f)
            
        return ae_losses, vae_losses
    except Exception as e:
        st.error(f"Error al cargar históricos de pérdidas: {e}")
        return None, None

def load_models(epochs, latent_dim, device, dataset_name="mnist", 
               input_channels=1, input_size=28, models_dir="./saved_models"):
    """
    Carga modelos entrenados previamente
    
    Args:
        epochs: Número de épocas
        latent_dim: Dimensión del espacio latente
        device: Dispositivo (CPU/GPU)
        dataset_name: Nombre del dataset
        input_channels: Número de canales de entrada
        input_size: Tamaño de la imagen
        models_dir: Directorio donde están los modelos
        
    Returns:
        tuple: (ae_model, vae_model, ae_losses, vae_losses) - Modelos y pérdidas
    """
    try:
        # Obtener directorio y rutas
        dataset_dir = get_model_directory(dataset_name, models_dir)
        paths = get_model_paths(dataset_dir, epochs)
        
        # Verificar si existen modelos guardados
        if not os.path.exists(paths['ae_model']):
            st.warning(f"No se encontraron modelos guardados para {dataset_name} con {epochs} épocas. Se entrenarán nuevos modelos.")
            return None, None, None, None
        
        # Cargar metadatos si existen
        metadata = load_metadata(paths)
        if metadata:
            # Usar metadatos guardados para dimensiones de entrada
            input_channels = metadata.get('input_channels', input_channels)
            input_size = metadata.get('input_size', input_size)
        
        # Crear instancias de modelos
        ae_model, vae_model = create_model_instances(
            latent_dim, device, input_channels, input_size, dataset_name
        )
        
        if ae_model is None or vae_model is None:
            return None, None, None, None
        
        # Cargar pesos
        ae_success = load_model_weights(ae_model, paths['ae_model'], device)
        vae_success = load_model_weights(vae_model, paths['vae_model'], device)
        
        if not (ae_success and vae_success):
            return None, None, None, None
        
        # Cargar históricos de pérdidas
        ae_losses, vae_losses = load_losses(paths)
        
        return ae_model, vae_model, ae_losses, vae_losses
    
    except Exception as e:
        st.error(f"Error al cargar modelos: {str(e)}")
        return None, None, None, None