"""
Funciones para visualizar generación de imágenes desde el espacio latente.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from .helpers import display_image
from .latent_space import get_embeddings

def visualize_grid_generation(ae_model, vae_model, device, latent_dim=2, channels=1, normalize_output=False):
    """
    Visualiza una rejilla de imágenes generadas desde el espacio latente
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        device: Dispositivo (CPU/GPU)
        latent_dim: Dimensión del espacio latente
        channels: Número de canales de la imagen (1 para escala de grises, 3 para RGB)
        normalize_output: Si True, los datos están normalizados en [-1,1] en lugar de [0,1]
    """
    try:
        if latent_dim != 2:
            st.info("La visualización de rejilla solo está disponible para espacios latentes de 2 dimensiones.")
            return
            
        st.subheader("Generación a partir de puntos en el espacio latente")
        
        # Crear rejilla de puntos y generar imágenes
        grid_points, ae_generated, vae_generated = generate_grid_images(
            ae_model, vae_model, device, latent_dim
        )
        
        # Mostrar rejilla de imágenes generadas
        display_grid_images(ae_generated, vae_generated, channels, normalize_output)
    except Exception as e:
        st.error(f"Error en la generación de rejilla: {str(e)}")

def generate_grid_images(ae_model, vae_model, device, latent_dim):
    """
    Genera imágenes a partir de una rejilla de puntos en el espacio latente
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        device: Dispositivo (CPU/GPU)
        latent_dim: Dimensión del espacio latente
        
    Returns:
        tuple: (grid_points, ae_generated, vae_generated)
    """
    # Crear rejilla de puntos en el espacio latente
    x = np.linspace(-3, 3, 5)
    y = np.linspace(-3, 3, 5)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    
    # Generar imágenes desde rejilla
    with torch.no_grad():
        # Autoencoder
        ae_grid_tensor = torch.FloatTensor(grid_points).to(device)
        ae_generated = ae_model.decode(ae_grid_tensor).cpu()
        
        # VAE
        vae_grid_tensor = torch.FloatTensor(grid_points).to(device)
        vae_generated = vae_model.decode(vae_grid_tensor).cpu()
    
    return grid_points, ae_generated, vae_generated

def display_grid_images(ae_generated, vae_generated, channels, normalize_output):
    """
    Muestra una rejilla de imágenes generadas
    
    Args:
        ae_generated: Imágenes generadas con Autoencoder
        vae_generated: Imágenes generadas con VAE
        channels: Número de canales de la imagen
        normalize_output: Si normalizar la salida o no
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Autoencoder Tradicional")
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            try:
                display_image(ax, ae_generated[i], channels=channels, normalize_output=normalize_output)
            except Exception as e:
                ax.text(0.5, 0.5, "Error", ha='center', va='center')
                ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Autoencoder Variacional")
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            try:
                display_image(ax, vae_generated[i], channels=channels, normalize_output=normalize_output)
            except Exception as e:
                ax.text(0.5, 0.5, "Error", ha='center', va='center')
                ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)

def visualize_interpolation(ae_model, vae_model, device, test_loader, channels=1, normalize_output=False):
    """
    Visualiza interpolación entre dos imágenes en el espacio latente
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        device: Dispositivo (CPU/GPU)
        test_loader: DataLoader con datos de test
        channels: Número de canales de la imagen (1 para escala de grises, 3 para RGB)
        normalize_output: Si True, los datos están normalizados en [-1,1] en lugar de [0,1]
    """
    try:
        st.subheader("Interpolación en el Espacio Latente")
        
        # Obtener embeddings y seleccionar puntos para interpolación
        selected_points, selected_classes = select_interpolation_points(ae_model, vae_model, device, test_loader)
        
        if selected_points is None:
            return
        
        # Generar secuencia de interpolación
        ae_interp_images, vae_interp_images = generate_interpolation_images(
            ae_model, vae_model, device, selected_points
        )
        
        # Mostrar interpolación
        display_interpolation_images(
            ae_interp_images, 
            vae_interp_images, 
            selected_classes, 
            channels, 
            normalize_output
        )
    except Exception as e:
        st.error(f"Error en la interpolación: {str(e)}")
        st.exception(e)

def select_interpolation_points(ae_model, vae_model, device, test_loader):
    """
    Selecciona puntos para interpolación entre dos clases específicas definidas en el código.
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        device: Dispositivo (CPU/GPU)
        test_loader: DataLoader con datos de test
        
    Returns:
        tuple: (selected_points, selected_classes) o (None, None) si no hay suficientes datos
    """
    # <<< Selección manual de clases aquí >>>
    class_pair = (9, 1)  # Cambia estos valores según tus necesidades

    # Obtener embeddings para interpolación
    with st.spinner("Obteniendo embeddings para interpolación..."):
        ae_embeddings, ae_labels = get_embeddings(ae_model, test_loader, device, is_vae=False)
        vae_embeddings, vae_labels = get_embeddings(vae_model, test_loader, device, is_vae=True)
    
    if len(ae_embeddings) < 2 or len(vae_embeddings) < 2:
        st.warning("No hay suficientes datos para la interpolación.")
        return None, None

    class_a, class_b = class_pair

    if class_a not in vae_labels or class_b not in vae_labels:
        st.warning(f"Las clases {class_a} y/o {class_b} no están presentes en el conjunto de test.")
        return None, None
    
    try:
        start_idx = np.where(vae_labels == class_a)[0][0]
        end_idx = np.where(vae_labels == class_b)[0][0]
    except IndexError:
        st.warning("No se encontraron suficientes muestras para las clases seleccionadas.")
        return None, None

    selected_points = {
        'ae_start': ae_embeddings[start_idx],
        'ae_end': ae_embeddings[end_idx],
        'vae_start': vae_embeddings[start_idx],
        'vae_end': vae_embeddings[end_idx]
    }

    return selected_points, (class_a, class_b)


def generate_interpolation_images(ae_model, vae_model, device, selected_points, steps=10):
    """
    Genera imágenes interpoladas entre dos puntos del espacio latente
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        device: Dispositivo (CPU/GPU)
        selected_points: Puntos seleccionados para interpolación
        steps: Número de pasos para la interpolación
        
    Returns:
        tuple: (ae_interp_images, vae_interp_images)
    """
    # Función para interpolar puntos
    def interpolate_points(start, end, steps=10):
        alpha = np.linspace(0, 1, steps)
        return np.array([start * (1 - a) + end * a for a in alpha])
    
    # Generar secuencia de interpolación
    ae_interp_points = interpolate_points(selected_points['ae_start'], selected_points['ae_end'], steps)
    vae_interp_points = interpolate_points(selected_points['vae_start'], selected_points['vae_end'], steps)
    
    # Generar imágenes
    with torch.no_grad():
        # Autoencoder
        ae_interp_tensor = torch.FloatTensor(ae_interp_points).to(device)
        ae_interp_images = ae_model.decode(ae_interp_tensor).cpu()
        
        # VAE
        vae_interp_tensor = torch.FloatTensor(vae_interp_points).to(device)
        vae_interp_images = vae_model.decode(vae_interp_tensor).cpu()
    
    return ae_interp_images, vae_interp_images

def display_interpolation_images(ae_interp_images, vae_interp_images, selected_classes, channels, normalize_output, steps=10):
    """
    Muestra imágenes de interpolación
    
    Args:
        ae_interp_images: Imágenes interpoladas con Autoencoder
        vae_interp_images: Imágenes interpoladas con VAE
        selected_classes: Clases seleccionadas para interpolación
        channels: Número de canales de la imagen
        normalize_output: Si normalizar la salida o no
        steps: Número de pasos para la interpolación
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Autoencoder: Interpolación entre clase {selected_classes[0]} y {selected_classes[1]}")
        fig, axes = plt.subplots(1, steps, figsize=(15, 2))
        for i, ax in enumerate(axes):
            display_image(ax, ae_interp_images[i], channels=channels, normalize_output=normalize_output)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown(f"### VAE: Interpolación entre clase {selected_classes[0]} y {selected_classes[1]}")
        fig, axes = plt.subplots(1, steps, figsize=(15, 2))
        for i, ax in enumerate(axes):
            display_image(ax, vae_interp_images[i], channels=channels, normalize_output=normalize_output)
        plt.tight_layout()
        st.pyplot(fig)

def visualize_random_generation(ae_model, vae_model, device, latent_dim, channels=1, normalize_output=False):
    """
    Visualiza imágenes generadas aleatoriamente desde el espacio latente
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        device: Dispositivo (CPU/GPU)
        latent_dim: Dimensión del espacio latente
        channels: Número de canales de la imagen (1 para escala de grises, 3 para RGB)
        normalize_output: Si True, los datos están normalizados en [-1,1] en lugar de [0,1]
    """
    try:
        st.subheader("Muestreo aleatorio del espacio latente")
        
        # Generar imágenes aleatorias
        ae_random_images, vae_random_images = generate_random_images(
            ae_model, vae_model, device, latent_dim
        )
        
        # Mostrar imágenes generadas
        display_random_images(ae_random_images, vae_random_images, channels, normalize_output)
    except Exception as e:
        st.error(f"Error en la generación aleatoria: {str(e)}")

def generate_random_images(ae_model, vae_model, device, latent_dim, num_samples=10):
    """
    Genera imágenes desde puntos aleatorios del espacio latente
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        device: Dispositivo (CPU/GPU)
        latent_dim: Dimensión del espacio latente
        num_samples: Número de imágenes a generar
        
    Returns:
        tuple: (ae_random_images, vae_random_images)
    """
    # Generar puntos aleatorios en el espacio latente
    np.random.seed(42)
    random_points = np.random.normal(0, 1, size=(num_samples, latent_dim))
    
    # Generar imágenes desde puntos aleatorios
    with torch.no_grad():
        # Autoencoder
        ae_random_tensor = torch.FloatTensor(random_points).to(device)
        ae_random_images = ae_model.decode(ae_random_tensor).cpu()
        
        # VAE
        vae_random_tensor = torch.FloatTensor(random_points).to(device)
        vae_random_images = vae_model.decode(vae_random_tensor).cpu()
    
    return ae_random_images, vae_random_images

def display_random_images(ae_random_images, vae_random_images, channels, normalize_output, num_samples=10):
    """
    Muestra imágenes generadas aleatoriamente
    
    Args:
        ae_random_images: Imágenes generadas con Autoencoder
        vae_random_images: Imágenes generadas con VAE
        channels: Número de canales de la imagen
        normalize_output: Si normalizar la salida o no
        num_samples: Número de imágenes a mostrar
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Autoencoder Tradicional")
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
        for i, ax in enumerate(axes):
            display_image(ax, ae_random_images[i], channels=channels, normalize_output=normalize_output)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Autoencoder Variacional")
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
        for i, ax in enumerate(axes):
            display_image(ax, vae_random_images[i], channels=channels, normalize_output=normalize_output)
        plt.tight_layout()
        st.pyplot(fig)