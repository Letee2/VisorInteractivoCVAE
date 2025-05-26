"""
Funciones para visualizar reconstrucciones de imágenes.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from .helpers import display_image, create_figure_grid

def visualize_reconstructions(ae_model, vae_model, test_loader, device, channels=1, normalize_output=False):
    """
    Visualiza reconstrucciones de imágenes originales
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        test_loader: DataLoader con datos de test
        device: Dispositivo (CPU/GPU)
        channels: Número de canales de la imagen (1 para escala de grises, 3 para RGB)
        normalize_output: Si True, los datos están normalizados en [-1,1] en lugar de [0,1]
    """
    try:
        st.header("Comparación de Reconstrucción")
        
        # Obtener imágenes de prueba y reconstruirlas
        test_images, test_labels, ae_reconstructed, vae_reconstructed = get_reconstructions(
            ae_model, vae_model, test_loader, device
        )
        
        # Determinar el número de imágenes a mostrar
        num_images = len(test_images)
        
        # Mostrar imágenes originales y reconstruidas
        display_reconstructions(
            test_images, 
            test_labels, 
            ae_reconstructed, 
            vae_reconstructed, 
            num_images, 
            channels, 
            normalize_output
        )
    except Exception as e:
        st.error(f"Error al visualizar reconstrucciones: {str(e)}")
        st.exception(e)

def get_reconstructions(ae_model, vae_model, test_loader, device):
    """
    Obtiene reconstrucciones de imágenes de prueba
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        test_loader: DataLoader con datos de test
        device: Dispositivo (CPU/GPU)
        
    Returns:
        tuple: (test_images, test_labels, ae_reconstructed, vae_reconstructed)
    """
    with torch.no_grad():
        # Seleccionar algunas imágenes de prueba
        test_images = []
        test_labels = []
        
        for images, labels in test_loader:
            # Limitar a 5 imágenes para evitar filas muy largas
            test_images.append(images[:5])
            test_labels.append(labels[:5])
            break
        
        test_images = test_images[0].to(device)
        test_labels = test_labels[0]
        
        # Reconstrucción con Autoencoder
        ae_reconstructed = ae_model(test_images)
        if isinstance(ae_reconstructed, tuple):
            ae_reconstructed = ae_reconstructed[0]  # En caso de que devuelva múltiples valores
        ae_reconstructed = ae_reconstructed.cpu()
        
        # Reconstrucción con VAE
        vae_output = vae_model(test_images)
        if len(vae_output) == 4:  # [rec_x, mu, logvar, z]
            vae_reconstructed = vae_output[0]
        else:
            vae_reconstructed = vae_output
        vae_reconstructed = vae_reconstructed.cpu()
        
        return test_images.cpu(), test_labels, ae_reconstructed, vae_reconstructed

def display_reconstructions(test_images, test_labels, ae_reconstructed, vae_reconstructed, 
                           num_images, channels, normalize_output):
    """
    Muestra imágenes originales y sus reconstrucciones
    
    Args:
        test_images: Imágenes originales
        test_labels: Etiquetas de las imágenes
        ae_reconstructed: Reconstrucciones del Autoencoder
        vae_reconstructed: Reconstrucciones del VAE
        num_images: Número de imágenes a mostrar
        channels: Número de canales de la imagen
        normalize_output: Si normalizar la salida o no
    """
    # Mostrar imágenes originales y reconstruidas
    fig, axes = plt.subplots(3, num_images, figsize=(15, 5))
    
    # Asegurarnos de que axes sea un array bidimensional incluso con una sola imagen
    if num_images == 1:
        axes = np.array([axes]).T
    
    # Labels para cada fila
    row_labels = ["Original", "AE", "VAE"]
    
    # Añadir labels de fila
    for i, label in enumerate(row_labels):
        if num_images > 1:
            axes[i, 0].set_ylabel(label, fontsize=14, rotation=90, va='center')
        else:
            axes[i].set_ylabel(label, fontsize=14, rotation=90, va='center')
    
    # Mostrar imágenes
    for i in range(num_images):
        # Determinar el índice correcto para el eje
        if num_images > 1:
            # Imágenes originales
            display_image(
                axes[0, i], 
                test_images[i], 
                title=f"Clase: {test_labels[i].item() if hasattr(test_labels[i], 'item') else test_labels[i]}",
                channels=channels,
                normalize_output=normalize_output
            )
            
            # Reconstrucciones del Autoencoder
            display_image(
                axes[1, i], 
                ae_reconstructed[i],
                channels=channels,
                normalize_output=normalize_output
            )
            
            # Reconstrucciones del VAE
            display_image(
                axes[2, i], 
                vae_reconstructed[i],
                channels=channels,
                normalize_output=normalize_output
            )
        else:
            # Caso de una sola imagen
            display_image(
                axes[0], 
                test_images[i], 
                title=f"Clase: {test_labels[i].item() if hasattr(test_labels[i], 'item') else test_labels[i]}",
                channels=channels,
                normalize_output=normalize_output
            )
            display_image(
                axes[1], 
                ae_reconstructed[i],
                channels=channels,
                normalize_output=normalize_output
            )
            display_image(
                axes[2], 
                vae_reconstructed[i],
                channels=channels,
                normalize_output=normalize_output
            )
    
    plt.tight_layout()
    st.pyplot(fig)