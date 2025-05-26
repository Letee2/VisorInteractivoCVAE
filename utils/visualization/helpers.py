"""
Funciones auxiliares para visualización.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

def plotly_clickdata_to_dict(click_result):
    """
    Extrae los datos de clic de un gráfico de Plotly en Streamlit.
    
    Args:
        click_result: Resultado del clic en un gráfico de Plotly en Streamlit
        
    Returns:
        Dict: Diccionario con las coordenadas 'x' e 'y' o None si no hay clic
    """
    try:
        if click_result and "clickData" in click_result:
            point = click_result["clickData"]["points"][0]
            return {"x": point["x"], "y": point["y"]}
    except (KeyError, TypeError, IndexError):
        pass
    return None

def display_image(ax, img, title=None, channels=1, normalize_output=False):
    """
    Muestra una imagen en un eje de matplotlib
    
    Args:
        ax: Eje de matplotlib
        img: Imagen como tensor de PyTorch
        title: Título opcional para el eje
        channels: Número de canales (1 para escala de grises, 3 para RGB)
        normalize_output: Si True, los datos están normalizados en [-1,1] en lugar de [0,1]
    """
    # Para imágenes de un solo canal (escala de grises)
    if channels == 1:
        img_display = img.squeeze().numpy()
        if normalize_output:
            img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
        ax.imshow(img_display, cmap='gray')
    # Para imágenes RGB
    else:
        img_display = img.permute(1, 2, 0).numpy()
        if normalize_output:
            img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
        img_display = np.clip(img_display, 0, 1)  # Asegurar rango válido
        ax.imshow(img_display)
    
    if title:
        ax.set_title(title)
    ax.axis('off')

def create_figure_grid(rows, cols, figsize=(15, 5)):
    """
    Crea una rejilla de subplots
    
    Args:
        rows: Número de filas
        cols: Número de columnas
        figsize: Tamaño de la figura en pulgadas
        
    Returns:
        tuple: (fig, axes) - Figura y ejes
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Asegurar que axes sea un array bidimensional incluso con una sola fila/columna
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
        
    return fig, axes