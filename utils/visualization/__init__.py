"""
Módulo de visualización para comparación de Autoencoder vs VAE.

Este módulo proporciona funciones para visualizar diferentes aspectos de
Autoencoders y VAEs, como espacios latentes, reconstrucciones, etc.
"""

# Importar todas las funciones públicas para facilitar el acceso
from .latent_space import get_embeddings, visualize_latent_space, interactive_latent_space
from .loss_curves import visualize_loss_curves
from .reconstruction import visualize_reconstructions
from .generation import (
    visualize_grid_generation,
    visualize_interpolation,
    visualize_random_generation
)
from .helpers import plotly_clickdata_to_dict

# Re-exportar todo para mantener la API compatible
__all__ = [
    'get_embeddings',
    'visualize_latent_space',
    'interactive_latent_space',
    'visualize_loss_curves',
    'visualize_reconstructions',
    'visualize_grid_generation',
    'visualize_interpolation',
    'visualize_random_generation',
    'plotly_clickdata_to_dict'
]