"""
Funciones para visualizar el espacio latente de Autoencoders y VAEs.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go

from .helpers import display_image

def get_embeddings(model, test_loader, device, is_vae=False):
    """
    Obtiene los embeddings del espacio latente para visualización
    
    Args:
        model: Modelo entrenado (Autoencoder o VAE)
        test_loader: DataLoader con datos de test
        device: Dispositivo (CPU/GPU)
        is_vae: Si es True, el modelo es un VAE, si no un Autoencoder
        
    Returns:
        embeddings: Matriz con los embeddings
        labels: Etiquetas correspondientes
    """
    try:
        model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                
                if is_vae:
                    mu, _ = model.encode(data)
                    embeddings.append(mu.cpu().numpy())
                else:
                    z = model.encode(data)
                    embeddings.append(z.cpu().numpy())
                
                labels.append(target.numpy())
        
        return np.vstack(embeddings), np.concatenate(labels)
    except Exception as e:
        st.error(f"Error al obtener embeddings: {str(e)}")
        # Devolver arrays vacíos con la dimensión correcta para evitar errores
        return np.array([]).reshape(0, 2), np.array([])

def visualize_latent_space(ae_embeddings, vae_embeddings, ae_labels, vae_labels, latent_dim):
    """
    Visualiza el espacio latente de Autoencoder y VAE
    
    Args:
        ae_embeddings: Embeddings del Autoencoder
        vae_embeddings: Embeddings del VAE
        ae_labels: Etiquetas para Autoencoder
        vae_labels: Etiquetas para VAE
        latent_dim: Dimensión del espacio latente
    """
    try:
        st.header("Visualización del Espacio Latente")
        
        # Verificar que tenemos suficientes datos
        if len(ae_embeddings) == 0 or len(vae_embeddings) == 0:
            st.warning("No hay suficientes datos para visualizar el espacio latente.")
            return
        
        if latent_dim == 2:
            _visualize_2d_latent_space(ae_embeddings, vae_embeddings, ae_labels, vae_labels)
        else:
            _visualize_high_dim_latent_space(ae_embeddings, vae_embeddings, ae_labels, vae_labels)
    except Exception as e:
        st.error(f"Error al visualizar el espacio latente: {str(e)}")

def _visualize_2d_latent_space(ae_embeddings, vae_embeddings, ae_labels, vae_labels):
    """
    Visualiza un espacio latente 2D directamente
    
    Args:
        ae_embeddings: Embeddings del Autoencoder
        vae_embeddings: Embeddings del VAE
        ae_labels: Etiquetas para Autoencoder
        vae_labels: Etiquetas para VAE
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Autoencoder Tradicional")
        fig = px.scatter(
            x=ae_embeddings[:, 0], 
            y=ae_embeddings[:, 1], 
            color=ae_labels.astype(str),
            labels={'x': 'Dimensión 1', 'y': 'Dimensión 2', 'color': 'Clase'},
            title="Distribución del Espacio Latente (Autoencoder)",
            color_discrete_sequence=px.colors.qualitative.G10
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar estadísticas
        st.markdown("#### Estadísticas del Espacio Latente")
        df_stats = pd.DataFrame({
            'Dimensión': ['Dim 1', 'Dim 2'],
            'Media': [ae_embeddings[:, 0].mean(), ae_embeddings[:, 1].mean()],
            'Desviación Estándar': [ae_embeddings[:, 0].std(), ae_embeddings[:, 1].std()],
            'Mínimo': [ae_embeddings[:, 0].min(), ae_embeddings[:, 1].min()],
            'Máximo': [ae_embeddings[:, 0].max(), ae_embeddings[:, 1].max()]
        })
        st.dataframe(df_stats)
    
    with col2:
        st.markdown("### Autoencoder Variacional")
        fig = px.scatter(
            x=vae_embeddings[:, 0], 
            y=vae_embeddings[:, 1], 
            color=vae_labels.astype(str),
            labels={'x': 'Dimensión 1', 'y': 'Dimensión 2', 'color': 'Clase'},
            title="Distribución del Espacio Latente (VAE)",
            color_discrete_sequence=px.colors.qualitative.G10
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar estadísticas
        st.markdown("#### Estadísticas del Espacio Latente")
        df_stats = pd.DataFrame({
            'Dimensión': ['Dim 1', 'Dim 2'],
            'Media': [vae_embeddings[:, 0].mean(), vae_embeddings[:, 1].mean()],
            'Desviación Estándar': [vae_embeddings[:, 0].std(), vae_embeddings[:, 1].std()],
            'Mínimo': [vae_embeddings[:, 0].min(), vae_embeddings[:, 1].min()],
            'Máximo': [vae_embeddings[:, 0].max(), vae_embeddings[:, 1].max()]
        })
        st.dataframe(df_stats)

def _visualize_high_dim_latent_space(ae_embeddings, vae_embeddings, ae_labels, vae_labels):
    """
    Visualiza un espacio latente de dimensión > 2 usando t-SNE
    
    Args:
        ae_embeddings: Embeddings del Autoencoder
        vae_embeddings: Embeddings del VAE
        ae_labels: Etiquetas para Autoencoder
        vae_labels: Etiquetas para VAE
    """
    # Si la dimensión latente es mayor a 2, usar t-SNE para visualización
    st.markdown("### Proyección t-SNE del Espacio Latente")
    
    # Solo ejecutar t-SNE si tenemos suficientes datos
    if len(ae_embeddings) > 5 and len(vae_embeddings) > 5:
        with st.spinner("Calculando proyección t-SNE..."):
            def apply_tsne(embeddings):
                tsne = TSNE(n_components=2, random_state=42)
                return tsne.fit_transform(embeddings)
            
            ae_tsne = apply_tsne(ae_embeddings)
            vae_tsne = apply_tsne(vae_embeddings)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Autoencoder Tradicional")
            fig = px.scatter(
                x=ae_tsne[:, 0], 
                y=ae_tsne[:, 1], 
                color=ae_labels.astype(str),
                labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Clase'},
                title="Proyección t-SNE del Espacio Latente (Autoencoder)",
                color_discrete_sequence=px.colors.qualitative.G10
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Autoencoder Variacional")
            fig = px.scatter(
                x=vae_tsne[:, 0], 
                y=vae_tsne[:, 1], 
                color=vae_labels.astype(str),
                labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Clase'},
                title="Proyección t-SNE del Espacio Latente (VAE)",
                color_discrete_sequence=px.colors.qualitative.G10
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insuficientes datos para calcular t-SNE. Se necesitan al menos 5 muestras.")

def interactive_latent_space(ae_model, vae_model, device, ae_embeddings, vae_embeddings, ae_labels, vae_labels, 
                           latent_dim=2, channels=1, normalize_output=False):
    """
    Crea una visualización interactiva del espacio latente donde el usuario puede
    seleccionar puntos y ver las imágenes generadas.
    
    Args:
        ae_model: Modelo Autoencoder entrenado
        vae_model: Modelo VAE entrenado
        device: Dispositivo (CPU/GPU)
        ae_embeddings: Embeddings del Autoencoder
        vae_embeddings: Embeddings del VAE
        ae_labels: Etiquetas para Autoencoder
        vae_labels: Etiquetas para VAE
        latent_dim: Dimensión del espacio latente
        channels: Número de canales de la imagen (1 para escala de grises, 3 para RGB)
        normalize_output: Si True, los datos están normalizados en [-1,1] en lugar de [0,1]
    """
    try:
        if latent_dim != 2:
            st.warning("La visualización interactiva solo está disponible para espacios latentes de 2 dimensiones.")
            return
        
        # Verificar si tenemos suficientes embeddings
        if len(ae_embeddings) == 0 or len(vae_embeddings) == 0:
            st.warning("No hay suficientes datos para la visualización interactiva.")
            return
        
        st.header("Exploración Interactiva del Espacio Latente")
        st.markdown("""
        ### Generación de imágenes desde el espacio latente
        
        Selecciona un punto en el espacio latente para ver qué imagen se genera.
        - Puedes usar los controles deslizantes para seleccionar coordenadas precisas
        - O puedes hacer clic directamente en el gráfico para seleccionar un punto
        
        Esto te permitirá entender cómo se organizan las clases en el espacio latente y 
        las diferencias entre el Autoencoder tradicional y el VAE.
        """)
        
        # Crear pestañas para elegir entre Autoencoder y VAE
        tab1, tab2 = st.tabs(["Autoencoder Tradicional", "Autoencoder Variacional (VAE)"])
        
        # Pestaña para Autoencoder
        with tab1:
            _interactive_model_tab(
                "ae", 
                ae_model, 
                ae_embeddings, 
                ae_labels, 
                device, 
                channels, 
                normalize_output
            )
        
        # Pestaña para VAE
        with tab2:
            _interactive_model_tab(
                "vae", 
                vae_model, 
                vae_embeddings, 
                vae_labels, 
                device, 
                channels, 
                normalize_output
            )
    except Exception as e:
        st.error(f"Error en el espacio latente interactivo: {str(e)}")
        st.exception(e)

def _interactive_model_tab(model_type, model, embeddings, labels, device, channels, normalize_output):
    """
    Renderiza una pestaña interactiva para un modelo específico
    
    Args:
        model_type: Tipo de modelo ("ae" o "vae")
        model: Instancia del modelo
        embeddings: Embeddings del modelo
        labels: Etiquetas
        device: Dispositivo (CPU/GPU)
        channels: Número de canales de la imagen
        normalize_output: Si normalizar la salida o no
    """
    # Inicializar estados de sesión para los puntos seleccionados si no existen
    if f'{model_type}_x' not in st.session_state:
        st.session_state[f'{model_type}_x'] = 0.0
    if f'{model_type}_y' not in st.session_state:
        st.session_state[f'{model_type}_y'] = 0.0
        
    # Determinar límites del espacio latente
    x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
    
    # Añadir un poco de margen
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    
    # Función de callback para actualizar el punto cuando se hace clic
    def update_point(x, y):
        st.session_state[f'{model_type}_x'] = x
        st.session_state[f'{model_type}_y'] = y
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Gráfico interactivo
        title = "Autoencoder Tradicional" if model_type == "ae" else "Autoencoder Variacional"
        fig = px.scatter(
            x=embeddings[:, 0], 
            y=embeddings[:, 1], 
            color=labels.astype(str),
            labels={'x': 'Dimensión 1', 'y': 'Dimensión 2', 'color': 'Clase'},
            title=f"Espacio Latente del {title} - Usa los sliders para generar una imagen",
            color_discrete_sequence=px.colors.qualitative.G10
        )
        
        # Configurar los límites del gráfico con margen
        fig.update_xaxes(range=[x_min - margin_x, x_max + margin_x])
        fig.update_yaxes(range=[y_min - margin_y, y_max + margin_y])
        
        # Añadir un punto al gráfico para mostrar la selección actual
        fig.add_trace(go.Scatter(
            x=[st.session_state[f'{model_type}_x']],
            y=[st.session_state[f'{model_type}_y']],
            mode='markers',
            marker=dict(color='red', size=12, symbol='x'),
            name='Punto seleccionado',
            hoverinfo='skip'
        ))
        
        # Configurar el gráfico con interactividad
        key = f"{model_type}_interactive"
        st.plotly_chart(fig, use_container_width=True, key=key)

        if key in st.session_state and "clickData" in st.session_state[key]:
            try:
                click_data = st.session_state[key]["clickData"]["points"][0]
                update_point(click_data["x"], click_data["y"])
            except (KeyError, IndexError):
                pass
    
    with col2:
        # Controles deslizantes para selección manual de coordenadas
        st.subheader("Coordenadas Latentes")
        
        # Actualizar valores de los deslizadores con los valores de session_state
        x_value = st.slider(
            "Dimensión 1", 
            float(x_min - margin_x), 
            float(x_max + margin_x), 
            float(st.session_state[f'{model_type}_x']),
            key=f"{model_type}_x_slider"
        )
        
        y_value = st.slider(
            "Dimensión 2", 
            float(y_min - margin_y), 
            float(y_max + margin_y), 
            float(st.session_state[f'{model_type}_y']),
            key=f"{model_type}_y_slider"
        )
        
        # Actualizar el session_state con los valores de los deslizadores
        update_point(x_value, y_value)
        
        # Mostrar las coordenadas seleccionadas
        st.write(f"Punto seleccionado: ({st.session_state[f'{model_type}_x']:.2f}, {st.session_state[f'{model_type}_y']:.2f})")
        
        # Generar imagen desde el punto seleccionado
        with torch.no_grad():
            latent_point = torch.tensor([[st.session_state[f'{model_type}_x'], st.session_state[f'{model_type}_y']]], 
                                       dtype=torch.float32).to(device)
            generated_img = model.decode(latent_point).cpu()
        
        # Mostrar la imagen generada
        st.subheader("Imagen Generada")
        fig, ax = plt.subplots(figsize=(5, 5))
        display_image(ax, generated_img[0], channels=channels, normalize_output=normalize_output)
        st.pyplot(fig)
        
        # Añadir un botón para resetear las coordenadas
        if st.button("Resetear Coordenadas", key=f"reset_{model_type}"):
            update_point(0.0, 0.0)