import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def visualize_loss_curves(ae_losses, vae_losses, epochs):
    """
    Visualiza las curvas de pérdida de los modelos
    
    Args:
        ae_losses: Histórico de pérdidas del Autoencoder
        vae_losses: Histórico de pérdidas del VAE (diccionario)
        epochs: Número de épocas
    """
    try:
        st.subheader("Curvas de Pérdida")
        col1, col2 = st.columns(2)
      
        with col1:
            st.markdown("### Autoencoder Tradicional")
            fig, ax = plt.subplots()
            ax.plot(range(1, epochs + 1), ae_losses)
            ax.set_xlabel('Época')
            ax.set_ylabel('Pérdida (BCE)')
            ax.set_title('Pérdida del Autoencoder')
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Autoencoder Variacional")
            fig, ax = plt.subplots()
            ax.plot(range(1, epochs + 1), vae_losses["total"], label='Total')
            ax.plot(range(1, epochs + 1), vae_losses["bce"], label='BCE')
            ax.plot(range(1, epochs + 1), vae_losses["kld"], label='KLD')
            ax.set_xlabel('Época')
            ax.set_ylabel('Pérdida')
            ax.set_title('Pérdida del VAE')
            ax.legend()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al visualizar curvas de pérdida: {str(e)}")

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
        else:
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
    except Exception as e:
        st.error(f"Error al visualizar el espacio latente: {str(e)}")

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
        
        # Reconstruir algunas imágenes de prueba
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
            ae_reconstructed, _ = ae_model(test_images)
            ae_reconstructed = ae_reconstructed.cpu()
            
            # Reconstrucción con VAE
            vae_reconstructed, _, _, _ = vae_model(test_images)
            vae_reconstructed = vae_reconstructed.cpu()
        
        # Determinar el número de imágenes a mostrar
        num_images = len(test_images)
        
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
        
        # Función para mostrar imágenes correctamente según el número de canales
        def display_image(ax, img, title=None):
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
        
        # Mostrar imágenes
        for i in range(num_images):
            # Determinar el índice correcto para el eje
            if num_images > 1:
                # Imágenes originales
                display_image(axes[0, i], test_images[i].cpu(), 
                             f"Clase: {test_labels[i].item() if hasattr(test_labels[i], 'item') else test_labels[i]}")
                
                # Reconstrucciones del Autoencoder
                display_image(axes[1, i], ae_reconstructed[i])
                
                # Reconstrucciones del VAE
                display_image(axes[2, i], vae_reconstructed[i])
            else:
                # Caso de una sola imagen
                display_image(axes[0], test_images[i].cpu(), 
                             f"Clase: {test_labels[i].item() if hasattr(test_labels[i], 'item') else test_labels[i]}")
                display_image(axes[1], ae_reconstructed[i])
                display_image(axes[2], vae_reconstructed[i])
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al visualizar reconstrucciones: {str(e)}")
        st.exception(e)

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
        
        # Función para mostrar imágenes correctamente según el número de canales
        def display_grid_images(ax, img):
            if channels == 1:
                img_display = img.squeeze().numpy()
                if normalize_output:
                    img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
                ax.imshow(img_display, cmap='gray')
            else:
                img_display = img.permute(1, 2, 0).numpy()
                if normalize_output:
                    img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
                img_display = np.clip(img_display, 0, 1)
                ax.imshow(img_display)
            ax.axis('off')
        
        # Mostrar rejilla de imágenes generadas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Autoencoder Tradicional")
            fig, axes = plt.subplots(5, 5, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                try:
                    display_grid_images(ax, ae_generated[i])
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
                    display_grid_images(ax, vae_generated[i])
                except Exception as e:
                    ax.text(0.5, 0.5, "Error", ha='center', va='center')
                    ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error en la generación de rejilla: {str(e)}")

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
        
        # Obtener embeddings para interpolación
        with st.spinner("Obteniendo embeddings para interpolación..."):
            ae_embeddings, ae_labels = get_embeddings(ae_model, test_loader, device, is_vae=False)
            vae_embeddings, vae_labels = get_embeddings(vae_model, test_loader, device, is_vae=True)
        
        # Verificar si tenemos suficientes datos
        if len(ae_embeddings) < 2 or len(vae_embeddings) < 2:
            st.warning("No hay suficientes datos para la interpolación.")
            return
        
        # Identificar las clases disponibles
        unique_labels = np.unique(vae_labels)
        if len(unique_labels) < 2:
            st.warning("Se necesitan al menos dos clases diferentes para la interpolación.")
            return
        
        # Seleccionar dos puntos de clases diferentes
        selected_classes = unique_labels[:2]  # Tomar las dos primeras clases disponibles
        
        start_idx = np.where(vae_labels == selected_classes[0])[0][0]
        end_idx = np.where(vae_labels == selected_classes[1])[0][0]
        
        start_point_ae = ae_embeddings[start_idx]
        end_point_ae = ae_embeddings[end_idx]
        
        start_point_vae = vae_embeddings[start_idx]
        end_point_vae = vae_embeddings[end_idx]
        
        # Crear interpolación
        def interpolate_points(start, end, steps=10):
            alpha = np.linspace(0, 1, steps)
            return np.array([start * (1 - a) + end * a for a in alpha])
        
        # Generar secuencia de interpolación
        steps = 10
        ae_interp_points = interpolate_points(start_point_ae, end_point_ae, steps)
        vae_interp_points = interpolate_points(start_point_vae, end_point_vae, steps)
        
        # Generar imágenes
        with torch.no_grad():
            # Autoencoder
            ae_interp_tensor = torch.FloatTensor(ae_interp_points).to(device)
            ae_interp_images = ae_model.decode(ae_interp_tensor).cpu()
            
            # VAE
            vae_interp_tensor = torch.FloatTensor(vae_interp_points).to(device)
            vae_interp_images = vae_model.decode(vae_interp_tensor).cpu()
        
        # Función para mostrar imágenes correctamente según el número de canales
        def display_interp_image(ax, img):
            if channels == 1:
                img_display = img.squeeze().numpy()
                if normalize_output:
                    img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
                ax.imshow(img_display, cmap='gray')
            else:
                img_display = img.permute(1, 2, 0).numpy()
                if normalize_output:
                    img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
                img_display = np.clip(img_display, 0, 1)
                ax.imshow(img_display)
            ax.axis('off')
        
        # Mostrar interpolación
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### Autoencoder: Interpolación entre clase {selected_classes[0]} y {selected_classes[1]}")
            fig, axes = plt.subplots(1, steps, figsize=(15, 2))
            for i, ax in enumerate(axes):
                display_interp_image(ax, ae_interp_images[i])
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown(f"### VAE: Interpolación entre clase {selected_classes[0]} y {selected_classes[1]}")
            fig, axes = plt.subplots(1, steps, figsize=(15, 2))
            for i, ax in enumerate(axes):
                display_interp_image(ax, vae_interp_images[i])
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error en la interpolación: {str(e)}")
        st.exception(e)

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
        
        # Generar puntos aleatorios en el espacio latente
        np.random.seed(42)
        random_points = np.random.normal(0, 1, size=(10, latent_dim))
        
        # Generar imágenes desde puntos aleatorios
        with torch.no_grad():
            # Autoencoder
            ae_random_tensor = torch.FloatTensor(random_points).to(device)
            ae_random_images = ae_model.decode(ae_random_tensor).cpu()
            
            # VAE
            vae_random_tensor = torch.FloatTensor(random_points).to(device)
            vae_random_images = vae_model.decode(vae_random_tensor).cpu()
        
        # Función para mostrar imágenes correctamente según el número de canales
        def display_random_image(ax, img):
            if channels == 1:
                img_display = img.squeeze().numpy()
                if normalize_output:
                    img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
                ax.imshow(img_display, cmap='gray')
            else:
                img_display = img.permute(1, 2, 0).numpy()
                if normalize_output:
                    img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
                img_display = np.clip(img_display, 0, 1)
                ax.imshow(img_display)
            ax.axis('off')
        
        # Mostrar imágenes generadas aleatoriamente
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Autoencoder Tradicional")
            fig, axes = plt.subplots(1, 10, figsize=(15, 2))
            for i, ax in enumerate(axes):
                display_random_image(ax, ae_random_images[i])
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Autoencoder Variacional")
            fig, axes = plt.subplots(1, 10, figsize=(15, 2))
            for i, ax in enumerate(axes):
                display_random_image(ax, vae_random_images[i])
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error en la generación aleatoria: {str(e)}")

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
        
        # Determinar límites del espacio latente para ambos modelos
        ae_x_min, ae_x_max = ae_embeddings[:, 0].min(), ae_embeddings[:, 0].max()
        ae_y_min, ae_y_max = ae_embeddings[:, 1].min(), ae_embeddings[:, 1].max()
        
        vae_x_min, vae_x_max = vae_embeddings[:, 0].min(), vae_embeddings[:, 0].max()
        vae_y_min, vae_y_max = vae_embeddings[:, 1].min(), vae_embeddings[:, 1].max()
        
        # Añadir un poco de margen
        ae_margin_x = (ae_x_max - ae_x_min) * 0.1
        ae_margin_y = (ae_y_max - ae_y_min) * 0.1
        
        vae_margin_x = (vae_x_max - vae_x_min) * 0.1
        vae_margin_y = (vae_y_max - vae_y_min) * 0.1
        
        # Inicializar estados de sesión para los puntos seleccionados si no existen
        if 'ae_x' not in st.session_state:
            st.session_state.ae_x = 0.0
        if 'ae_y' not in st.session_state:
            st.session_state.ae_y = 0.0
        if 'vae_x' not in st.session_state:
            st.session_state.vae_x = 0.0
        if 'vae_y' not in st.session_state:
            st.session_state.vae_y = 0.0
        
        # Función de callback para actualizar el punto cuando se hace clic
        def update_ae_point(x, y):
            st.session_state.ae_x = x
            st.session_state.ae_y = y
        
        def update_vae_point(x, y):
            st.session_state.vae_x = x
            st.session_state.vae_y = y
        
        # Pestaña para Autoencoder
        with tab1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Gráfico interactivo para Autoencoder
                fig = px.scatter(
                    x=ae_embeddings[:, 0], 
                    y=ae_embeddings[:, 1], 
                    color=ae_labels.astype(str),
                    labels={'x': 'Dimensión 1', 'y': 'Dimensión 2', 'color': 'Clase'},
                    title="Espacio Latente del Autoencoder - Haz clic para generar una imagen",
                    color_discrete_sequence=px.colors.qualitative.G10
                )
                
                # Configurar los límites del gráfico con margen
                fig.update_xaxes(range=[ae_x_min - ae_margin_x, ae_x_max + ae_margin_x])
                fig.update_yaxes(range=[ae_y_min - ae_margin_y, ae_y_max + ae_margin_y])
                
                # Añadir un punto al gráfico para mostrar la selección actual
                fig.add_trace(go.Scatter(
                    x=[st.session_state.ae_x],
                    y=[st.session_state.ae_y],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='x'),
                    name='Punto seleccionado',
                    hoverinfo='skip'
                ))
                
                # Configurar el gráfico con interactividad
                st.plotly_chart(fig, use_container_width=True, key="ae_interactive")

                if "ae_interactive" in st.session_state and "clickData" in st.session_state["ae_interactive"]:
                    try:
                        click_data = st.session_state["ae_interactive"]["clickData"]["points"][0]
                        update_ae_point(click_data["x"], click_data["y"])
                    except (KeyError, IndexError):
                        pass
                            
            with col2:
                # Controles deslizantes para selección manual de coordenadas
                st.subheader("Coordenadas Latentes")
                
                # Actualizar valores de los deslizadores con los valores de session_state
                ae_x = st.slider(
                    "Dimensión 1", 
                    float(ae_x_min - ae_margin_x), 
                    float(ae_x_max + ae_margin_x), 
                    float(st.session_state.ae_x),
                    key="ae_x_slider"
                )
                
                ae_y = st.slider(
                    "Dimensión 2", 
                    float(ae_y_min - ae_margin_y), 
                    float(ae_y_max + ae_margin_y), 
                    float(st.session_state.ae_y),
                    key="ae_y_slider"
                )
                
                # Actualizar el session_state con los valores de los deslizadores
                update_ae_point(ae_x, ae_y)
                
                # Mostrar las coordenadas seleccionadas
                st.write(f"Punto seleccionado: ({st.session_state.ae_x:.2f}, {st.session_state.ae_y:.2f})")
                
                # Generar imagen desde el punto seleccionado
                with torch.no_grad():
                    latent_point = torch.tensor([[st.session_state.ae_x, st.session_state.ae_y]], dtype=torch.float32).to(device)
                    generated_img = ae_model.decode(latent_point).cpu()
                
                # Mostrar la imagen generada
                st.subheader("Imagen Generada")
                fig, ax = plt.subplots(figsize=(5, 5))
                
                # Mostrar imagen según el número de canales
                if channels == 1:
                    img_display = generated_img[0].squeeze().numpy()
                    if normalize_output:
                        img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
                    ax.imshow(img_display, cmap='gray')
                else:
                    img_display = generated_img[0].permute(1, 2, 0).numpy()
                    if normalize_output:
                        img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
                    img_display = np.clip(img_display, 0, 1)
                    ax.imshow(img_display)
                
                ax.axis('off')
                st.pyplot(fig)
                
                # Añadir un botón para resetear las coordenadas
                if st.button("Resetear Coordenadas", key="reset_ae"):
                    update_ae_point(0.0, 0.0)
        
        # Pestaña para VAE
        with tab2:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Gráfico interactivo para VAE
                fig = px.scatter(
                    x=vae_embeddings[:, 0], 
                    y=vae_embeddings[:, 1], 
                    color=vae_labels.astype(str),
                    labels={'x': 'Dimensión 1', 'y': 'Dimensión 2', 'color': 'Clase'},
                    title="Espacio Latente del VAE - Haz clic para generar una imagen",
                    color_discrete_sequence=px.colors.qualitative.G10
                )
                
                # Configurar los límites del gráfico con margen
                fig.update_xaxes(range=[vae_x_min - vae_margin_x, vae_x_max + vae_margin_x])
                fig.update_yaxes(range=[vae_y_min - vae_margin_y, vae_y_max + vae_margin_y])
                
                # Añadir un punto al gráfico para mostrar la selección actual
                fig.add_trace(go.Scatter(
                    x=[st.session_state.vae_x],
                    y=[st.session_state.vae_y],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='x'),
                    name='Punto seleccionado',
                    hoverinfo='skip'
                ))
                
                # Configurar el gráfico con interactividad
                st.plotly_chart(fig, use_container_width=True, key="vae_interactive")

                if "vae_interactive" in st.session_state and "clickData" in st.session_state["vae_interactive"]:
                    try:
                        click_data = st.session_state["vae_interactive"]["clickData"]["points"][0]
                        update_vae_point(click_data["x"], click_data["y"])
                    except (KeyError, IndexError):
                        pass
            with col2:
                # Controles deslizantes para selección manual de coordenadas
                st.subheader("Coordenadas Latentes")
                
                # Actualizar valores de los deslizadores con los valores de session_state
                vae_x = st.slider(
                    "Dimensión 1", 
                    float(vae_x_min - vae_margin_x), 
                    float(vae_x_max + vae_margin_x), 
                    float(st.session_state.vae_x),
                    key="vae_x_slider"
                )
                
                vae_y = st.slider(
                    "Dimensión 2", 
                    float(vae_y_min - vae_margin_y), 
                    float(vae_y_max + vae_margin_y), 
                    float(st.session_state.vae_y),
                    key="vae_y_slider"
                )
                
                # Actualizar el session_state con los valores de los deslizadores
                update_vae_point(vae_x, vae_y)
                
                # Mostrar las coordenadas seleccionadas
                st.write(f"Punto seleccionado: ({st.session_state.vae_x:.2f}, {st.session_state.vae_y:.2f})")
                
                # Generar imagen desde el punto seleccionado
                with torch.no_grad():
                    latent_point = torch.tensor([[st.session_state.vae_x, st.session_state.vae_y]], dtype=torch.float32).to(device)
                    generated_img = vae_model.decode(latent_point).cpu()
                
                # Mostrar la imagen generada
                st.subheader("Imagen Generada")
                fig, ax = plt.subplots(figsize=(5, 5))
                
                # Mostrar imagen según el número de canales
                if channels == 1:
                    img_display = generated_img[0].squeeze().numpy()
                    if normalize_output:
                        img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
                    ax.imshow(img_display, cmap='gray')
                else:
                    img_display = generated_img[0].permute(1, 2, 0).numpy()
                    if normalize_output:
                        img_display = (img_display + 1) / 2  # De [-1,1] a [0,1] para visualización
                    img_display = np.clip(img_display, 0, 1)
                    ax.imshow(img_display)
                
                ax.axis('off')
                st.pyplot(fig)
                
                # Añadir un botón para resetear las coordenadas
                if st.button("Resetear Coordenadas", key="reset_vae"):
                    update_vae_point(0.0, 0.0)
    except Exception as e:
        st.error(f"Error en el espacio latente interactivo: {str(e)}")
        st.exception(e)

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