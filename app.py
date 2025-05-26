import streamlit as st
import torch
import numpy as np
import os
import pandas as pd

# Importar módulos propios
from models import (
    train_convolutional_autoencoder,
    train_convolutional_vae
)
from utils.data_loader import load_data
from utils.model_utils import save_models, load_models, models_exist
from utils.visualization import (
    get_embeddings,
    visualize_latent_space,
    visualize_loss_curves,
    visualize_reconstructions,
    visualize_grid_generation,
    visualize_interpolation,
    visualize_random_generation, 
    interactive_latent_space
)

# Configuración de página
st.set_page_config(page_title="VAE vs Autoencoder - Comparación", layout="wide")

# Directorio para modelos guardados
MODELS_DIR = "./saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def initialize_session_state():
    """Inicializar estados de sesión para persistencia"""
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
        st.session_state.ae_model = None
        st.session_state.vae_model = None
        st.session_state.ae_losses = None
        st.session_state.vae_losses = None
        st.session_state.ae_embeddings = None
        st.session_state.vae_embeddings = None
        st.session_state.ae_labels = None
        st.session_state.vae_labels = None
        st.session_state.input_channels = 1
        st.session_state.input_size = 28
        st.session_state.normalize_output = True

def render_header():
    """Renderizar el título y descripción de la aplicación"""
    st.title("Comparación entre VAE y Autoencoder Tradicional")
    st.markdown("""
    Esta aplicación interactiva muestra las diferencias entre un Autoencoder tradicional y un Autoencoder Variacional (VAE)
    al trabajar con diferentes datasets de imágenes.

    Explora cómo cada modelo aprende a representar y generar imágenes, y descubre las ventajas y desventajas de cada enfoque.
    """)

def render_sidebar():
    """
    Renderizar la barra lateral con opciones de configuración
    
    Returns:
        dict: Configuración seleccionada por el usuario
    """
    st.sidebar.header("Configuración")

    # Selector de dataset
    dataset_options = ["MNIST", "Fashion MNIST", "CIFAR10", "Cats"]
    DATASET = st.sidebar.selectbox("Dataset:", dataset_options, index=0)

    # Convertir el nombre del dataset al formato esperado por las funciones
    dataset_name_map = {
        "MNIST": "mnist",
        "Fashion MNIST": "fashion_mnist",
        "CIFAR10": "cifar10",
        "Cats": "cats"
    }
    dataset_name = dataset_name_map[DATASET]

    # Parámetros globales según el dataset
    LATENT_DIM = st.sidebar.slider("Dimensión del Espacio Latente:", 2, 512, 2)
    BATCH_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Selector de número de épocas
    epochs_options = [1, 3, 5, 10, 15, 20,30,50,100]
    EPOCHS = st.sidebar.selectbox("Número de épocas:", epochs_options, index=epochs_options.index(5))

    # Información sobre el dispositivo
    st.sidebar.markdown(f"""
    ### Dispositivo: {DEVICE}
    Los modelos se entrenarán en {DEVICE}.
    """)

    return {
        'dataset': DATASET,
        'dataset_name': dataset_name,
        'latent_dim': LATENT_DIM,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'device': DEVICE
    }

def render_dataset_info(config):
    """
    Renderizar información del dataset seleccionado mostrando un ejemplo de cada clase
    
    Args:
        config (dict): Configuración actual
        
    Returns:
        tuple: (data_loaders, dataset_info) - Dataloaders y información del dataset
    """
    # Cargar datos según el dataset seleccionado
    train_loader, test_loader, train_dataset, test_dataset, img_size, channels, normalize_output = load_data(
        dataset_name=config['dataset_name'], 
        batch_size=config['batch_size']
    )

    # Actualizar estado con las dimensiones del dataset
    st.session_state.input_channels = channels
    st.session_state.input_size = img_size
    st.session_state.normalize_output = normalize_output

    # Información general del dataset
    with st.expander("Información del Dataset", expanded=True):
        st.header(f"Información del Dataset: {config['dataset']}")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
            st.write(f"Tamaño del conjunto de prueba: {len(test_dataset)}")
            st.write(f"Dimensiones de las imágenes: {img_size}x{img_size} píxeles ({channels} canal{'es' if channels > 1 else ''})")
            
            # Detectar todas las clases disponibles en el dataset
            available_classes = get_dataset_classes(train_dataset, test_loader)
            st.write(f"Número de clases: {len(available_classes)}")
            class_names = get_class_names(config['dataset_name'])
            if class_names:
                class_info = ", ".join([f"{i}: {name}" for i, name in enumerate(class_names)])
                st.write(f"Clases: {class_info}")

    # Mostrar un ejemplo de cada clase
    with st.expander("Ejemplos de cada clase", expanded=True):
        st.subheader("Ejemplos de imágenes por clase")
        display_class_examples(train_dataset, available_classes, channels, class_names)
    
    data_loaders = {
        'train_loader': train_loader, 
        'test_loader': test_loader
    }
    
    dataset_info = {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'img_size': img_size, 
        'channels': channels,
        'normalize_output': normalize_output,
        'classes': available_classes,
        'class_names': class_names
    }
    
    return data_loaders, dataset_info

def get_dataset_classes(train_dataset, test_loader):
    """
    Obtiene las clases disponibles en el dataset
    
    Args:
        train_dataset: Dataset de entrenamiento
        test_loader: DataLoader de prueba
        
    Returns:
        list: Lista de clases únicas disponibles
    """
    try:
        # Intentar obtener clases del dataset directamente
        if hasattr(train_dataset, 'classes'):
            return list(range(len(train_dataset.classes)))
        
        # Si no es posible, extraer clases únicas de las etiquetas
        labels = []
        
        # Método 1: Intentar con el dataset directamente
        try:
            for i in range(min(1000, len(train_dataset))):  # Limitar a 1000 para eficiencia
                if isinstance(train_dataset[i], tuple):
                    _, label = train_dataset[i]
                    labels.append(label)
        except Exception:
            # Método 2: Intentar con el dataloader
            for _, batch_labels in test_loader:
                labels.extend(batch_labels.numpy())
                break  # Solo necesitamos un lote para encontrar clases
                
        return sorted(list(set(labels)))
    except Exception as e:
        st.warning(f"No se pudieron determinar las clases del dataset: {e}")
        return list(range(10))  # Valor por defecto para la mayoría de datasets

def get_class_names(dataset_name):
    """
    Obtiene los nombres de las clases para datasets conocidos
    
    Args:
        dataset_name: Nombre del dataset
        
    Returns:
        list: Lista de nombres de clases o None si no está disponible
    """
    if dataset_name.lower() == "mnist":
        return [str(i) for i in range(10)]  # Dígitos 0-9
    elif dataset_name.lower() == "fashion_mnist":
        return [
            "Camiseta/Top", "Pantalones", "Jersey", "Vestido", "Abrigo",
            "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"
        ]
    elif dataset_name.lower() == "cifar10":
        return [
            "Avión", "Automóvil", "Pájaro", "Gato", "Ciervo",
            "Perro", "Rana", "Caballo", "Barco", "Camión"
        ]
    elif dataset_name.lower() == "cats":
        return ["Gato"]
    else:
        return None

def display_class_examples(dataset, classes, channels, class_names=None):
    """
    Muestra un ejemplo de cada clase del dataset
    
    Args:
        dataset: Dataset que contiene las imágenes
        classes: Lista de clases disponibles
        channels: Número de canales de las imágenes
        class_names: Nombres de las clases (opcional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    # Determinar el número de filas y columnas para la cuadrícula
    n_classes = len(classes)
    n_cols = min(5, n_classes)  # Máximo 5 columnas
    n_rows = math.ceil(n_classes / n_cols)
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    
    # Asegurarse de que axes sea un array 2D incluso con una sola fila/columna
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Diccionario para almacenar ejemplos encontrados por clase
    class_examples = {}
    
    # Buscar un ejemplo de cada clase
    for i, item in enumerate(dataset):
        if i >= 10000:  # Limitar búsqueda para datasets grandes
            break
            
        if isinstance(item, tuple):
            img, label = item
            
            # Convertir tensor/numpy a int si es necesario
            if hasattr(label, 'item'):
                label = label.item()
                
            # Si aún no tenemos un ejemplo de esta clase, guardarlo
            if label not in class_examples:
                class_examples[label] = img
                
            # Si ya tenemos ejemplos de todas las clases, salir del bucle
            if len(class_examples) == len(classes):
                break
    
    # Mostrar ejemplos de cada clase
    for idx, class_label in enumerate(classes):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # Verificar si tenemos un ejemplo para esta clase
        if class_label in class_examples:
            img = class_examples[class_label]
            
            # Mostrar imagen según número de canales
            if channels == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                # Convertir tensor a formato de imagen
                if hasattr(img, 'permute'):
                    img_np = img.permute(1, 2, 0).numpy()
                    # Asegurarse de que los valores están en [0, 1]
                    img_np = np.clip(img_np, 0, 1)
                else:
                    img_np = img
                ax.imshow(img_np)
            
            # Título con clase y nombre (si está disponible)
            if class_names and class_label < len(class_names):
                title = f"Clase {class_label}: {class_names[class_label]}"
            else:
                title = f"Clase {class_label}"
                
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, f"No hay ejemplos\nde la clase {class_label}", 
                   ha='center', va='center')
            
        ax.axis('off')
    
    # Ocultar ejes vacíos si hay más celdas que clases
    for idx in range(len(classes), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def render_model_architecture(config, dataset_info):
    """
    Renderizar la sección de arquitectura de los modelos
    
    Args:
        config (dict): Configuración actual
        dataset_info (dict): Información del dataset
    """
    with st.expander("Arquitectura de los Modelos", expanded=True):
        st.subheader("Arquitectura de los Modelos")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Autoencoder Tradicional")
            st.markdown(f"""
            - **Encoder**: Capas convolucionales con activación ReLU
            - **Espacio Latente**: Vector determinista (dimensión {config['latent_dim']})
            - **Decoder**: Capas transpuestas convolucionales con activación ReLU y Sigmoid final
            - **Función de Pérdida**: Binary Cross Entropy (BCE)
            - **Canales de entrada**: {dataset_info['channels']}
            - **Tamaño de imagen**: {dataset_info['img_size']}x{dataset_info['img_size']}
            """)

        with col2:
            st.markdown("### Autoencoder Variacional (VAE)")
            st.markdown(f"""
            - **Encoder**: Capas convolucionales con activación ReLU que producen μ y σ²
            - **Espacio Latente**: Distribución normal parametrizada por μ y σ (dimensión {config['latent_dim']})
            - **Reparametrización**: z = μ + ε⋅σ, donde ε ~ N(0,1)
            - **Decoder**: Capas transpuestas convolucionales con activación ReLU y Sigmoid final
            - **Función de Pérdida**: BCE + KL Divergence
            - **Canales de entrada**: {dataset_info['channels']}
            - **Tamaño de imagen**: {dataset_info['img_size']}x{dataset_info['img_size']}
            """)
            
            st.markdown("""
            #### La Diferencia Clave:
            El Autoencoder tradicional codifica cada imagen como un punto específico en el espacio latente, 
            mientras que el VAE codifica cada imagen como una distribución de probabilidad, permitiendo 
            generación y muestreo más robusto.
            """)

def load_or_train_models(config, data_loaders, dataset_info):
    """
    Cargar o entrenar modelos según la configuración
    
    Args:
        config (dict): Configuración actual
        data_loaders (dict): DataLoaders para entrenamiento y prueba
        dataset_info (dict): Información del dataset
        
    Returns:
        dict: Información de los modelos entrenados/cargados
    """
    try:
        # Verificar si ya existen modelos entrenados
        models_trained = models_exist(
            config['epochs'],
            config['latent_dim'], 
            dataset_name=config['dataset_name'], 
            models_dir=MODELS_DIR
        )
        
        if models_trained:
            with st.spinner(f"Cargando modelos entrenados con {config['epochs']} épocas para {config['dataset']}..."):
                ae_model, vae_model, ae_losses, vae_losses = load_models(
                    config['epochs'], 
                    config['latent_dim'], 
                    config['device'],
                    dataset_name=config['dataset_name'],
                    input_channels=dataset_info['channels'],
                    input_size=dataset_info['img_size'],
                    models_dir=MODELS_DIR
                )
            st.success(f"Modelos cargados correctamente con {config['epochs']} épocas para {config['dataset']}.")
        else:
            with st.spinner(f"Entrenando modelos con {config['epochs']} épocas para {config['dataset']}..."):
                with st.spinner("Entrenando Autoencoder..."):
                    ae_model, ae_losses = train_convolutional_autoencoder(
                        data_loaders['train_loader'], 
                        config['device'], 
                        config['latent_dim'], 
                        config['epochs'],
                        input_channels=dataset_info['channels'],
                        input_size=dataset_info['img_size'],
                        normalize_output=dataset_info['normalize_output'],
                        dataset_name=config['dataset_name']
                    )
                
                with st.spinner("Entrenando VAE..."):
                    vae_model, vae_losses = train_convolutional_vae(
                        data_loaders['train_loader'], 
                        config['device'], 
                        config['latent_dim'], 
                        config['epochs'],
                        input_channels=dataset_info['channels'],
                        input_size=dataset_info['img_size'],
                        normalize_output=dataset_info['normalize_output'],
                        dataset_name=config['dataset_name']
                    )
                
                # Guardar modelos y pérdidas
                save_models(
                    ae_model, 
                    vae_model, 
                    ae_losses, 
                    vae_losses, 
                    config['epochs'],
                    config['latent_dim'], 
                    dataset_name=config['dataset_name'],
                    input_channels=dataset_info['channels'],
                    input_size=dataset_info['img_size'],
                    models_dir=MODELS_DIR
                )
            
            st.success(f"¡Entrenamiento completado y modelos guardados con {config['epochs']} épocas para {config['dataset']}!")
        
        # Guardar modelos en session state para usarlos en el sampling de latent space
        st.session_state.ae_model = ae_model
        st.session_state.vae_model = vae_model
        st.session_state.ae_losses = ae_losses
        st.session_state.vae_losses = vae_losses
        st.session_state.models_loaded = True
        
        # Mostrar pérdidas
        visualize_loss_curves(ae_losses, vae_losses, config['epochs'])
        
        # Obtener embeddings
        with st.spinner("Generando visualizaciones del espacio latente..."):
            ae_embeddings, ae_labels = get_embeddings(
                ae_model, 
                data_loaders['test_loader'], 
                config['device'], 
                is_vae=False
            )
            vae_embeddings, vae_labels = get_embeddings(
                vae_model, 
                data_loaders['test_loader'], 
                config['device'], 
                is_vae=True
            )
            
            # Guardar embeddings en session state
            st.session_state.ae_embeddings = ae_embeddings
            st.session_state.vae_embeddings = vae_embeddings
            st.session_state.ae_labels = ae_labels
            st.session_state.vae_labels = vae_labels
            
            # Visualizar espacio latente
            visualize_latent_space(ae_embeddings, vae_embeddings, ae_labels, vae_labels, config['latent_dim'])
            
        return {
            'ae_model': ae_model,
            'vae_model': vae_model,
            'ae_losses': ae_losses,
            'vae_losses': vae_losses,
            'ae_embeddings': ae_embeddings,
            'vae_embeddings': vae_embeddings,
            'ae_labels': ae_labels,
            'vae_labels': vae_labels
        }
    except Exception as e:
        st.error(f"Error al cargar o entrenar los modelos: {str(e)}")
        st.exception(e)
        return None

def render_model_training(config, data_loaders, dataset_info):
    """
    Renderizar la sección de entrenamiento/carga de modelos
    
    Args:
        config (dict): Configuración actual
        data_loaders (dict): DataLoaders para entrenamiento y prueba
        dataset_info (dict): Información del dataset
        
    Returns:
        dict or None: Información de los modelos si se entrenan/cargan
    """
    # Botón para cargar/entrenar modelos
    if st.button("Entrenar/Cargar Modelos", type="primary"):
        return load_or_train_models(config, data_loaders, dataset_info)
    return None

def render_visualizations(models_info, config, data_loaders, dataset_info):
    """
    Renderizar visualizaciones una vez que los modelos están cargados
    
    Args:
        models_info (dict): Información de los modelos
        config (dict): Configuración actual
        data_loaders (dict): DataLoaders para entrenamiento y prueba
        dataset_info (dict): Información del dataset
    """
    if models_info or st.session_state.models_loaded:
        try:
            # Usar los modelos del parámetro o de session state
            ae_model = models_info['ae_model'] if models_info else st.session_state.ae_model
            vae_model = models_info['vae_model'] if models_info else st.session_state.vae_model
            ae_embeddings = models_info['ae_embeddings'] if models_info else st.session_state.ae_embeddings
            vae_embeddings = models_info['vae_embeddings'] if models_info else st.session_state.vae_embeddings
            ae_labels = models_info['ae_labels'] if models_info else st.session_state.ae_labels
            vae_labels = models_info['vae_labels'] if models_info else st.session_state.vae_labels
            
            # Visualización interactiva del espacio latente
            interactive_latent_space(
                ae_model, 
                vae_model, 
                config['device'], 
                ae_embeddings, 
                vae_embeddings, 
                ae_labels, 
                vae_labels, 
                config['latent_dim'],
                channels=dataset_info['channels'],
                normalize_output=dataset_info['normalize_output']
            )
            
            # Visualizaciones adicionales
            with st.expander("Generación de Imágenes Adicionales", expanded=False):
                st.header("Generación de Imágenes")
                st.markdown("""
                Aquí puedes ver diferentes formas de generar imágenes a partir del espacio latente.
                """)
                
                if config['latent_dim'] == 2:
                    # Generación a partir de puntos específicos
                    visualize_grid_generation(
                        ae_model, 
                        vae_model, 
                        config['device'], 
                        config['latent_dim'],
                        channels=dataset_info['channels'],
                        normalize_output=dataset_info['normalize_output']
                    )
                    
                    # Interpolación entre clases
                    visualize_interpolation(
                        ae_model, 
                        vae_model, 
                        config['device'], 
                        data_loaders['test_loader'],
                        channels=dataset_info['channels'],
                        normalize_output=dataset_info['normalize_output']
                    )
                    
                    # Muestreo aleatorio
                    visualize_random_generation(
                        ae_model, 
                        vae_model, 
                        config['device'], 
                        config['latent_dim'],
                        channels=dataset_info['channels'],
                        normalize_output=dataset_info['normalize_output']
                    )
            
            # Mostrar reconstrucciones
            with st.expander("Comparación de Reconstrucción", expanded=False):
                visualize_reconstructions(
                    ae_model, 
                    vae_model, 
                    data_loaders['test_loader'], 
                    config['device'],
                    channels=dataset_info['channels'],
                    normalize_output=dataset_info['normalize_output']
                )
        except Exception as e:
            st.error(f"Error al mostrar visualizaciones: {str(e)}")
            st.exception(e)

def render_conclusions():
    """
    Renderizar la sección de conclusiones
    """
    with st.expander("Diferencias y Conclusiones", expanded=True):
        st.header("Diferencias y Conclusiones")
        
        st.markdown("""
        ### Principales diferencias entre Autoencoder y VAE
        
        #### Espacio Latente
        - **Autoencoder Tradicional**: Produce un espacio latente discreto y no estructurado. Los puntos se distribuyen de manera irregular y pueden tener grandes espacios vacíos.
        - **VAE**: Genera un espacio latente continuo y suave, con una distribución más regular que se aproxima a una distribución normal. Esto facilita la generación de nuevas muestras mediante interpolación o muestreo aleatorio.
        
        #### Generación de Nuevas Imágenes
        - **Autoencoder Tradicional**: Al generar imágenes desde puntos aleatorios del espacio latente, las imágenes pueden ser poco realistas o no interpretables si esos puntos caen en regiones no cubiertas por los datos de entrenamiento.
        - **VAE**: La regularización KL fuerza el espacio latente a seguir una distribución normal, lo que permite generar nuevas imágenes realistas muestreando de una distribución normal.
        
        #### Interpolación
        - **Autoencoder Tradicional**: La interpolación puede producir transiciones menos suaves entre imágenes.
        - **VAE**: La estructura continua y suave del espacio latente permite interpolaciones más coherentes entre elementos.
        
        #### Función de Pérdida
        - **Autoencoder Tradicional**: Solo optimiza la reconstrucción de las imágenes originales.
        - **VAE**: Balanceo entre calidad de reconstrucción y regularización del espacio latente (KL Divergence).
        
        ### ¿Cuál elegir?
        
        La elección entre un Autoencoder tradicional y un VAE depende del objetivo:
        
        - Si el principal objetivo es **compresión de datos** o **reducción de dimensionalidad**, un Autoencoder tradicional puede ser suficiente.
        - Si el objetivo es **generar nuevas muestras** o tener un espacio latente con propiedades específicas (como continuidad o interpolación suave), un VAE es la mejor opción.
        
        ### Aplicaciones
        
        - **Autoencoders**: Reducción de dimensionalidad, detección de anomalías, denoising (eliminación de ruido).
        - **VAEs**: Generación de imágenes, aprendizaje de representaciones, síntesis de datos, transferencia de estilo.
        """)

def main():
    """Función principal de la aplicación"""
    # Semilla para reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Inicializar estados de sesión
    initialize_session_state()
    
    # Título principal y descripción
    render_header()
    
    # Obtener configuración de la sidebar
    config = render_sidebar()
    
    # Información del dataset
    data_loaders, dataset_info = render_dataset_info(config)
    
    # Sección de modelos y entrenamiento
    st.header("Modelos y Entrenamiento")
    
    # Arquitectura de los modelos
    render_model_architecture(config, dataset_info)
    
    # Entrenamiento o carga de modelos
    models_info = render_model_training(config, data_loaders, dataset_info)
    
    # Si los modelos ya están cargados, mostrar visualizaciones
    if models_info or st.session_state.models_loaded:
        render_visualizations(models_info, config, data_loaders, dataset_info)
        render_conclusions()
    else:
        st.info("👆 Presiona el botón para entrenar/cargar los modelos y ver las comparaciones. El proceso puede tardar unos minutos si se necesita entrenar desde cero.")

if __name__ == "__main__":
    # Importación condicional para evitar errores de circular import
    import matplotlib.pyplot as plt
    main()