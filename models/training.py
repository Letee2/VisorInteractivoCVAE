import torch
import torch.optim as optim
import pandas as pd
import streamlit as st
import numpy as np
from models.autoencoder import ae_loss_function
from models.vae import vae_loss_function

def weights_init(m):
    """Inicializar pesos de la red neuronal"""
    import torch.nn as nn
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train_convolutional_autoencoder(train_loader, device, latent_dim, epochs, input_channels=1, input_size=28, normalize_output=False, dataset_name="mnist"):
    """
    Entrenamiento del autoencoder con optimizaciones adaptativas según el dataset
    
    Args:
        train_loader: DataLoader con datos de entrenamiento
        device: Dispositivo (CPU/GPU)
        latent_dim: Dimensión del espacio latente
        epochs: Número de épocas
        input_channels: Número de canales de entrada (1 para escala de grises, 3 para color)
        input_size: Tamaño de la imagen (ancho=alto)
        normalize_output: Si True, los datos están normalizados en [-1,1]
        dataset_name: Nombre del dataset para adaptaciones específicas
        
    Returns:
        model: Modelo entrenado
        losses: Lista de pérdidas durante el entrenamiento
    """
    try:
        from models.adaptive_models import create_autoencoder
        
        # Determinar si es un dataset complejo
        is_complex_dataset = (dataset_name.lower() == "cifar10" or
                            (input_channels == 3 and input_size >= 32))
        
        # Crear modelo adaptado al dataset
        model = create_autoencoder(
            latent_dim=latent_dim,
            input_channels=input_channels,
            input_size=input_size,
            dataset_name=dataset_name
        ).to(device)
        
        # Inicializar pesos
        model.apply(weights_init)
        
        # Configurar optimizer y scheduler según la complejidad del dataset
        if is_complex_dataset:
            # Para datasets complejos: tasa de aprendizaje más alta y weight decay
            optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
        else:
            # Para datasets simples: configuración básica
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = None
        
        # Entrenamiento
        model.train()
        losses = []
        
        # Crear barra de progreso y área para mostrar métricas
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.empty()
        temp_losses = []
        
        # Determinar factor de normalización para pérdidas para datasets complejos
        # Permite presentar valores más significativos para el usuario
        loss_normalization = 1.0
        if is_complex_dataset and normalize_output:
            # Para datasets como CIFAR, normalizar la pérdida según el número de píxeles
            loss_normalization = input_size * input_size * input_channels
        
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                
                reconstructed, _ = model(data)
                loss = ae_loss_function(reconstructed, data, normalize_output)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Normalizar la pérdida para mostrar valores más interpretables
            avg_loss = train_loss / (len(train_loader.dataset) * loss_normalization)
            losses.append(avg_loss)
            temp_losses.append(avg_loss)
            
            # Actualizar scheduler si existe
            if scheduler:
                scheduler.step(avg_loss)
                current_lr = optimizer.param_groups[0]['lr']
                lr_info = f" - LR: {current_lr:.6f}"
            else:
                lr_info = ""
            
            # Actualizar interfaz
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Época {epoch+1}/{epochs} - Pérdida: {avg_loss:.6f}{lr_info}")
            
            # Actualizar gráfico cada época
            if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
                chart_data = pd.DataFrame({"Época": list(range(1, len(temp_losses) + 1)), "Pérdida": temp_losses})
                loss_chart.line_chart(chart_data.set_index("Época"))
        
        # Limpiar componentes temporales
        progress_bar.empty()
        status_text.empty()
        loss_chart.empty()
        
        return model, losses
        
    except Exception as e:
        st.error(f"Error en el entrenamiento del autoencoder: {str(e)}")
        st.exception(e)
        return None, []

def train_convolutional_vae(train_loader, device, latent_dim, epochs, input_channels=1, input_size=28, normalize_output=False, dataset_name="mnist"):
    """
    Entrenamiento del VAE con optimizaciones adaptativas según el dataset
    
    Args:
        train_loader: DataLoader con datos de entrenamiento
        device: Dispositivo (CPU/GPU)
        latent_dim: Dimensión del espacio latente
        epochs: Número de épocas
        input_channels: Número de canales de entrada (1 para escala de grises, 3 para color)
        input_size: Tamaño de la imagen (ancho=alto)
        normalize_output: Si True, los datos están normalizados en [-1,1]
        dataset_name: Nombre del dataset para adaptaciones específicas
        
    Returns:
        model: Modelo entrenado
        losses: Diccionario con histórico de pérdidas
    """
    try:
        from models.adaptive_models import create_vae
        
        # Determinar si es un dataset complejo
        is_complex_dataset = (dataset_name.lower() == "cifar10" or
                            (input_channels == 3 and input_size >= 32))
        
        # Crear modelo adaptado al dataset
        model = create_vae(
            latent_dim=latent_dim,
            input_channels=input_channels,
            input_size=input_size,
            dataset_name=dataset_name
        ).to(device)
        
        # Inicializar pesos
        model.apply(weights_init)
        
        # Configurar optimizer, scheduler y beta-VAE según la complejidad del dataset
        if is_complex_dataset:
            # Para datasets complejos: tasa de aprendizaje más alta, weight decay y beta reducido
            optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
            beta = 0.5  # Valor más bajo para priorizar reconstrucción en datasets complejos
        else:
            # Para datasets simples: configuración básica
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = None
            beta = 1.0  # Valor estándar para datasets simples
        
        # Entrenamiento
        model.train()
        losses = {"total": [], "bce": [], "kld": []}
        
        # Crear barra de progreso y área para mostrar métricas
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.empty()
        
        # Determinar factor de normalización para pérdidas
        loss_normalization = 1.0
        if is_complex_dataset and normalize_output:
            # Para datasets como CIFAR, normalizar la pérdida según el número de píxeles
            loss_normalization = input_size * input_size * input_channels
        
        # Función de pérdida VAE modificada para usar beta
        def beta_vae_loss(reconstructed, x, mu, logvar, normalize_output):
            rec_loss, kl_loss = vae_loss_function(reconstructed, x, mu, logvar, normalize_output)[1:]
            # Ajustar el peso del término KL con beta
            return rec_loss + beta * kl_loss, rec_loss, kl_loss
        
        for epoch in range(epochs):
            train_loss = 0
            bce_loss = 0
            kld_loss = 0
            
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                
                reconstructed, mu, logvar, _ = model(data)
                loss, bce, kld = beta_vae_loss(reconstructed, data, mu, logvar, normalize_output)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                bce_loss += bce.item()
                kld_loss += kld.item()
            
            # Normalizar las pérdidas para mostrar valores más interpretables
            avg_loss = train_loss / (len(train_loader.dataset) * loss_normalization)
            avg_bce = bce_loss / (len(train_loader.dataset) * loss_normalization)
            avg_kld = kld_loss / (len(train_loader.dataset) * loss_normalization)
            
            losses["total"].append(avg_loss)
            losses["bce"].append(avg_bce)
            losses["kld"].append(avg_kld)
            
            # Actualizar scheduler si existe
            if scheduler:
                scheduler.step(avg_loss)
                current_lr = optimizer.param_groups[0]['lr']
                lr_info = f" - LR: {current_lr:.6f}"
            else:
                lr_info = ""
            
            # Actualizar interfaz
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Época {epoch+1}/{epochs} - Total: {avg_loss:.4f}, Rec: {avg_bce:.4f}, KLD: {avg_kld:.4f}{lr_info}")
            
            # Actualizar gráfico en cada época
            if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
                chart_data = pd.DataFrame({
                    "Época": list(range(1, epoch + 2)),
                    "Total": losses["total"],
                    "Rec": losses["bce"],
                    "KLD": losses["kld"]
                })
                loss_chart.line_chart(chart_data.set_index("Época"))
        
        # Limpiar componentes temporales
        progress_bar.empty()
        status_text.empty()
        loss_chart.empty()
        
        return model, losses
        
    except Exception as e:
        st.error(f"Error en el entrenamiento del VAE: {str(e)}")
        st.exception(e)
        return None, {"total": [], "bce": [], "kld": []}