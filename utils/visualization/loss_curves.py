"""
Funciones para visualizar curvas de pérdida durante el entrenamiento.
"""
import matplotlib.pyplot as plt
import streamlit as st

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
        
        print("AAAAAAAAAAAAAAAA")
        print(vae_losses["bce"])
        print(ae_losses)
        

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