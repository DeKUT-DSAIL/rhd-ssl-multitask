import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np
import torch
import logging
import seaborn as sns

def plot_training_loss(losses, filename='training_loss.png'):
    """
    Plots and saves the training loss curve.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    logging.info(f"Training loss plot saved as {filename}")

def get_embeddings(model, dataloader, device, label_type):
    """
    Extracts embeddings and labels from the model for the given dataloader.
    """
    model.eval()
    embeddings = []
    labels = []

    logging.info(f"Extracting embeddings for {label_type} visualization...")

    with torch.no_grad():
        for x_i, x_j, labels_dict in dataloader:
            images = x_i.to(device)
            h, _ = model(images)
            embeddings.append(h.cpu().numpy())
            labels.extend(labels_dict[label_type])

    embeddings = np.concatenate(embeddings, axis=0)
    logging.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings, labels

def reduce_embeddings(embeddings, n_components=50):
    """
    Reduces embeddings dimensionality using PCA.
    """
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    logging.info(f"Reduced embeddings to {n_components} components using PCA.")
    return reduced_embeddings, pca

def visualize_embeddings(embeddings, labels, method, filename, label_type):
    """
    Reduces embeddings to 2D using the specified method and visualizes them.
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        logging.info("Performing t-SNE dimensionality reduction.")
    elif method == 'umap':
        reducer = umap.UMAP(random_state=42)
        logging.info("Performing UMAP dimensionality reduction.")
    else:
        raise ValueError(f"Unknown visualization method: {method}")

    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and create color map
    unique_labels = sorted(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    
    # Plot each class separately
    for label in unique_labels:
        mask = np.array(labels) == label
        plt.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            c=[label_to_color[label]],
            label=label,
            alpha=0.6
        )
    
    plt.title(f'{method.upper()} Visualization of {label_type.capitalize()} Classes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Embeddings visualization ({method}) saved as {filename}")

def plot_confusion_matrix(cm, classes, filename, label_type):
    """
    Plots and saves a confusion matrix.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {label_type.capitalize()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Confusion matrix saved as {filename}")