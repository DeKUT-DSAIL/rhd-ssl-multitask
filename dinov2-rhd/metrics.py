import random
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

    
def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = (y_true == y_pred).mean()
    
    # Calculate specificity for each class
    cm = confusion_matrix(y_true, y_pred)
    specificity = []
    for i in range(len(cm)):
        true_neg = np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + cm[i,i]
        false_pos = np.sum(cm[:,i]) - cm[i,i]
        specificity.append(true_neg / (true_neg + false_pos))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': np.mean(specificity),
        'f1': f1
    }

def plot_confusion_matrix(y_true, y_pred, labels, task_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    
    figsize = (len(labels) * 1.5, len(labels) * 1.2)  # Adjust these multipliers as needed
    plt.figure(figsize=figsize)

    # Plot the confusion matrix with adjusted parameters for better visibility
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 14}) # Increased annotation font size


    plt.title(f'Confusion Matrix - {task_name} Classification', fontsize=20) # Increased title font size
    plt.ylabel('True Label', fontsize=16) # Increased label font size
    plt.xlabel('Predicted Label', fontsize=16) # Increased label font size

    plt.xticks(rotation=45, ha='right', fontsize=15) # Rotated x-axis labels and set font size
    plt.yticks(rotation=0, fontsize=15) # Set y-axis label font size

    #plt.tight_layout()

    if save_path:
        plt.savefig(save_path + ".png", bbox_inches='tight', dpi=400)
        plt.savefig(save_path + ".svg")
    plt.close()

def visualize_embeddings(embeddings, labels, video_ids, method='umap', save_path=None, label_type='view', color_scheme='video', label_decoders=None):
    """
    Visualize embeddings using t-SNE or UMAP with different color schemes based on video IDs or classification labels.
    """
    plt.figure(figsize=(20, 16))
    
    # Determine the number of components for PCA
    n_samples, n_features = embeddings.shape
    n_components = min(50, n_samples, n_features)
    
    # Perform PCA to reduce to n_components
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(random_state=42)
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    reduced = reducer.fit_transform(pca_embeddings)
    
    if color_scheme == 'video':
        if video_ids is None:
            raise ValueError("video_ids cannot be None when color_scheme is 'video'")
        
        # Ensure video IDs are sorted numerically if possible
        try:
            unique_ids = sorted(set(video_ids), key=lambda x: int(''.join(filter(str.isdigit, x))))
        except ValueError:
            unique_ids = sorted(set(video_ids))
        
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_ids)))
        color_map = dict(zip(unique_ids, colors))
        
        # Create a mapping from actual video IDs to shorter labels
        video_id_mapping = {vid: f"Video {i+1}" for i, vid in enumerate(unique_ids)}
        short_video_ids = np.array([video_id_mapping[vid] for vid in video_ids])
        
        for vid in unique_ids:
            mask = np.array(video_ids) == vid
            plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                       c=[color_map[vid]], label=video_id_mapping[vid], 
                       alpha=0.6, s=50)
        
        title = f'{method.upper()} - {label_type.capitalize()} (By Video ID)'
    elif color_scheme == 'label':
        unique_labels = sorted(set(labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))
        
        if label_decoders and label_type in label_decoders:
            decoded_labels = [label_decoders[label_type].get(label, f"Unknown ({label})") for label in unique_labels]
        else:
            decoded_labels = unique_labels
        
        for label, decoded_label in zip(unique_labels, decoded_labels):
            mask = np.array(labels) == label
            plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                       c=[color_map[label]], label=f"{decoded_label}", 
                       alpha=0.6, s=50)
        
        title = f'{method.upper()} - {label_type.capitalize()} (By {label_type.capitalize()} Label)'
    else:
        raise ValueError(f"Unknown color scheme: {color_scheme}")
    
    plt.title(title, fontsize=24, weight='bold')
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=16, framealpha=0.3)  
    plt.tight_layout()

    if save_path:
        for ext in ['.png', '.svg']:
            full_path = save_path + ext
            plt.savefig(full_path, dpi=400, bbox_inches='tight')
            logging.info(f"Saved {title} visualization to {full_path}")
    
    plt.show()  # Display the plot
    plt.close()

