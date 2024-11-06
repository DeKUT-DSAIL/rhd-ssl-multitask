import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

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
    plt.figure(figsize=(12, 10))
    
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    # Add task-specific title and detailed class information
    if task_name.upper() == 'VIEW':
        plt.title('Confusion Matrix - View Classification\n\n' + 
                 'PLAX: Parasternal long axis (724 samples)\n' +
                 'PSAX: Parasternal short axis (1448 samples)\n' +
                 'A4C: Apical Four Chamber (483 samples)')
    elif task_name.upper() == 'CONDITION':
        plt.title('Confusion Matrix - Condition Classification\n\n' +
                 'AVR+PVR: Aortic & Pulmonary Valve Regurgitation (121)\n' +
                 'AVR: Aortic Valve Regurgitation (121)\n' +
                 'MVP: Mitral Valve Prolapse (241)\n' +
                 'MVR: Mitral Valve Regurgitation (121)\n' +
                 'NA: Not Applicable (1809)\n' +
                 'PVR: Pulmonary Valve Regurgitation (121)\n' +
                 'TVR: Tricuspid Valve Regurgitation (121)')
    elif task_name.upper() == 'SEVERITY':
        plt.title('Confusion Matrix - Severity Classification\n\n' +
                 'Borderline RHD (242 samples)\n' +
                 'Definite RHD (241 samples)\n' +
                 'Not Applicable (2172 samples)')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=330)
    plt.close()

def visualize_embeddings(embeddings, labels, label_names, task_name, method='umap', save_path=None):
    """
    Visualize embeddings using UMAP or t-SNE
    
    Args:
        embeddings: numpy array of embeddings
        labels: numpy array of labels
        label_names: dictionary mapping label indices to label names
        task_name: string indicating the task (view/condition/severity)
        method: 'umap' or 'tsne'
        save_path: path to save the visualization
    """
    # Set up the reducer
    if method == 'umap':
        reducer = umap.UMAP(random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    
    # Convert embeddings and reduce dimensionality
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(15, 10))
    
    # Ensure labels are integers
    labels = np.array(labels).astype(int)
    
    # Get unique labels and their counts
    unique_labels = np.unique(labels)
    counts = {label: np.sum(labels == label) for label in unique_labels}
    
    # Create scatter plot
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab20')
    
    # Create legend elements
    if label_names is not None:
        legend_elements = []
        for i in unique_labels:
            # Convert to int to use as dictionary key
            idx = int(i)
            if idx in label_names:
                legend_elements.append(
                    plt.Line2D([0], [0], 
                              marker='o', 
                              color='w',
                              markerfacecolor=plt.cm.tab20(idx/len(unique_labels)),
                              label=f'{label_names[idx]} ({counts[idx]} samples)',
                              markersize=10)
                )
        
        if legend_elements:  # Only add legend if we have elements
            plt.legend(handles=legend_elements,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left',
                      title=f'{task_name} Classes',
                      fontsize=10)
    else:
        plt.colorbar(scatter)
    
    plt.title(f'{method.upper()} visualization of embeddings - {task_name} Classification')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=330)
    plt.close()