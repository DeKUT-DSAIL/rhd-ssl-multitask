import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import logging

def visualize_samples(dataloader, num_samples=5, label_decoders=None, save_path=None):
    """
    Visualize samples with decoded labels.
    """
    data_iter = iter(dataloader)
    images, labels, video_ids = next(data_iter)
    view_labels, condition_labels, severity_labels = labels
    
    fig, axes = plt.subplots(1, min(num_samples, len(images)), figsize=(30, 10))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for i in range(min(num_samples, len(images))):
        # Convert and normalize image
        image = images[i].permute(1, 2, 0)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = image * std + mean
        image = torch.clamp(image, 0, 1).numpy()
        
        # Get decoded labels
        if label_decoders:
            view = label_decoders['view'].get(view_labels[i].item(), f"Unknown ({view_labels[i].item()})")
            condition = label_decoders['condition'].get(condition_labels[i].item(), f"Unknown ({condition_labels[i].item()})")
            severity = label_decoders['severity'].get(severity_labels[i].item(), f"Unknown ({severity_labels[i].item()})")
            
            # Format the labels nicely
            view = view.replace("(", "\n(")
            condition = condition.replace("[", "").replace("]", "").replace("'", "").replace(",","\n,")
            severity = severity.replace("rhd", "RHD")
            
            label_text = (
                f'View: {view}\n'
                f'Condition: {condition}\n'
                f'Severity: {severity}'
            )
        else:
            label_text = (
                f'View: {view_labels[i].item()}\n'
                f'Condition: {condition_labels[i].item()}\n'
                f'Severity: {severity_labels[i].item()}'
            )

        # Plot image
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(label_text, fontsize=18, wrap=True)

    plt.subplots_adjust(wspace=.9, hspace=.5)
    
    if save_path:
        png_save_path = save_path + ".png"
        svg_save_path = save_path + ".svg"

        plt.savefig(png_save_path, dpi=400, bbox_inches='tight') # save as png and svg
        plt.savefig(svg_save_path)
    plt.close()


def visualize_test_samples(images, actual_labels, predicted_labels, label_decoders, num_samples=3, save_path=None, project_title=""):
    """
    Visualize test samples with their actual and predicted labels.
    """
    tasks = ['view', 'condition', 'severity']
    num_tasks = len(tasks)
    
    # Create figure with more height to accommodate labels
    fig, axes = plt.subplots(num_tasks, num_samples, figsize=(7*num_samples, 9*num_tasks))
    plt.subplots_adjust(hspace=.8, wspace=0.5)  # Increased spacing
    
    # Convert images if they're tensors
    if torch.is_tensor(images[0]):
        images = [img.cpu().numpy() for img in images[:num_samples]]
    images = np.array(images)[:num_samples]
    actual_labels = np.array(actual_labels)[:num_samples]
    
    task_titles = {
        'view': 'Echocardiographic View',
        'condition': 'RHD Condition',
        'severity': 'RHD Severity'
    }
    
    # Add a legend for true/predicted labels
    legend_elements = [
        plt.Line2D([0], [0], color='w', label='True: True Label'),
        plt.Line2D([0], [0], color='w', label='Pred: Predicted Label')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=16)
    
    
    for i, task in enumerate(tasks):
        # Add row title
        fig.text(0.04, 0.80 - (i * 0.33), task_titles[task], 
            fontsize=20, fontweight='bold', rotation=90, 
            ha='center', va='center')
        
        for j in range(num_samples):
            ax = axes[i, j]
            
            # Show image in all rows
            img = images[j].transpose(1, 2, 0)
            # Denormalize image
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            
            # Get true and predicted labels
            true_label_idx = actual_labels[j, i]
            pred_label_idx = predicted_labels[task][j]
            
            # Decode labels
            try:
                true_label = label_decoders[task][true_label_idx]
                pred_label = label_decoders[task][pred_label_idx]
                
                # Format condition labels (which might be strings or lists)
                if task == 'condition':
                    if isinstance(true_label, str) and true_label.startswith('['):
                        true_label = '\n'.join(eval(true_label))
                    if isinstance(pred_label, str) and pred_label.startswith('['):
                        pred_label = '\n'.join(eval(pred_label))
                
                # Set colors based on correctness
                color = 'green' if true_label_idx == pred_label_idx else 'red'
                
                # Add white background to text for better visibility
                bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
                
                # Add labels with more spacing and background
                ax.text(0.5, -0.1, f'True: {true_label}', 
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       transform=ax.transAxes,
                       fontsize=14,
                       wrap=True,
                       bbox=bbox_props)
                
                ax.text(0.5, -0.2, f'Pred: {pred_label}',  # Increased spacing
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       transform=ax.transAxes,
                       fontsize=14,
                       color=color,
                       wrap=True,
                       bbox=bbox_props)
                
            except Exception as e:
                logging.error(f"Error processing labels for task {task}: {str(e)}")
    
    plt.suptitle(project_title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    if save_path:
        for ext in ['.png', '.svg']:
            full_path = save_path + ext
            plt.savefig(full_path, dpi=400, bbox_inches='tight', pad_inches=0.5)
            logging.info(f"Test samples visualization saved to {full_path}")
    
    plt.close()

def visualize_two_videos(embeddings, video_ids, method='umap', save_path=None):
    """
    Visualize embeddings of frames from two randomly selected videos.
    """
    plt.figure(figsize=(16, 12))
    
    # Randomly select two unique video IDs
    unique_video_ids = list(set(video_ids))
    selected_videos = random.sample(unique_video_ids, 2)
    
    # Filter embeddings and video IDs for the selected videos
    mask = np.isin(video_ids, selected_videos)
    filtered_embeddings = embeddings[mask]
    filtered_video_ids = np.array(video_ids)[mask]
    
    # Perform PCA to reduce to 50 components
    n_samples, n_features = filtered_embeddings.shape
    n_components = min(50, n_samples, n_features)
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(filtered_embeddings)
    
    # Apply UMAP or t-SNE
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(random_state=42)
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    reduced = reducer.fit_transform(pca_embeddings)
    
    # Create a mapping from actual video IDs to shorter labels
    video_id_mapping = {vid: f"Video {i+1}" for i, vid in enumerate(selected_videos)}
    short_video_ids = np.array([video_id_mapping[vid] for vid in filtered_video_ids])
    
    # Plot the embeddings
    unique_labels = sorted(set(short_video_ids))
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))
    
    for label in unique_labels:
        mask = short_video_ids == label
        plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                   c=[color_map[label]], label=label, 
                   alpha=0.6, s=50)
    
    title = f'{method.upper()} - Embeddings of Two Randomly Selected Videos'
    plt.title(title, fontsize=24, weight='bold')
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.xticks([])
    plt.yticks([])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=14, framealpha=0.5)  
    plt.tight_layout()
    

    if save_path:
        for ext in ['.png', '.svg']:
            full_path = save_path + ext
            plt.savefig(full_path, dpi=400, bbox_inches='tight')
            logging.info(f"Saved {title} visualization to {full_path}")
    
    plt.show()  # Display the plot
    plt.close()