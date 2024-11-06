import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np

def visualize_samples(dataloader, num_samples=5, label_decoders=None, save_path=None):
    """
    Visualize samples with decoded labels.
    """
    images, labels = next(iter(dataloader))
    view_labels, condition_labels, severity_labels = labels
    
    fig, axes = plt.subplots(1, min(num_samples, len(images)), figsize=(20, 6))
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
            condition = condition.replace("[", "").replace("]", "").replace("'", "")
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
        axes[i].set_title(label_text, fontsize=8, wrap=True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()