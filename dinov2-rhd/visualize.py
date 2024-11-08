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
        axes[i].set_title(label_text, fontsize=12, wrap=True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=330, bbox_inches='tight')
    plt.close()


def visualize_test_samples(images, actual_labels, predicted_labels, label_decoders=None, num_samples=5, save_path=None):
    """
    Visualize test samples with actual and predicted labels.
    """
    print(f"Number of images: {len(images)}")
    print(f"Number of actual labels: {len(actual_labels)}")
    print(f"Number of predicted labels: {len(predicted_labels)}")

    # Ensure the number of samples to visualize does not exceed the available data
    num_samples = min(num_samples, len(images), len(actual_labels), len(predicted_labels))

    fig, axes = plt.subplots(1, min(num_samples, len(images)), figsize=(20, 6))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for i in range(min(num_samples, len(images))):
        # Check if images is a numpy array and convert to tensor if necessary
        if isinstance(images, np.ndarray):
            image = torch.tensor(images[i])  # Convert to tensor
        else:
            image = images[i]  # Assume it's already a tensor

        # Ensure the image is in the correct format (C, H, W)
        if image.dim() == 3:  # If it's a tensor
            image = image.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = image * std + mean
        image = torch.clamp(image, 0, 1).numpy()

        # Get decoded labels
        actual_label = actual_labels[i]
        predicted_label = predicted_labels[i]

        # Ensure actual_label and predicted_label are single elements
        if isinstance(actual_label, np.ndarray):
            actual_label = actual_label.item() if actual_label.size == 1 else actual_label[0]
        if isinstance(predicted_label, np.ndarray):
            predicted_label = predicted_label.item() if predicted_label.size == 1 else predicted_label[0]

        actual_view = label_decoders['view'].get(int(actual_label), f"Unknown ({actual_label})")
        predicted_view = label_decoders['view'].get(int(predicted_label), f"Unknown ({predicted_label})")
        
        label_text = (
            f'Actual: {actual_view}\n'
            f'Predicted: {predicted_view}'
        )

        # Plot image
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(label_text, fontsize=12, wrap=True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=330, bbox_inches='tight')
    plt.close()