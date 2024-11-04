import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging

def evaluate_classifier(encoder, classifier, dataloader, label_encoder, label_type, device):
    """
    Evaluate the classifier on the given dataloader.
    """
    encoder.eval()
    classifier.eval()
    
    all_preds = []
    all_true_labels = []
    
    logging.info(f"Evaluating {label_type} classification...")
    
    with torch.no_grad():
        for x_i, x_j, labels_dict in dataloader:
            # Get labels for current task
            batch_labels = labels_dict[label_type]
            
            # Convert labels to indices using label encoder
            true_indices = label_encoder.transform(batch_labels)
            true_indices = torch.tensor(true_indices).to(device)
            
            # Forward pass
            images = x_i.to(device)
            features = encoder(images)
            outputs = classifier(features)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_true_labels.extend(true_indices.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_true_labels, all_preds, label_encoder, label_type)
    
    return metrics

def calculate_metrics(y_true, y_pred, label_encoder, label_type):
    """
    Calculate various classification metrics.
    """
    # Convert numeric predictions back to labels if needed
    if hasattr(label_encoder, 'inverse_transform'):
        y_true_labels = label_encoder.inverse_transform(y_true)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
    else:
        y_true_labels = y_true
        y_pred_labels = y_pred
    
    # Calculate basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true_labels, y_pred_labels),
        'precision': precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
        'recall': recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
        'f1': f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)
    
    # Calculate specificity for each class
    n_classes = len(label_encoder.classes_)
    specificities = []
    
    for i in range(n_classes):
        # True Negatives are all the samples that are not the current class and were predicted as not the current class
        tn = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
        # False Positives are all the samples that are not the current class but were predicted as the current class
        fp = np.sum(np.delete(cm[:, i], i))
        
        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    
    # Add average specificity to metrics
    metrics['specificity'] = np.mean(specificities)
    
    return metrics