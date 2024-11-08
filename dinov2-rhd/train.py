import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from metrics import calculate_metrics, plot_confusion_matrix, visualize_embeddings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
      


def evaluate_model(model, test_loader, criterion, device, save_dir, num_samples=5):
    model.eval()
    test_loss = 0
    test_preds = {task: [] for task in ['view', 'condition', 'severity']}
    test_labels = {task: [] for task in ['view', 'condition', 'severity']}
    test_embeddings = []
    test_images = []  # To store images for visualization
    actual_labels = []  # To store actual labels for visualization
    predicted_labels = []  # To store predicted labels for visualization

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on test set"):
            images = images.to(device)
            view_labels = torch.tensor(labels[0], dtype=torch.long).to(device)
            condition_labels = torch.tensor(labels[1], dtype=torch.long).to(device)
            severity_labels = torch.tensor(labels[2], dtype=torch.long).to(device)

            outputs, embeddings = model(images, return_features=True)
            test_embeddings.append(embeddings.cpu())

            loss = (criterion(outputs['view'], view_labels) + 
                   criterion(outputs['condition'], condition_labels) + 
                   criterion(outputs['severity'], severity_labels))
            
            test_loss += loss.item()

            for task, task_labels in zip(['view', 'condition', 'severity'], 
                                       [view_labels, condition_labels, severity_labels]):
                preds = outputs[task].argmax(dim=1)
                test_preds[task].extend(preds.cpu().numpy())
                test_labels[task].extend(task_labels.cpu().numpy())


            # Store images and labels for visualization
            if len(test_images) < num_samples:
                test_images.extend(images.cpu().numpy())  # Store the images
                actual_labels.extend(labels)  # Store the actual labels
                predicted_labels.extend(outputs['view'].argmax(dim=1).cpu().numpy())  # Store predicted labels

    test_embeddings = torch.cat(test_embeddings, dim=0).numpy()
    # Convert lists to numpy arrays for easier handling
    test_images = np.array(test_images)
    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate metrics and plot confusion matrices
    test_metrics = {}
    # Get label decoders, handling DataParallel case
    if isinstance(model, torch.nn.DataParallel):
        label_decoders = model.module.get_label_decoders()
    else:
        label_decoders = model.get_label_decoders()

    for task in ['view', 'condition', 'severity']:
        # Calculate metrics
        test_metrics[task] = calculate_metrics(
            np.array(test_labels[task]), 
            np.array(test_preds[task])
        )
        
        # Get labels for current task
        if label_decoders:
            labels = [label_decoders[task][i] for i in range(len(label_decoders[task]))]
            
            # Plot confusion matrix
            plot_confusion_matrix(
                np.array(test_labels[task]),
                np.array(test_preds[task]),
                labels=labels,
                task_name=task,
                save_path=os.path.join(save_dir, f'{task}_confusion_matrix.png')
            )

            # Visualize embeddings for both UMAP and t-SNE
            for method in ['umap', 'tsne']:
                try:
                    visualize_embeddings(
                        embeddings=test_embeddings,
                        labels=np.array(test_labels[task]).astype(int),  # Ensure integer labels
                        label_names=label_decoders[task],  # Pass decoder dictionary directly
                        task_name=task,
                        method=method,
                        save_path=os.path.join(save_dir, f'test_embeddings_{task}_{method}.png')
                    )
                except Exception as e:
                    logging.error(f"Error visualizing {task} embeddings with {method}: {str(e)}")
                    logging.error(f"Labels shape: {np.array(test_labels[task]).shape}")
                    logging.error(f"Embeddings shape: {test_embeddings.shape}")
                    logging.error(f"Label decoder keys: {label_decoders[task].keys()}")
        else:
            # Fallback to numeric labels
            logging.warning("Label decoders not available, using numeric labels")
            labels = range(len(np.unique(test_labels[task])))
            
            plot_confusion_matrix(
                np.array(test_labels[task]),
                np.array(test_preds[task]),
                labels=labels,
                task_name=task,
                save_path=os.path.join(save_dir, f'{task}_confusion_matrix.png')
            )

            for method in ['umap', 'tsne']:
                visualize_embeddings(
                    embeddings=test_embeddings,
                    labels=np.array(test_labels[task]).astype(int),
                    label_names={i: str(i) for i in range(len(labels))},  # Create simple numeric mapping
                    task_name=task,
                    method=method,
                    save_path=os.path.join(save_dir, f'test_embeddings_{task}_{method}.png')
                )

    return test_images, actual_labels, predicted_labels, test_metrics, test_embeddings

def calculate_silhouette_scores(embeddings, num_clusters):
    """Calculate silhouette scores for each task."""
    silhouette_scores = {}
    
    for task, (embeddings_task, n_clusters) in zip(['view', 'condition', 'severity'], zip(embeddings, num_clusters)):
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_task)

        # Calculate the silhouette score
        silhouette_avg = silhouette_score(embeddings_task, cluster_labels)
        silhouette_scores[task] = silhouette_avg
        print(f'Silhouette Score for {task.capitalize()}: {silhouette_avg:.4f}')
        
        # Log the silhouette score
        logging.info(f'Silhouette Score for {task.capitalize()}: {silhouette_avg:.4f}')
    
    return silhouette_scores
    
def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    unlabelled_loader,
    device,
    num_epochs=100,
    patience=7,
    learning_rate=1e-4,
    save_dir='outputs'
):
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if model is wrapped in DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    # Initialize optimizer
    optimizer = optim.AdamW([
        {'params': model_module.heads.parameters(), 'lr': learning_rate},
        {'params': model_module.backbone.parameters(), 'lr': learning_rate * 0.1}
    ])
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    early_stopping = EarlyStopping(patience=patience)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': {'view': [], 'condition': [], 'severity': []},
        'val_metrics': {'view': [], 'condition': [], 'severity': []}
    }

    # Extract embeddings from unlabelled data first
    unlabelled_embeddings = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(unlabelled_loader, desc="Extracting unlabelled embeddings"):
            images = images.to(device)
            if isinstance(model, torch.nn.DataParallel):
                embeddings = model.module.backbone.forward_features(images)['x_norm_clstoken']
            else:
                embeddings = model.backbone.forward_features(images)['x_norm_clstoken']
            unlabelled_embeddings.append(embeddings.cpu())
    
    unlabelled_embeddings = torch.cat(unlabelled_embeddings, dim=0)


    # Calculate silhouette scores for the unlabeled embeddings
    num_clusters = [3, 7, 3]  # Adjust the number of clusters for each task as needed
    silhouette_scores = calculate_silhouette_scores(
        [unlabelled_embeddings.numpy() for _ in range(3)],  # Use the same embeddings for all tasks
        num_clusters
    )
    
    # Get label decoders from model
    if isinstance(model, torch.nn.DataParallel):
        label_decoders = model.module.get_label_decoders()
    else:
        label_decoders = model.get_label_decoders()
        
    visualize_embeddings(
        unlabelled_embeddings.numpy(), 
        np.zeros(len(unlabelled_embeddings)),  # dummy labels
        label_names={0: 'Unlabelled'},  # dummy label names
        task_name='Unlabelled',
        method='umap',
        save_path=f'{save_dir}/unlabelled_embeddings_umap.png'
    )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = {task: [] for task in ['view', 'condition', 'severity']}
        train_labels = {task: [] for task in ['view', 'condition', 'severity']}

        # Training phase
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            images = images.to(device)
            
            # Convert labels to tensors and move to device
            view_labels = torch.tensor(labels[0], dtype=torch.long).to(device)
            condition_labels = torch.tensor(labels[1], dtype=torch.long).to(device)
            severity_labels = torch.tensor(labels[2], dtype=torch.long).to(device)

            # Skip samples with invalid labels (-1)
            if -1 in view_labels or -1 in condition_labels or -1 in severity_labels:
                continue

            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = (criterion(outputs['view'], view_labels) + 
                       criterion(outputs['condition'], condition_labels) + 
                       criterion(outputs['severity'], severity_labels))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            
            # Store predictions and labels
            for task, task_labels in zip(['view', 'condition', 'severity'], 
                                       [view_labels, condition_labels, severity_labels]):
                preds = outputs[task].argmax(dim=1)
                train_preds[task].extend(preds.cpu().numpy())
                train_labels[task].extend(task_labels.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = {task: [] for task in ['view', 'condition', 'severity']}
        val_labels = {task: [] for task in ['view', 'condition', 'severity']}

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                 # Convert labels to tensors and move to device
                view_labels = torch.tensor(labels[0], dtype=torch.long).to(device)
                condition_labels = torch.tensor(labels[1], dtype=torch.long).to(device)
                severity_labels = torch.tensor(labels[2], dtype=torch.long).to(device)

                outputs = model(images)
                loss = (criterion(outputs['view'], view_labels) + 
                       criterion(outputs['condition'], condition_labels) + 
                       criterion(outputs['severity'], severity_labels))
                
                val_loss += loss.item()

                # Store predictions and labels
                for task, task_labels in zip(['view', 'condition', 'severity'], 
                                           [view_labels, condition_labels, severity_labels]):
                    preds = outputs[task].argmax(dim=1)
                    val_preds[task].extend(preds.cpu().numpy())
                    val_labels[task].extend(task_labels.cpu().numpy())

        # Calculate and store metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        for task in ['view', 'condition', 'severity']:
            train_metrics = calculate_metrics(
                np.array(train_labels[task]), 
                np.array(train_preds[task])
            )
            val_metrics = calculate_metrics(
                np.array(val_labels[task]), 
                np.array(val_preds[task])
            )
            
            history['train_metrics'][task].append(train_metrics)
            history['val_metrics'][task].append(val_metrics)

        # Print epoch metrics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        for task in ['view', 'condition', 'severity']:
            print(f'{task.capitalize()} - Train Acc: {train_metrics["accuracy"]:.4f}, '
                  f'Val Acc: {val_metrics["accuracy"]:.4f}')

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        scheduler.step()

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_images, actual_labels, predicted_labels, test_metrics, test_embeddings = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        save_dir=save_dir
    )

    # Save test metrics
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)

    return history, test_metrics

def plot_training_curves(history, save_dir):
    """Plot and save training curves"""
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

    # Plot metrics for each task
    tasks = ['view', 'condition', 'severity']
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    
    for task in tasks:
        plt.figure(figsize=(12, 8))
        for metric in metrics:
            train_metric = [m[metric] for m in history['train_metrics'][task]]
            val_metric = [m[metric] for m in history['val_metrics'][task]]
            plt.plot(train_metric, label=f'Train {metric}')
            plt.plot(val_metric, label=f'Val {metric}')
        plt.title(f'{task.capitalize()} Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{task}_metrics.png'))
        plt.close()
    return history, save_dir
