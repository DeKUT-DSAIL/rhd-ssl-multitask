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
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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
      
def pretrain_model(model, unlabelled_loader, device, optimizer, num_epochs=10, learning_rate=1e-4, criterion=None):  # Add optimizer
    model.train()  # Set model to training mode
    if criterion is None:  # Only initialize if not passed
        criterion = nn.CrossEntropyLoss().to(device) # Move criterion to device

    for epoch in range(num_epochs):
        for images, _, _ in tqdm(unlabelled_loader, desc=f'Pretraining Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'): # Use autocast for pretraining too
                outputs = model(images)
                pseudo_labels = outputs['view'].argmax(dim=1) # Get highest probability class
                loss = criterion(outputs['view'], pseudo_labels) # Calculate CrossEntropyLoss

            loss.backward()
            optimizer.step()
            

def evaluate_model(model, test_loader, criterion, device, save_dir, num_samples=3, tasks=['view', 'condition', 'severity']):
    model.eval()
    test_loss = 0
    test_preds = {task: [] for task in tasks}  # Predictions for all tasks
    test_labels = {task: [] for task in tasks}  # Ground truth for all tasks
    test_embeddings = []
    test_images = []  # To store images for visualization
    actual_labels = []  # To store actual labels for visualization
    predicted_labels = {task: [] for task in tasks}  # To store predicted labels for visualization
    video_ids = []  # To store video IDs for visualization

    with torch.no_grad():
        for images, labels, vids in tqdm(test_loader, desc="Evaluating on test set"):  # Unpack three values
            images = images.to(device)
            
            # view_labels = torch.tensor(labels[0], dtype=torch.long).to(device)
            # condition_labels = torch.tensor(labels[1], dtype=torch.long).to(device)
            # severity_labels = torch.tensor(labels[2], dtype=torch.long).to(device)

            view_labels = labels[0].clone().detach().to(device)
            condition_labels = labels[1].clone().detach().to(device)
            severity_labels = labels[2].clone().detach().to(device)

            outputs, embeddings = model(images, return_features=True)
            test_embeddings.append(embeddings.cpu())
            video_ids.extend(vids)  # Collect video IDs

            loss = (criterion(outputs['view'], view_labels) + 
                   criterion(outputs['condition'], condition_labels) + 
                   criterion(outputs['severity'], severity_labels))
            
            test_loss += loss.item()

            for task_idx, task in enumerate(tasks):  # Iterate through each task
                preds = outputs[task].argmax(dim=1)  # Get predictions for the current task
                test_preds[task].extend(preds.cpu().numpy())  # Store predictions for metrics
                test_labels[task].extend(labels[task_idx].cpu().numpy())  # Store true labels

                if len(predicted_labels[task]) < num_samples: # Correctly extend each prediction
                    predicted_labels[task].extend(preds.cpu().numpy())  # Store predictions for visualization

            if len(test_images) < num_samples: # Ensure correct shapes and dimensions for images
                test_images.extend(images.cpu().numpy())
                actual_labels.extend([label.cpu().numpy() for label in labels])  # Collect actual labels

    for task in tasks:
        predicted_labels[task] = np.array(predicted_labels[task]) # Convert prediction list to array for each task
    test_embeddings = torch.cat(test_embeddings, dim=0).numpy() # Convert embeddings

    # Convert actual_labels to a numpy array
    actual_labels = np.array(actual_labels)
    actual_labels = actual_labels.reshape(-1, len(tasks))

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
            confusion_matrix_path = os.path.join(save_dir, f'{task}_confusion_matrix')
            plot_confusion_matrix(
                np.array(test_labels[task]),
                np.array(test_preds[task]),
                labels=labels,
                task_name=task,
                save_path=confusion_matrix_path
            )

            # Visualize embeddings for both UMAP and t-SNE
            for method in ['umap', 'tsne']:
                try:
                    # Visualize by classification label
                    visualize_embeddings(
                        embeddings=test_embeddings,
                        labels=np.array(test_labels[task]).astype(int),  # Ensure integer labels
                        video_ids=video_ids,
                        label_type=task,
                        method=method,
                        color_scheme='label',
                        save_path=os.path.join(save_dir, f'test_embeddings_{task}_{method}_label'),
                        label_decoders=label_decoders  # Pass the label decoders
                    )
                    # Visualize by video ID
                    visualize_embeddings(
                        embeddings=test_embeddings,
                        labels=video_ids,  # Use video IDs for labels
                        video_ids=video_ids,
                        label_type=task,
                        method=method,
                        color_scheme='video',
                        save_path=os.path.join(save_dir, f'test_embeddings_{task}_{method}_video'),
                        label_decoders=label_decoders  # Pass the label decoders
                    )
                except Exception as e:
                    logging.error(f"Error visualizing {task} embeddings with {method}: {str(e)}")
                    logging.error(f"Labels shape: {np.array(test_labels[task]).shape}")
                    logging.error(f"Embeddings shape: {test_embeddings.shape}")
                    logging.error(f"Label decoder keys: {label_decoders[task].keys()}")
        else:
            # Fallback to numeric labels
            logging.warning("Label decoders not available, using numeric labels")
            # Add debugging logs before visualization
            logging.info(f"Test embeddings shape: {test_embeddings.shape}")
            for task in tasks:
                logging.info(f"\nTask: {task}")
                logging.info(f"Number of test labels: {len(test_labels[task])}")
                logging.info(f"Unique labels: {np.unique(test_labels[task])}")
                logging.info(f"Label distribution: {np.bincount(np.array(test_labels[task]).astype(int))}")
                
                if label_decoders:
                    logging.info(f"Label decoder keys for {task}: {label_decoders[task].keys()}")
                    
            # Before visualization loop
            logging.info(f"\nNumber of video IDs: {len(video_ids)}")
            logging.info(f"Sample of video IDs: {video_ids[:5]}")
            
            labels = range(len(np.unique(test_labels[task])))
            
            confusion_matrix_path = os.path.join(save_dir, f'{task}_confusion_matrix')
            plot_confusion_matrix(
                np.array(test_labels[task]),
                np.array(test_preds[task]),
                labels=labels,
                task_name=task,
                save_path=confusion_matrix_path
            )

            for method in ['umap', 'tsne']:
                visualize_embeddings(
                    embeddings=test_embeddings,
                    labels=np.array(test_labels[task]).astype(int),
                    video_ids=video_ids,  # Pass video IDs
                    method=method,
                    save_path=os.path.join(save_dir, f'test_embeddings_{task}_{method}_label'),
                    label_type=task,
                    color_scheme='label'
                )
                visualize_embeddings(
                    embeddings=test_embeddings,
                    labels=video_ids,  # Use video IDs for labels
                    video_ids=video_ids,  # Pass video IDs
                    method=method,
                    save_path=os.path.join(save_dir, f'test_embeddings_{task}_{method}_video'),
                    label_type=task,
                    color_scheme='video'
                )

    return test_images, actual_labels, predicted_labels, test_metrics, test_embeddings, video_ids

def calculate_silhouette_scores(task_embeddings, num_clusters):
    """
    Calculate silhouette scores for each task using agglomerative clustering.
    """
    silhouette_scores = {}
    linkage_methods = ['ward', 'complete', 'average', 'single']

    for task_idx, (task, n_clusters) in enumerate(zip(['view', 'condition', 'severity'], num_clusters)):
        best_score = -1
        best_method = None

        for linkage in linkage_methods:
            agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            cluster_labels = agglomerative.fit_predict(task_embeddings[task])

            silhouette_avg = silhouette_score(task_embeddings[task], cluster_labels)

            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_method = linkage

        print(f'Best Silhouette Score for {task.capitalize()} (using {best_method}): {best_score:.4f}')
        silhouette_scores[task] = best_score
        logging.info(f'Best Silhouette Score for {task.capitalize()} (using {best_method}): {best_score:.4f}')

    return silhouette_scores
# def calculate_silhouette_scores(task_embeddings, cluster_range, linkage_methods=['ward', 'complete', 'average', 'single']):
#     """
#     Calculate silhouette scores for each task using agglomerative clustering with a grid search over the number of clusters.
#     """
#     silhouette_scores = {}
#     best_n_clusters = {}
#     best_linkage_methods = {}

#     for task, embeddings in task_embeddings.items():
#         best_score = -1
#         best_n_cluster = None
#         best_method = None

#         for n_clusters in cluster_range:
#             for linkage in linkage_methods:
#                 try:
#                     agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
#                     cluster_labels = agglomerative.fit_predict(embeddings)
#                     silhouette_avg = silhouette_score(embeddings, cluster_labels)

#                     if silhouette_avg > best_score:
#                         best_score = silhouette_avg
#                         best_n_cluster = n_clusters
#                         best_method = linkage
#                 except Exception as e:
#                     logging.error(f"Error calculating silhouette score for {task} with {n_clusters} clusters and {linkage} linkage: {str(e)}")

#         silhouette_scores[task] = best_score
#         best_n_clusters[task] = best_n_cluster
#         best_linkage_methods[task] = best_method

#         logging.info(f'Best Silhouette Score for {task.capitalize()} (using {best_method} with {best_n_cluster} clusters): {best_score:.4f}')
#         print(f'Best Silhouette Score for {task.capitalize()} (using {best_method} with {best_n_cluster} clusters): {best_score:.4f}')

#     return silhouette_scores, best_n_clusters, best_linkage_methods


    
def train_model(model,  train_loader, val_loader, test_loader, unlabelled_loader, device, num_epochs=100, patience=7, 
                learning_rate=1e-4, save_dir='outputs', pretrain_epochs=5, pretrain_lr=1e-4, optimizer=None, criterion=None, tasks=['view', 'condition', 'severity']):
    
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
    
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = GradScaler('cuda')
    early_stopping = EarlyStopping(patience=patience)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': {'view': [], 'condition': [], 'severity': []},
        'val_metrics': {'view': [], 'condition': [], 'severity': []}
    }
    
    # Pretrain the model, passing the optimizer and criterion
    if pretrain_epochs > 0:  # Conditional pretraining
        pretrain_model(model, unlabelled_loader, device, optimizer=optimizer, num_epochs=pretrain_epochs, learning_rate=pretrain_lr, criterion=criterion)


    # Extract *task-specific* embeddings:
    task_embeddings = {}  # Store embeddings for each task separately
    for task in tasks:
        task_embeddings[task] = [] # Initialize array to accumulate each task's embeddings over batches
        with torch.no_grad():
            for images, _, _ in tqdm(unlabelled_loader, desc=f"Extracting embeddings for {task}"):
                images = images.to(device)
                features = model.module.backbone.forward_features(images)['x_norm_clstoken'] if isinstance(model, torch.nn.DataParallel) else model.backbone.forward_features(images)['x_norm_clstoken']
                task_embeddings_batch = model.module.heads[task](features).cpu().numpy() if isinstance(model, torch.nn.DataParallel) else model.heads[task](features).cpu().numpy()

                task_embeddings[task].append(task_embeddings_batch) # Accumulate embeddings

        task_embeddings[task] = np.concatenate(task_embeddings[task]) # Concatenate after all batches

    # Save task embeddings
    for task in tasks:
        np.save(os.path.join(save_dir, f'{task}_embeddings.npy'), task_embeddings[task])

    # Calculate silhouette scores using agglomerative clustering with task-specific embeddings
    num_clusters = [3, 7, 3]  # Number of clusters for view, condition, severity
    silhouette = calculate_silhouette_scores(task_embeddings, num_clusters)
    cluster_labels = {}
    
    # Get label decoders from model
    if isinstance(model, torch.nn.DataParallel):
        label_decoders = model.module.get_label_decoders()
    else:
        label_decoders = model.get_label_decoders()
    
    for task_idx, (task, n_clusters) in enumerate(zip(tasks, num_clusters)):
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels[task] = agglomerative.fit_predict(task_embeddings[task])

    # Save cluster labels
    for task in tasks:
        np.save(os.path.join(save_dir, f'{task}_cluster_labels.npy'), cluster_labels[task])

    # Visualize clustered embeddings
    for task in tasks:
        for method in ['umap', 'tsne']:
            visualize_embeddings(
                task_embeddings[task],
                cluster_labels[task],  # Use agglomerative cluster labels here
                video_ids=None,  # No video IDs needed for unlabelled data
                label_type=task,
                method=method,
                save_path=f'{save_dir}/unlabelled_embeddings_{method}_{task}',
                color_scheme='label'
            )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = {task: [] for task in ['view', 'condition', 'severity']}
        train_labels = {task: [] for task in ['view', 'condition', 'severity']}

        # Training phase
        for batch_idx, (images, labels, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
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
            for images, labels, _ in val_loader:
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
    test_images, actual_labels, predicted_labels, test_metrics, test_embeddings, video_ids = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        save_dir=save_dir
    )

    # Save test embeddings and labels
    np.save(os.path.join(save_dir, 'test_embeddings.npy'), test_embeddings)
    np.save(os.path.join(save_dir, 'test_labels.npy'), actual_labels)
    np.save(os.path.join(save_dir, 'predicted_labels.npy'), predicted_labels)
    np.save(os.path.join(save_dir, 'video_ids.npy'), video_ids)

    # Collect embeddings from the training set for KNN evaluation
    train_embeddings = []
    train_y = {task: [] for task in ['view', 'condition', 'severity']} # Fix: collect multi-task labels

    for images, labels, _ in train_loader:
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                embeddings = model.module.get_embeddings(images.to(device)).cpu().numpy()
            else:
                embeddings = model.get_embeddings(images.to(device)).cpu().numpy()
            train_embeddings.append(embeddings)
            for i, task in enumerate(['view', 'condition', 'severity']):
                train_y[task].extend(np.array(labels[i])) # Fix: extend with numpy arrays


    train_embeddings = np.concatenate(train_embeddings)
    for i, task in enumerate(tasks):
        train_y[task] = np.array(train_y[task])


    # Save train embeddings and labels
    np.save(os.path.join(save_dir, 'train_embeddings.npy'), train_embeddings)
    for task in tasks:
        np.save(os.path.join(save_dir, f'train_labels_{task}.npy'), train_y[task])



     # Evaluate KNN (modified to handle multi-task labels)
    knn_models = {}
    knn_accuracies = {}  # Store accuracy for each task

    for i, task in enumerate(tasks):
        knn = KNeighborsClassifier(n_neighbors=5)  # Initialize KNN
        knn.fit(train_embeddings, train_y[task])  # Fit to training embeddings and labels
        knn_models[task] = knn  # Store fitted KNN model
        
    # Optionally, you can evaluate KNN on the test set as well (modified)
    test_embeddings = []
    test_y = {task: [] for task in ['view', 'condition', 'severity']}

    for images, labels, _ in test_loader:
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                embeddings = model.module.get_embeddings(images.to(device)).cpu().numpy()
            else:
                embeddings = model.get_embeddings(images.to(device)).cpu().numpy()
            test_embeddings.append(embeddings)
            for i, task in enumerate(tasks):
                test_y[task].extend(np.array(labels[i]))

    test_embeddings = np.concatenate(test_embeddings)
    for i, task in enumerate(tasks):
        test_y[task] = np.array(test_y[task])

    # Save test embeddings and labels for KNN
    np.save(os.path.join(save_dir, 'test_embeddings_knn.npy'), test_embeddings)
    for task in tasks:
        np.save(os.path.join(save_dir, f'test_labels_knn_{task}.npy'), test_y[task])

    # Predict using KNN (modified)
    for task in ['view', 'condition', 'severity']:
        knn_predictions = knn_models[task].predict(test_embeddings)
        knn_accuracy = accuracy_score(test_y[task], knn_predictions)
        knn_accuracies[task] = knn_accuracy
        print(f"KNN Accuracy on Test Set ({task.capitalize()}): {knn_accuracy:.4f}")

    
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
        plt.savefig(os.path.join(save_dir, f'{task}_metrics.svg'))
        plt.close()
    return history, save_dir
