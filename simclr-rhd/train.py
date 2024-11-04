import torch
import torch.nn as nn
import torch.optim as optim
import logging
from losses import nt_xent_loss
from torch.utils.data import DataLoader
import numpy as np

def train_simclr(model, dataloader, optimizer, device, num_epochs=10, patience=5, save_path='best_simclr_model.pth'):
    """
    Trains the SimCLR model.
    """
    best_loss = float('inf')
    wait = 0
    losses = []

    logging.info("Starting SimCLR training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (x_i, x_j, _) in enumerate(dataloader):
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            # Forward pass
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            # Compute loss
            loss = nt_xent_loss(z_i, z_j)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            # Save the model state
            torch.save(model.state_dict(), save_path)
            logging.info(f"Best model saved at epoch {epoch+1} with loss {avg_loss:.4f}")
        else:
            wait += 1
            logging.info(f"No improvement in loss for {wait} epoch(s).")
            if wait >= patience:
                logging.info(f"Early stopping triggered after epoch {epoch+1}")
                break

    logging.info("SimCLR training completed.")
    return losses

def train_classifier(encoder, classifier, dataloader, optimizer, criterion, device, num_epochs, save_path, label_type, label_encoder):
    """
    Train a linear classifier for a specific label type.
    """
    classifier.train()
    losses = []
    valid_batches = 0
    
    logging.info(f"Starting training for {label_type} classification")
    logging.info(f"Label encoder classes: {label_encoder.classes_}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (x_i, x_j, labels_dict) in enumerate(dataloader):
            try:
                images = x_i.to(device)
                batch_labels = labels_dict[label_type]

                # Debug information for first batch
                if epoch == 0 and batch_idx == 0:
                    logging.info(f"First batch {label_type} labels: {batch_labels}")

                # Transform labels
                labels = label_encoder.transform(batch_labels)
                labels = torch.tensor(labels, dtype=torch.long).to(device)

                # Get features from encoder
                with torch.no_grad():
                    features = encoder(images)
                
                # Forward pass through classifier
                outputs = classifier(features)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                valid_batches += 1

            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                logging.error(f"Problematic {label_type} labels: {batch_labels}")
                continue

        # Calculate average loss only for valid batches
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            if (epoch + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            losses.append(avg_loss)

    # Save the trained classifier
    torch.save(classifier.state_dict(), save_path)
    logging.info(f"Classifier saved to {save_path}")
    return losses