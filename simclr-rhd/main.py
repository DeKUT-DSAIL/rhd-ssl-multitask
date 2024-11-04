import argparse
import os
import logging
from dataset import GCPBucketDataset
from models import SimCLR, LinearClassifier
from train import train_simclr, train_classifier
from evaluate import evaluate_classifier
from visualize import (
    plot_training_loss, 
    get_embeddings, 
    reduce_embeddings, 
    visualize_embeddings,
    plot_confusion_matrix
)
from utils import set_seed, setup_logging, get_label_encoders, get_encoder
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

def main(args):
    # Setup Logging and Output Directory
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, 'simclr_rhd.log')
    setup_logging(log_file=log_file)

    # Set seed for reproducibility
    set_seed(args.seed)

    # Set environment variables
    os.environ["GCLOUD_PROJECT"] = "rhd-diagnosis"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "rhd-diagnosis-17affaa34749.json"
    logging.info("Environment variables set.")

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Define Transformation Pipeline
    data_transforms = transforms.Compose([
        transforms.Pad(padding=(40, 60, 550, 400), fill=0),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    logging.info("Data transformations defined.")

    # Define all possible classes for each label type
    all_classes = {
        'view': ['Parasternal long axis (PLAX)', 'Parasternal short axis(PSAX)', 
                 'Apical Four Chamber(A4C)', 'Not Applicable'],
        'condition': ['Not Applicable', 'Aortic Valve Regurgitation', 
                     'Mitral Valve Prolapse', 'Mitral Valve Regurgitation',
                     'Pulmonary Valve Regurgitation', 'Tricuspid Valve Regurgitation',
                     'Aortic Valve Regurgitation, Pulmonary Valve Regurgitation'],
        'severity': ['Not Applicable', 'Borderline rhd', 'Definite rhd']
    }

    # Create and fit label encoders
    label_encoders = get_label_encoders(all_classes)

    # Create Datasets
    bucket_name = 'rhdiag'
    labelled_prefix = 'supervised/annotated-final'
    unlabelled_prefix = 'unsupervised-videos/unlabelled-frames'
    csv_file = 'supervised/sorted_file.csv'

    # Create datasets
    labelled_dataset = GCPBucketDataset(
        bucket_name=bucket_name,
        prefix=labelled_prefix,
        csv_file=csv_file,
        transform=data_transforms
    )
    
    unlabelled_dataset = GCPBucketDataset(
        bucket_name=bucket_name,
        prefix=unlabelled_prefix,
        csv_file=csv_file,
        transform=data_transforms
    )
    logging.info("Datasets created.")

    # Split labeled dataset
    train_size = int(0.7 * len(labelled_dataset))
    val_size = int(0.15 * len(labelled_dataset))
    test_size = len(labelled_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        labelled_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logging.info(f"Dataset split: {train_size} train, {val_size} val, {test_size} test")

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    unlabelled_dataloader = DataLoader(
        unlabelled_dataset,
        batch_size=args.batch_size * 2,  # Larger batch size for unlabeled data
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    logging.info("DataLoaders created.")

    # Initialize SimCLR Model
    simclr = SimCLR(output_dim=512, projection_dim=128).to(device)
    if torch.cuda.device_count() > 1:
        simclr = nn.DataParallel(simclr)
        logging.info(f"Using {torch.cuda.device_count()} GPUs.")

    # Define SimCLR Optimizer
    optimizer = optim.Adam(simclr.parameters(), lr=args.learning_rate)
    logging.info("SimCLR model and optimizer initialized.")

    # Train SimCLR
    simclr_losses = train_simclr(
        model=simclr,
        dataloader=unlabelled_dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        save_path=os.path.join(args.save_dir, 'best_simclr_model.pth')
    )

    # Plot SimCLR training loss
    plot_training_loss(
        simclr_losses, 
        filename=os.path.join(args.save_dir, 'simclr_training_loss.png')
    )

    # Train and evaluate classifiers for each label type
    for label_type in ['view', 'condition', 'severity']:
        logging.info(f"\nTraining classifier for {label_type}")
        
        # Create classifier
        num_classes = len(all_classes[label_type])
        classifier = LinearClassifier(input_dim=512, num_classes=num_classes).to(device)
        
        # Define optimizer and criterion
        optimizer_cls = optim.Adam(classifier.parameters(), lr=args.classifier_lr)
        criterion = nn.CrossEntropyLoss()

        # Train classifier
        classifier_losses = train_classifier(
            encoder=get_encoder(simclr),
            classifier=classifier,
            dataloader=train_dataloader,
            optimizer=optimizer_cls,
            criterion=criterion,
            device=device,
            num_epochs=args.linear_epochs,
            save_path=os.path.join(args.save_dir, f'best_classifier_{label_type}.pth'),
            label_type=label_type,
            label_encoder=label_encoders[label_type]
        )

        # Plot classifier training loss
        plot_training_loss(
            classifier_losses,
            filename=os.path.join(args.save_dir, f'classifier_{label_type}_loss.png')
        )

        # Evaluate on validation set
        val_metrics = evaluate_classifier(
            encoder=get_encoder(simclr),
            classifier=classifier,
            dataloader=val_dataloader,
            label_encoder=label_encoders[label_type],
            label_type=label_type,
            device=device
        )

        # Log validation metrics
        logging.info(f"\n{label_type.capitalize()} Validation Metrics:")
        for metric_name, value in val_metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")

        # Evaluate on test set
        test_metrics = evaluate_classifier(
            encoder=get_encoder(simclr),
            classifier=classifier,
            dataloader=test_dataloader,
            label_encoder=label_encoders[label_type],
            label_type=label_type,
            device=device
        )

        # Log test metrics
        logging.info(f"\n{label_type.capitalize()} Test Metrics:")
        for metric_name, value in test_metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")

        # Generate visualizations
        embeddings, labels = get_embeddings(simclr, val_dataloader, device, label_type)
        reduced_embeddings, _ = reduce_embeddings(embeddings, n_components=50)
        
        # Create t-SNE visualization
        visualize_embeddings(
            reduced_embeddings,
            labels,
            method='tsne',
            filename=os.path.join(args.save_dir, f'tsne_{label_type}.png'),
            label_type=label_type
        )
        
        # Create UMAP visualization
        visualize_embeddings(
            reduced_embeddings,
            labels,
            method='umap',
            filename=os.path.join(args.save_dir, f'umap_{label_type}.png'),
            label_type=label_type
        )

    logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SimCLR Training and Evaluation Pipeline')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for SimCLR training')
    parser.add_argument('--linear-epochs', type=int, default=50, help='Number of epochs for linear classifier training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate for SimCLR')
    parser.add_argument('--classifier-lr', type=float, default=1e-3, help='Learning rate for classifier')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    # System parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models and plots')

    args = parser.parse_args()
    main(args)