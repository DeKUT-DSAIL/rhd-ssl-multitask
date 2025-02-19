import os
from datetime import datetime
import logging
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.amp import GradScaler, autocast
from dataset import GCPBucketDataset
from utils import set_seed
from visualize import visualize_samples, visualize_test_samples, visualize_two_videos
from metrics import calculate_metrics, plot_confusion_matrix, visualize_embeddings
from model import DINOv2MultiTask
from train import train_model, plot_training_curves, evaluate_model
import json
from datetime import datetime


def setup_logging(save_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Script for training and evaluating a model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--pretrain_epochs', type=int, default=100, help='Number of pretrain epochs')
    parser.add_argument('--patience', type=int, default=7, help='Patience for early stopping')
    parser.add_argument('--model_name', type=str, default='dinov2_vits14', help='DINOv2 model variant')
    parser.add_argument('--freeze_backbone', type=bool, default=True, help='Whether to freeze backbone')
    
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('outputs', f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    # Set the environment variable for authentication
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "rhd-diagnosis-17affaa34749.json"

    # Set seed for reproducibility
    set_seed(args.seed)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
    # Define the dataset parameters
    bucket_name = 'rhdiag'
    labelled_prefix = 'supervised/annotated-final'
    unlabelled_prefix = 'unsupervised-videos/unlabelled-frames'
    csv_file = 'supervised/sorted_file.csv'
    
    # CustomPad class definition (unchanged)
    class CustomPad:
        def __init__(self, x_min, y_min, x_max, y_max, pad_value=0):
            self.x_min = x_min
            self.y_min = y_min
            self.x_max = x_max
            self.y_max = y_max
            self.pad_value = pad_value
    
        def __call__(self, image):
            if isinstance(image, torch.Tensor):
                image = F.to_pil_image(image)
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([self.x_min, self.y_min, self.x_max, self.y_max], fill=255)
            image = Image.composite(image, Image.new('RGB', image.size, 
                                  (self.pad_value, self.pad_value, self.pad_value)), mask)
            return image

    # Define data augmentation
    data_transforms = transforms.Compose([
        CustomPad(x_min=40, y_min=60, x_max=550, y_max=400, pad_value=0),
        transforms.Resize((224, 224)),  # DINOv2 expected input size
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    logging.info("Creating datasets...")
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
    
    # Get number of classes
    num_classes = labelled_dataset.get_num_classes()
    logging.info(f"Number of classes: {num_classes}")

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
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    unlabelled_dataloader = DataLoader(
        unlabelled_dataset,
        batch_size=args.batch_size * 2,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logging.info("DataLoaders created.")

    # Initialize model
    model = DINOv2MultiTask(
        model_name=args.model_name,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone
    ).to(device)
    
    # Wrap with DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    
    
    # Get label decoders from dataset
    label_decoders = labelled_dataset.get_label_decoders()
    
    # Log the label mappings
    logging.info("\nLabel Decoders:")
    for task, decoder in label_decoders.items():
        logging.info(f"\n{task.upper()}:")
        for idx, label in decoder.items():
            logging.info(f"{idx}: {label}")
            
    # Visualize initial samples
    visualize_samples(
        train_dataloader, 
        num_samples=5, 
        label_decoders=label_decoders,  # Pass the decoders here
        save_path=os.path.join(output_dir, 'initial_samples')
    )
    
    logging.info(f"Initialized {args.model_name} model")


    optimizer = optim.AdamW([
        {'params': model.module.heads.parameters() if isinstance(model, nn.DataParallel) else model.heads.parameters(), 'lr': args.learning_rate},
        {'params': model.module.backbone.parameters() if isinstance(model, nn.DataParallel) else model.backbone.parameters(), 'lr': args.learning_rate * 0.1}
    ])
    
    # Define the criterion (loss function)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Train the model
    logging.info("Starting training...")
    
    history, test_metrics = train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        unlabelled_loader=unlabelled_dataloader,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        optimizer=optimizer,  
        criterion=criterion, 
        pretrain_epochs=args.pretrain_epochs, 
        pretrain_lr=1e-4, 
        save_dir=output_dir
    )

    # Save the training curves
    plot_training_curves(history, output_dir)

    logging.info("\nTest Metrics:")
    for task in ['view', 'condition', 'severity']:
        logging.info(f"\n{task.upper()}:")
        for metric, value in test_metrics[task].items():
            logging.info(f"{metric}: {value:.4f}")
            
    # Save test metrics
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)

    # Visualize test samples
    test_images, actual_labels, predicted_labels, test_metrics, test_embeddings, video_ids = evaluate_model(
        model=model,
        test_loader=test_dataloader,
        criterion=criterion,
        device=device,
        save_dir=output_dir
    )
    
    # Visualize test samples
    visualize_test_samples(
        images=test_images, 
        actual_labels=actual_labels, 
        predicted_labels=predicted_labels, 
        label_decoders=label_decoders, 
        num_samples=3,  # Adjust the number of samples to visualize
        save_path=os.path.join(output_dir, 'test_samples_visualization'),
        project_title="Echocardiographic Image Classification"
    )

    # Visualize embeddings of two randomly selected videos
    visualize_two_videos(test_embeddings, video_ids, method='umap', save_path=os.path.join(output_dir, 'two_videos'))

    logging.info("Training completed. Results saved in: " + output_dir)

if __name__ == '__main__':
    main()