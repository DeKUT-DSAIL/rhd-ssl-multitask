import os
import io
from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd
from google.cloud import storage
from torchvision import transforms
import logging

class GCPBucketDataset(Dataset):
    def __init__(self, bucket_name, prefix, csv_file, transform=None):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.transform = transform
        
        # Connect to Google Cloud Storage
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        # Read and process the CSV file
        self.labels_df = self._read_csv(csv_file)
        
        # Get list of files in the bucket
        self.file_paths = [blob.name for blob in self.bucket.list_blobs(prefix=self.prefix)]
        
        logging.info(f"Total files in bucket prefix '{self.prefix}': {len(self.file_paths)} (All included)")

    def _read_csv(self, csv_file):
        """Read and process the CSV file from GCP bucket."""
        blob = self.bucket.blob(csv_file)
        
        with blob.open("rb") as f:
            labels_df = pd.read_csv(f)
        
        # Clean up condition labels
        if 'CONDITION' in labels_df.columns:
            labels_df['CONDITION'] = labels_df['CONDITION'].astype(str).str.strip("[]").str.replace("'", "")
        
        # Create separate dictionaries for each label type
        self.view_labels = dict(zip(labels_df['FILENAME'], labels_df['VIEW-APP']))
        self.condition_labels = dict(zip(labels_df['FILENAME'], labels_df['CONDITION']))
        self.severity_labels = dict(zip(labels_df['FILENAME'], labels_df['SEVERITY']))
        
        # Count and log unique classes
        unique_classes = {
            'view': labels_df['VIEW-APP'].unique(),
            'condition': labels_df['CONDITION'].unique(),
            'severity': labels_df['SEVERITY'].unique()
        }
        
        logging.info("Number of unique classes:")
        for label_type, classes in unique_classes.items():
            logging.info(f"{label_type}: {len(classes)}")
            logging.info(f"{label_type.capitalize()} labels: {classes}")
        
        return labels_df

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        filename = os.path.basename(file_path)
        
        try:
            # Load image from GCP
            blob = self.bucket.blob(file_path)
            with blob.open("rb") as f:
                image = Image.open(io.BytesIO(f.read())).convert('RGB')
            
            # Apply transformations
            if self.transform:
                x_i = self.transform(image)
                x_j = self.transform(image)
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                x_i = transform(image)
                x_j = transform(image)
            
            # Get labels for each task
            view_label = self.view_labels.get(filename, 'Not Applicable')
            condition_label = self.condition_labels.get(filename, 'Not Applicable')
            severity_label = self.severity_labels.get(filename, 'Not Applicable')
            
            return x_i, x_j, {
                'view': view_label,
                'condition': condition_label,
                'severity': severity_label
            }
            
        except Exception as e:
            logging.error(f"Error loading image {file_path}: {e}")
            # Return zero tensors and default labels on error
            zero_tensor = torch.zeros(3, 224, 224)
            return zero_tensor, zero_tensor, {
                'view': 'Not Applicable',
                'condition': 'Not Applicable',
                'severity': 'Not Applicable'
            }

    def get_labels(self, label_type):
        """Get all labels for a specific type."""
        if label_type == 'view':
            return list(self.view_labels.values())
        elif label_type == 'condition':
            return list(self.condition_labels.values())
        elif label_type == 'severity':
            return list(self.severity_labels.values())
        else:
            raise ValueError(f"Unknown label type: {label_type}")

    def get_num_classes(self, label_type):
        """Get number of unique classes for a specific label type."""
        return len(set(self.get_labels(label_type)))