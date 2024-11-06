import random
import numpy as np
import torch
import logging
import sys
import os
from sklearn.preprocessing import LabelEncoder

def set_seed(seed):
    """
    Sets the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

def setup_logging(log_file='simclr_rhd.txt'):
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging is configured.")

def get_label_encoders(all_classes):
    """
    Creates and fits label encoders for each label type.
    """
    label_encoders = {}
    for label_type, classes in all_classes.items():
        encoder = LabelEncoder()
        encoder.fit(classes)
        label_encoders[label_type] = encoder
        logging.info(f"Label encoder fitted for {label_type}. Classes: {encoder.classes_}")
    return label_encoders

def get_encoder(model):
    """
    Retrieves the encoder from the model, handling DataParallel wrappers.
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module.encoder
    return model.encoder