import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2MultiTask(nn.Module):
    def __init__(self, model_name='dinov2_vitb14', num_classes=None, freeze_backbone=True):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # Get feature dimension based on model
        feature_dim = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536,
        }[model_name]
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Task-specific heads
        self.heads = nn.ModuleDict({
            'view': nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes['view'])
            ),
            'condition': nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes['condition'])
            ),
            'severity': nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes['severity'])
            )
        })
        # Add label decoders
        self.label_decoders = {
            'view': {
                0: 'Apical Four Chamber(A4C)',
                1: 'Parasternal long axis (PLAX)',
                2: 'Parasternal short axis(PSAX)'
            },
            'condition': {
                0: ['Aortic Valve Regurgitation', 'Pulmonary Valve Regurgitation'],
                1: ['Aortic Valve Regurgitation'],
                2: ['Mitral Valve Prolapse'],
                3: ['Mitral Valve Regurgitation'],
                4: ['Not Applicable'],
                5: ['Pulmonary Valve Regurgitation'],
                6: ['Tricuspid Valve Regurgitation']
            },
            'severity': {
                0: 'Borderline rhd',
                1: 'Definite rhd',
                2: 'Not Applicable'
            }
        }
        
    
    
    def get_embeddings(self, x):
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            return features['x_norm_clstoken']

    def forward(self, x, return_features=False):
        features = self.backbone.forward_features(x)
        embeddings = features['x_norm_clstoken']
        
        outputs = {
            'view': self.heads['view'](embeddings),
            'condition': self.heads['condition'](embeddings),
            'severity': self.heads['severity'](embeddings)
        }
        
        if return_features:
            return outputs, embeddings
        return outputs

    def get_label_decoders(self):
        """Return the label decoders dictionary"""
        return self.label_decoders