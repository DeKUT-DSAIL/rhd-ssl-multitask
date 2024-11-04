import torch
import torch.nn as nn

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    """
    N = z1.size(0)  # Batch size
    z = torch.cat([z1, z2], dim=0)  # 2N x D
    z = nn.functional.normalize(z, dim=1)  # Normalize to unit sphere

    # Compute pairwise cosine similarities
    sim = torch.mm(z, z.T) / temperature  # (2N x 2N)

    # Mask out self-similarities
    sim_i_j = torch.diag(sim, N)  # Positive pairs
    sim_j_i = torch.diag(sim, -N)  # Positive pairs

    # Negative samples mask out positives
    positive_mask = torch.zeros_like(sim)
    positive_mask[range(N), range(N, 2*N)] = 1
    positive_mask[range(N, 2*N), range(N)] = 1
    negative_mask = ~positive_mask.bool()

    # Fill diagonal with large negative values
    sim = sim.masked_fill(torch.eye(2*N, device=sim.device).bool(), -9e15)

    # Compute positive and negative pairs
    positives = torch.cat([sim_i_j, sim_j_i]).reshape(2*N, 1)
    negatives = sim[negative_mask].reshape(2*N, -1)

    # Concatenate positive and negative similarities
    logits = torch.cat([positives, negatives], dim=1)  # (2N x (1 + 2N-2))

    # Labels: positives are the first index
    labels = torch.zeros(2 * N, dtype=torch.long, device=z.device)

    # Cross-Entropy Loss
    loss = nn.CrossEntropyLoss()(logits, labels)
    
    return loss