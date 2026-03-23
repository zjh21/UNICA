import torch.nn as nn

class ActionEmbeddings(nn.Module):
    def __init__(self, num_actions=5, embedding_dim=768):
        super().__init__()
        self.embeddings = nn.Embedding(num_actions, embedding_dim)
        # Initialize embeddings
        nn.init.normal_(self.embeddings.weight, std=0.02)
    
    def forward(self, motion_indices):
        # motion_indices: (Batch,) tensor of motion type indices
        embeds = self.embeddings(motion_indices)  # (Batch, 768)
        embeds = embeds.unsqueeze(1)  # (Batch, 1, 768)
        return embeds