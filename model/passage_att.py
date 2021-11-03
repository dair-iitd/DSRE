import torch
from torch import nn, optim
from torch.nn import functional as F

class PassageAttention(nn.Module):
    """
    token-level attention for passage-level relation extraction.
    """

    def __init__(self, 
                passage_encoder, 
                num_class, 
                rel2id):
        """
        Args:
            passage_encoder: encoder for whole passage (bag of sentences)
            num_class: number of classes
        """
        super().__init__()
        self.passage_encoder = passage_encoder
        self.embed_dim = self.passage_encoder.hidden_size
        self.num_class = num_class
        self.fc = nn.Linear(self.embed_dim, 1)
        self.relation_embeddings = nn.Parameter(torch.empty(self.num_class, self.embed_dim))
        nn.init.xavier_normal_(self.relation_embeddings)
        self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def forward(self, token, mask, train=True):
        """
        Args:
            token: (nsum, L), index of tokens
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)
        """
        batch_size = token.shape[0]
        #max_len = token.shape[-1]
        if mask is not None:
            rep = self.passage_encoder(token, mask) # (B, max_len, H)
        else:
            rep = self.passage_encoder(token)  # (nsum, H)
        if train:
            att_mat = self.relation_embeddings.repeat(batch_size,1,1)  # (B, N, emb_dim)
            att_scores = torch.bmm(rep, att_mat.transpose(1,2)).transpose(1,2)    # (B, max_len, emb_dim)* (B, emb_dim, N) = (B, max_len, N) -> (B, N, max_len)
            att_scores = self.softmax(att_scores) #(B, N, max_len) 
            rel_logits = torch.bmm(att_scores,rep) # (B, N, max_len) * (B, max_len, H) -> (B, N, H)    
            rel_scores = self.sigm(self.fc(rel_logits).squeeze(-1)) # (B, N, H) -> (B, N, 1) -> (B, N)
            
        else:
            with torch.no_grad():
                att_mat = self.relation_embeddings.repeat(batch_size,1,1)  # (B, N, emb_dim)
                att_scores = torch.bmm(rep, att_mat.transpose(1,2)).transpose(1,2)    # (B, max_len, emb_dim)* (B, emb_dim, N) = (B, max_len, N) -> (B, N, max_len)
                att_scores = self.softmax(att_scores) #(B, N, max_len) 
                rel_logits = torch.bmm(att_scores,rep) # (B, N, max_len) * (B, max_len, H) -> (B, N, H)    
                rel_scores = self.sigm(self.fc(rel_logits).squeeze(-1)) # (B, N, H) -> (B, N, 1) -> (B, N)
                
        return rel_scores
