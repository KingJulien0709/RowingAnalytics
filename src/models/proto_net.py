import torch
import torch.nn.functional as F

class ProtoNet(torch.nn.Module):
    def __init__(self, features_dim):
        super(ProtoNet, self).__init__()
        self.features_dim = features_dim

    def cosine_similarity(self, x, y):
        return F.cosine_similarity(x, y, dim=-1)
    
    def forward(self, support_set, support_labels, query_set):
        prototypes = []
        unique_classes = torch.unique(support_labels)
        for c in unique_classes:
            class_prototype = torch.mean(support_set[support_labels == c], dim=0)
            prototypes.append(class_prototype)
        prototypes = torch.stack(prototypes)

        similarities = self.cosine_similarity(query_set, prototypes)
        log_p_y = F.log_softmax(similarities, dim=-1)
        return log_p_y
    
