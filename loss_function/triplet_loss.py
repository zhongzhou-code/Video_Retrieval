import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.

        Reference:
            Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

        Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

        Args:
            margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        # Compute the similarity among two tensor,if: the distance > margin,then: loss > 0, else: loss = 0
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        #  inputs = tensor([[ 1,  2,  3,  4], [ 5,  6,  7,  8], [ 9, 10, 11, 12]])
        n = inputs.size(0)  # batch_size

        # Compute pairwise distance, replace by the official when merged
        # dist = tensor([[ 30,  30,  30], [174, 174, 174], [446, 446, 446]])
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = tensor([[ 60, 204, 476], [204, 348, 620], [476, 620, 892]])
        dist = dist + dist.t()
        # dist = tensor([[  0,  64, 256], [ 64,   0,  64], [256,  64,   0]])
        dist.addmm_(1, -2, inputs, inputs.t())
        # dist = tensor([[ 0.,  8., 16.], [ 8.,  0.,  8.], [16.,  8.,  0.]])
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        # if mask[i][j]=1 represented i and j have the same label.
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        #
        dist_ap, dist_an = [], []
        # Calculate the maximum value of distances that belong to the same sample
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
