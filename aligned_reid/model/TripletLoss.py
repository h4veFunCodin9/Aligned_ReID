from torch import nn
from torch.autograd import Variable


class TripletLoss(object):
  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
  Loss for Person Re-Identification'."""
  def __init__(self, margin=None, margin_in=None):
    self.margin = margin
    #self.margin_in = margin_in
    #self.beta = 0.01
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
      #self.ranking_loss_in = nn.MarginRankingLoss(margin=0)
    else:
      self.ranking_loss = nn.SoftMarginLoss()

  def __call__(self, dist_ap, dist_an):
    """
    Args:
      dist_ap: pytorch Variable, distance between anchor and positive sample, 
        shape [N]
      dist_an: pytorch Variable, distance between anchor and negative sample, 
        shape [N]
    Returns:
      loss: pytorch Variable, with shape [1]
    """
    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    #in_y = Variable(dist_ap.data.new().resize_as_(dist_ap.data).fill_(1))
    #in_margins = Variable(dist_ap.data.new().resize_as_(dist_ap.data).fill_(self.margin_in))
    if self.margin is not None:
      loss = self.ranking_loss(dist_an, dist_ap, y)
      #loss += self.beta * self.ranking_loss_in(in_margins, dist_ap, in_y)
    else:
      loss = self.ranking_loss(dist_an - dist_ap, y)
    return loss

