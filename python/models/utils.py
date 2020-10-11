import torch
from torch.autograd import Variable


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())


def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max


def binary_cross_entropy(r, x, eps):
    return -torch.mean(torch.sum(x*torch.log(r + eps) + (1 - x)*torch.log(1 - r + eps), dim=-1))

def ikatura_saito_divergence(r, x, eps):
    #return torch.sum((x + eps)/(r + eps) - torch.log((x + eps)/(r+ eps)) - 1, dim=-1)
    #return torch.sum((x + eps)/(r + eps) - torch.log(x + eps) + torch.log(r+ eps) - 1, dim=-1)
    return torch.sum(x/r - torch.log(x + eps) + torch.log(r) - 1, dim=-1)

def elbo(x, r, mu, logvar, eps): 
    recon = torch.mean(torch.sum(x/r - torch.log(x + eps) + torch.log(r) - 1, dim=-1))
    KL = -0.5 * torch.mean(torch.sum(logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return recon + KL, recon, KL

def L_loss(x, r, mu, logvar, eps): 
    recon = torch.sum(x/r - torch.log(x + eps) + torch.log(r) - 1, dim=-1)
    KL = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return recon + KL, recon, KL

def U_loss(x, r, mu, logvar, y_hat_soft, eps):
    recon = torch.sum(x/r - torch.log(x + eps) + torch.log(r) - 1, dim=-1)
    KL = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp(), dim=-1)

    L = recon + KL
    L = L.view_as(y_hat_soft.t()).t()

    # Calculate entropy H(q(y|x)) and sum over all labels
    H = -torch.mul(y_hat_soft, torch.log(y_hat_soft + eps)) - torch.mul(1-y_hat_soft, torch.log(1-y_hat_soft + eps))
    L = torch.sum(torch.mul(y_hat_soft, L), dim=-1)

    # Equivalent to U(x)
    #U = torch.mean(L + H[:,0]) # wrong sign
    U = torch.mean(L - H[:,0])
    L = torch.mean(L)
    return U, L, torch.mean(recon), torch.mean(KL)

def f1_loss(y_hat_hard:torch.Tensor, y:torch.Tensor, epsilon=1e-8) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    y_pred = y_hat_hard.detach()
    y_true = y.detach()

    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    #f1.requires_grad = is_training
    return f1, tp, tn, fp, fn
