import sklearn.metrics
from sklearn.metrics import roc_auc_score

def data1D(ax, real_data, fake_data):
    
    bins = 30
    xlo, xhi = -1.5, 1.5
    _ = ax.hist(real_data, bins=bins, alpha=0.7, range=(xlo, xhi), label='real data')
    _ = ax.hist(fake_data, bins=bins, alpha=0.7, range=(xlo, xhi), label='fake data')
    _ = ax.set_xlim(xlo, xhi)
    _ = ax.legend(loc='best')
    _ = ax.set_xlabel('False positive rate')


def roc_curve(ax, labels, logits):
    
    auroc = roc_auc_score(labels, logits)
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, logits)
    _ = ax.plot(fpr, tpr, label='AUROC={:2.2f}'.format(auroc))
    _ = ax.set_xlabel('False positive rate')
    _ = ax.set_ylabel('True positive rate')
    _ = ax.legend(loc='best')
    

