import numpy as np
import sklearn.metrics
from sklearn.metrics import roc_auc_score


def data1D(ax, real_data, fake_data):
    
    bins = 30
    xlo, xhi = -0.25, 1.25
    _ = ax.hist(real_data, bins=bins, alpha=0.7, range=(xlo, xhi), label='real data')
    _ = ax.hist(fake_data, bins=bins, alpha=0.7, range=(xlo, xhi), label='fake data')
    _ = ax.set_xlim(xlo, xhi)
    _ = ax.legend(loc='best')
    ax.set_xlim(xlo, xhi)


def ratio1D(ax, real_data, fake_data):

    bins = 30
    xlo, xhi = -0.25, 1.25
    rh, edges = np.histogram(real_data, bins=bins, range=(xlo, xhi))
    fh, edges = np.histogram(fake_data, bins=bins, range=(xlo, xhi))
    upper, lower = edges[1:], edges[:-1]
    centres = 0.5*(upper+lower)
    _ = ax.hist(centres, weights=rh/fh, color='k', histtype='step', bins=bins, range=(xlo, xhi))
    _ = ax.plot([0, 1], [1.0, 1.0], 'r')
    _ = ax.set_xlabel('modelled variable')
    ax.set_xlim(xlo, xhi)


def roc_curve(ax, labels, logits):
    
    auroc = roc_auc_score(labels, logits)
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, logits)
    _ = ax.plot(fpr, tpr, label='AUROC={:2.2f}'.format(auroc))
    _ = ax.set_xlabel('False positive rate')
    _ = ax.set_ylabel('True positive rate')
    _ = ax.legend(loc='best')
    

