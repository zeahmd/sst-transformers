from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluation_metrics(Y_true, Y_pred, split='test'):
    metrics = dict()
    metrics[split+'_accurac_score'] = accuracy_score(Y_true, Y_pred)
    metrics[split+'_precision_score'] = precision_score(Y_true, Y_pred, average='macro')
    metrics[split+'_recall_score'] = recall_score(Y_true, Y_pred, average='macro')
    metrics[split+'_f1_score'] = f1_score(Y_true, Y_pred, average='macro')

    return metrics