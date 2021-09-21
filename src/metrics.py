import numpy as np
from sklearn.metrics import confusion_matrix

from constants import img_rows, img_cols


def compute_metrics(y_true, y_pred):
    smooth = 1.  # smoothing value to deal zero denominators.
    y_true_f = y_true.reshape([1, img_rows * img_cols])
    y_pred_f = y_pred.reshape([1, img_rows * img_cols])

    print(y_true_f.unique())
    print(y_pred_f.unique())

    tn, fp, fn, tp = confusion_matrix(y_true_f[0], y_pred_f[0]).ravel()

    acc = (tn + tp) / (tn + tp + fp + fn)  # Accuracy (all correct / all)
    precision = tp / (tp + fp)  # Precision (true positives / predicted positives)
    sensitivity = tp / (tp + fn)  # Sensitivity aka Recall (true positives / all actual positives)
    fpr = fp / (fp + tn)  # False Positive Rate (Type I error)
    specificity = tn / (tn + fp)  # Specificity (true negatives / all actual negatives)
    error = (fn + fp) / (tn + tp + fp + fn)  # Misclassification (all incorrect / all)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)

    acc = check_nan(acc)
    precision = check_nan(precision)
    sensitivity = check_nan(sensitivity)
    fpr = check_nan(fpr)
    specificity = check_nan(specificity)
    error = check_nan(error)
    f1 = check_nan(f1)

    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    return {
        'dice': dice,
        'acc': acc,
        'precision': precision,
        'sensitivity': sensitivity,
        'fpr': fpr,
        'specificity': specificity,
        'error': error,
        'f1': f1
    }


def check_nan(t):
    isnan = np.isnan(t)
    if isnan.any():
        t[isnan == 1] = 0
    return t
