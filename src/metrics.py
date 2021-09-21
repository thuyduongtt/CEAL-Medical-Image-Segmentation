import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
from data2 import load_data

from constants import img_rows, img_cols, global_path


def validate(model, X_val, y_val, iteration, n_samples, n_labeled_used):
    predictions = model.predict(X_val)
    # print(f'X_val: {X_val.shape}')  # (30, 6, 256, 256)
    # print(f'y_val: {y_val.shape}')  # (30, 1, 256, 256)
    # print(f'Predictions: {predictions.shape}')  # (30, 1, 256, 256)
    metrics = {}
    for index in range(len(X_val)):
        sample_prediction = cv2.threshold(predictions[index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
        # print(sample_prediction.shape)  # (1, 256, 256)
        sample_true = cv2.threshold(y_val[index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')

        cv2.imwrite(f'val_{iteration}_{index}_pred.bmp', sample_prediction)
        cv2.imwrite(f'val_{iteration}_{index}_true.bmp', sample_true)

        sample_metrics = compute_metrics(sample_true, sample_prediction)
        for k in sample_metrics:
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(sample_metrics[k])
    for k in metrics:
        metrics[k] = np.asarray(metrics[k]).mean()

    with open(global_path + "logs/metrics.txt", 'a') as f:
        print(f'========================= Active Iteration {iteration}', file=f)
        print(f'Num of samples: {n_labeled_used} / {n_samples} ==> {n_labeled_used / n_samples * 100:.0f}%', file=f)
        print(metrics, file=f)
        print('\n\n')


def test(model):
    X_test, y_test = load_data('test')

    predictions = model.predict(X_test)
    metrics = {}
    for index in range(len(X_test)):
        sample_prediction = cv2.threshold(predictions[index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
        # print(sample_prediction.shape)  # (1, 256, 256)
        sample_true = cv2.threshold(y_test[index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')

        cv2.imwrite(f'test_{index}_pred.bmp', sample_prediction)
        cv2.imwrite(f'test_{index}_true.bmp', sample_true)

        sample_metrics = compute_metrics(sample_true, sample_prediction)
        for k in sample_metrics:
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(sample_metrics[k])
    for k in metrics:
        metrics[k] = np.asarray(metrics[k]).mean()

    with open(global_path + "logs/test.txt", 'a') as f:
        print(metrics, file=f)


def compute_metrics(y_true, y_pred):
    smooth = 1.  # smoothing value to deal zero denominators.
    y_true_f = y_true.reshape([1, img_rows * img_cols])
    y_pred_f = y_pred.reshape([1, img_rows * img_cols])

    # print(np.unique(y_true_f))
    # print(np.unique(y_pred_f))

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
    return 0 if np.isnan(t) else t
