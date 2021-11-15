import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score
import cv2
from data2 import load_data

from constants import img_rows, img_cols, global_path
from visualization import print_log


def validate(model, X_val, y_val, iteration, n_samples, n_labeled_used):
    predictions = model.predict(X_val)
    # print(f'X_val: {X_val.shape}')  # (30, 6, 256, 256)
    # print(f'y_val: {y_val.shape}')  # (30, 1, 256, 256)
    # print(f'Predictions: {predictions.shape}')  # (30, 1, 256, 256)
    metrics = {}
    avg_metrics = {}
    with open(global_path + f"results/metrics_{iteration}.txt", 'a') as f:
        print_log(f'========================= Active Iteration {iteration}', file=f)
        print_log(f'Num of samples: {n_labeled_used} / {n_samples} ==> {n_labeled_used / n_samples * 100:.0f}%', file=f)

        for index in range(len(X_val)):
            sample_pred = cv2.threshold(predictions[index], 0.5, 1, cv2.THRESH_BINARY)[1]
            # print(sample_pred.shape)  # (1, 256, 256)
            sample_true = cv2.threshold(y_val[index], 0.5, 1, cv2.THRESH_BINARY)[1]

            print(f'==================== {index}', file=f)

            if iteration % 10 == 0:
                save_img(f'val_{iteration}_{index}_pred.png', sample_pred)
                save_img(f'val_{iteration}_{index}_true.png', sample_true)

            sample_pred_int = sample_pred.astype('uint8')
            sample_true_int = sample_true.astype('uint8')

            sample_metrics = compute_metrics(sample_true_int, sample_pred_int)
            print_log(sample_metrics)
            print_log(f'===== Sample {index}', file=f)
            print_log(sample_metrics, file=f)

            for k in sample_metrics:
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(sample_metrics[k])

        for k in metrics:
            avg_metrics[k] = np.asarray(metrics[k]).mean()

        print_log(f'===== AVERAGE of {len(X_val)} samples', file=f)
        print_log(avg_metrics, file=f)
        print_log('\n\n', file=f)

    return avg_metrics


def test(model):
    X_test, y_test = load_data('test')

    predictions = model.predict(X_test)
    metrics = {}
    for index in range(len(X_test)):
        sample_pred = cv2.threshold(predictions[index], 0.5, 1, cv2.THRESH_BINARY)[1]
        # print(sample_pred.shape)  # (1, 256, 256)
        sample_true = cv2.threshold(y_test[index], 0.5, 1, cv2.THRESH_BINARY)[1]

        save_img(f'test_{index}_pred.png', sample_pred)
        save_img(f'test_{index}_true.png', sample_true)

        sample_pred_int = sample_pred.astype('uint8')
        sample_true_int = sample_true.astype('uint8')

        sample_metrics = compute_metrics(sample_true_int, sample_pred_int)
        for k in sample_metrics:
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(sample_metrics[k])
    for k in metrics:
        metrics[k] = np.asarray(metrics[k]).mean()

    with open(global_path + "results/test.txt", 'a') as f:
        print(metrics, file=f)


def compute_metrics(y_true, y_pred):
    smooth = 1.  # smoothing value to deal zero denominators.
    y_true_f = y_true.reshape([1, img_rows * img_cols])
    y_pred_f = y_pred.reshape([1, img_rows * img_cols])

    f1 = f1_score(y_true_f[0], y_pred_f[0])
    accuracy = accuracy_score(y_true_f[0], y_pred_f[0])
    precision = precision_score(y_true_f[0], y_pred_f[0])
    recall = recall_score(y_true_f[0], y_pred_f[0])
    jaccard = jaccard_score(y_true_f[0], y_pred_f[0])

    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    dice2 = 2 * jaccard / (1 + jaccard)

    print(f'Their dice: {dice:.4f}')
    print(f'My dice: {dice2:.4f}')

    return {
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'jaccard': jaccard,
        'dice': dice2,
    }


def check_nan(t):
    return 0 if np.isnan(t) else t


def save_img(name, im):
    c, h, w = im.shape
    img = im.reshape(h, w, c) * 255.
    cv2.imwrite(global_path + 'results/' + name, img)


def debug(arr, name):
    mn = arr.min()
    mx = arr.max()
    unique, counts = np.unique(arr, return_counts=True)
    n_unique = len(unique)
    if n_unique == 2:
        print(f'{name}: {mn} ({counts[0]}), {mx} ({counts[1]}), {n_unique}')
    else:
        print(f'{name}: {mn}, {mx}, {n_unique}')
