from __future__ import print_function

from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np

from data2 import load_data
from utils import *
from metrics import compute_metrics

create_paths()
log_file = open(global_path + "logs/log_file.txt", 'a')

# CEAL data definition
X_train, y_train = load_data('train')
labeled_index = np.arange(0, nb_labeled)
unlabeled_index = np.arange(nb_labeled, len(X_train))

X_val, y_val = load_data('val')

# (1) Initialize model
model = get_unet(dropout=True)
# model.load_weights(initial_weights_path)

if initial_train:
    model_checkpoint = ModelCheckpoint(initial_weights_path, monitor='loss', save_best_only=True)

    if apply_augmentation:
        for initial_epoch in range(0, nb_initial_epochs):
            history = model.fit_generator(
                data_generator().flow(X_train[labeled_index], y_train[labeled_index], batch_size=32, shuffle=True),
                steps_per_epoch=len(labeled_index), nb_epoch=1, verbose=1, callbacks=[model_checkpoint])

            model.save(initial_weights_path)
            log(history, initial_epoch, log_file)
    else:
        history = model.fit(X_train[labeled_index], y_train[labeled_index], batch_size=32, epochs=nb_initial_epochs,
                            verbose=1, shuffle=True, callbacks=[model_checkpoint])

        log(history, 0, log_file)
else:
    model.load_weights(initial_weights_path)

# Active loop
model_checkpoint = ModelCheckpoint(final_weights_path, monitor='loss', save_best_only=True)

for iteration in range(1, nb_iterations + 1):
    if iteration == 1:
        weights = initial_weights_path

    else:
        weights = final_weights_path

    # (2) Labeling
    computed_sets = compute_train_sets(X_train, y_train, labeled_index, unlabeled_index, weights, iteration)
    if computed_sets is None:
        break
    X_labeled_train, y_labeled_train, labeled_index, unlabeled_index = computed_sets

    # (3) Training
    history = model.fit(X_labeled_train, y_labeled_train, batch_size=32, epochs=nb_active_epochs, verbose=1,
                        shuffle=True, callbacks=[model_checkpoint])

    log(history, iteration, log_file)
    model.save(global_path + "models/active_model" + str(iteration) + ".h5")

    # validate
    predictions = model.predict(X_val)
    print(f'X_val: {X_val.shape}')
    print(f'y_val: {y_val.shape}')
    print(f'Predictions: {predictions.shape}')
    metrics = {}
    for index in range(predictions.shape[0]):
        sample_prediction = cv2.threshold(predictions[index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
        sample_metrics = compute_metrics(y_val[index][0], sample_prediction)
        for k in sample_metrics:
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(sample_metrics[k])
    print(metrics)
    for k in metrics:
        metrics[k] = np.asarray(metrics[k]).mean()
    print(metrics)


log_file.close()
