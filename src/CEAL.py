from __future__ import print_function

from keras.callbacks import ModelCheckpoint

from data2 import load_data
from metrics import validate, test
import time
from utils import *
from visualization import print_log, sec_to_time

create_paths()
log_file = open(global_path + "logs/log_file.txt", 'w')
print_log(f'====== START TRAINING ({time.strftime("%d/%m/%Y %H:%M:%S")}) ======', file=log_file)
start_time = time.time()

# CEAL data definition
X_train, y_train = load_data('train')
n_samples = len(X_train)
labeled_index = np.arange(0, nb_labeled)
unlabeled_index = np.arange(nb_labeled, n_samples)

X_val, y_val = load_data('val')

# (1) Initialize model
model = get_unet(dropout=True)
# model.load_weights(initial_weights_path)

if initial_train:
    print_log(f'Initial training with {nb_labeled} samples', file=log_file)
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

n_labeled_used = nb_labeled  # how many labeled samples have been used
validate(model, X_val, y_val, 0, n_samples, n_labeled_used)

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
    X_labeled_train, y_labeled_train, labeled_index, unlabeled_index, oracle_size = computed_sets
    n_labeled_used += oracle_size

    # (3) Training
    history = model.fit(X_labeled_train, y_labeled_train, batch_size=32, epochs=nb_active_epochs, verbose=1,
                        shuffle=True, callbacks=[model_checkpoint])

    log(history, iteration, log_file)
    model.save(global_path + "models/active_model" + str(iteration) + ".h5")

    validate(model, X_val, y_val, iteration, n_samples, n_labeled_used)

end_time = time.time()
total_time = end_time - start_time
print_log('============================\n', file=log_file)
print_log(f'====== END TRAINING ({time.strftime("%d/%m/%Y %H:%M:%S")}) - {sec_to_time(total_time)} ======\n', file=log_file)
log_file.close()

test(model)
