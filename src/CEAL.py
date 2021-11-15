from __future__ import print_function

from keras.callbacks import ModelCheckpoint

from data2 import load_data
from metrics import validate, test
from utils import *
from visualization import *

create_paths()
log_file_path = global_path + "logs/log_file.txt"
log_file = open(log_file_path, 'w')
print_log(f'====== START TRAINING ({time.strftime("%d/%m/%Y %H:%M:%S")}) ======', file=log_file)
log_file.close()
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

# early_stopper = EarlyStopping(monitor='loss', mode='min', patience=30)
# tensorboard = TensorBoard(log_dir=f'{global_path}logs')

all_loss = []
all_metrics = {}

if initial_train:
    print_log(f'Initial training with {nb_labeled} samples', file_path=log_file_path)
    iter_start_time = time.time()
    model_checkpoint = ModelCheckpoint(initial_weights_path, monitor='loss', save_best_only=True)

    if apply_augmentation:
        for initial_epoch in range(0, nb_initial_epochs):
            history = model.fit_generator(
                data_generator().flow(X_train[labeled_index], y_train[labeled_index], batch_size=batch_size, shuffle=True),
                steps_per_epoch=len(labeled_index), nb_epoch=1, verbose=1, callbacks=[model_checkpoint])

            model.save(initial_weights_path)
            # log(history, initial_epoch, log_file)
            all_loss.extend(history.history['loss'])

            # history.history includes: 'loss' and 'dice_coef'
            # history.history['loss'] and history.history['dice_coef'] are list with 'num_of_epoch' elements
            plot([all_loss], title='Losses after initial training', labels=['loss'],
                 output_dir=f'{global_path}plots', output_name=f'init_train_{initial_epoch}')

    else:
        history = model.fit(X_train[labeled_index], y_train[labeled_index], batch_size=batch_size, epochs=nb_initial_epochs,
                            verbose=1, shuffle=True, callbacks=[model_checkpoint])

        all_loss.extend(history.history['loss'])

        # log(history, 0, log_file)
        plot([all_loss], title='Losses after initial training', labels=['loss'], output_dir=f'{global_path}plots', output_name='init_train')

    iter_end_time = time.time()
    iter_time = iter_end_time - iter_start_time
    print_log(f'Initial iteration: {iter_time:.0f}s - {sec_to_time(iter_time)}', file_path=log_file_path)

else:
    model.load_weights(initial_weights_path)

n_labeled_used = nb_labeled  # how many labeled samples have been used
metrics = validate(model, X_val, y_val, 0, n_samples, n_labeled_used)
for k in metrics:
    if k not in all_metrics:
        all_metrics[k] = []
    all_metrics[k].append(metrics[k])
plot([all_metrics['f1'], all_metrics['accuracy'], all_metrics['precision'], all_metrics['recall'], all_metrics['dice'], all_metrics['jaccard']],
     f'Validation Scores at initial iteration', labels=['F1', 'Accuracy', 'Precision', 'Recall', 'Dice', 'Jaccard'],
     output_dir=f'{global_path}plots', output_name=f'val_init', ylabel='value')

# Active loop
model_checkpoint = ModelCheckpoint(final_weights_path, monitor='loss', save_best_only=True)

for iteration in range(1, nb_iterations + 1):
    iter_start_time = time.time()

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
    history = model.fit(X_labeled_train, y_labeled_train, batch_size=batch_size, epochs=nb_active_epochs, verbose=1,
                        shuffle=True, callbacks=[model_checkpoint])

    all_loss.extend(history.history['loss'])

    # log(history, iteration, log_file)
    plot([all_loss], title=f'Losses after iteration {iteration}', labels=['loss'], output_dir=f'{global_path}plots', output_name=f'loss_{iteration}')

    model.save(global_path + "models/active_model" + str(iteration) + ".h5")

    metrics = validate(model, X_val, y_val, iteration, n_samples, n_labeled_used)
    for k in metrics:
        if k not in all_metrics:
            all_metrics[k] = []
        all_metrics[k].append(metrics[k])
    plot([all_metrics['f1'], all_metrics['accuracy'], all_metrics['precision'], all_metrics['recall'], all_metrics['dice'], all_metrics['jaccard']],
         f'Validation Scores at Iteration {iteration}', labels=['F1', 'Accuracy', 'Precision', 'Recall', 'Dice', 'Jaccard'],
         output_dir=f'{global_path}plots', output_name=f'val_{iteration}', ylabel='value')

    iter_end_time = time.time()
    iter_time = iter_end_time - iter_start_time
    print_log(f'Iteration {iteration}: {iter_time:.0f}s - {sec_to_time(iter_time)}', file_path=log_file_path)

end_time = time.time()
total_time = end_time - start_time
print_log('============================\n', file=log_file)
print_log(f'====== END TRAINING ({time.strftime("%d/%m/%Y %H:%M:%S")}) - {sec_to_time(total_time)} ======\n', file_path=log_file_path)
log_file.close()

test(model)
