# PATH definition
global_path = "output_"
initial_weights_path = "models/init_weights.hdf5"
final_weights_path = "models/output_weights.hdf5"

# Data definition
# img_rows = 64 * 3
# img_cols = 80 * 3
img_rows = 256
img_cols = 256
n_channel = 6

nb_total = 1411
nb_train = 1350
nb_labeled = 270
nb_unlabeled = nb_train - nb_labeled

# CEAL parameters
apply_edt = True
nb_iterations = 10

nb_step_predictions = 20

nb_no_detections = 10
nb_random = 15
nb_most_uncertain = 10
most_uncertain_rate = 5

pseudo_epoch = 5
nb_pseudo_initial = 20
pseudo_rate = 20

initial_train = True
apply_augmentation = False
nb_initial_epochs = 10
nb_active_epochs = 2
batch_size = 128
