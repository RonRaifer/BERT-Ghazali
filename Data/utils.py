params = {
    'Name': "",
    'Niter': 10,
    'ACCURACY_THRESHOLD': 0.93,
    'BERT_INPUT_LENGTH': 510,
    'F1': 0.3,
    'SILHOUETTE_THRESHOLD': 0.75,
    'TEXT_DIVISION_METHOD': 'Fixed-Size',
    'F': 'minority',

    'KERNELS': 30,
    'LEARNING_RATE': 0.001,
    'NB_EPOCHS': 10,
    '1D_CONV_KERNEL': {1: 2, 2: 3, 3: 4, 4: 5},
    'OUTPUT_CLASSES': 1,
    'STRIDES': 1,
    'BATCH_SIZE': 20,
    'ACTIVATION_FUNC': 'Relu',
    'DROPOUT_RATE': 0.5,
}

heat_map = None
heat_map_plot = None
kmeans_plot = None
labels = None
silhouette_calc = None
log_content = None
progress_bar = None
original_stdout = None
stopped = False


def LoadDefaultCNNConfig():
    params['KERNELS'] = 30
    params['LEARNING_RATE'] = 0.001
    params['NB_EPOCHS'] = 10
    params['1D_CONV_KERNEL'] = {1: 2, 2: 3, 3: 4, 4: 5}
    params['OUTPUT_CLASSES'] = 1
    params['STRIDES'] = 1
    params['BATCH_SIZE'] = 20
    params['ACTIVATION_FUNC'] = 'Relu'
    params['DROPOUT_RATE'] = 0.3


def LoadDefaultGeneralConfig():
    params['Niter'] = 10
    params['ACCURACY_THRESHOLD'] = 0.93
    params['BERT_INPUT_LENGTH'] = 510
    params['F1'] = 0.3
    params['SILHOUETTE_THRESHOLD'] = 0.75
    params['TEXT_DIVISION_METHOD'] = 'Fixed-Size'
    params['F'] = 'minority'


