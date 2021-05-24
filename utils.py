params = {
    'Name': "",
    'Niter': 10,
    'ACCURACY_THRESHOLD': 0.93,
    'BERT_INPUT_LENGTH': 510,
    'F1': 0.3,
    'SILHOUETTE_THRESHOLD': 0.75,
    'TEXT_DIVISION_METHOD': 'Fixed-Size',
    'F': 'minority',

    'KERNELS': 3,
    'LEARNING_RATE': 0.001,
    'NB_EPOCHS': 10,
    '1D_CONV_KERNEL': {1: 3, 2: 6, 3: 12},
    'OUTPUT_CLASSES': 1,
    'STRIDES': 1,
    'BATCH_SIZE': 32,
    'ACTIVATION_FUNC': 'Relu',
    'DROPOUT_RATE': 0.3,
}

heat_map = None
heat_map_plot = None
kmeans_plot = None
labels = None
silhouette_calc = None


def LoadDefaultCNNConfig():
    params['KERNELS'] = 3
    # params['CNN_FILTERS'] = 500
    params['LEARNING_RATE'] = 0.001
    params['NB_EPOCHS'] = 10
    params['1D_CONV_KERNEL'] = {1: 3, 2: 6, 3: 12}
    # params['POOLING_SIZE'] = 500
    # params['DECAY'] = 1
    params['OUTPUT_CLASSES'] = 1
    params['STRIDES'] = 1
    params['BATCH_SIZE'] = 32
    # params['MOMENTUM'] = 0.9
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


progress_bar = None


def isint_and_inrange(n, start, end):
    x = False
    try:
        int(n)
        x = True if start <= int(n) <= end else False
        return x
    except ValueError:
        return x


def isfloat_and_inrange(n, start, end):
    x = False
    try:
        float(n)
        x = True if start < float(n) < end else False
        return x
    except ValueError:
        return x


def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

