from bonfire.model.benchmark import crc_models, count_mnist_models, dgr_models, four_mnist_models, masati_models,\
    musk_models, sival_models, tef_models

model_names = ['InstanceSpaceNN', 'EmbeddingSpaceNN', 'AttentionNN', 'MultiHeadAttentionNN', 'ClusterGNN', 'MiLstm']


def get_model_clz(dataset_name, model_name):
    if dataset_name == 'crc':
        model_clzs = crc_models.get_model_clzs()
    elif dataset_name == 'count_mnist':
        model_clzs = count_mnist_models.get_model_clzs()
    elif dataset_name == 'dgr':
        model_clzs = dgr_models.get_model_clzs()
    elif dataset_name == 'four-mnist':
        model_clzs = four_mnist_models.get_model_clzs()
    elif dataset_name == 'masati':
        model_clzs = masati_models.get_model_clzs()
    elif dataset_name == 'musk':
        model_clzs = musk_models.get_model_clzs()
    elif dataset_name == 'sival':
        model_clzs = sival_models.get_model_clzs()
    elif dataset_name in ['tiger', 'elephant', 'fox']:
        model_clzs = tef_models.get_model_clzs()
    else:
        raise ValueError('No models registered for dataset {:}'.format(dataset_name))

    for model_clz in model_clzs:
        if model_name == model_clz.__base__.__name__:
            return model_clz
    raise ValueError('No model with name {:s} found for dataset {:s}'.format(model_name, dataset_name))
