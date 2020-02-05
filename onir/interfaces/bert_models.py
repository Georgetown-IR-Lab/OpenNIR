import os
import shutil
from glob import glob
import torch
import pytorch_pretrained_bert
from pytorch_pretrained_bert import BertForPreTraining, BertConfig
from onir import util


def _hugging_handler(name, base_path, logger):
    # Just use the default huggingface handler for model
    return name


def _scibert_handler(url_seg):
    url = f'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/{url_seg}'
    def wrapped(name, base_path, logger):
        path = os.path.join(base_path, name)
        if not os.path.exists(path):
            _download_tarball(url, path, logger)

            weights_tarball = os.path.join(path, 'weights.tar.gz')
            util.extract_tarball(weights_tarball, path, logger, reset_permissions=True)
            os.remove(weights_tarball)
        return path
    return wrapped


def _biobert_handler(url_seg):
    url = f'https://github.com/naver/biobert-pretrained/releases/download/{url_seg}'
    def wrapped(name, base_path, logger):
        path = os.path.join(base_path, name)
        if not os.path.exists(path):
            _download_tarball(url, path, logger)

            _convert_tf_checkpoint_to_pytorch(os.path.join(path, 'biobert_model.ckpt'),
                                              os.path.join(path, 'bert_config.json'),
                                              os.path.join(path, 'pytorch_model.bin'))
            for file in os.listdir(path):
                if file not in ('bert_config.json', 'pytorch_model.bin', 'vocab.txt'):
                    if os.path.isfile(os.path.join(path, file)):
                        os.remove(os.path.join(path, file))
                    else:
                        os.rmdir(os.path.join(path, file))
        return path
    return wrapped


# locations of pre-trained BERT configurations
MODEL_MAP = {
    'scibert-scivocab-uncased':_scibert_handler('pytorch_models/scibert_scivocab_uncased.tar'),
    'scibert-scivocab-cased':_scibert_handler('pytorch_models/scibert_scivocab_cased.tar.gz'),
    'scibert-basevocab-uncased':_scibert_handler('pytorch_models/scibert_basevocab_uncased.tar.gz'),
    'scibert-basevocab-cased':_scibert_handler('pytorch_models/scibert_basevocab_cased.tar.gz'),
    'biobert-pubmed-pmc':_biobert_handler('v1.0-pubmed-pmc/biobert_pubmed_pmc.tar.gz'),
    'biobert-pubmed':_biobert_handler('v1.0-pubmed/biobert_pubmed.tar.gz'),
    'biobert-pmc':_biobert_handler('v1.0-pmc/biobert_pmc.tar.gz'),
    **{m:_hugging_handler for m in pytorch_pretrained_bert.modeling.PRETRAINED_MODEL_ARCHIVE_MAP},
}


# shortcuts for "recommended" configurations
MODEL_ALIAS = {
    'scibert': 'scibert-scivocab-uncased',
    'biobert': 'biobert-pubmed-pmc'
}


def get_model(name, logger):
    name = MODEL_ALIAS.get(name, name) # replace with alias
    if name not in MODEL_MAP:
        raise ValueError(f'Unknown bert model {name}')
    base_path = os.path.join(util.get_working(), 'bert_models')
    os.makedirs(base_path, exist_ok=True)
    return MODEL_MAP[name](name, base_path, logger)


def _download_tarball(url, path, logger):
    util.download_if_needed(url, path + '.tar.gz')
    util.extract_tarball(path + '.tar.gz', path, logger, reset_permissions=True)
    os.remove(path + '.tar.gz')
    for file in glob(path + '/*/*') + glob(path + '/*/.*'):
        shutil.move(file, path)


def _load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    Adapted from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L52
    """
    SKIP_KEYS = {"adam_v", "adam_m", "BERTAdam", "BERTAdam_1"}
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please "
              "see https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    # print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, _ in init_vars:
        # print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if name[0] != 'bert' or any(n in SKIP_KEYS for n in name):
            # print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
            if m_name[-11:] == '_embeddings':
                pointer = getattr(pointer, 'weight')
            elif m_name == 'kernel':
                array = np.transpose(array)
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            pointer.data = torch.from_numpy(array)
    return model


def _convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # adapated from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py#L30
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    _load_tf_weights_in_bert(model, tf_checkpoint_path)

    # Save pytorch-model
    torch.save(model.state_dict(), pytorch_dump_path)
