# -*- coding: utf-8 -*-
"""MatchZoo interface

This module interfaces with the MatchZoo-py library. For instance, it provides a general
translation between MatchZoo models and onir.rankers. It also supports operations such as inferring
the required inputs from a MatchZoo model implementation.

Attributes:
    is_available (bool): Indicates whether matchzoo-py installed and available for use.
    If `is_available == False`, try installing matchzoo-py with `pip install matchzoo-py==1.0`.

    mz (module): Direct interface to the matchzoo module. Is `None` if `is_available == False`.
"""

import numpy as np
from onir import rankers


try:
    import matchzoo as mz
    is_available = True
except ImportError:
    is_available = False
    mz = None


_CONFIG_IGNORED_PARAMS = ('model_class', 'task', 'mask_value', 'with_embedding', 'embedding',
                          'embedding_input_dim', 'embedding_output_dim', 'embedding_freeze',
                          'with_multi_layer_perceptron')


def generate_mz_ranker(MatchZooModel: type) -> type:
    """Generates an onir.rankers.Ranker class for the given MatchZooModel.

    Args:
        MatchZooModel (type): The MatchZoo model class

    Returns:
        type: A onir.rankers.Ranker subclass wrapping MatchZooModel

    Examples:
        >>> generate_mz_ranker(mz.models.KNRM)
        <class 'onir.interfaces.matchzoo.generate_mz_ranker.<locals>.MatchZooBaseRanker'>

    """
    default_config = generate_default_config(MatchZooModel)

    class MatchZooBaseRanker(rankers.Ranker):
        @staticmethod
        def default_config() -> dict:
            return dict(default_config)

        def __init__(self, config, random, logger, vocab):
            super().__init__(config, random)
            self.vocab = vocab
            self.logger = logger
            encoder = vocab.encoder()
            assert encoder.enc_spec()['supports_forward'], \
                   "Joint query/document encoding not supported by MatchZoo rankers."
            assert encoder.emb_views() == 1, \
                   "Multiple embedding views not supported by MatchZoo rankers."
            params = MatchZooModel.get_default_params()
            params.update(config)
            params['embedding'] = np.zeros((vocab.lexicon_size(), encoder.dim()))
            params['task'] = mz.tasks.Ranking()
            if 'mask_value'  in params:
                params['mask_value'] = -1
            self.mz_model = MatchZooModel(params)
            self.mz_model.build()
            self.mz_model.embedding = encoder
            self._fields = set()
            self._input_adapters = {}
            for mz_input in guess_mz_model_inputs(self.mz_model):
                if mz_input in _INPUT_ADAPTERS:
                    self._input_adapters[mz_input] = _INPUT_ADAPTERS[mz_input][0]
                    self._fields.update(_INPUT_ADAPTERS[mz_input][1])
                else:
                    raise ValueError(f'Unsupported MatchZoo input `{mz_input}`.')

        def input_spec(self):
            result = super().input_spec()
            result['fields'].update(self._fields)
            return result

        def _forward(self, **inputs):
            mz_inputs = {key: adapter(inputs) for key, adapter in self._input_adapters.items()}
            mz_output = self.mz_model(mz_inputs)
            return mz_output

        def path_segment(self):
            result = '{name}'.format(name=self.name, **self.config)
            # include non-default configuration entries in path segment
            for key, val in sorted(self.config.items()):
                if key != '' and default_config[key] != val:
                    result += f'_{key}-{val}'
            return result

    MatchZooBaseRanker.__doc__ = f"""
    Interface to MatchZoo's {MatchZooModel.__name__} model.
    """

    return MatchZooBaseRanker



def generate_default_config(MatchZooModel: type) -> dict:
    """Generates a default config for the given MatchZooModel, based on its default params.

    Args:
        MatchZooModel (type): The MatchZoo model class

    Returns:
        dict: A dictionary containing the configuration keys and default values, and any default
        config items from onir.rankers.Ranker.default_config().

    Examples:
        >>> generate_default_config(mz.models.KNRM)
        {'qlen': 20, 'dlen': 2000, 'add_runscore': False, 'kernel_num': 11, 'sigma': 0.1, 'exact_sigma': 0.001}
    """
    default_config = rankers.Ranker.default_config()
    default_params = MatchZooModel.get_default_params()
    for param in default_params.keys():
        if param not in _CONFIG_IGNORED_PARAMS:
            default_config[param] = default_params[param]
    return default_config


def guess_mz_model_inputs(mz_model) -> set:
    """Guesses the inputs needed to execute a particular mz_model.

    Examples:
        >>> guess_mz_model_inputs(mz.models.KNRM)
        {'text_right', 'text_left'}
    """
    class InputProbe:
        def __init__(self):
            self.keys = set()
        def __getitem__(self, key):
            self.keys.add(key)
            return None

    probe = InputProbe()
    try:
        mz_model(probe)
    except Exception: # TODO: less generic except block here?
        # model will fail when it actually uses inputs, which are just placeholder None
        pass

    return probe.keys


_INPUT_ADAPTERS = {
    'text_left': (lambda inputs: inputs['query_tok'], {'query_tok'}),
    'text_right': (lambda inputs: inputs['doc_tok'], {'doc_tok'}),
    # Add more input adapters here as needed
}
