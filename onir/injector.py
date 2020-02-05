import inspect
import numpy as np
import colorlog
from terminaltables import AsciiTable
from onir import config, log, util



def load(configuration, extra_args=None, pretty=False):
    a = config.args()
    if extra_args:
        b = a
        a = b
        a.update(b)
    context = {}
    for name, obj in configuration.items():
        # handle modules with registry
        if hasattr(obj, 'registry'):
            obj = obj.registry

        if isinstance(obj, util.Registry):
            registry = obj
            registered = registry.registered
            default = registry.default
            if default is None and name not in a:
                raise ValueError(f'Missing argument for `{name}`')
            if a.get(name, default) not in registered:
                raise ValueError('no `{}` found for a `{}`'.format(a.get(name, default), name))
            cls = registered[a.get(name, default)]
        else:
            cls = obj

        spec = inspect.getargspec(cls.__init__)
        args = []
        for arg in spec.args:
            # HACK
            if arg == 'dataset' and name == 'valid_pred':
                arg = 'valid_ds'
            if arg == 'dataset' and name == 'test_pred':
                arg = 'test_ds'

            if arg == 'self':
                continue
            elif arg in context:
                args.append(context[arg])
            elif arg == 'random':
                args.append(np.random.RandomState(int(config.args()['random_seed'])))
            elif arg == 'logger':
                logger_name = (name+':'+a.get(name, default)).rstrip(':')
                args.append(log.Logger(logger_name))
            elif arg == 'config':
                args.append(config.apply_config(name, a, cls))
            else:
                raise ValueError(f'cannot match argument `{arg}` for `{cls}`')
        context[name] = cls(*args)

    if pretty:
        _log_config(configuration.keys(), context)

    return context


LIGHT = colorlog.escape_codes['thin_white']
YELLOW = colorlog.escape_codes['yellow']
RESET = colorlog.escape_codes['reset']


def _log_row(row, default_config):
    key, value = row
    if value == '':
        return key, f'{LIGHT}[empty]{RESET}'
    default = default_config.get(key)
    if isinstance(default, config.ConfigValue):
        default = default.value
    if default != value:
        return key, f'{YELLOW}{value}{RESET}'
    return key, value


def _log_config(keys, context):
    result = ''
    for key in keys:
        c = context[key].config
        items = list(c.items())
        default_config = context[key].default_config()
        tab = [(key, context[key].name, "", "", "", "", "", "")]
        for i in range((len(items) + 2) // 3):
            row_l = _log_row(items[i*3], default_config)
            if i * 3 + 1 < len(items):
                row_m = ("|", *_log_row(items[i*3+1], default_config))
            else:
                row_m = ("", "", "")
            if i * 3 + 2 < len(items):
                row_r = ("|", *_log_row(items[i*3+2], default_config))
            else:
                row_r = ("", "", "")
            tab.append((*row_l, *row_m, *row_r))
        table = AsciiTable(tab)
        table.inner_column_border = False
        table.outer_border = False
        result += f'\n{table.table}\n'
    log.easy().debug(f'Configuration:{RESET}{result}')
