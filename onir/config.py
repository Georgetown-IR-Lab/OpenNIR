import sys
import os
import functools
from datetime import datetime
import random
from pathlib import Path
import shlex


@functools.lru_cache()
def logger():
    from onir import log
    return log.easy()


@functools.lru_cache()
def args():
    result = {
        # runid for logging, etc.
        'runid': datetime.now().strftime(f'%Y%m%dT%H%M%S-{random.randint(1000, 9999)}')
    }

    # default configuration
    result.update(_parse_args(['./config']))

    # configuration from ONIR_ARGS
    result.update(_parse_args(shlex.split(os.environ.get('ONIR_ARGS', ''))))

    # configuration from command line arguments
    if os.environ.get('ONIR_IGNORE_ARGV', '').lower() != 'true':
        result.update(_parse_args(sys.argv[1:]))

    return result


def _parse_args(arg_iter, cd=None):
    for arg in arg_iter:
        if '=' in arg:
            # assignment, i.e., key=value
            key, value = arg.split('=', 1)
            yield key, value
        elif arg == '###':
            # stop reading args once ### is encountered. This is useful when using argparse or
            # similar in addition to or instead of this argument parsing mechanism.
            break
        else:
            # reference to file, i.e., config/something
            path = Path(arg if cd is None else os.path.join(cd, arg))
            while path.is_dir():
                path = path / '_dir'
            if not path.exists():
                raise FileNotFoundError(f'configuraiton file not found: {path}')
            with open(path, 'rt') as f:
                yield from _parse_args(shlex.split(f.read()), path.parent)


def apply_config(name, args, cls):
    result = {}
    for key, value in cls.default_config().items():
        if not isinstance(value, ConfigValue):
            value = TypeConstraint(type(value), value)
        arg_key = f'{name}.{key}'
        if arg_key in args:
            value.override_value(args[arg_key])
        result[key] = value.realize()
    for key, val in args.items():
        if key.startswith(f'{name}.'):
            subkey = key.split('.', 1)[1]
            if subkey not in result:
                logger().warn(f'Argument {key}={val} set but no matching setting in {cls}')
    return result


class ConfigValue:
    def __init__(self, desc=None):
        self.value = None
        self.desc = desc

    def override_value(self, value):
        self.value = value

    def realize(self):
        raise NotImplementedError

    def __str__(self):
        if self.desc is not None:
            return f'{self.vaue} - {self.desc}'
        return f'{self.value}'


class TypeConstraint(ConfigValue):
    def __init__(self, type, value=None, allow_none=False, desc=None):
        super().__init__(desc=desc)
        self.type = type
        self.value = value
        self.allow_none = allow_none

    def realize(self):
        if self.value is None:
            if self.allow_none:
                return None
            raise ValueError('Got `None` when not allow_none')
        if self.type is bool and isinstance(self.value, str):
            return self.value.lower() in ('true', 't', 'yes', 'y', '1')
        return self.type(self.value)

    def __str__(self):
        st = f'{self.value} (type: {self.type})'
        if self.desc is not None:
            st = f'{st} - {self.desc}'
        return st


class ChoiceConstraint(ConfigValue):
    def __init__(self, choices, default=NotImplemented, strict=False, desc=None):
        super().__init__(desc=desc)
        self.choices = choices
        self.strict = strict
        if default is NotImplemented:
            self.value = choices[0]
        else:
            self.value = default

    def realize(self):
        if self.value not in self.choices:
            err = f'value {self.value} not found in {self.choices}'
            if self.strict:
                raise ValueError(err)
            else:
                logger().warn(err)
        return self.value

    def __str__(self):
        st = f'{self.value} (choices: {self.choices})'
        if self.desc is not None:
            st = f'{st} - {self.desc}'
        return st



class RankerConstraint(ConfigValue):
    def __init__(self, default='bm25', desc=None, strict=False):
        super().__init__(desc=desc)
        self.value = default
        self.strict = strict

    def _raise_or_warn(self, err):
        if self.strict:
            raise ValueError(err)
        else:
            logger().warn(err)

    def realize(self):
        parts = self.value.split('_')
        model, m_args = parts[0], parts[1:]
        if model == 'bm25':
            for arg in m_args:
                parts = arg.split('-')
                if len(parts) == 1:
                    if parts[0] not in ('rm3',):
                        self._raise_or_warn(f'unknown bm25 arg {arg}')
                elif len(parts) != 2:
                    if parts[0] not in ('k1', 'b', 'rm3.fbTerms', 'rm3.fbDocs'):
                        self._raise_or_warn(f'unknown bm25 arg {arg}')
                else:
                    self._raise_or_warn(f'unknown bm25 arg {arg}')
        elif model == 'ql':
            for arg in m_args:
                parts = arg.split('-')
                if len(parts) != 2:
                    if parts[0] not in ('mu',):
                        self._raise_or_warn(f'unknown ql arg {arg}')
                else:
                    self._raise_or_warn(f'unknown ql arg {arg}')
        elif model == 'sdm':
            for arg in m_args:
                parts = arg.split('-')
                if len(parts) != 2:
                    if parts[0] not in ('mu', 'sdm.tw', 'sdm.ow', 'sdm.uw'):
                        self._raise_or_warn(f'unknown sdm arg {arg}')
                else:
                    self._raise_or_warn(f'unknown sdm arg {arg}')
        elif model == 'vsm':
            if m_args:
                self._raise_or_warn(f'unknown vsm args {m_args}')
        else:
            self._raise_or_warn(f'unknown model {model}')
        return self.value


Choices = ChoiceConstraint
Typed = TypeConstraint
Ranker = RankerConstraint
