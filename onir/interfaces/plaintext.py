from onir import util


def read_sv(file, sep):
    if hasattr(file, 'read'):
        for line in file:
            # TODO: handle values with sep, blank values, etc?
            yield line.rstrip('\r\n').split(sep)
    else:
        with open(file, 'rt') as f:
            yield from read_sv(f, sep)


def write_sv(file, data, sep):
    if hasattr(file, 'write'):
        for values in data:
            # TODO: handle values with sep, blank values, etc?
            file.write(sep.join(map(str, values)) + '\n')
        file.flush()
    else:
        with open(file, 'wt') as f:
            write_sv(f, data, sep)


def read_tsv(file):
    return read_sv(file, sep='\t')


def write_tsv(file, data):
    write_sv(file, data, sep='\t')
