import torch


def apply_spec_batch(batch, input_spec, device=None):
    result = {}
    for k in batch:
        result[k] = []
        seq_type, len_mode, maxlen, minlen = None, None, None, None
        if k.startswith('query_'):
            maxlen = input_spec['qlen']
            minlen = input_spec.get('qlen_min')
            len_mode = input_spec['qlen_mode']
        if k.startswith('doc_'):
            maxlen = input_spec['dlen']
            minlen = input_spec.get('dlen_min')
            len_mode = input_spec['dlen_mode']

        if k in ('runscore', 'relscore'):
            seq_type = 'float'
        elif k.endswith('_len'):
            seq_type = 'int'
        elif k.endswith('_tok'):
            seq_type = 'list[int]'
        elif k.endswith('_idf') or k.endswith('_score'):
            seq_type = 'list[float]'
        elif k.endswith('_id') or k.endswith('_rawtext'):
            seq_type = 'str'
        elif k.endswith('_text'):
            seq_type = 'list[str]'

        if seq_type is None:
            raise ValueError(f'unsupported input type {k}')

        for data in batch[k]:
            # apply maximum length (if needed)
            if k.endswith('_len'):
                data = min(data, maxlen)
            result[k].append(data)

        if seq_type in ('str', 'list[str]'):
            # strings are converted to tensors
            continue

        if seq_type in ('list[int]', 'list[float]'):
            if len_mode == 'strict':
                result[k] = [clip_crop(r, maxlen) for r in result[k]]
            elif len_mode == 'max':
                max_seq_len = min(maxlen, max(len(r) for r in result[k]))
                result[k] = [clip_crop(r, max_seq_len) for r in result[k]]
            elif len_mode == 'none':
                pass
            else:
                raise ValueError(f'unkonwn len_mode {len_mode}')
            if minlen is not None:
                result[k] = [pad_min_len(r, minlen) for r in result[k]]

        result[k] = torch.tensor(result[k])
        if seq_type == 'list[float]':
            result[k] = result[k].float()
        elif seq_type == 'list[int]':
            result[k] = result[k].long()
        if device:
            result[k] = result[k].to(device)
    return result



def clip_crop(seq, maxlen, pad_val=-1):
    seq = seq[:maxlen] # clip
    seq += [pad_val] * (maxlen - len(seq)) # pad
    return seq


def pad_min_len(seq, min_len, pad_val=-1):
    if len(seq) < min_len:
        seq += [pad_val] * (min_len - len(seq)) # pad
    return seq
