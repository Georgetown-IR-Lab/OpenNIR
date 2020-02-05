import torch
from tqdm import tqdm
import onir
from onir import injector, util, spec


def main():
    onir.rankers.base.global_memcache_enable = False
    context = injector.load({
        'vocab': onir.vocab,
        'dataset': onir.datasets.registry.copy(default='random'),
        'ranker': onir.rankers,
    })

    logger = onir.log.easy()
    logger.debug(f'vocab: {context["vocab"].config}')
    logger.debug(f'dataset: {context["dataset"].config}')
    logger.debug(f'ranker: {context["ranker"].config}')

    ds = context['dataset']
    ranker = context['ranker']

    batch_size = int(onir.config.args().get('batch_size', '512'))
    repeat = int(onir.config.args().get('repeat', '5'))
    gpu = onir.config.args().get('gpu', 'true').lower() == 'true'
    device = torch.device('cuda') if gpu else torch.device('cpu')

    logger.debug(f'batch_size: {batch_size}')
    logger.debug(f'repeat: {repeat}')
    logger.debug(f'device: {device}')

    input_spec = ranker.input_spec()
    batches = [{field: [] for field in input_spec['fields']}]
    some_field = next(iter(input_spec['fields']))
    record_count = 0
    for record in tqdm(ds.iter_records(input_spec['fields']), desc='loading data', leave=False):
        if len(batches[-1][some_field]) == batch_size:
            batches[-1] = spec.apply_spec_batch(batches[-1], input_spec)
            batches.append({field: [] for field in input_spec['fields']})
        for k, v in record.items():
            batches[-1][k].append(v)
        record_count += 1
    batches[-1] = spec.apply_spec_batch(batches[-1], input_spec)

    if gpu:
        ranker = ranker.to(device)
    ranker.eval()
    times = []
    times_per_1k = []
    for i in range(repeat):
        with torch.no_grad():
            timer = util.HierTimer(gpu_sync=gpu)
            for batch in tqdm(batches, leave=False, ncols=80, desc=str(i)):
                with timer.time('model'):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ranker(**batch).cpu()
            time = timer.durations['model'] * 1000
            time_per_1k = time / record_count * 1000
            logger.debug(f'{i} time={time:.2f}ms record_count={record_count} time_per_1k={time_per_1k:.2f}ms')
            times.append(time)
            times_per_1k.append(time_per_1k)
    avg = lambda vals: sum(vals) / len(vals)
    med = lambda vals: list(sorted(vals))[len(vals)//2]
    logger.info(f'max time={max(times):.2f}ms record_count={record_count} time_per_1k={max(times_per_1k):.2f}ms')
    logger.info(f'avg time={avg(times):.2f}ms record_count={record_count} time_per_1k={avg(times_per_1k):.2f}ms')
    logger.info(f'med time={med(times):.2f}ms record_count={record_count} time_per_1k={med(times_per_1k):.2f}ms')
    logger.info(f'min time={min(times):.2f}ms record_count={record_count} time_per_1k={min(times_per_1k):.2f}ms')


if __name__ == '__main__':
    main()
