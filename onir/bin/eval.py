import os
os.environ['ONIR_IGNORE_ARGV'] = 'true'
import json
import argparse
from onir import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('qrels')
    parser.add_argument('run')
    parser.add_argument('--each_topic', '-q', action='store_true')
    parser.add_argument('--nosummary', '-n', action='store_true')
    parser.add_argument('--json_output', '-j', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('metrics', nargs='+')
    args = parser.parse_args()
    result = metrics.calc(args.qrels, args.run, args.metrics, verbose=args.verbose)
    if args.json_output:
        print(json.dumps(result))
    elif result:
        if args.each_topic:
            for qid in result[args.metrics[0]]:
                for metric in args.metrics:
                    print(f'{metric}\t{qid}\t{result[metric][qid]:.4f}')
        if not args.nosummary:
            for metric, mean in metrics.mean(result).items():
                print(f'{metric}\tall\t{mean:.4f}')


if __name__ == '__main__':
    main()
