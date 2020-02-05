import os
import json
import socket
import itertools
from tqdm import tqdm
from decimal import Decimal
import matplotlib
matplotlib.use('Agg') # no display
import matplotlib.pyplot as plt
import seaborn as sns
import ternary
import pandas as pd
import onir
from onir import trainers, util, config, pipelines


@pipelines.register('grid_search')
class GridSearchPipeline(pipelines.BasePipeline):
    name = None

    @staticmethod
    def default_config():
        return {
            'rankfn': 'bm25',
            'params': 'k1-0.1:4.0:0.1_b-0.00:1.00:0.05',
            'val_metric': 'map',
            'skip_ds_init': False,
            'test': False,
            'cond': '',
            'heatmap_2d': '',
            'heatmap_ternary': ''
        }

    def __init__(self, config, trainer, valid_pred, test_pred, logger):
        super().__init__(config, logger)
        self.trainer = trainer
        self.valid_pred = valid_pred
        self.test_pred = test_pred
        assert not valid_pred.config['preload']

    def run(self):
        if not self.config['skip_ds_init']:
            self.valid_pred.dataset.init(force=False)
            if self.config['test']:
                self.test_pred.dataset.init(force=False)

        train_ctxt = next(self.trainer.iter_train())
        top_params, top_value, top_valid_ctxt = None, None, None

        params = [p.split('-', 1) for p in self.config['params'].split('_')]
        params = {k: list(_dec_range(r)) for k, r in params}

        params = [[(k, v) for v in vals] for k, vals in params.items()]

        combinations = list(itertools.product(*params))

        if self.config['cond']:
            combinations = [comb for comb in combinations if self._condition(comb)]

        self.logger.debug(f'{len(combinations)} parameter configurations')

        default_rankfn = self.config['rankfn']

        df = []

        with tqdm(combinations, ncols=80) as pbar:
            for combo in pbar:
                df.append({k: v for k, v in combo})
                combo = '_'.join((default_rankfn,) + tuple(f'{k}-{v}' for k, v in combo))
                pbar.set_description(combo)

                self.valid_pred.dataset.config['rankfn'] = combo

                # gahhhhh, this screwed me up forever
                if hasattr(self.valid_pred.dataset, '_record_cache'):
                    self.valid_pred.dataset._record_cache.clear()

                valid_ctxt = self.valid_pred.run(train_ctxt)
                message = self._build_valid_msg(valid_ctxt, combo)

                df[-1]['metric'] = valid_ctxt['metrics'][self.config['val_metric']]

                if top_value is None or valid_ctxt['metrics'][self.config['val_metric']] > top_value:
                    message += ' <---'
                    top_params = combo
                    top_value = valid_ctxt['metrics'][self.config['val_metric']]
                    top_valid_ctxt = valid_ctxt
                if valid_ctxt['cached']:
                    self.logger.debug('[valid] [cached] ' + message)
                else:
                    self.logger.info('[valid] ' + message)

        self.logger.info('top validation {} {}={}'.format(top_params, self.config['val_metric'], top_value))

        if self.config['test']:
            self.test_pred.dataset.config['rankfn'] = top_params

            with self.logger.duration('testing'):
                test_ctxt = self.test_pred.run(train_ctxt)

        self.logger.info('valid run at {}'.format(valid_ctxt['run_path']))
        if self.config['test']:
            self.logger.info('test run at {}'.format(test_ctxt['run_path']))
        self.logger.info('valid ' + self._build_valid_msg(top_valid_ctxt, top_params))
        if self.config['test']:
            self.logger.info('test  ' + self._build_valid_msg(test_ctxt, top_params))

        if self.config['heatmap_2d']:
            self.heatmap_2d(df, top_valid_ctxt)
        if self.config['heatmap_ternary']:
            self.heatmap_ternary(params, df, top_valid_ctxt)

    def _build_valid_msg(self, ctxt, params):
        message = [params]
        for metric, value in sorted(ctxt['metrics'].items()):
            message.append('{}={:.4f}'.format(metric, value))
            if metric == self.config['val_metric']:
                message[-1] = '[' + message[-1] + ']'
        return ' '.join(message)

    def _condition(self, comb):
        cond = self.config['cond']
        for k, v in comb:
            cond = cond.replace(k, f'{v:0.8f}')
        # using Decimal representations is very important here to avoid floating point errors
        return util.matheval(cond, Decimal)

    def heatmap_2d(self, df, top_valid_ctxt):
        axes = self.config['heatmap_2d'].split(',')
        assert len(axes) == 2
        heatmap_path = os.path.join(top_valid_ctxt['base_path'], 'gs-heatmap_2d-{val_metric}.pdf'.format(**self.config))
        sns.heatmap(pd.DataFrame(df).pivot(axes[0], axes[1], "metric"))

        # The following lines do not work, not sure why... Places point incorrectly on heatmap
        # max_point = max(df, key=lambda x: x['metric'])
        # max_point = (max_point[axes[0]], max_point[axes[1]])
        # plt.scatter([max_point[0]], [max_point[1]], marker='.', color='black', zorder=100, label='${}={}$\n${}={}$'.format(axes[0], max_point[0], axes[1], max_point[1]))
        # plt.legend()

        plt.title(self.config['val_metric'])
        plt.tight_layout()
        plt.savefig(heatmap_path)
        self.logger.debug('heatmap: ' + heatmap_path)

    def heatmap_ternary(self, params, df, top_valid_ctxt):
        axes = self.config['heatmap_ternary'].split(',')
        assert len(axes) == 3
        heatmap_path = os.path.join(top_valid_ctxt['base_path'], 'gs-heatmap_ternary-{val_metric}.pdf'.format(**self.config))
        step = (params[0][1][1] - params[0][0][1])
        scale = int(1.0 / float(step))
        d = {(float(v[axes[0]]), float(v[axes[1]]), float(v[axes[2]])): v['metric'] for v in df}
        def fn(p):
            return d.get(tuple(p))
        _, tax = ternary.figure(scale=scale)
        tax.heatmapf(fn, boundary=True, style="triangular", scientific=False)
        max_point = max(d.items(), key=lambda x: (x[1], x[0]))[0]
        max_point_x = [x*scale for x in max_point]
        tax.scatter([max_point_x], marker='.', color='black', zorder=100, label='${}={}$\n${}={}$\n${}={}$'.format(axes[0], max_point[0], axes[1], max_point[1], axes[2], max_point[2]))
        tax.legend()
        tax.bottom_axis_label(f"$\\rightarrow$ {axes[0]} $\\rightarrow$", offset=-0.1)
        tax.right_axis_label(f"$\\leftarrow$ {axes[1]} $\\leftarrow$", offset=0.05)
        tax.left_axis_label(f"$\\leftarrow$ {axes[2]} $\\leftarrow$", offset=0.05)
        tax._redraw_labels() # hack from <https://github.com/marcharper/python-ternary/issues/36>

        # tax.ticks(axis='lbr', multiple=5, linewidth=1, offset=0.025)
        tax.set_title("{val_metric}".format(**self.config))
        tax.get_axes().axis('off')
        tax.ticks(ticks=[0, 1], locations=[0, scale], axis='lbr', linewidth=0.5, offset=0.015)
        tax.boundary(linewidth=0.5)
        tax.gridlines(multiple=scale/5, color="black")
        plt.savefig(heatmap_path)
        self.logger.debug('heatmap: ' + heatmap_path)



def _dec_range(range_str):
    if ':' in range_str:
        start, stop, skip = range_str.split(':')
        start, stop, skip = Decimal(start), Decimal(stop), Decimal(skip)
        value = start
        while value <= stop:
            yield value
            value += skip
    else:
        for value in range_str.split('-'):
            yield Decimal(value)
