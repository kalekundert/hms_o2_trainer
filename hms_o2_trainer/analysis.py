"""\
Compare metrics between different training runs.

Usage:
    hot_plot <logs>... [-m <metrics>] [-p <hparams>] [-k <regex>]
        [-o <path>] [-fcst] [-ASTVL]

Arguments:
    <logs>
        Paths to the log files containing the data to plot.  Paths to 
        directories can also be specified, in which case the directories will 
        be searched recursively for log files.

Options:
    -m --metrics <strs>
        A comma separated list of the metrics to plot.  Glob-style patterns are 
        supported.  By default, the following metrics will be displayed if 
        present: loss, RMSE or MAE, pearson R, accuracy

    -p --hparams <csv>      [default: hparams.csv]
        A path to a CSV file describing the hyperparameters for each training 
        run.  The first row gives the name of each hyperparameter, and the 
        following rows give the corresponding values for each run. The table 
        must contain one value called "name", which gives the name of the 
        directory containing the logs for the run in question.

    -k --select <sql>
        Only show models that are selected by the given SQL expression.  The 
        expression will be used in the following SQL statement:
            
            SELECT * from models WHERE ...

        The `...` is what will be replaced by the given expression.  The 
        `models` tables contains one row for each model and, if `--hparams` is 
        given, one column for each hyperparameter.

    -o --output <path>
        Write the resulting plot to the given path.  If not specified, the plot 
        will be displayed in a GUI instead.

    -f --force-reload
        Ignore any caches and parse the training data from scratch.

    -c --concat-hparams
        Instead of making a separate row of plots for each hyperparameter, plot 
        everything in a single row.  For smaller experiments, this might be 
        easier to understand than the default.

    -s --steps
        Plot the number of steps on the x-axis, instead of the epoch.

    -t --elapsed-time
        Plot elapsed time on the x-axis, instead of the epoch.

    -A --hide-raw
        Don't plot raw data points; only plot smoothed curves.

    -S --hide-smooth
        Don't plot smoothed curves; only plot raw data points.

    -T --hide-train
        Only plot the validation metrics, not the epoch-level training metrics.
        Note that this option is ignored if `--metrics` is specified.

    -V --hide-val
        Only plot the epoch-level training metrics, not the validation metrics.
        Note that this option is ignored if `--metrics` is specified.

    -L --hide-loss
        Don't plot the loss function.  This option is ignored if `--metrics` is 
        specified.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import fnmatch

from pathlib import Path
from itertools import product
from more_itertools import one, unique_everseen as unique
from operator import itemgetter

def main():
    try:
        import docopt

        args = docopt.docopt(__doc__)
        df = load_tensorboard_logs(
                log_paths=map(Path, args['<logs>']),
                refresh=args['--force-reload'],
        )

        if args['--metrics']:
            metrics = pick_metrics(df, args['--metrics'])
        else:
            metrics = pick_default_metrics(
                    df,
                    include_train=not args['--hide-train'],
                    include_val=not args['--hide-val'],
                    include_loss=not args['--hide-loss'],
            )

        if Path(args['--hparams']).exists():
            hparam_df = load_hparams(args['--hparams'])
            df = join_hparams(df, hparam_df)
            hparams = [x for x in hparam_df.columns if x != 'name']
        else:
            hparams = ['name']

        if k := args['--select']:
            df = df.sql(f'SELECT * from self WHERE {k}')

        if args['--steps']:
            x = 'step'
        elif args['--elapsed-time']:
            x = 'elapsed_time'
        else:
            x = 'epoch'

        plot_training_metrics(
                df, metrics, hparams,
                x=x,
                show_raw=not args['--hide-raw'],
                show_smooth=not args['--hide-smooth'],
                concat_hparams=args['--concat-hparams'],
        )

        if out_path := args['--output']:
            plt.savefig(out_path)
        else:
            plt.show()

    except KeyboardInterrupt:
        pass

def load_tensorboard_logs(log_paths, cache=True, refresh=False):
    return pl.concat([
        load_tensorboard_log(log_path, cache=cache, refresh=refresh)
        for log_path in log_paths
    ])

def load_tensorboard_log(log_path, cache=True, refresh=False):
    if not refresh:
        if log_path.suffix == '.feather':
            return pl.read_ipc(log_path, memory_map=False)
        if (p := log_path / 'cache.feather').exists():
            return pl.read_ipc(p, memory_map=False)

    from tbparse import SummaryReader

    reader = SummaryReader(
            log_path,
            extra_columns={'dir_name', 'wall_time'},
    )

    df = (
            pl.from_pandas(reader.scalars)
            .rename({
                'dir_name': 'name',
                'tag': 'metric',
            })
    )
    epochs = (
            df
            .lazy()
            .filter(metric='epoch')
            .select(
                'name', 'step',
                epoch=pl.col('value').cast(int),
            )
            .group_by('name', 'step')
            .min()
    )
    df = (
            df
            .lazy()
            .filter(
                pl.col('step') > 0,
                pl.col('metric') != 'epoch',
                pl.col('metric').str.ends_with('_step') == False,
            )
            .filter(
                pl.col('step').len().over('name', 'metric') > 1,
            )
            .join(
                epochs,
                on=['name', 'step'],
                how='left',
            )
            .with_columns(
                pl.concat_str(
                    pl.lit(str(log_path)),
                    pl.col('name'),
                    separator='/',
                ).alias('name'),
            )
            .with_columns(
                pl.col('wall_time')
                  .map_elements(
                      lambda t: pl.Series(infer_elapsed_time(t.to_numpy())),
                      return_dtype=pl.List(float),
                  )
                  .over(['name', 'metric'])
                  .alias('elapsed_time')
            )
            .sort('name', 'metric', 'step')
            .select(
                'name',
                'metric',
                'epoch',
                'step',
                'elapsed_time',
                'value',
            )
            .collect()
    )

    if cache:
        if log_path.is_dir():
            cache_path = log_path / 'cache.feather'
        else:
            cache_path = log_path.with_suffix('.feather')

        df.write_ipc(cache_path)

    return df

def plot_training_metrics(
        df, metrics, hparams, *,
        x='step',
        show_raw=True,
        show_smooth=True,
        concat_hparams=False,
):
    if not show_raw and not show_smooth:
        raise ValueError("nothing to show; both raw and smooth plots disabled")

    if concat_hparams:
        df = df.with_columns(
                hparams=pl.concat_list(hparams).list.join('; ')
        )
        hparams = ['hparams']

    ncols = len(metrics) + 1
    nrows = len(hparams)

    fig, axes = plt.subplots(
            figsize=(ncols * 3, nrows * 3),
            ncols=ncols,
            nrows=nrows,
            constrained_layout=True,
            squeeze=False,
            sharex=True,
    )

    x_labels = {
            'epoch': 'epochs',
            'step': 'steps (×1000)',
            'elapsed_time': 'elapsed time (h)',
    }
    x_getters = {
            'epoch': lambda x: x + 1,
            'step': lambda x: (x + 1) / 1000,
            'elapsed_time': lambda x: x / 3600,
    }

    df_by_metric = df.partition_by('metric', as_dict=True)
    hparam_colors = _pick_hparam_colors(df, hparams)

    for i, metric in enumerate(metrics):
        t_raw = []
        y_raw = []
        color_raw = []

        if metric not in df_by_metric:
            did_you_mean = '\n'.join('- ' + k for k in sorted(df_by_metric))
            raise ValueError(f"can't find metric: {metric}\n\nDid you mean:\n{did_you_mean}")

        # This loop plots the "primary" curves, i.e. the ones that are labeled 
        # and solidly colored.  This could be either the raw data or the 
        # smoothed data, depending on what the user asked for.

        for df_i in df_by_metric[metric].partition_by('name'):
            t = x_getters[x](df_i[x].to_numpy())
            y = df_i['value'].to_numpy()

            t_raw.append(t)
            y_raw.append(y)
            color_raw.append([])

            if show_smooth:
                t, y = _apply_smoothing(t, y)

            for j, hparam in enumerate(hparams):
                ax = axes[j,i]

                hparam_value = one(df_i[hparam].unique())
                if hparam_value is not None:
                    color = hparam_colors[hparam, hparam_value]
                    color_raw[-1].append((j, color))

                    ax.plot(
                            t, y,
                            label=f'{hparam_value}',
                            color=color,
                    )

                if j == 0:
                    ax.set_title(metric)
                if j == len(hparams) - 1:
                    ax.set_xlabel(x_labels[x])

        # This loops plots the "secondary" curves, i.e. the ones that are 
        # unlabeled, mostly transparent, and have no effect on the y-limits.  
        # These curves are only needed if both raw data and smoothed curves are 
        # requested.

        if show_raw and show_smooth:
            ylim = axes[0,i].get_ylim()

            for t, y, colors in zip(t_raw, y_raw, color_raw):
                for j, color in colors:
                    axes[j,i].plot(t, y, color=color, alpha=0.2)
                    axes[j,i].set_ylim(*ylim)

    t = x_getters[x](df[x].to_numpy())
    axes[0,0].set_xlim(t.min(), t.max())

    for i, ax_row in enumerate(axes):
        labels = ax_row[0].get_legend_handles_labels()

        if any(labels):
            h, l = zip(
                    *unique(
                        zip(*labels),
                        key=itemgetter(1),
                    )
            )

            ax_row[-1].legend(
                    h, l,
                    borderaxespad=0,
                    title=hparams[i],
                    alignment='left',
                    frameon=False,
                    loc='center left',
            )

        ax_row[-1].axis('off')

def infer_elapsed_time(t):
    from sklearn.ensemble import IsolationForest

    # - The wall time data includes both the time it takes to process an 
    #   example and the time spent waiting between job requeues.  We only care 
    #   about the former.  So the purpose of this function is to detect the 
    #   latter, and to replace those data points with the average of the 
    #   former.  Note that this only works if all the jobs run on the same GPU.
    #
    # - I compared a number of different outlier detection algorithms to 
    #   distinguish these two time steps.  I found that isolation forests 
    #   performed the best; classifying the data points exactly as I would on 
    #   the datasets I was experimenting with.  The local outlier factor 
    #   algorithm also performed well, but classified some true time steps as 
    #   outliers.

    dt = np.diff(t)

    outlier_detector = IsolationForest(random_state=0)
    labels = outlier_detector.fit_predict(dt.reshape(-1, 1))

    inlier_mask = (labels == 1)
    outlier_mask = (labels == -1)

    dt_mean = np.mean(dt[inlier_mask])
    dt[outlier_mask] = dt_mean

    return _cumsum0(dt)

def pick_metrics(df, spec):
    known_metrics = sorted(set(df['metric']))

    metrics = []
    for pattern in spec.split(','):
        metrics += fnmatch.filter(known_metrics, pattern)

    return list(unique(metrics))

def pick_default_metrics(
        df,
        *,
        include_train=False,
        include_val=True,
        include_loss=True,
):
    known_metrics = set(df['metric'])

    stages = []

    if include_val:
        stages += ['val/{}']

    if include_train:
        stages += ['train/{}_epoch']

    metrics = []

    if include_loss:
        metrics.append('loss')

    if 'val/rmse' in known_metrics:
        metrics.append('rmse')
    elif 'val/mae' in known_metrics:
        metrics.append('mae')

    if 'val/pearson_r' in known_metrics:
        metrics.append('pearson_r')
    if 'val/accuracy' in known_metrics:
        metrics.append('accuracy')

    return [
            stage.format(metric)
            for stage, metric in product(stages, metrics)
    ]

def load_hparams(path):
    import csv

    # Polars doesn't have the option to not infer data types, if the number of 
    # columns isn't known.  Since we want to ensure that the model names are 
    # strings even if they're composed entirely of digits, we need to parse the 
    # file ourselves.

    with open(path) as f:
        rows = list(csv.reader(f))

    head = rows[0]
    body = rows[1:]

    if 'name' not in head:
        raise ValueError(f"hyperparameter table must contain 'name' column: {path}")

    return pl.DataFrame(body, head, orient='row')

def join_hparams(df, hparams):
    return (
            df
            .join(
                hparams.with_row_index(),
                left_on=pl.col('name').str.split('/').list.last(),
                right_on='name',
                how='left',
            )
            .sort('index')
            .drop('index')
    )

def _pick_hparam_colors(df, hparams):
    hparam_colors = {}

    for hparam in hparams:
        hparam_values = df[hparam].unique(maintain_order=True).drop_nulls()
        for i, value in enumerate(hparam_values):
            hparam_colors[hparam, value] = f'C{i}'

    return hparam_colors

def _apply_smoothing(x, y):
    from sklearn.neighbors import LocalOutlierFactor
    from scipy.signal import savgol_filter

    window_length = max(len(x) // 10, 15)

    lof = LocalOutlierFactor(2)
    labels = lof.fit_predict(y.reshape(-1, 1))
    inlier_mask = (labels == 1)

    x_inlier = x[inlier_mask]
    y_inlier = y[inlier_mask]

    if len(x_inlier) < window_length:
        return x, y
    if len(x_inlier) < 2 * window_length:
        x_inlier = x
        y_inlier = y

    y_smooth = savgol_filter(
            y_inlier,
            window_length=window_length,
            polyorder=2,
    )

    return x_inlier, y_smooth

def _cumsum0(x):
    # https://stackoverflow.com/questions/27258693/how-to-make-numpy-cumsum-start-after-the-first-value
    y = np.empty(len(x) + 1, dtype=x.dtype)
    y[0] = 0
    np.cumsum(x, out=y[1:])
    return y


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

