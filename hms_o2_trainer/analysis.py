"""\
Compare metrics between different training runs.

Usage:
    hot_plot [<logs>...] [-c <config>] [-m <metrics>] [-n <names>]
        [-k <regex>] [-o <path>] [-f1st] [-ASTVL]

Arguments:
    <logs>
        Paths to the log files containing the data to plot.  Paths to 
        directories can also be specified, in which case the directories will 
        be searched recursively for log files.

Options:
    -c --config <nt>        [default: hot_plot.nt]
        A path to a NestedText [1] file that containing various plotting 
        parameters.  See the `Configuration` section below for more information 
        on the format of this file.
        
    -m --metrics <strs>
        A comma separated list of the metrics to plot.  Glob-style patterns are 
        supported.  By default, the following metrics will be displayed if 
        present: loss, RMSE or MAE, pearson R, accuracy

    -n --models <names>
        A comma-separated list of model names to plot.

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

    -1 --squash-hparams
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

Configuration:

    Most of the options that can be specified on the command line (and some 
    that can't) can be specified via a configuration file.  The file is 
    typically named `hot_plot.nt`, but the `--config` option can be used to 
    specify a different file.  The file should be in the NestedText [1] format, 
    and should have the following structure.  All values are optional:

        logs: See `<logs>`.

        hparams: A CSV-formatted string describing the hyperparameters for each 
            training run.  The first row gives the name of each hyperparameter, 
            and the following rows give the corresponding values for each run. 
            The table must contain one value called "name", which gives the 
            name of the directory containing the logs for the run in question.

        models:
            names: See `--models`, except that the names should be specified as 
                a list rather than a comma-separated string.

            sql: See `--select`.

        metrics:
          <metric name>:
            title: How to label this metric on the plot.
            ylim: A space-separated pair of two values: the low and high limits 
                of the y-axis for this metric.  The full python floating point 
                syntax (including most binary operators) can be used for each 
                value.

        options:
            x unit: Which unit to display on the x-axis.  Must be one of 
                'step', 'elapsed_time', or 'epoch'.  See `--steps` and 
                `--elapsed-time`.

            squash hparams: See `--squash-hparams`.  Must be 'yes' or 'no'.
            hide raw: See `--hide-raw`.  Must be 'yes' or 'no'.
            hide smooth: See `--hide-smooth`.  Must be 'yes' or 'no'.
            hide train: See `--hide-train`.  Must be 'yes' or 'no'.
            hide val: See `--hide-val`.  Must be 'yes' or 'no'.
            hide loss: See `--hide-loss`.  Must be 'yes' or 'no'.

References:
[1] https://nestedtext.org
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import byoc

from byoc import NtConfig, DocoptConfig, Key, Func
from voluptuous import Schema, Any
from itertools import product
from more_itertools import one, unique_everseen as unique
from contextlib import nullcontext
from operator import itemgetter
from pathlib import Path
from io import StringIO

def hparams_from_csv(csv: str):
    from io import StringIO
    csv_io = StringIO(csv)
    return load_hparams(csv_io)

def hparams_from_default_file():
    # This function is meant to maintain backwards compatibility with my old 
    # API, which allowed hyperparameters to be specified via a file called 
    # `hparams.csv`.  New code should use `hot_plot.nt`.

    path = Path('hparams.csv')
    
    if not path.exists():
        return None

    return load_hparams(path)

def get_sql_name_expr(names):
    if not names:
        raise ValueError("found empty list of names to select")
    return "name IN ('" + "', '".join(names) + "')"

def join_sql_exprs(exprs):
    return ' AND '.join(exprs)

def parse_escapes(x):
    return x.encode('raw_unicode_escape').decode('unicode_escape')

def parse_paths(paths):
    return [Path(x) for x in paths]

def parse_ylim(ylim: str):
    fields = ylim.split()

    if len(fields) != 2:
        raise ValueError(f"expected 2 ylim values (low and high), got: {ylim!r}")

    return {
            k: byoc.float_eval(v)
            for k, v in zip(['bottom', 'top'], fields, strict=True)
            if v != '-'
    }

def parse_comma_list(x):
    return x.split(',')

def parse_bool(x):
    if x == 'yes':
        return True
    if x == 'no':
        return False

    raise ValueError(f"expected 'yes' or 'no', got: {x!r}")

class App:
    __config__ = [
            NtConfig.setup(
                path_getter=lambda app: app.config_path,
                schema=Schema({
                    'logs': [str],
                    'hparams': str,
                    'models': {
                        'names': [str],
                        'sql': str,
                    },
                    'metrics': {
                        str: {
                            'title': parse_escapes,
                            'ylim': parse_ylim,
                        },
                    },
                    'options': {
                        'x unit': Any('step', 'elapsed_time', 'epoch'),
                        'squash hparams': parse_bool,
                        'hide raw': parse_bool,
                        'hide smooth': parse_bool,
                        'hide train': parse_bool,
                        'hide val': parse_bool,
                        'hide loss': parse_bool,
                    },
                }),
            ),
            DocoptConfig.setup(usage_getter=lambda app: __doc__),
    ]

    config_path = byoc.param(
            Key(DocoptConfig, '--config', cast=Path),
            default=Path('hot_plot.nt'),
    )
    output_path = byoc.param(
            Key(DocoptConfig, '--output', cast=Path),
            default=None,
    )
    logs = byoc.param(
            Key(DocoptConfig, '<logs>'),
            Key(NtConfig, 'logs'),
            cast=parse_paths,
            default_factory=lambda: [Path('.')],
    )
    force_reload = byoc.param(
            Key(DocoptConfig, '--force-reload'),
            default=False,
    )
    models_sql = byoc.param(
            Key(DocoptConfig, '--models', cast=[parse_comma_list, get_sql_name_expr]),
            Key(DocoptConfig, '--select'),
            Key(NtConfig, ['models', 'names'], cast=get_sql_name_expr),
            Key(NtConfig, ['models', 'sql']),
            pick=join_sql_exprs,
    )
    metric_globs = byoc.param(
            Key(DocoptConfig, '--metrics', cast=parse_comma_list),
            Key(NtConfig, 'metrics', cast=lambda x: list(x.keys())),
            default=None,
    )
    metric_styles = byoc.param(
            Key(NtConfig, 'metrics'),
            default_factory=dict,
    )
    hparams = byoc.param(
            Key(NtConfig, 'hparams', cast=hparams_from_csv),
            Func(hparams_from_default_file),
            default=None,
    )
    squash_hparams = byoc.param(
            Key(DocoptConfig, '--squash-hparams'),
            Key(NtConfig, ['options', 'squash hparams']),
            default=False,
    )
    hide_raw = byoc.param(
            Key(DocoptConfig, '--hide-raw'),
            Key(NtConfig, ['options', 'hide raw']),
            default=False,
    )
    hide_smooth = byoc.param(
            Key(DocoptConfig, '--hide-smooth'),
            Key(NtConfig, ['options', 'hide smooth']),
            default=False,
    )
    hide_train = byoc.param(
            Key(DocoptConfig, '--hide-train'),
            Key(NtConfig, ['options', 'hide train']),
            default=False,
    )
    hide_val = byoc.param(
            Key(DocoptConfig, '--hide-val'),
            Key(NtConfig, ['options', 'hide val']),
            default=False,
    )
    hide_loss = byoc.param(
            Key(DocoptConfig, '--hide-loss'),
            Key(NtConfig, ['options', 'hide loss']),
            default=False,
    )
    x_unit = byoc.param(
            Key(DocoptConfig, '--steps', cast=lambda x: 'step'),
            Key(DocoptConfig, '--elapsed-time', cast=lambda x: 'elapsed_time'),
            Key(NtConfig, ['options', 'x unit']),
            default='epoch',
    )

def main():
    try:
        app = App()
        byoc.load(app)

        df = load_tensorboard_logs(
                log_paths=app.logs,
                refresh=app.force_reload,
        )

        if app.metric_globs:
            metrics = pick_custom_metrics(df, app.metric_globs)
        else:
            metrics = pick_default_metrics(
                    df,
                    include_train=not app.hide_train,
                    include_val=not app.hide_val,
                    include_loss=not app.hide_loss,
            )

        if app.hparams is not None:
            df = join_hparams(df, app.hparams)
            hparams = [x for x in app.hparams.columns if x != 'name']
        else:
            app.hparams = df['name'].unique()
            hparams = ['name']

        if app.models_sql:
            df = df.sql(f'SELECT * from self WHERE {app.models_sql}')

        plot_training_metrics(
                df, metrics, hparams,
                x=app.x_unit,
                metric_styles=app.metric_styles,
                show_raw=not app.hide_raw,
                show_smooth=not app.hide_smooth,
                squash_hparams=app.squash_hparams,
        )

        if app.output_path:
            plt.savefig(app.output_path)
        else:
            launch_pyplot_gui(app.hparams)

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
                'dir_name': 'root',
                'tag': 'metric',
            })
    )
    epochs = (
            df
            .lazy()
            .filter(metric='epoch')
            .select(
                'root', 'step',
                epoch=pl.col('value').cast(int),
            )
            .group_by('root', 'step')
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
                pl.col('step').len().over('root', 'metric') > 1,
            )
            .join(
                epochs,
                on=['root', 'step'],
                how='left',
            )
            .with_columns(
                root=pl.concat_str(
                    pl.lit(str(log_path)),
                    pl.col('root'),
                    separator='/',
                ),
            )
            .with_columns(
                name=pl.col('root').str.split('/').list.last(),
            )
            .with_columns(
                pl.col('wall_time')
                  .map_elements(
                      lambda t: pl.Series(infer_elapsed_time(t.to_numpy())),
                      return_dtype=pl.List(float),
                  )
                  .over(['root', 'metric'])
                  .alias('elapsed_time')
            )
            .sort('name', 'metric', 'step')
            .select(
                'root',
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
        metric_styles={},
        show_raw=True,
        show_smooth=True,
        squash_hparams=False,
):
    if not show_raw and not show_smooth:
        raise ValueError("nothing to show; both raw and smooth plots disabled")

    if squash_hparams:
        col = '; '.join(hparams)
        df = df.with_columns(
                pl.concat_list(hparams).list.join('; ').alias(col)
        )
        hparams = [col]

    ncols = len(metrics) + 1
    nrows = len(hparams)

    fig, axes = plt.subplots(
            figsize=(ncols * 3, nrows * 3),
            ncols=ncols,
            nrows=nrows,
            constrained_layout=True,
            squeeze=False,
            width_ratios=(ncols - 1) * [15] + [1],
    )

    for ax_row in axes:
        for ax in ax_row[:-1]:
            ax.sharex(axes[0, 0])

    for ax_col in axes.T[:-1]:
        for ax in ax_col[1:]:
            ax.sharey(ax_col[0])

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

    df_by_metric = {
            one(k): v
            for k, v in df.partition_by('metric', as_dict=True).items()
    }
    hparam_colors = _pick_hparam_colors(df, hparams)

    def plot_with_meta(ax, *args, name, **kwargs):
        line = one(ax.plot(*args, **kwargs))
        line.meta = dict(
                name=name,
                foreground=True,
        )
        

    for i, metric in enumerate(metrics):
        t_raw = []
        y_raw = []
        color_raw = []
        name_raw = []

        if metric not in df_by_metric:
            did_you_mean = '\n'.join('- ' + k for k in sorted(df_by_metric))
            raise ValueError(f"can't find metric: {metric}\n\nDid you mean:\n{did_you_mean}")

        df_i = df_by_metric[metric]

        # This loop plots the "primary" curves, i.e. the ones that are labeled 
        # and solidly colored.  This could be either the raw data or the 
        # smoothed data, depending on what the user asked for.

        for (name,), df_ij in df_i.partition_by('name', as_dict=True).items():
            t = x_getters[x](df_ij[x].to_numpy())
            y = df_ij['value'].to_numpy()

            t_raw.append(t)
            y_raw.append(y)
            color_raw.append([])
            name_raw.append(name)

            if show_smooth:
                t, y = _apply_smoothing(t, y)

            for j, hparam in enumerate(hparams):
                ax = axes[j,i]

                hparam_value = one(df_ij[hparam].unique())
                if hparam_value is not None:
                    color = hparam_colors[hparam, hparam_value]
                    color_raw[-1].append((j, color))

                    plot_with_meta(
                            ax, t, y,
                            label=f'{hparam_value}',
                            color=color,
                            name=name,
                    )

                if j == 0:
                    style = metric_styles.get(metric, {})
                    ax.set_title(style.get('title', metric))
                    ax.set_ylim(**style.get('ylim', dict(auto=True)))

                if j == len(hparams) - 1:
                    ax.set_xlabel(x_labels[x])

        # This loops plots the "secondary" curves, i.e. the ones that are 
        # unlabeled, mostly transparent, and have no effect on the y-limits.  
        # These curves are only needed if both raw data and smoothed curves are 
        # requested.

        if show_raw and show_smooth:
            ylim = axes[0,i].get_ylim()

            for t, y, colors, name in zip(t_raw, y_raw, color_raw, name_raw):
                for j, color in colors:
                    plot_with_meta(
                            axes[j,i], t, y,
                            color=color,
                            alpha=0.2,
                            name=name,
                    )
                    axes[j,i].set_ylim(*ylim)

    t = x_getters[x](df[x].to_numpy())
    axes[0,0].set_xlim(t.min(), t.max())

    for i, ax_row in enumerate(axes):
        hparam = hparams[i]
        labels = ax_row[0].get_legend_handles_labels()

        if not any(labels):
            continue

        if hparam in hparam_colors:
            xlim = ax_row[-1].get_xlim()
            fig.colorbar(
                    mappable=hparam_colors[hparam],
                    cax=ax_row[-1],
                    label=hparam,
            )
            ax_row[-1].set_xlim(xlim)
            #ax_row[-1].axis('off')

        else:
            h, l = zip(
                    *unique(
                        zip(*labels),
                        key=itemgetter(1),
                    )
            )
            ax_row[-1].legend(
                    h, l,
                    borderaxespad=0,
                    title=hparam,
                    alignment='left',
                    frameon=False,
                    loc='center left',
            )
            ax_row[-1].axis('off')

def launch_pyplot_gui(hparams_df):
    import pyperclip
    from matplotlib.backend_bases import MouseButton

    def on_click(event):
        if event.button is not MouseButton.LEFT:
            return
        if not event.inaxes:
            return

        # This lock indicates that some other UI action is pending, e.g. a 
        # click-and-drag zoom.
        if event.inaxes.figure.canvas.widgetlock.locked():
            return

        # Find the line closest to where the user clicked:

        click_names = set()

        for line in event.inaxes.get_lines():
            if not hasattr(line, 'meta'):
                continue

            line.set_pickradius(3)
            hit, details = line.contains(event)

            if hit:
                click_names.add(line.meta['name'])

        # Highlight the selected lines:

        if not click_names:
            mode = 'all'
        elif not event.key:
            mode = 'reset'
        elif 'control' in event.key:
            mode = 'remove'
        elif 'shift' in event.key:
            mode = 'add'
        else:
            mode = 'reset'

        highlight_trajectories(click_names, mode)

        # Copy the selected hyperparameters to the clipboard:

        fg_names = get_foreground_names()
        hparams_df\
                .filter(pl.col('name').is_in(fg_names))\
                .write_ndjson(fp := StringIO())

        pyperclip.copy(fp.getvalue())

    plt.connect('button_release_event', on_click)
    plt.show()

def highlight_trajectories(names, mode):

    def move_to_foreground(line):
        if line.meta['foreground']:
            return
        else:
            line.meta['foreground'] = True
            color = line.meta.pop('color')
            line.set_color(color)
            line.zorder += 100

    def move_to_background(line):
        if line.meta['foreground']:
            line.meta['foreground'] = False
            line.meta['color'] = line.get_color()
            line.set_color('#dddddd')
            line.zorder -= 100
        else:
            return

    def no_change(line):
        pass

    # The keys are (mode, is selected) tuples.
    actions = {
            ('all', True): move_to_foreground,
            ('all', False): move_to_foreground,

            ('reset', True): move_to_foreground,
            ('reset', False): move_to_background,

            ('add', True): move_to_foreground,
            ('add', False): no_change,

            ('remove', True): move_to_background,
            ('remove', False): no_change,
    }

    for ax in plt.gcf().get_axes():
        for line in ax.get_lines():
            if not hasattr(line, 'meta'):
                continue

            action = actions[mode, line.meta['name'] in names]
            action(line)

    plt.draw()

def get_foreground_names():
    return {
            line.meta['name']
            for ax in plt.gcf().get_axes()
            for line in ax.get_lines()
            if hasattr(line, 'meta') and line.meta['foreground']
    }

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

def pick_custom_metrics(df, globs):
    known_metrics = sorted(set(df['metric']))

    metrics = []
    for glob in globs:
        metrics += fnmatch.filter(known_metrics, glob)

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

    candidates = [
            stage.format(metric)
            for stage, metric in product(stages, metrics)
    ]
    candidates += [
            'gen/accuracy',
            'gen/frechet_dist',
    ]

    return [x for x in candidates if x in known_metrics]

def load_hparams(path_or_io):
    import csv

    # Polars doesn't have the option to not infer data types, if the number of 
    # columns isn't known.  Since we want to ensure that the model names are 
    # strings even if they're composed entirely of digits, we need to parse the 
    # file ourselves.

    if isinstance(path_or_io, Path):
        open_csv = open(path_or_io)
    else:
        open_csv = nullcontext(path_or_io)

    with open_csv as f:
        rows = list(csv.reader(f))

    head = rows[0]
    body = rows[1:]

    if 'name' not in head:
        raise ValueError(f"hyperparameter table must contain 'name' column: {path_or_io}")

    return pl.DataFrame(body, head, orient='row')

def join_hparams(df, hparams):
    return (
            df
            .join(
                hparams.with_row_index(),
                on='name',
                how='inner',
            )
            .sort('index')
            .drop('index')
    )

def _pick_hparam_colors(df, hparams):
    hparam_colors = {}

    for hparam in hparams:
        hparam_values = df[hparam].unique(maintain_order=True).drop_nulls()

        # For now, I just hard-coded these two parameters in.  Later, I want to 
        # add the ability to parse hyperparameter options from a config file.

        if hparam in ['flops', 'params']:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            hparam_floats = hparam_values.cast(float)

            norm = Normalize(min(hparam_floats), max(hparam_floats))
            cm = ScalarMappable(norm, 'viridis')

            hparam_colors[hparam] = cm
            for value in hparam_values:
                hparam_colors[hparam, value] = cm.to_rgba(float(value))

        else:
            for i, value in enumerate(hparam_values):
                hparam_colors[hparam, value] = f'C{i}'

    return hparam_colors

def _apply_smoothing(x, y):
    from sklearn.neighbors import LocalOutlierFactor
    from scipy.signal import savgol_filter

    # Remove missing data:
    i = np.isfinite(y)
    x = x[i]
    y = y[i]

    if len(x) < 5:
        return x, y

    # Remove outliers:
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

    # Create smoothed curve:
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

def _remove_nan_inf_rows(x):
    return x[np.isfinite(x).all(axis=1)]


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

