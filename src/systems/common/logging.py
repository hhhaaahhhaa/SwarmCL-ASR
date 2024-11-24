from lightning.pytorch.loggers import TensorBoardLogger, CometLogger
import matplotlib.pylab as plt


def log_text(logger, text, step, tag):
    if isinstance(logger, CometLogger):
        logger.experiment.log_text(
            text=text,
            step=step,
            metadata={"tag": tag}
        )
    elif isinstance(logger, TensorBoardLogger):
        logger.experiment.add_text(tag, text, step)


def log_figure(logger, fig, step, tag):
    if isinstance(logger, CometLogger):
        logger.experiment.log_figure(
            figure_name=tag,
            figure=fig,
            step=step,
        )
    elif isinstance(logger, TensorBoardLogger):
        logger.experiment.add_figure(tag, plt.gcf(), step)
    plt.close(fig)
