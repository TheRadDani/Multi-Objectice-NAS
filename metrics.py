from ax.metrics.tensorboard import TensorboardMetric
from tensorboard.backend.event_processing import plugin_event_multiplexer \
as event_multiplexer
from pathlib import Path
from runner import *

class MyTensorboardMetric(TensorboardMetric):

    def _get_event_multiplexer_for_trial(self, trial):
        mul = event_multiplexer.EventMultiplexer(max_reload_threads=20)
        mul.AddRunsFromDirectory(Path(log_dir).joinpath(str(trial.index)).as_posix(), None)
        mul.Reload()

        return mul
    
    @classmethod
    def is_available_while_running(cls):
        return False
    
val_acc = MyTensorboardMetric(
    name="val_acc",
    tag="val_acc",
    lower_is_better=False,
)

model_num_params = MyTensorboardMetric(
    name="num_params",
    tag="num_params",
    lower_is_better=True,
)