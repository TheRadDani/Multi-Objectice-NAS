from ax.service.scheduler import Scheduler, SchedulerOptions
from experiment import experiment
from generation_strategy import *

scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=gs,
    options=SchedulerOptions(
        total_trials=num_trials,
        max_pending_trials=4,
    ),
)