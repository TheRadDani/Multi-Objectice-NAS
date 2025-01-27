from ax.modelbridge.dispatch_utils import choose_generation_strategy
from experiment import experiment

num_trials = 48 # total of evaluation budget


gs = choose_generation_strategy(
    search_space=experiment.search_space,
    optimization_config=experiment.optimization_config,
    num_trials= num_trials,
)