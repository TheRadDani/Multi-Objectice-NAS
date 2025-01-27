from ax.core import Experiment
from search_space import search_space
from optim_config import optim_config
from runner import ax_runner

experiment = Experiment(
    name="nas_mnist",
    search_space=search_space,
    optimization_config=optim_config,
    runner=ax_runner,
)