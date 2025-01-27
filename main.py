from scheduler import scheduler
from generation_strategy import gs
from experiment import experiment
from ax.service.utils.report_utils import exp_to_df
from ax.service.utils.report_utils import _pareto_frontier_scatter_2d_plotly
from ax.modelbridge.cross_validation import compute_diagnostics, cross_validate
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.plot.contour import interact_contour_plotly
import webbrowser
import tempfile
import time

def open_plot_in_new_tab(plot, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        plot.write_html(tmpfile.name)
        webbrowser.open_new_tab(tmpfile.name)

if __name__ == "__main__":

    start_time = time.time()
    scheduler.run_all_trials()
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    df = exp_to_df(experiment)
    print(df.head(10))

    pareto_plot  = _pareto_frontier_scatter_2d_plotly(experiment)
    open_plot_in_new_tab(pareto_plot, "pareto_frontier.html")

    cv = cross_validate(model=gs.model)  # The surrogate model is stored on the ``GenerationStrategy``
    compute_diagnostics(cv)

    cv_plot = interact_cross_validation_plotly(cv)
    open_plot_in_new_tab(cv_plot, "cross_validation.html")

    contour_plot_val_acc = interact_contour_plotly(model=gs.model, metric_name="val_acc")
    open_plot_in_new_tab(contour_plot_val_acc, "contour_val_acc.html")

    contour_plot_num_params = interact_contour_plotly(model=gs.model, metric_name="num_params")
    open_plot_in_new_tab(contour_plot_num_params, "contour_num_params.html")