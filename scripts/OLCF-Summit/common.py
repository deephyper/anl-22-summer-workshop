import pathlib
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from deephyper.search.hps import CBO
from sst import problem

NUM_WORKERS = 512 - 1  # $SLURM_JOBSIZE * $RANKS_PER_NODE - 1
SEARCH_TIMEOUT = 200
RUN_SLEEP = 1


def execute_search(evaluator):
    t = time.time()
    init_duration = t - evaluator.timestamp
    evaluator.timestamp = t
    
    print(f"Creation of the Evaluator done with {evaluator.num_workers} worker(s)")
    print("Creation of the search instance...")
    

    # search = CBO(problem, evaluator, surrogate_model="DUMMY", filter_duplicated=False)
    search = CBO(problem, evaluator, initial_points=[problem.default_configuration], log_dir="cbo-results", random_state=42)

    
    print("Creation of the search done")
    print("Starting the search...")

    # results = search.search(timeout=SEARCH_TIMEOUT)
    results = search.search(max_evals=20)

    print("Search is done")
    results.to_csv("results.csv")
    print('results saved to csv')

    return init_duration


def get_profile_from_hist(hist):
    n_processes = 0
    profile_dict = dict(t=[0], n_processes=[0])
    for e in sorted(hist):
        t, incr = e
        n_processes += incr
        profile_dict["t"].append(t)
        profile_dict["n_processes"].append(n_processes)
    profile = pd.DataFrame(profile_dict)
    return profile


def get_perc_util(profile):
    csum = 0
    for i in range(len(profile) - 1):
        csum += (profile.loc[i + 1, "t"] - profile.loc[i, "t"]) * profile.loc[
            i, "n_processes"
        ]
    perc_util = csum / (profile["t"].iloc[-1] * 6)
    return perc_util


def plot_profile(ax, profile, ylabel="None", color="blue"):
    ax.step(profile["t"], profile["n_processes"], where="post", color=color)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, NUM_WORKERS + 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid()


def plot_sum_up(name, init_duration):

    pathlib.Path("plots").mkdir(parents=False, exist_ok=True)
    results = pd.read_csv("results.csv")

    # compute profiles from results.csv
    jobs_hist = []
    runs_hist = []
    for _, row in results.iterrows():
        jobs_hist.append((row["timestamp_submit"], 1))
        jobs_hist.append((row["timestamp_gather"], -1))
        runs_hist.append((row["timestamp_start"], 1))
        runs_hist.append((row["timestamp_end"], -1))

    jobs_profile = get_profile_from_hist(jobs_hist)
    runs_profile = get_profile_from_hist(runs_hist)

    # compute average job and run durations
    job_avrg_duration = (
        results["timestamp_gather"] - results["timestamp_submit"]
    ).mean()

    # compute perc_util
    jobs_perc_util = get_perc_util(jobs_profile)
    runs_perc_util = get_perc_util(runs_profile)

    # compute total number of evaluations
    total_num_eval = len(results)

    # plot
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(name, fontsize=17)
    plot_profile(axs[0], jobs_profile, ylabel="# jobs submitted", color="blue")
    plot_profile(axs[1], runs_profile, ylabel="# jobs running", color="crimson")
    fig.text(0.1, -0.1, f"init_duration: {init_duration:.2f}s.", fontsize=12)
    fig.text(0.1, -0.2, f"job_avrg_duration: {job_avrg_duration:.2f}s.", fontsize=12)
    fig.text(0.1, -0.3, f"total_num_eval: {total_num_eval}", fontsize=12)
    fig.text(
        0.6,
        -0.1,
        f"jobs_perc_util: {100*jobs_perc_util:.1f}%",
        fontsize=12,
        color="blue",
    )
    fig.text(
        0.6,
        -0.2,
        f"runs_perc_util: {100*runs_perc_util:.1f}%",
        fontsize=12,
        color="crimson",
    )
    fig.tight_layout()
    plt.savefig(f"plots/{name}.jpg", bbox_inches="tight")


def evaluate_and_plot(evaluator, name):

    init_duration = execute_search(evaluator)

    plot_sum_up(name, init_duration)
