"""Phase 5: fault detection in No.14WT and No.39WT with the kernel model.

The kernel model selected in Phase 4 is refitted on all healthy No.2WT
observations and the faulty turbines are projected onto it. Alarms, fault
periods and alarm proportions are calculated as in Phases 2 and 3 and
compared with linear PCA (the Phase 1 model) evaluated with the same limit
formulas: the F-distribution T2 limit and the scaled chi-square Q limit.

Note: Nothing is fitted on the faulty turbines. The Phase 2 and 3 results,
which use the Jackson-Mudholkar Q limit, are not changed; the linear model
is only re-evaluated here with the kernel model's limit formulas.
"""

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernel_monitoring import (
    KernelModel,
    KernelProjection,
    fit_kernel_model,
    kernel_limits,
    load_kernel_settings,
    project_kernel,
    verify_partial_scores,
)
from pca_monitoring import Projection, project, verify_reference_projection
from phase2_fault_detection import MONITORING_STATISTICS_FILE, load_final_model
from phase3_diagnostics import VARIABLE_13_TRANSITION, find_alarm_episodes
from phase4_kernel_tuning import LIMIT_TYPE, SELECTED_KERNEL_FILE
from pretreatment import FAULTY_TURBINES, HEALTHY_TURBINE, OUTPUT_DIR, load_pca_data

KERNEL_STATISTICS_FILE = OUTPUT_DIR / "phase5_kernel_monitoring_statistics.csv"
MODELS = ["kernel", "linear"]
STATISTICS = ["T2", "Q", "T2 or Q"]


def main() -> None:
    pca_x, _ = load_pca_data()
    healthy_x = pca_x[HEALTHY_TURBINE]
    turbines = [HEALTHY_TURBINE, *FAULTY_TURBINES]

    # ============================ Rebuild the kernel and linear models.
    settings = load_kernel_settings(SELECTED_KERNEL_FILE)
    assert settings["limit_type"] == LIMIT_TYPE
    kernel_model = fit_kernel_model(healthy_x, settings["family"], settings["width"])
    linear_model, linear_components, phase1_limits = load_final_model()
    models = {
        "kernel": (kernel_model, settings["n_components"], project_kernel),
        "linear": (linear_model, linear_components, project),
    }

    projections = {}
    limits = {}
    for model_name, (model, n_components, project_function) in models.items():
        projections[model_name] = {
            name: project_function(model, pca_x[name], n_components) for name in turbines
        }
        healthy = projections[model_name][HEALTHY_TURBINE]
        verify_reference_projection(model, healthy)
        # Both models use the same limit formulas: the F-distribution T2
        # limit and the scaled chi-square Q limit fitted to the healthy Q.
        limits[model_name] = kernel_limits(model, healthy, LIMIT_TYPE)

    # The refitted kernel model reproduces the Phase 4 limits, and the
    # linear model reproduces the Phase 1 T2 limit and the Phase 2 statistics.
    assert np.isclose(limits["kernel"]["t2"], settings["limits"]["t2"], rtol=1e-9)
    assert np.isclose(limits["kernel"]["q"], settings["limits"]["q"], rtol=1e-9)
    assert np.isclose(limits["linear"]["t2"], phase1_limits["t2_f"], rtol=1e-12)
    phase2_statistics = pd.read_csv(MONITORING_STATISTICS_FILE)
    for name in FAULTY_TURBINES:
        saved = phase2_statistics[phase2_statistics["turbine"] == name]
        assert np.allclose(projections["linear"][name].t2, saved["t2"], rtol=1e-12)
        assert np.allclose(projections["linear"][name].q, saved["q"], rtol=1e-12)
        verify_partial_scores(kernel_model, projections["kernel"][name])

    flags = {
        model_name: {
            name: alarm_flags(projections[model_name][name], limits[model_name])
            for name in turbines
        }
        for model_name in MODELS
    }

    print("\n========================================================")
    print(
        f"\nKernel model: {settings['family']}, width {settings['width']:g}, "
        f"{settings['n_components']} components; limits T2 = {limits['kernel']['t2']:.3f}, "
        f"Q = {limits['kernel']['q']:.4f}"
    )
    print(
        f"Linear model (Phase 1): {linear_components} components; limits T2 = "
        f"{limits['linear']['t2']:.3f}, Q = {limits['linear']['q']:.3f}"
    )
    print("Both: 99% F-distribution T2 limit and scaled chi-square Q limit")

    # ============================ Detection: statistics, alarm summary and control charts.
    statistics = []
    for name in FAULTY_TURBINES:
        table = {"turbine": name, "observation": np.arange(1, len(pca_x[name]) + 1)}
        for model_name in MODELS:
            projection = projections[model_name][name]
            table[f"{model_name}_t2"] = projection.t2
            table[f"{model_name}_q"] = projection.q
            for statistic, column in zip(STATISTICS, ["t2_alarm", "q_alarm", "t2_or_q_alarm"]):
                table[f"{model_name}_{column}"] = flags[model_name][name][statistic]
        statistics.append(pd.DataFrame(table))
    pd.concat(statistics, ignore_index=True).to_csv(KERNEL_STATISTICS_FILE, index=False)

    summary = []
    for model_name in MODELS:
        n_components = models[model_name][1]
        for name in turbines:
            either = flags[model_name][name]["T2 or Q"]
            observations = np.arange(1, len(either) + 1)
            summary.append(
                {
                    "model": model_name,
                    "turbine": name,
                    "role": "healthy in-sample" if name == HEALTHY_TURBINE else "faulty evaluation",
                    "observations": len(either),
                    "n_components": n_components,
                    "t2_limit": limits[model_name]["t2"],
                    "q_limit": limits[model_name]["q"],
                    "t2_alarms": int(flags[model_name][name]["T2"].sum()),
                    "q_alarms": int(flags[model_name][name]["Q"].sum()),
                    "t2_or_q_alarms": int(either.sum()),
                    "t2_or_q_alarm_percent": 100 * either.mean(),
                    "first_alarm_observation": int(observations[either][0]) if either.any() else None,
                    "last_alarm_observation": int(observations[either][-1]) if either.any() else None,
                }
            )
    summary = pd.DataFrame(summary)
    summary.to_csv(OUTPUT_DIR / "phase5_kernel_alarm_summary.csv", index=False)

    for name in FAULTY_TURBINES:
        create_kernel_control_charts(
            projections["kernel"][name],
            limits["kernel"],
            f"{name} projected onto the {HEALTHY_TURBINE} kernel model "
            f"({settings['family']}, width {settings['width']:g}, "
            f"{settings['n_components']} components)",
            OUTPUT_DIR / f"phase5_kernel_control_charts_{name}.png",
        )

    print("\n========================================================")
    print("\nOut-of-control observations (T2 or Q)")
    for row in summary.itertuples():
        print(
            f"{row.model:6s} {row.turbine} ({row.role}): {row.t2_or_q_alarms} / "
            f"{row.observations} ({row.t2_or_q_alarm_percent:.1f}%); first alarm "
            f"{row.first_alarm_observation}, last alarm {row.last_alarm_observation}"
        )

    # ============================ Fault periods and alarm proportions.
    period_rows = []
    proportion_rows = []
    for name in FAULTY_TURBINES:
        transition = VARIABLE_13_TRANSITION[name]
        variable_13 = pca_x[name][13].to_numpy()
        assert np.median(variable_13[: transition - 1]) > 80
        assert variable_13[transition:].max() < 1
        observations = np.arange(1, len(pca_x[name]) + 1)

        for model_name in MODELS:
            model_flags = flags[model_name][name]
            episodes = find_alarm_episodes(model_flags["T2 or Q"])
            in_episode = np.zeros(len(observations), dtype=bool)
            for start, end in episodes:
                in_episode[start - 1 : end] = True
            periods = {
                "A: fault period (before variable 13 transition)": observations < transition,
                "A: transition observation": observations == transition,
                "A: after transition": observations > transition,
                "B: alarm episodes": in_episode,
                "B: outside alarm episodes": ~in_episode,
            }

            period_rows.append(
                {
                    "turbine": name,
                    "model": model_name,
                    "method": "A: variable 13 transition",
                    "fault_start": 1,
                    "fault_end": transition - 1,
                    "observations": transition - 1,
                    "alarm_percent": 100 * model_flags["T2 or Q"][: transition - 1].mean(),
                }
            )
            for start, end in episodes:
                period_rows.append(
                    {
                        "turbine": name,
                        "model": model_name,
                        "method": "B: alarm episode",
                        "fault_start": start,
                        "fault_end": end,
                        "observations": end - start + 1,
                        "alarm_percent": 100 * model_flags["T2 or Q"][start - 1 : end].mean(),
                    }
                )
            for period, mask in periods.items():
                if not mask.any():
                    continue
                for statistic in STATISTICS:
                    proportion_rows.append(
                        {
                            "turbine": name,
                            "model": model_name,
                            "period": period,
                            "statistic": statistic,
                            "observations": int(mask.sum()),
                            "alarms": int(model_flags[statistic][mask].sum()),
                            "alarm_percent": 100 * model_flags[statistic][mask].mean(),
                        }
                    )
    fault_periods = pd.DataFrame(period_rows)
    fault_periods.to_csv(OUTPUT_DIR / "phase5_kernel_fault_periods.csv", index=False)
    proportions = pd.DataFrame(proportion_rows)
    proportions.to_csv(OUTPUT_DIR / "phase5_kernel_alarm_proportions.csv", index=False)

    print("\n========================================================")
    print("\nFault periods (method B: start at 3 consecutive alarms, end at the last alarm "
          "before 30 alarm-free observations)")
    for row in fault_periods.itertuples():
        print(
            f"{row.turbine}, {row.model:6s}, {row.method}: observations {row.fault_start}-"
            f"{row.fault_end} ({row.observations} observations, {row.alarm_percent:.1f}% alarms)"
        )

    print("\nT2 or Q alarm proportions")
    selected = proportions[proportions["statistic"] == "T2 or Q"]
    for row in selected.itertuples():
        print(
            f"{row.turbine}, {row.model:6s}, {row.period}: {row.alarms} / "
            f"{row.observations} ({row.alarm_percent:.1f}%)"
        )

    # ============================ Agreement between the kernel and linear alarms.
    agreement = []
    for name in FAULTY_TURBINES:
        kernel = flags["kernel"][name]["T2 or Q"]
        linear = flags["linear"][name]["T2 or Q"]
        agreement.append(
            {
                "turbine": name,
                "observations": len(kernel),
                "both_alarm": int(np.sum(kernel & linear)),
                "kernel_only": int(np.sum(kernel & ~linear)),
                "linear_only": int(np.sum(~kernel & linear)),
                "neither": int(np.sum(~kernel & ~linear)),
                "agreement_percent": 100 * np.mean(kernel == linear),
                "cohen_kappa": cohen_kappa(kernel, linear),
            }
        )
    agreement = pd.DataFrame(agreement)
    agreement.to_csv(OUTPUT_DIR / "phase5_alarm_agreement.csv", index=False)

    print("\n========================================================")
    print("\nAgreement of the kernel and linear T2 or Q alarms")
    for row in agreement.itertuples():
        print(
            f"{row.turbine}: both {row.both_alarm}, kernel only {row.kernel_only}, linear only "
            f"{row.linear_only}, neither {row.neither}; agreement {row.agreement_percent:.1f}%, "
            f"Cohen's kappa {row.cohen_kappa:.3f}"
        )

    create_kernel_score_plots(kernel_model, projections["kernel"], settings)


def alarm_flags(
    projection: Projection | KernelProjection, limits: dict[str, float]
) -> dict[str, np.ndarray]:
    """Observations above the T2 limit, the Q limit, and either of them."""
    t2 = projection.t2 > limits["t2"]
    q = projection.q > limits["q"]
    return {"T2": t2, "Q": q, "T2 or Q": t2 | q}


def cohen_kappa(first: np.ndarray, second: np.ndarray) -> float:
    """
    Agreement of two alarm series beyond chance: kappa = (p_o - p_e) / (1 - p_e).

    p_o is the share of observations on which the series agree and
    p_e = p1 p2 + (1 - p1)(1 - p2) the agreement expected from their alarm
    rates p1 and p2 alone.
    """
    observed = np.mean(first == second)
    p1, p2 = first.mean(), second.mean()
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    assert expected < 1, "Kappa is undefined when both series are constant and equal."
    return (observed - expected) / (1 - expected)


def create_kernel_control_charts(
    projection: KernelProjection, limits: dict[str, float], title: str, path
) -> None:
    """
    T2 and Q charts in observation order with the 99% limits.

    The y-axes are logarithmic because both statistics are positive and
    strongly skewed.
    """
    observations = np.arange(1, len(projection.t2) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for axis, statistic, label, limit_name in zip(
        axes,
        ["t2", "q"],
        ["Hotelling's T2", "Q (SPEx)"],
        ["F-distribution", "scaled chi-square"],
    ):
        limit = limits[statistic]
        axis.plot(observations, getattr(projection, statistic), linewidth=0.7, color="tab:blue")
        axis.axhline(limit, color="tab:red", linewidth=1.2,
                     label=f"99% limit, {limit_name} ({limit:.4g})")
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Observation order")
    fig.suptitle(f"{title}: T2 and Q control charts")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_kernel_score_plots(
    model: KernelModel, projections: dict[str, KernelProjection], settings: dict
) -> None:
    """
    KPC1-KPC2 scores of each faulty turbine over the healthy scores.

    Faulty observations are coloured by observation order. Bounded one-sensor
    kernels bound score magnitudes, but faulty scores can extend beyond the
    observed healthy range. Healthy and faulty scores use common axes. Kernel
    components have no per-sensor loading arrows.
    """
    explained = 100 * model.eigenvalues / model.eigenvalues.sum()
    healthy = projections[HEALTHY_TURBINE].scores

    fig, axes = plt.subplots(1, len(FAULTY_TURBINES), figsize=(14, 6), constrained_layout=True)
    for axis, name in zip(axes, FAULTY_TURBINES):
        scores = projections[name].scores
        axis.scatter(healthy[:, 0], healthy[:, 1], color="0.6", s=6, alpha=0.5,
                     label=f"{HEALTHY_TURBINE} (healthy)")
        points = axis.scatter(scores[:, 0], scores[:, 1], c=np.arange(1, len(scores) + 1),
                              cmap="viridis", s=9, alpha=0.8, label=name)
        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.axvline(0, color="0.5", linewidth=0.8)
        axis.set_xlabel(f"KPC1 scores ({explained[0]:.2f}%)")
        axis.set_ylabel(f"KPC2 scores ({explained[1]:.2f}%)")
        axis.set_title(name)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
        fig.colorbar(points, ax=axis, label=f"{name} observation order", shrink=0.9)

    fig.suptitle(
        f"Faulty turbines on the KPC1-KPC2 plane of the {HEALTHY_TURBINE} kernel model "
        f"({settings['family']}, width {settings['width']:g})"
    )
    fig.savefig(OUTPUT_DIR / "phase5_kernel_score_plots.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
