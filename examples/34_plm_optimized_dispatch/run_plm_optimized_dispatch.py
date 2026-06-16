"""
This example simulates a Battery Energy Storage system controller
to demonstrate demand-response and peak-load management dispatch
using a rolling-horizon MILP controller.
The battery is scheduled to discharge during high-LMP peak hours
to maximize incentives and to minimize the operation cost during
off-peak hours.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from h2integrate.core.utilities import build_time_series_from_plant_config
from h2integrate.core.h2integrate_model import H2IntegrateModel

EXAMPLE_DIR = Path(__file__).parent


# Run H2Ingegrate
model = H2IntegrateModel(EXAMPLE_DIR / "34_plm_optimized_dispatch.yaml")
model.setup()
model.run()

# Read inputs from config
control_params = model.technology_config["technologies"]["battery"]["model_inputs"][
    "control_parameters"
]
n_timesteps = int(model.plant_config["plant"]["simulation"]["n_timesteps"])
lmp = np.array(control_params["lmp_signal"])[:n_timesteps]
demand = np.array(control_params["demand_signal"])[:n_timesteps]
dt_seconds = int(model.plant_config["plant"]["simulation"]["dt"])
time_index = pd.DatetimeIndex(build_time_series_from_plant_config(model.plant_config))

event_dur_cfg = control_params.get("event_duration")
half_td = None
if event_dur_cfg is not None:
    half_td = pd.Timedelta(value=event_dur_cfg["val"], unit=event_dur_cfg["units"]) / 2

# Read H2Integrate output
battery_power = model.prob.get_val("battery.storage_electricity_discharge", units="kW")
soc_pct = model.prob.get_val("battery.SOC", units="percent")

# Read controller output
controller = model.control_strategies[0]
pw_start, pw_end = controller._parse_peak_window()
pw_start_h = pw_start.hour
pw_end_h = pw_end.hour

# Intermediate MILP decision variables
u_discharge1 = controller.discharge1_bin_history
u_discharge2 = controller.discharge2_bin_history
v_charge = controller.charge_bin_history
p_discharge1 = controller.p_discharge1_history
p_discharge2 = controller.p_discharge2_history
p_charge = controller.p_charge_history
p_tocoop = controller.p_tocoop_history

eventlogmask = [False] * n_timesteps
for i in range(n_timesteps):
    if i == 0:
        eventlogmask[i] = False
    elif u_discharge1[i] == 1 and p_discharge1[i-1] ==0 :
        eventlogmask[i] = True

# Plot outputs
plotdays = 4
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 11))
days = pd.date_range(time_index[0].normalize(), periods=plotdays, freq="D", tz=time_index.tz)
plot_time_window = min(n_timesteps, int(plotdays * 24 * 3600 / dt_seconds))  # 14 days


def shade_peaks(ax):
    for day in days:
        ax.axvspan(
            day + pd.Timedelta(hours=pw_start_h),
            day + pd.Timedelta(hours=pw_end_h),
            color="orange",
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )
        if half_td is None:
            continue
        pw_start_ts = day + pd.Timedelta(hours=pw_start_h)
        pw_end_ts = day + pd.Timedelta(hours=pw_end_h)
        in_pw = (time_index >= pw_start_ts) & (time_index <= pw_end_ts)
        if not in_pw.any():
            continue
        peak_idx = np.where(in_pw)[0][np.argmax(lmp[in_pw])]
        peak_ts = time_index[peak_idx]
        ax.axvspan(
            peak_ts - half_td,
            peak_ts + half_td,
            color="darkorange",
            alpha=0.30,
            linewidth=0,
            zorder=0,
        )

ax = axes[0]
shade_peaks(ax)
ax.plot(time_index[:plot_time_window], lmp[:plot_time_window], color="steelblue", linewidth=1.0)
ax.plot(
    time_index[:plot_time_window][eventlogmask[:plot_time_window]],
    lmp[:plot_time_window][eventlogmask[:plot_time_window]],
    "r*",
    markersize=8,
    zorder=5,
)
ax.set_ylabel("LMP ($/MWh)", fontsize=8)
ax.set_ylim(bottom=0)

ax = axes[1]
shade_peaks(ax)
ax.plot(time_index[:plot_time_window], soc_pct[:plot_time_window], color="g", linewidth=1.0)
ax.axhline(90, color="gray", linestyle=":", linewidth=0.7)
ax.axhline(10, color="gray", linestyle=":", linewidth=0.7)
ax.set_ylabel("SOC (%)", fontsize=8)
ax.set_ylim([0, 105])

# Subplot 3: battery power flows (charge plotted negative for visual separation)
ax = axes[2]
shade_peaks(ax)
ax.plot(
    time_index[:plot_time_window],
    p_discharge1[:plot_time_window],
    color="darkorange",
    label="p_discharge1 (G&T)",
    linewidth=1.0,
)
ax.plot(
    time_index[:plot_time_window],
    p_discharge2[:plot_time_window],
    color="green",
    label="p_discharge2 (Co-Op)",
    linewidth=1.0,
)
ax.plot(
    time_index[:plot_time_window],
    -p_charge[:plot_time_window],
    color="steelblue",
    label="p_charge (neg)",
    linewidth=1.0,
)
ax.axhline(0, color="k", linewidth=0.5)
ax.set_ylabel("Battery power (kW)", fontsize=8)
ax.legend(fontsize=7, loc="upper right", frameon=False)

# Subplot 4: grid / Co-Op flows vs demand
ax = axes[3]
shade_peaks(ax)
ax.plot(
    time_index[:plot_time_window],
    demand[:plot_time_window],
    color="k",
    label="demand",
    linewidth=1.0,
)
ax.plot(
    time_index[:plot_time_window],
    p_tocoop[:plot_time_window],
    color="teal",
    label="p_tocoop",
    linewidth=1.0,
    linestyle="--",
)
ax.set_ylabel("Grid / Co-Op (kW)", fontsize=8)
ax.set_xlabel("Time")
ax.legend(fontsize=7, loc="upper right", frameon=False)

pass
plt.tight_layout()
plt.savefig(EXAMPLE_DIR / "plm_optimized_dispatch.png", dpi=150, bbox_inches="tight")
