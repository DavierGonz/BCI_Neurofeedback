"""Análisis offline de Theta/Beta y ERD en imaginación motora.

Procesa un sujeto del dataset GDF continuo utilizado en BCI_Exploration. El cue
LEFT/RIGHT es t=0; se compara baseline [-3, -1) s con MI [1, 4) s.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "BCI_Exploration" / "BCI Database" / "Signals" / "DATA A"
DEFAULT_OUTPUT_ROOT = ROOT / "BCI_Neurofeedback" / "results"
CHANNELS = ("C3", "Cz", "C4")
CLASS_EVENTS = {"769": "LEFT", "770": "RIGHT"}
BASELINE_SECONDS = (-3.0, -1.0)
MI_SECONDS = (1.0, 4.0)
LINE_FREQUENCY_HZ = 50.0
FILTER_BAND_HZ = (1.0, 40.0)
ARTIFACT_THRESHOLD_UV = 150.0
BANDS = {
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
EPSILON = np.finfo(float).eps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="A18", help="Sujeto, por ejemplo A18.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include-artifacts", action="store_true",
                        help="Incluye trials que superen ±150 µV (no recomendado).")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def band_power(frequencies: np.ndarray, psd: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Integra la PSD para cada canal dentro de [fmin, fmax)."""
    fmin, fmax = limits
    mask = (frequencies >= fmin) & (frequencies < fmax)
    if mask.sum() < 2:
        raise ValueError(f"No hay suficientes bins para integrar {fmin}-{fmax} Hz.")
    return np.trapezoid(psd[:, mask], frequencies[mask], axis=1)


def spectral_powers(window_volts: np.ndarray, sampling_rate: float) -> dict[str, np.ndarray]:
    """Welch comparable entre ventanas usando segmentos fijos de un segundo."""
    segment_samples = int(round(sampling_rate))
    frequencies, psd = welch(
        window_volts,
        fs=sampling_rate,
        window="hann",
        nperseg=segment_samples,
        noverlap=segment_samples // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    return {name: band_power(frequencies, psd, limits) for name, limits in BANDS.items()}


def prepare_raw(path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    typed_channels = {
        name: "eog" for name in ("EOG1", "EOG2", "EOG3") if name in raw.ch_names
    }
    typed_channels.update({
        name: "emg" for name in ("EMGg", "EMGd") if name in raw.ch_names
    })
    if typed_channels:
        raw.set_channel_types(typed_channels, verbose="ERROR")
    missing = sorted(set(CHANNELS) - set(raw.ch_names))
    if missing:
        raise RuntimeError(f"{path.name} no contiene los canales {missing}.")

    # El registro proviene de Francia: se elimina red a 50 Hz antes del low-pass.
    raw.notch_filter(
        freqs=[LINE_FREQUENCY_HZ], picks="eeg", method="iir",
        verbose="ERROR",
    )
    raw.filter(
        FILTER_BAND_HZ[0], FILTER_BAND_HZ[1], picks="eeg", method="iir",
        iir_params={"order": 4, "ftype": "butter"}, verbose="ERROR",
    )
    # Referencia promedio usando EEG; EOG y EMG fueron excluidos mediante sus tipos.
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


def extract_subject_trials(subject_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    files = sorted(subject_dir.glob("*.gdf"))
    if not files:
        raise FileNotFoundError(f"No se encontraron GDF en {subject_dir}.")

    channel_rows: list[dict] = []
    trial_rows: list[dict] = []
    rejected = 0
    discovered = 0

    for path in files:
        print(f"Procesando {path.name}...")
        raw = prepare_raw(path)
        sampling_rate = float(raw.info["sfreq"])
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        descriptions = {code: description for description, code in event_id.items()}
        trial_number = 0

        for cue_sample, _, code in events:
            description = descriptions.get(int(code))
            if description not in CLASS_EVENTS:
                continue
            trial_number += 1
            discovered += 1
            class_name = CLASS_EVENTS[description]

            baseline_start = cue_sample + int(round(BASELINE_SECONDS[0] * sampling_rate))
            baseline_stop = cue_sample + int(round(BASELINE_SECONDS[1] * sampling_rate))
            mi_start = cue_sample + int(round(MI_SECONDS[0] * sampling_rate))
            mi_stop = cue_sample + int(round(MI_SECONDS[1] * sampling_rate))
            if baseline_start < 0 or mi_stop > raw.n_times:
                rejected += 1
                continue

            baseline = raw.get_data(
                picks=list(CHANNELS), start=baseline_start, stop=baseline_stop
            )
            mi = raw.get_data(picks=list(CHANNELS), start=mi_start, stop=mi_stop)
            peak_uv = float(max(np.abs(baseline).max(), np.abs(mi).max()) * 1e6)
            artifact = peak_uv > ARTIFACT_THRESHOLD_UV

            baseline_power = spectral_powers(baseline, sampling_rate)
            mi_power = spectral_powers(mi, sampling_rate)
            session = path.stem

            for channel_index, channel in enumerate(CHANNELS):
                theta_baseline = baseline_power["theta"][channel_index]
                theta_mi = mi_power["theta"][channel_index]
                beta_baseline = baseline_power["beta"][channel_index]
                beta_mi = mi_power["beta"][channel_index]
                mu_baseline = baseline_power["mu"][channel_index]
                mu_mi = mi_power["mu"][channel_index]
                ratio_baseline = theta_baseline / max(beta_baseline, EPSILON)
                ratio_mi = theta_mi / max(beta_mi, EPSILON)
                channel_rows.append({
                    "subject": subject_dir.name,
                    "session": session,
                    "trial": trial_number,
                    "class": class_name,
                    "channel": channel,
                    "artifact": artifact,
                    "peak_abs_uv": peak_uv,
                    "theta_baseline": theta_baseline,
                    "theta_mi": theta_mi,
                    "beta_baseline": beta_baseline,
                    "beta_mi": beta_mi,
                    "mu_baseline": mu_baseline,
                    "mu_mi": mu_mi,
                    "theta_beta_baseline": ratio_baseline,
                    "theta_beta_mi": ratio_mi,
                    "theta_beta_change_pct": 100 * (ratio_mi - ratio_baseline) / max(ratio_baseline, EPSILON),
                    # Convención: ERD positivo representa reducción durante MI.
                    "mu_erd_pct": 100 * (mu_baseline - mu_mi) / max(mu_baseline, EPSILON),
                    "beta_erd_pct": 100 * (beta_baseline - beta_mi) / max(beta_baseline, EPSILON),
                })

            # Resumen por trial: primero promedia potencia entre canales y luego
            # calcula razones; no promedia cocientes con denominadores diferentes.
            aggregate = {
                f"{band}_{phase}": float(np.mean(power[band]))
                for band in BANDS
                for phase, power in (("baseline", baseline_power), ("mi", mi_power))
            }
            ratio_baseline = aggregate["theta_baseline"] / max(aggregate["beta_baseline"], EPSILON)
            ratio_mi = aggregate["theta_mi"] / max(aggregate["beta_mi"], EPSILON)
            trial_rows.append({
                "subject": subject_dir.name,
                "session": session,
                "trial": trial_number,
                "class": class_name,
                "artifact": artifact,
                "peak_abs_uv": peak_uv,
                **aggregate,
                "theta_beta_baseline": ratio_baseline,
                "theta_beta_mi": ratio_mi,
                "theta_beta_change_pct": 100 * (ratio_mi - ratio_baseline) / max(ratio_baseline, EPSILON),
                "mu_erd_pct": 100 * (aggregate["mu_baseline"] - aggregate["mu_mi"]) / max(aggregate["mu_baseline"], EPSILON),
                "beta_erd_pct": 100 * (aggregate["beta_baseline"] - aggregate["beta_mi"]) / max(aggregate["beta_baseline"], EPSILON),
            })

    metadata = {
        "files": [path.name for path in files],
        "trials_discovered": discovered,
        "trials_outside_recording": rejected,
    }
    return pd.DataFrame(channel_rows), pd.DataFrame(trial_rows), metadata


def paired_statistics(trials: pd.DataFrame) -> dict:
    baseline = trials["theta_beta_baseline"].to_numpy()
    mi = trials["theta_beta_mi"].to_numpy()
    statistic, p_value = wilcoxon(mi, baseline, alternative="two-sided")
    return {
        "n_trials": int(len(trials)),
        "theta_beta_baseline_mean": float(baseline.mean()),
        "theta_beta_baseline_median": float(np.median(baseline)),
        "theta_beta_mi_mean": float(mi.mean()),
        "theta_beta_mi_median": float(np.median(mi)),
        "theta_beta_change_pct_mean": float(trials["theta_beta_change_pct"].mean()),
        "theta_beta_change_pct_median": float(trials["theta_beta_change_pct"].median()),
        "mu_erd_pct_mean": float(trials["mu_erd_pct"].mean()),
        "mu_erd_pct_median": float(trials["mu_erd_pct"].median()),
        "beta_erd_pct_mean": float(trials["beta_erd_pct"].mean()),
        "beta_erd_pct_median": float(trials["beta_erd_pct"].median()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value": float(p_value),
    }


def create_summary_figure(trials: pd.DataFrame, channels: pd.DataFrame, subject: str):
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    rng = np.random.default_rng(42)

    ax = axes[0, 0]
    for _, row in trials.iterrows():
        ax.plot([0, 1], [row.theta_beta_baseline, row.theta_beta_mi],
                color="#9ca3af", alpha=0.12, linewidth=0.7)
    means = [trials.theta_beta_baseline.mean(), trials.theta_beta_mi.mean()]
    ax.plot([0, 1], means, color="#111827", marker="o", linewidth=3,
            markersize=8, label="Promedio del sujeto")
    ax.set_xticks([0, 1], ["Baseline", "MI"])
    ax.set_ylabel("Razón θ/β")
    ax.set_title("A. θ/β por trial y promedio")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for class_name, color in (("LEFT", "#2563eb"), ("RIGHT", "#dc2626")):
        values = trials.loc[trials["class"] == class_name, "theta_beta_change_pct"]
        ax.hist(values, bins=24, alpha=0.45, color=color, label=class_name)
        ax.axvline(values.median(), color=color, linestyle="--", linewidth=2)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_xlabel("Cambio θ/β respecto al baseline (%)")
    ax.set_ylabel("Trials")
    ax.set_title("B. Cambio porcentual de θ/β")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    def erd_panel(axis, column: str, title: str):
        positions, values, labels, colors = [], [], [], []
        position = 1
        for channel in CHANNELS:
            for class_name, color in (("LEFT", "#2563eb"), ("RIGHT", "#dc2626")):
                series = channels.loc[
                    (channels.channel == channel) & (channels["class"] == class_name), column
                ].to_numpy()
                positions.append(position)
                values.append(series)
                labels.append(f"{channel}\n{class_name}")
                colors.append(color)
                position += 1
            position += 0.5
        boxes = axis.boxplot(values, positions=positions, widths=0.7, patch_artist=True,
                             showfliers=False, medianprops={"color": "black"})
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.45)
        for pos, series, color in zip(positions, values, colors):
            jitter = rng.normal(0, 0.06, len(series))
            axis.scatter(pos + jitter, series, s=8, color=color, alpha=0.18)
        axis.axhline(0, color="#111827", linewidth=1)
        axis.set_xticks(positions, labels, fontsize=8)
        axis.set_ylabel("ERD (%) · positivo = reducción en MI")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)

    erd_panel(axes[1, 0], "mu_erd_pct", "C. ERD en μ (8–13 Hz)")
    erd_panel(axes[1, 1], "beta_erd_pct", "D. ERD en β (13–30 Hz)")
    figure.suptitle(
        f"Theta/Beta y ERD durante imaginación motora — {subject}\n"
        "Baseline: −3 a −1 s · MI: +1 a +4 s respecto al cue",
        fontsize=16, fontweight="bold",
    )
    return figure


def save_results(subject: str, output_root: Path, channels: pd.DataFrame,
                 trials: pd.DataFrame, metadata: dict, include_artifacts: bool):
    output_dir = output_root / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    all_trial_count = len(trials)
    artifact_count = int(trials.artifact.sum())
    valid_channels = channels if include_artifacts else channels.loc[~channels.artifact].copy()
    valid_trials = trials if include_artifacts else trials.loc[~trials.artifact].copy()
    if valid_trials.empty:
        raise RuntimeError("Todos los trials fueron rechazados por artefactos.")

    channel_path = output_dir / "theta_beta_erd_por_canal.csv"
    trial_path = output_dir / "theta_beta_erd_por_trial.csv"
    figure_path = output_dir / "resumen_theta_beta_erd.png"
    report_path = output_dir / "resumen.json"
    channels.to_csv(channel_path, index=False)
    trials.to_csv(trial_path, index=False)

    figure = create_summary_figure(valid_trials, valid_channels, subject)
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")

    report = {
        "subject": subject,
        "channels": list(CHANNELS),
        "classes": valid_trials["class"].value_counts().to_dict(),
        "sampling_rate_hz": 512.0,
        "baseline_seconds_relative_to_cue": list(BASELINE_SECONDS),
        "mi_seconds_relative_to_cue": list(MI_SECONDS),
        "bands_hz": {name: list(limits) for name, limits in BANDS.items()},
        "welch": {"nperseg_seconds": 1.0, "overlap": 0.5, "window": "hann"},
        "preprocessing": {
            "notch_hz": LINE_FREQUENCY_HZ,
            "bandpass_hz": list(FILTER_BAND_HZ),
            "reference": "average EEG reference before selecting C3/Cz/C4",
            "artifact_threshold_uv": ARTIFACT_THRESHOLD_UV,
        },
        "erd_formula": "100 * (power_baseline - power_mi) / power_baseline",
        "artifact_trials": artifact_count,
        "trials_total": all_trial_count,
        "trials_analyzed": int(len(valid_trials)),
        "artifacts_included_in_statistics": include_artifacts,
        "source": metadata,
        "overall": paired_statistics(valid_trials),
        "by_class": {
            class_name: paired_statistics(group)
            for class_name, group in valid_trials.groupby("class")
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_dir, report, figure


def main() -> None:
    args = parse_args()
    subject_dir = args.dataset_root / args.subject
    channels, trials, metadata = extract_subject_trials(subject_dir)
    output_dir, report, figure = save_results(
        args.subject, args.output_root, channels, trials, metadata, args.include_artifacts
    )
    overall = report["overall"]
    print("\nAnálisis finalizado")
    print(f"Trials: {report['trials_analyzed']}/{report['trials_total']}")
    print(f"Theta/Beta promedio: baseline={overall['theta_beta_baseline_mean']:.4f}, "
          f"MI={overall['theta_beta_mi_mean']:.4f}")
    print(f"Cambio Theta/Beta mediano: {overall['theta_beta_change_pct_median']:.2f} %")
    print(f"ERD mu mediano: {overall['mu_erd_pct_median']:.2f} %")
    print(f"ERD beta mediano: {overall['beta_erd_pct_median']:.2f} %")
    print(f"Resultados: {output_dir.resolve()}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
