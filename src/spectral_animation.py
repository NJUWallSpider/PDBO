"""Interactive visualization of spectral modes and primal-dual dynamics."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.spectral import histogram_edges


class SpectralAnimation:
    def __init__(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        bins: int,
        mode_label: str = "exact spectrum",
    ) -> None:
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.bins = bins
        self.mode_label = mode_label
        self.bin_indices = None
        self.figure = None
        self.axes = None
        self.dual_axes = None
        self.z_axes = None
        self.objective_axes = None
        self.bars = None
        self.dual_bars = None
        self.z_bars = None
        self.lagrangian_line = None
        self.objective_line = None
        self.dual_edges = None
        self.z_edges = None
        self.dual = None
        self.z = None
        self.objective_sample_keys = []
        self.objective_iterations = []
        self.lagrangian_history = []
        self.objective_history = []
        self.plt = None
        self.next_button = None
        self.run_button = None
        self.step_requested = False
        self.run_continuous = False

    def mode_bin_means(self, primal: np.ndarray) -> np.ndarray:
        if self.bin_indices is None:
            raise RuntimeError("spectral animation has not been initialized")
        centered = np.asarray(primal, dtype=np.float64) - 0.5
        coefficients = centered @ self.eigenvectors
        mode_lengths = np.sqrt(np.mean(coefficients**2, axis=0))
        sums = np.bincount(
            self.bin_indices,
            weights=mode_lengths,
            minlength=self.bins,
        )
        counts = np.bincount(self.bin_indices, minlength=self.bins)
        return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    @staticmethod
    def _dual_histogram(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
        """Return stable histogram edges and counts, including constant data."""
        finite = np.asarray(values, dtype=np.float64).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return np.linspace(-0.5, 0.5, bins + 1), np.zeros(bins, dtype=np.int64)

        lower = float(np.min(finite))
        upper = float(np.max(finite))
        if lower == upper:
            padding = max(abs(lower) * 0.05, 1e-3)
            lower -= padding
            upper += padding
        else:
            padding = max((upper - lower) * 0.05, 1e-12)
            lower -= padding
            upper += padding
        edges = np.linspace(lower, upper, bins + 1, dtype=np.float64)
        counts, _ = np.histogram(finite, bins=edges)
        return edges, counts

    def _update_histogram(
        self,
        values: np.ndarray,
        edges_name: str,
        bars_name: str,
        axes_name: str,
    ) -> None:
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        edges = getattr(self, edges_name)
        if finite.size == 0:
            counts = np.zeros(self.bins, dtype=np.int64)
        else:
            # Keep the x-axis stable until a new value falls outside it.
            if (
                edges is None
                or np.min(finite) < edges[0]
                or np.max(finite) > edges[-1]
            ):
                edges, counts = self._dual_histogram(values, self.bins)
                setattr(self, edges_name, edges)
            else:
                counts, _ = np.histogram(finite, bins=edges)
        bars = getattr(self, bars_name)
        axes = getattr(self, axes_name)
        if bars is None:
            return
        if edges is None:
            edges = np.linspace(-0.5, 0.5, self.bins + 1)
            setattr(self, edges_name, edges)
        for bar, left, right, count in zip(
            bars,
            edges[:-1],
            edges[1:],
            counts,
        ):
            bar.set_x(float(left))
            bar.set_width(float(right - left))
            bar.set_height(float(count))
        axes.set_xlim(float(edges[0]), float(edges[-1]))
        required_upper = max(float(np.max(counts)) * 1.15, 1.0)
        current_upper = float(axes.get_ylim()[1])
        if required_upper > current_upper:
            axes.set_ylim(0.0, required_upper)

    def _update_dual_histogram(self, dual: np.ndarray) -> None:
        self._update_histogram(dual, "dual_edges", "dual_bars", "dual_axes")

    def _update_z_histogram(self, z: np.ndarray) -> None:
        self._update_histogram(z, "z_edges", "z_bars", "z_axes")

    def _update_objective_history(
        self,
        phase: int,
        phase_iteration: int,
        total_iteration: int,
        lagrangian_value: float,
        objective_value: float,
    ) -> None:
        key = (phase, phase_iteration, total_iteration)
        if self.objective_sample_keys and self.objective_sample_keys[-1] == key:
            self.lagrangian_history[-1] = float(lagrangian_value)
            self.objective_history[-1] = float(objective_value)
        else:
            self.objective_sample_keys.append(key)
            self.objective_iterations.append(total_iteration)
            self.lagrangian_history.append(float(lagrangian_value))
            self.objective_history.append(float(objective_value))

        if self.lagrangian_line is None or self.objective_line is None:
            return
        self.lagrangian_line.set_data(
            self.objective_iterations,
            self.lagrangian_history,
        )
        self.objective_line.set_data(
            self.objective_iterations,
            self.objective_history,
        )
        self.objective_axes.relim()
        self.objective_axes.autoscale_view()

    def request_step(self, _event=None) -> None:
        if not self.run_continuous:
            self.step_requested = True

    def toggle_run(self, _event=None) -> None:
        self.run_continuous = not self.run_continuous
        if self.run_button is not None:
            label = "Pause" if self.run_continuous else "Run"
            self.run_button.label.set_text(label)
        if self.figure is not None:
            self.figure.canvas.draw_idle()

    def wait_for_step(self) -> float:
        wait_start = time.perf_counter()
        while self.figure is not None and not self.run_continuous and not self.step_requested:
            if not self.plt.fignum_exists(self.figure.number):
                self.figure = None
                break
            self.plt.pause(0.05)
        self.step_requested = False
        return time.perf_counter() - wait_start

    def setup(self, primal: np.ndarray, dual: Optional[np.ndarray] = None) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button
        except ImportError as error:
            raise RuntimeError(
                "spectral animation requires matplotlib; install with "
                'pip install -e ".[visualization]"'
            ) from error

        edges = histogram_edges(
            float(self.eigenvalues[0]),
            float(self.eigenvalues[-1]),
            self.bins,
        )
        bin_indices = np.searchsorted(edges, self.eigenvalues, side="right") - 1
        self.bin_indices = np.clip(bin_indices, 0, self.bins - 1)

        plt.ion()
        figure, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
        spectral_axes = axes[0, 0]
        dual_axes = axes[0, 1]
        z_axes = axes[1, 0]
        objective_axes = axes[1, 1]
        figure.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.25)
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title("PDBO primal-dual dynamics")
        heights = self.mode_bin_means(primal)
        bars = spectral_axes.bar(edges[:-1], heights, width=np.diff(edges), align="edge")
        spectral_axes.set_xlim(float(edges[0]), float(edges[-1]))
        spectral_axes.set_ylim(0.0, max(float(np.max(heights)) * 1.15, 1e-12))
        spectral_axes.set_xlabel("Eigenvalue")
        spectral_axes.set_ylabel(
            "Mean projection length"
            if self.mode_label == "exact spectrum"
            else "Mean sampled projection length"
        )
        spectral_axes.set_title(self.mode_label)
        spectral_axes.grid(axis="y", alpha=0.25)

        self.dual = np.zeros_like(primal, dtype=np.float64) if dual is None else np.asarray(dual)
        self.dual_edges, dual_counts = self._dual_histogram(self.dual, self.bins)
        dual_bars = dual_axes.bar(
            self.dual_edges[:-1],
            dual_counts,
            width=np.diff(self.dual_edges),
            align="edge",
            color="tab:orange",
        )
        dual_axes.set_xlim(float(self.dual_edges[0]), float(self.dual_edges[-1]))
        dual_axes.set_ylim(0.0, max(float(np.max(dual_counts)) * 1.15, 1.0))
        dual_axes.set_xlabel("y")
        dual_axes.set_ylabel("Count")
        dual_axes.set_title("Dual variable distribution")
        dual_axes.grid(axis="y", alpha=0.25)

        self.z = np.asarray(primal, dtype=np.float64) - 0.5
        self.z_edges, z_counts = self._dual_histogram(self.z, self.bins)
        z_bars = z_axes.bar(
            self.z_edges[:-1],
            z_counts,
            width=np.diff(self.z_edges),
            align="edge",
            color="tab:green",
        )
        z_axes.set_xlim(float(self.z_edges[0]), float(self.z_edges[-1]))
        z_axes.set_ylim(0.0, max(float(np.max(z_counts)) * 1.15, 1.0))
        z_axes.set_xlabel("z = x - 0.5")
        z_axes.set_ylabel("Count")
        z_axes.set_title("z distribution")
        z_axes.grid(axis="y", alpha=0.25)

        lagrangian_line, = objective_axes.plot(
            [],
            [],
            color="tab:red",
            label="L(x, y)",
        )
        objective_line, = objective_axes.plot(
            [],
            [],
            color="tab:blue",
            label="f(x)",
        )
        objective_axes.set_xlabel("Iteration t")
        objective_axes.set_ylabel("Batch mean objective")
        objective_axes.set_title("Objective values")
        objective_axes.grid(alpha=0.25)
        objective_axes.legend()

        next_axes = figure.add_axes((0.76, 0.025, 0.1, 0.05))
        run_axes = figure.add_axes((0.88, 0.025, 0.1, 0.05))
        self.next_button = Button(next_axes, "Next step")
        self.run_button = Button(run_axes, "Run")
        self.next_button.on_clicked(self.request_step)
        self.run_button.on_clicked(self.toggle_run)

        self.figure = figure
        self.axes = spectral_axes
        self.dual_axes = dual_axes
        self.z_axes = z_axes
        self.objective_axes = objective_axes
        self.bars = bars
        self.dual_bars = dual_bars
        self.z_bars = z_bars
        self.lagrangian_line = lagrangian_line
        self.objective_line = objective_line
        self.plt = plt
        figure.show()
        figure.canvas.draw_idle()
        figure.canvas.flush_events()

    def update(
        self,
        primal: np.ndarray,
        phase: int,
        phases: int,
        phase_iteration: int,
        dual: Optional[np.ndarray] = None,
        total_iteration: Optional[int] = None,
        lagrangian_value: Optional[float] = None,
        objective_value: Optional[float] = None,
    ) -> None:
        if self.figure is None or not self.plt.fignum_exists(self.figure.number):
            self.figure = None
            return

        heights = self.mode_bin_means(primal)
        if dual is not None:
            self.dual = np.asarray(dual)
        if self.dual is not None:
            self._update_dual_histogram(self.dual)
        self.z = np.asarray(primal, dtype=np.float64) - 0.5
        self._update_z_histogram(self.z)
        for bar, height in zip(self.bars, heights):
            bar.set_height(float(height))

        current_upper = float(self.axes.get_ylim()[1])
        required_upper = max(float(np.max(heights)) * 1.15, 1e-12)
        if required_upper > current_upper:
            self.axes.set_ylim(0.0, required_upper)
        self.axes.set_title(
            f"round {phase}/{phases} | iteration {phase_iteration} | {self.mode_label}"
        )
        if self.dual_axes is not None:
            self.dual_axes.set_title(
                f"y distribution | round {phase}/{phases} | iteration {phase_iteration}"
            )
        if self.z_axes is not None:
            self.z_axes.set_title(
                f"z distribution | round {phase}/{phases} | iteration {phase_iteration}"
            )
        if (
            lagrangian_value is not None
            and objective_value is not None
        ):
            self._update_objective_history(
                phase,
                phase_iteration,
                phase_iteration if total_iteration is None else total_iteration,
                lagrangian_value,
                objective_value,
            )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.001)

    def finish(
        self,
        primal: np.ndarray,
        phase: int,
        phases: int,
        phase_iteration: int,
        dual: Optional[np.ndarray] = None,
        total_iteration: Optional[int] = None,
        lagrangian_value: Optional[float] = None,
        objective_value: Optional[float] = None,
    ) -> None:
        if self.figure is None:
            return
        self.update(
            primal,
            phase,
            phases,
            phase_iteration,
            dual=dual,
            total_iteration=total_iteration,
            lagrangian_value=lagrangian_value,
            objective_value=objective_value,
        )
        if self.figure is not None:
            self.plt.ioff()
            self.plt.show()
