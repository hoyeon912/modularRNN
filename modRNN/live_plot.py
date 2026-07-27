import matplotlib.pyplot as plt

_FIXED_YLIM = {"accuracy": (0, 1), "reward": (0, 500)}
_COLOR = {"accuracy": "tab:green", "reward": "tab:orange"}


class LiveTrainingPlot:
    """Live-updating metrics window. Disables itself (no crash) if no GUI backend is available."""

    def __init__(self, title: str, metrics: tuple[str, ...] = ("loss", "accuracy")):
        self.enabled = True
        self.metrics = list(metrics)
        self.epochs = []
        self.history = {metric: [] for metric in self.metrics}
        try:
            plt.ion()
            self.fig, axes = plt.subplots(1, len(self.metrics), figsize=(5 * len(self.metrics), 4))
            self.fig.suptitle(title)
            if len(self.metrics) == 1:
                axes = [axes]

            self.axes = {}
            self.lines = {}
            for ax, metric in zip(axes, self.metrics):
                ax.set_xlabel("epoch")
                ax.set_ylabel(metric)
                if metric in _FIXED_YLIM:
                    ax.set_ylim(*_FIXED_YLIM[metric])
                (line,) = ax.plot([], [], marker="o", color=_COLOR.get(metric))
                self.axes[metric] = ax
                self.lines[metric] = line

            self.fig.tight_layout()
            self.fig.canvas.draw()
            plt.pause(0.001)
        except Exception as e:
            self.enabled = False
            print(f"live plot disabled (no GUI backend available): {e}")

    def update(self, epoch: int, *values: float) -> None:
        if not self.enabled:
            return

        self.epochs.append(epoch)
        for metric, value in zip(self.metrics, values):
            self.history[metric].append(value)
            self.lines[metric].set_data(self.epochs, self.history[metric])
            ax = self.axes[metric]
            if metric in _FIXED_YLIM:
                ax.set_xlim(0.5, max(self.epochs) + 0.5)
            else:
                ax.relim()
                ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
