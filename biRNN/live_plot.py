import matplotlib.pyplot as plt


class LiveTrainingPlot:
    """Live-updating loss/accuracy window. Disables itself (no crash) if no GUI backend is available."""

    def __init__(self, title: str):
        self.enabled = True
        self.epochs = []
        self.losses = []
        self.accuracies = []
        try:
            plt.ion()
            self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
            self.fig.suptitle(title)

            self.ax_loss.set_xlabel("epoch")
            self.ax_loss.set_ylabel("loss")
            (self.loss_line,) = self.ax_loss.plot([], [], marker="o")

            self.ax_acc.set_xlabel("epoch")
            self.ax_acc.set_ylabel("accuracy")
            self.ax_acc.set_ylim(0, 1)
            (self.acc_line,) = self.ax_acc.plot([], [], marker="o", color="tab:green")

            self.fig.tight_layout()
            self.fig.canvas.draw()
            plt.pause(0.001)
        except Exception as e:
            self.enabled = False
            print(f"live plot disabled (no GUI backend available): {e}")

    def update(self, epoch: int, loss: float, accuracy: float) -> None:
        if not self.enabled:
            return

        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)

        self.loss_line.set_data(self.epochs, self.losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.acc_line.set_data(self.epochs, self.accuracies)
        self.ax_acc.set_xlim(0.5, max(self.epochs) + 0.5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
