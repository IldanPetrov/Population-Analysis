import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Для GUI

class PlotServer:
    def __init__(self):
        self.queue = Queue()
        self.ack_queue = Queue()
        self.process = Process(target=self._worker, args=(self.queue,), daemon=True)
        self.process.start()

    def _worker(self, queue: Queue):
        import matplotlib.pyplot as plt

        figures = []
        plt.ion()

        while True:
            task = queue.get()
            if task == 'STOP':
                break

            df, columns, x_column, title = task

            columns = df.columns if columns is None else columns
            x = df[x_column] if x_column else df.index
            xlabel = x_column or (df.index.name or "Index")

            fig, ax = plt.subplots(figsize=(12, 6))
            for col in columns:
                ax.plot(x, df[col], label=col)
            ax.set_title(title or "Plot")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Value")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            fig.tight_layout()
            fig.show()
            figures.append(fig)
            plt.pause(0.01)
            self.ack_queue.put("done")  # подтверждение

        print("Plot server exiting.")
        plt.ioff()
        plt.show()

    def plot_df(self, df, columns=None, x_column=None, title=None):
        self.queue.put((df, columns, x_column, title))
        self.ack_queue.get()  # дождаться подтверждения отрисовки

    def close(self):
        self.queue.put("STOP")
        self.process.join()
