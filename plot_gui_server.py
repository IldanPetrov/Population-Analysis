import tkinter as tk
from tkinter import ttk
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import pickle


def plot_gui_worker(conn):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import pickle

    class PlotApp:
        def __init__(self, master):
            self.master = master
            self.closed = False

            self.master.title("Plot Viewer")
            self.master.geometry("1000x700")

            self.figures = []
            self.names = []

            self.listbox = tk.Listbox(master, width=30)
            self.listbox.pack(side=tk.LEFT, fill=tk.Y)

            self.canvas_frame = tk.Frame(master)
            self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            self.canvas = None
            self.current_fig = None

            self.listbox.bind("<<ListboxSelect>>", self.on_select)

            self.after_id = self.master.after(100, self.check_pipe)
            self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        def check_pipe(self):
            if self.closed:
                return

            while conn.poll():
                task = conn.recv()
                if isinstance(task, dict) and task.get("cmd") == "STOP":
                    self.master.destroy()
                    return

                df = pickle.loads(task['df'])
                columns = task.get('columns') or df.columns
                x_column = task.get('x_column')
                title = task.get('title') or "Plot"
                figsize = task.get('figsize', (8, 6))

                x = df[x_column] if x_column else df.index

                fig, ax = plt.subplots(figsize=figsize)
                for col in columns:
                    ax.plot(x, df[col], label=col)
                ax.set_title(title)
                ax.set_xlabel(x_column or df.index.name or "Index")
                ax.grid(True)
                ax.legend()

                self.figures.append(fig)
                self.names.append(title)
                self.listbox.insert(tk.END, title)

                if len(self.figures) == 1:
                    self.display_figure(0)
                    self.listbox.select_set(0)
                else:
                    # !!! Показываем только что добавленный график
                    self.display_figure(len(self.figures) - 1)
                    self.listbox.select_clear(0, tk.END)
                    self.listbox.select_set(len(self.figures) - 1)
                    self.listbox.see(len(self.figures) - 1)

            self.after_id = self.master.after(100, self.check_pipe)

        def display_figure(self, index):
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                if hasattr(self, "toolbar"):
                    self.toolbar.destroy()
            self.current_fig = self.figures[index]
            self.canvas = FigureCanvasTkAgg(self.current_fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
            self.toolbar.update()
            self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        def on_select(self, event):
            selection = event.widget.curselection()
            if selection:
                self.display_figure(selection[0])

        def on_close(self):
            self.closed = True
            if hasattr(self, 'after_id'):
                self.master.after_cancel(self.after_id)
            self.master.destroy()
            self.master.quit()

    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()
    import sys
    sys.exit(0)


class PlotClient:
    def __init__(self, conn):
        self.conn = conn

    def plot_df(self, df, columns=None, x_column=None, title=None, figsize=(12, 6)):
        # сериализуем DataFrame
        self.conn.send({
            'df': pickle.dumps(df),  # !!!
            'columns': columns,
            'x_column': x_column,
            'title': title,
            'figsize': figsize
        })


def start_gui_plot_server():
    q = Queue()
    p = Process(target=plot_gui_worker, args=(q,), daemon=False)
    p.start()
    return q, p
