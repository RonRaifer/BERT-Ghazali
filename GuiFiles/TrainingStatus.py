from Data import utils
from GuiFiles import ViewResults, CNNConfigurations

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True

import os.path
import sys
from concurrent import futures

# Create thread pool to execute long tasks.
thread_pool_executor = futures.ThreadPoolExecutor(max_workers=1)


def training_status_start():
    """
    Calls TrainingStatus_Screen class, and creates mainloop thread. It also overrides the default exit,
    to handle thread termination.
    """
    global root, top, original_stdout
    original_stdout = sys.stdout
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", exit_handler)
    top = TrainingStatus_Screen(root)
    root.mainloop()


def show_results_button_click():
    """
    Called on 'Show Results' button click.

         It destroys the current view, and brings the 'View Results' screen.
    """
    global root
    utils.log_content = top.output_Text.get('1.0', 'end')
    x.cancel()
    root.destroy()
    ViewResults.view_results_start("CNN")
    root = None


def proc_start():
    """
    Configures 'Start' button style after pressing on it, and sets the button to be disabled.
    """
    top.output_Text.insert('end', 'Starting...\n')
    top.start_training_button.configure(text='''Processing..''')
    top.start_training_button.configure(activebackground="#ececec")
    top.start_training_button.configure(activeforeground="#000000")
    top.start_training_button.configure(background="#c0c0c0")
    top.start_training_button.configure(disabledforeground="#a3a3a3")
    top.start_training_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
    top.start_training_button.configure(foreground="#ffffff")
    top.start_training_button.configure(highlightbackground="#d9d9d9")
    top.start_training_button.configure(highlightcolor="#000000")
    top.start_training_button.configure(pady="0")
    top.start_training_button.configure(relief="flat")
    top.start_training_button.configure(state='disabled')


def proc_end():
    """
    Configures 'View Results' button style when training finished, and sets the button to be enabled.
    """
    top.view_results_button.configure(activebackground="#ececec")
    top.view_results_button.configure(activeforeground="#000000")
    top.view_results_button.configure(background="#629b1c")
    top.view_results_button.configure(disabledforeground="#a3a3a3")
    top.view_results_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
    top.view_results_button.configure(foreground="#ffffff")
    top.view_results_button.configure(highlightbackground="#d9d9d9")
    top.view_results_button.configure(highlightcolor="#000000")
    top.view_results_button.configure(state='normal')
    top.start_training_button.configure(text='''DONE''')


def run_attributer():
    """
        Create the 'BERTGhazali_Attributer' object, and call it's 'run' function to start the process.
    """
    from Logic.Classification.bert_ghazali import BERTGhazali_Attributer
    utils.stopped = False
    gatt = BERTGhazali_Attributer(
        bert_model_name="aubmindlab/bert-large-arabertv2",
        text_division_method=utils.params['TEXT_DIVISION_METHOD'],
        text_console=top.output_Text)
    gatt.run()
    proc_end()


def start_training_click():
    """
    Called on 'Start' button click.

         It updates the buttons style and state, and submits the thread function 'run_attributer'.
    """
    global x
    proc_start()
    utils.stopped = False
    utils.progress_bar = top.progress_bar
    x = thread_pool_executor.submit(run_attributer)


def exit_handler():
    """
        Called on 'exit window' event.

            It terminates the thread, if exists, and destroys the window, and exits the program.
    """
    try:
        thread_pool_executor.shutdown(wait=False)
    except NameError:
        pass
    root.destroy()


def back_button_click():
    """
        Called on 'Back' button click.

            It revert the stdout to its original, cancel the thread is exists.
            Then it destroys the current view, and brings the 'CNN Configurations' screen.
    """
    global root, original_stdout, x
    try:
        utils.stopped = True
        x.cancel()
        sys.stdout = original_stdout
    except NameError:
        pass
    root.destroy()
    CNNConfigurations.cnn_configurations_start()
    root = None


class TrainingStatus_Screen:
    def __init__(self, top=None):
        """
            This class configures and populates the 'Training Status' window.
            top is the toplevel containing window.

            In this screen, the user can start the training process, and follow the progress.
            After the training ends, the user will be able to watch the obtained results by clicking on 'View Results'.
        """

        w = 886
        h = 363
        ws = top.winfo_screenwidth()
        hs = top.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        top.geometry('%dx%d+%d+%d' % (w, h, x, y))
        top.resizable(False, False)
        top.title("Al-Ghazali's Authorship Attribution")
        top.configure(background="#ffffff")

        self.TSeparator1 = ttk.Separator(top)
        self.TSeparator1.place(x=380, y=10, height=50)
        self.TSeparator1.configure(orient="vertical")

        self.main_ghazali_label = tk.Label(top)
        self.main_ghazali_label.place(x=80, y=20, height=27, width=245)
        self.main_ghazali_label.configure(activebackground="#f9f9f9")
        self.main_ghazali_label.configure(activeforeground="black")
        self.main_ghazali_label.configure(anchor='nw')
        self.main_ghazali_label.configure(background="#ffffff")
        self.main_ghazali_label.configure(disabledforeground="#a3a3a3")
        self.main_ghazali_label.configure(font="-family {Segoe UI} -size 14 -weight bold")
        self.main_ghazali_label.configure(foreground="#629b1c")
        self.main_ghazali_label.configure(highlightbackground="#d9d9d9")
        self.main_ghazali_label.configure(highlightcolor="black")
        self.main_ghazali_label.configure(text='''Progress''')

        self.TLabel1 = ttk.Label(top)
        self.TLabel1.place(x=10, y=5, height=60, width=70)
        self.TLabel1.configure(background="#ffffff")
        self.TLabel1.configure(foreground="#000000")
        self.TLabel1.configure(font="TkDefaultFont")
        self.TLabel1.configure(relief="flat")
        self.TLabel1.configure(anchor='center')
        self.TLabel1.configure(justify='center')
        photo_location = os.path.join("GuiFiles/Al-Ghazali-Top.png")
        global _img0
        _img0 = tk.PhotoImage(file=photo_location)
        self.TLabel1.configure(image=_img0)

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(x=0, y=300, height=65, width=885)
        self.Frame1.configure(background="#eeeeee")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label2 = tk.Label(top)
        self.Label2.place(x=390, y=23, height=21, width=324)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#ffffff")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font="-family {Segoe UI} -size 10")
        self.Label2.configure(foreground="#9d9d9d")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Here you can see the training progress''')

        self.start_training_button = tk.Button(top, command=start_training_click)
        self.start_training_button.place(x=340, y=315, height=33, width=188)
        self.start_training_button.configure(activebackground="#ececec")
        self.start_training_button.configure(activeforeground="#000000")
        self.start_training_button.configure(background="#629b1c")
        self.start_training_button.configure(disabledforeground="#a3a3a3")
        self.start_training_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.start_training_button.configure(foreground="#ffffff")
        self.start_training_button.configure(highlightbackground="#d9d9d9")
        self.start_training_button.configure(highlightcolor="#000000")
        self.start_training_button.configure(pady="0")
        self.start_training_button.configure(text='''Start''')

        self.view_results_button = tk.Button(top, command=show_results_button_click)
        self.view_results_button.place(x=680, y=315, height=33, width=188)
        self.view_results_button.configure(activebackground="#ececec")
        self.view_results_button.configure(activeforeground="#000000")
        self.view_results_button.configure(background="#c0c0c0")
        self.view_results_button.configure(disabledforeground="#a3a3a3")
        self.view_results_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.view_results_button.configure(foreground="#ffffff")
        self.view_results_button.configure(highlightbackground="#d9d9d9")
        self.view_results_button.configure(highlightcolor="#000000")
        self.view_results_button.configure(pady="0")
        self.view_results_button.configure(relief="flat")
        self.view_results_button.configure(state='disabled')
        self.view_results_button.configure(text='''View Results''')

        self.TSeparator2 = ttk.Separator(top)
        self.TSeparator2.place(x=20, y=72, width=840)

        self.output_Text = tk.Text(top)
        self.output_Text.place(x=40, y=140, height=144, width=804)
        self.output_Text.configure(background="white")
        self.output_Text.configure(font="TkTextFont")
        self.output_Text.configure(foreground="black")
        self.output_Text.configure(highlightbackground="#d9d9d9")
        self.output_Text.configure(highlightcolor="black")
        self.output_Text.configure(insertbackground="black")
        self.output_Text.configure(selectbackground="blue")
        self.output_Text.configure(selectforeground="white")
        self.output_Text.configure(wrap="word")

        self.progress_bar = ttk.Progressbar(top)
        self.progress_bar.place(x=240, y=97, width=400, height=22)
        self.progress_bar.configure(length="400")

        self.back_button = tk.Button(top, command=back_button_click)
        self.back_button.place(x=20, y=315, height=33, width=188)
        self.back_button.configure(activebackground="#ececec")
        self.back_button.configure(activeforeground="#000000")
        self.back_button.configure(background="#a5b388")
        self.back_button.configure(disabledforeground="#a3a3a3")
        self.back_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.back_button.configure(foreground="#ffffff")
        self.back_button.configure(highlightbackground="#d9d9d9")
        self.back_button.configure(highlightcolor="#000000")
        self.back_button.configure(pady="0")
        self.back_button.configure(relief="flat")
        self.back_button.configure(text='''Back''')

        utils.log_content = None
