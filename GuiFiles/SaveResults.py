from Logic.Analyze import Analyzer
from Data import utils

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


def save_results_start(view_results_window):
    """
        Call SaveResults_Screen class, and creates mainloop thread.
        Also, it overrides the default exit button, in order to bring back the previous screen,
        instead of exiting the whole program.

        Params:
            view_results_window (`tk.Tk()`):
                A tkinter window to be appeared after exiting the window.
    """
    global root, top, view_results
    view_results = view_results_window
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", cancel_button_click)
    top = SaveResults_Screen(root)
    root.mainloop()


def save_button_click():
    """
        Called when clicking on 'Save' button.

        The func first checks if the name already exists, and if so, an error label with proper message will be shown.
        If name does not exists, the model will be added to the Json file, and a Log file with
        Numpy matrix will be saved with the same name the user chose.
    """
    global top
    utils.params['Name'] = top.results_name_value.get()
    database = Analyzer.read_json()
    for p in database:
        if p['Name'] == utils.params['Name']:
            top.error_label.configure(text='''Woops, Name already exists...''')
            return
    Analyzer.save_results()
    top.error_label.configure(text='''Saved!''')
    top.save_button.configure(state='disabled')
    top.cancel_button.configure(text='''Done''')


def cancel_button_click():
    """
        Called when clicking on 'Cancel' button.

            This destroys the current window, and retrieves back the 'View Results' screen.
    """
    global root, view_results
    view_results.deiconify()
    root.destroy()
    root = None


class SaveResults_Screen:
    def __init__(self, top=None):
        """
            This class configures and populates the Save Results window.
            top is the toplevel containing window.

            In this screen, we configure two buttons: 'Save' and 'Cancel'.
            This screen purpose is allowing the user to save the results of the model just trained.
        """

        w = 495
        h = 246
        ws = top.winfo_screenwidth()
        hs = top.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        top.geometry('%dx%d+%d+%d' % (w, h, x, y))
        top.resizable(False, False)
        top.title("Al-Ghazali's Authorship Attribution")
        top.configure(background="#ffffff")

        self.TSeparator1 = ttk.Separator(top)
        self.TSeparator1.place(x=253, y=10, height=50)
        self.TSeparator1.configure(orient="vertical")

        self.main_ghazali_label = tk.Label(top)
        self.main_ghazali_label.place(x=70, y=20, height=27, width=145)
        self.main_ghazali_label.configure(activebackground="#f9f9f9")
        self.main_ghazali_label.configure(activeforeground="black")
        self.main_ghazali_label.configure(background="#ffffff")
        self.main_ghazali_label.configure(disabledforeground="#a3a3a3")
        self.main_ghazali_label.configure(font="-family {Segoe UI} -size 14 -weight bold")
        self.main_ghazali_label.configure(foreground="#629b1c")
        self.main_ghazali_label.configure(highlightbackground="#d9d9d9")
        self.main_ghazali_label.configure(highlightcolor="black")
        self.main_ghazali_label.configure(text='''Save Results''')

        self.TLabel1 = ttk.Label(top)
        self.TLabel1.place(x=10, y=5, height=60, width=70)
        self.TLabel1.configure(background="#ffffff")
        self.TLabel1.configure(foreground="#000000")
        self.TLabel1.configure(font="TkDefaultFont")
        self.TLabel1.configure(relief="flat")
        self.TLabel1.configure(anchor='center')
        self.TLabel1.configure(justify='center')

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(x=0, y=190, height=55, width=495)
        self.Frame1.configure(background="#eeeeee")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.save_button = tk.Button(self.Frame1, command=save_button_click)
        self.save_button.place(x=350, y=10, height=33, width=120)
        self.save_button.configure(activebackground="#ececec")
        self.save_button.configure(activeforeground="#000000")
        self.save_button.configure(background="#629b1c")
        self.save_button.configure(cursor="fleur")
        self.save_button.configure(disabledforeground="#a3a3a3")
        self.save_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.save_button.configure(foreground="#ffffff")
        self.save_button.configure(highlightbackground="#d9d9d9")
        self.save_button.configure(highlightcolor="#000000")
        self.save_button.configure(pady="0")
        self.save_button.configure(text='''Save''')

        self.Label2 = tk.Label(top)
        self.Label2.place(x=260, y=25, height=21, width=224)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#ffffff")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font="-family {Segoe UI} -size 10")
        self.Label2.configure(foreground="#9d9d9d")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Save obtained results for future use''')

        self.cancel_button = tk.Button(top, command=cancel_button_click)
        self.cancel_button.place(x=20, y=200, height=33, width=120)
        self.cancel_button.configure(activebackground="#ececec")
        self.cancel_button.configure(activeforeground="#000000")
        self.cancel_button.configure(background="#a5b388")
        self.cancel_button.configure(disabledforeground="#a3a3a3")
        self.cancel_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.cancel_button.configure(foreground="#ffffff")
        self.cancel_button.configure(highlightbackground="#d9d9d9")
        self.cancel_button.configure(highlightcolor="#000000")
        self.cancel_button.configure(pady="0")
        self.cancel_button.configure(relief="flat")
        self.cancel_button.configure(text='''Cancel''')

        self.TSeparator2 = ttk.Separator(top)
        self.TSeparator2.place(x=20, y=72, width=470)
        self.TSeparator2.configure(cursor="fleur")

        self.error_label = tk.Label(top)
        self.error_label.place(x=150, y=150, height=26, width=221)
        self.error_label.configure(activebackground="#f9f9f9")
        self.error_label.configure(activeforeground="black")
        self.error_label.configure(anchor='nw')
        self.error_label.configure(background="#ffffff")
        self.error_label.configure(disabledforeground="#a3a3a3")
        self.error_label.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.error_label.configure(foreground="#ff0000")
        self.error_label.configure(highlightbackground="#d9d9d9")
        self.error_label.configure(highlightcolor="black")
        self.error_label.configure(text='''''')

        self.Label1_5_2 = tk.Label(top)
        self.Label1_5_2.place(x=40, y=110, height=26, width=141)
        self.Label1_5_2.configure(activebackground="#f9f9f9")
        self.Label1_5_2.configure(activeforeground="black")
        self.Label1_5_2.configure(anchor='nw')
        self.Label1_5_2.configure(background="#ffffff")
        self.Label1_5_2.configure(disabledforeground="#a3a3a3")
        self.Label1_5_2.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5_2.configure(foreground="#525252")
        self.Label1_5_2.configure(highlightbackground="#d9d9d9")
        self.Label1_5_2.configure(highlightcolor="black")
        self.Label1_5_2.configure(text='''Name Your Results:''')

        self.results_name_value = tk.Entry(top)
        self.results_name_value.place(x=180, y=110, height=30, width=264)
        self.results_name_value.configure(background="white")
        self.results_name_value.configure(cursor="fleur")
        self.results_name_value.configure(disabledforeground="#a3a3a3")
        self.results_name_value.configure(font="TkFixedFont")
        self.results_name_value.configure(foreground="#000000")
        self.results_name_value.configure(insertbackground="black")

