import pathlib
from GuiFiles import HomeScreen, CNNConfigurations, gui_helper
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

import os.path

root = None
top = None


def general_configurations_start():
    """
       Calls GeneralConfigurations_Screen class, updates parameters, and creates mainloop thread.
    """
    global root, top
    root = tk.Tk()
    top = GeneralConfigurations_Screen(root)
    read_params_values()
    root.mainloop()


def back_button_click():
    """
    Called on 'Back' button click.

         It destroys the current view, and brings the 'Home' screen.
    """
    global root
    root.destroy()
    HomeScreen.home_screen_start()
    root = None


def next_button_click():
    """
    Called on 'Next' button click.

         Checks if parameters inserted are valid, raises an error if not. If valid, updates the parameters in utils.
         Checks if embeddings chosen are already exists, if not, show an info message indicates it will take time to produce.
         Then it destroys the current view, and brings the 'CNN Configurations' screen.

    """
    global root
    msg = validate_fields_values()
    if msg != "":
        from tkinter import messagebox as mb
        mb.showerror("Errors", msg)
        return

    update_params()
    embeddingsFile = "FS" if utils.params['TEXT_DIVISION_METHOD'] == "Fixed-Size" else "BU"
    embeddingsFile += str(utils.params['BERT_INPUT_LENGTH'])
    p = pathlib.Path(os.getcwd() + r"\Data\PreviousRuns\Embeddings\\" + embeddingsFile + ".zip")
    if not os.path.exists(p):  # embedding do not exist
        # raise pop up to user it's going to take a while.
        from tkinter import messagebox as mb
        res = mb.askyesno("Notice", "The word embedding configurations you chose does not exist in the system, "
                                    "producing word embedding might take a while.\nAre you sure you want to continue?")
        if res is True:
            root.destroy()
            CNNConfigurations.cnn_configurations_start()
            root = None
    else:
        root.destroy()
        CNNConfigurations.cnn_configurations_start()
        root = None


def load_defaults_click():
    """
    Called on 'Load Defaults' button click.

         Populates the entries with default values from utils.
    """
    utils.LoadDefaultGeneralConfig()
    top.reset_entries()
    read_params_values()


def read_params_values():
    """
    Populates entries with the parameters (params) from utils.
    """
    top.niter_value.delete(0, tk.END)
    top.niter_value.insert(0, utils.params['Niter'])

    top.acc_thresh_value.delete(0, tk.END)
    top.acc_thresh_value.insert(0, utils.params['ACCURACY_THRESHOLD'])

    top.bert_input_length_value.delete(0, tk.END)
    top.bert_input_length_value.insert(0, utils.params['BERT_INPUT_LENGTH'])

    top.f1_value.delete(0, tk.END)
    top.f1_value.insert(0, utils.params['F1'])

    top.f_value.delete(0, tk.END)
    top.f_value.insert(0, utils.params['F'])
    if utils.params['TEXT_DIVISION_METHOD'] == 'Fixed-Size':
        top.division_method_value.current(0)
    else:
        top.division_method_value.current(1)

    top.silhouette_thresh_value.delete(0, tk.END)
    top.silhouette_thresh_value.insert(0, utils.params['SILHOUETTE_THRESHOLD'])


def update_params():
    """
    Updates the params in utils, with the parameters from the entries.
    """
    utils.params['Niter'] = int(top.niter_value.get())
    utils.params['ACCURACY_THRESHOLD'] = float(top.acc_thresh_value.get())
    utils.params['BERT_INPUT_LENGTH'] = int(top.bert_input_length_value.get())
    utils.params['F1'] = float(top.f1_value.get())
    utils.params['SILHOUETTE_THRESHOLD'] = float(top.silhouette_thresh_value.get())
    utils.params['TEXT_DIVISION_METHOD'] = top.division_method_value.get()
    utils.params['F'] = top.f_value.get()


def validate_fields_values():
    """
    Validates the inputs from the entries, and returns message with errors.
    """
    msg = ""
    if not gui_helper.isint_and_inrange(top.niter_value.get(), 1, 99):
        msg += "Niter must be an integer in range [1,99]\n"
        top.niter_value.configure(highlightbackground="red", highlightcolor="red")
    if not gui_helper.isfloat_and_inrange(top.acc_thresh_value.get(), 0, 1):
        msg += "Accuracy Threshold must be float in range (0,1)\n"
        top.acc_thresh_value.configure(highlightbackground="red", highlightcolor="red")
    if not gui_helper.isfloat_and_inrange(top.f1_value.get(), 0, 1):
        msg += "F1 must be a float in range (0,1)\n"
        top.f1_value.configure(highlightbackground="red", highlightcolor="red")
    if not gui_helper.isint_and_inrange(top.bert_input_length_value.get(), 20, 510):
        msg += "Bert input length must be an integer in range [20,510]\n"
        top.bert_input_length_value.configure(highlightbackground="red", highlightcolor="red")
    if not gui_helper.isfloat_and_inrange(top.silhouette_thresh_value.get(), 0, 1):
        msg += "Silhouette threshold must be a float in range (0,1)\n"
        top.silhouette_thresh_value.configure(highlightbackground="red", highlightcolor="red")
    return msg


class GeneralConfigurations_Screen:
    def __init__(self, top=None):
        """
            This class configures and populates the General Configurations window.
            top is the toplevel containing window.

            In this screen, the user can set his own parameters to be trained later. Validation step will make sure
            the fields were filled correctly, and raise errors if needed.
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
        self.gh = gui_helper
        self.TSeparator1 = ttk.Separator(top)
        self.TSeparator1.place(x=380, y=10, height=50)
        self.TSeparator1.configure(orient="vertical")
        self.TSeparator1.configure(cursor="fleur")

        self.main_ghazali_label = tk.Label(top)
        self.main_ghazali_label.place(x=70, y=20, height=27, width=245)
        self.main_ghazali_label.configure(activebackground="#f9f9f9")
        self.main_ghazali_label.configure(activeforeground="black")
        self.main_ghazali_label.configure(background="#ffffff")
        self.main_ghazali_label.configure(disabledforeground="#a3a3a3")
        self.main_ghazali_label.configure(font="-family {Segoe UI} -size 14 -weight bold")
        self.main_ghazali_label.configure(foreground="#629b1c")
        self.main_ghazali_label.configure(highlightbackground="#d9d9d9")
        self.main_ghazali_label.configure(highlightcolor="black")
        self.main_ghazali_label.configure(text='''General Configurations''')

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
        self.Label2.configure(text='''Choose your preferred parameters, or load defaults.''')

        self.next_button = tk.Button(top, command=next_button_click)
        self.next_button.place(x=680, y=315, height=33, width=188)
        self.next_button.configure(activebackground="#ececec")
        self.next_button.configure(activeforeground="#000000")
        self.next_button.configure(background="#629b1c")
        self.next_button.configure(disabledforeground="#a3a3a3")
        self.next_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.next_button.configure(foreground="#ffffff")
        self.next_button.configure(highlightbackground="#d9d9d9")
        self.next_button.configure(highlightcolor="#000000")
        self.next_button.configure(pady="0")
        self.next_button.configure(text='''Next''')

        self.load_defaults_button = tk.Button(top, command=load_defaults_click)
        self.load_defaults_button.place(x=360, y=315, height=33, width=188)
        self.load_defaults_button.configure(activebackground="#ececec")
        self.load_defaults_button.configure(activeforeground="#000000")
        self.load_defaults_button.configure(background="#a5b388")
        self.load_defaults_button.configure(disabledforeground="#a3a3a3")
        self.load_defaults_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.load_defaults_button.configure(foreground="#ffffff")
        self.load_defaults_button.configure(highlightbackground="#d9d9d9")
        self.load_defaults_button.configure(highlightcolor="#000000")
        self.load_defaults_button.configure(pady="0")
        self.load_defaults_button.configure(relief="flat")
        self.load_defaults_button.configure(text='''Load Defaults''')
        self.gh.tooltip_message(self.load_defaults_button, "Load the suggested values")

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

        self.TSeparator2 = ttk.Separator(top)
        self.TSeparator2.place(x=20, y=72, width=840)

        self.niter_value = tk.Entry(top)
        self.niter_value.place(x=40, y=120, height=24, width=204)
        self.gh.entry_widget_defaults(self.niter_value)
        self.gh.tooltip_message(self.niter_value, "Number of iterations. Integer in range [1,99]")

        self.acc_thresh_value = tk.Entry(top)
        self.acc_thresh_value.place(x=340, y=120, height=24, width=204)
        self.gh.entry_widget_defaults(self.acc_thresh_value)
        self.gh.tooltip_message(self.acc_thresh_value, "The accuracy of the net. Float in range (0,1)")

        self.bert_input_length_value = tk.Entry(top)
        self.bert_input_length_value.place(x=640, y=120, height=24, width=204)
        self.gh.entry_widget_defaults(self.bert_input_length_value)
        self.gh.tooltip_message(self.bert_input_length_value, "The length of the input. Integer in range [20,510]")

        self.f1_value = tk.Entry(top)
        self.f1_value.place(x=40, y=180, height=24, width=204)
        self.gh.entry_widget_defaults(self.f1_value)
        self.gh.tooltip_message(self.f1_value, "The balancing ratio for majority. Float in range (0,1)")

        self.silhouette_thresh_value = tk.Entry(top)
        self.silhouette_thresh_value.place(x=340, y=180, height=24, width=204)
        self.gh.entry_widget_defaults(self.silhouette_thresh_value)
        self.gh.tooltip_message(self.silhouette_thresh_value, "The classification measurement value. Float in range ("
                                                              "0,1)")

        self.f_value = tk.Entry(top)
        self.f_value.place(x=40, y=240, height=24, width=204)
        self.gh.entry_widget_defaults(self.f_value)
        self.gh.tooltip_message(self.f_value, "Oversampling method. Defaults to minority")

        self.division_method_value = ttk.Combobox(top)
        # Adding combobox drop down list
        self.division_method_value['values'] = ('Fixed-Size', 'Bottom-Up')
        self.division_method_value.current(0)
        self.division_method_value.place(x=640, y=180, height=24, width=204)
        self.division_method_value.configure(font="-family {Segoe UI} -size 11")
        self.division_method_value.configure(foreground="#525252")
        self.division_method_value.configure(takefocus="")
        self.gh.tooltip_message(self.division_method_value, "The text division method")

        self.niter_label = tk.Label(top)
        self.niter_label.place(x=37, y=94, height=26, width=191)
        self.niter_label.configure(background="#ffffff")
        self.niter_label.configure(cursor="fleur")
        self.niter_label.configure(disabledforeground="#a3a3a3")
        self.niter_label.configure(font="-family {Segoe UI} -size 11")
        self.niter_label.configure(foreground="#525252")
        self.niter_label.configure(text='''Niter: (number of iterations)''')

        self.acc_thresh_label = tk.Label(top)
        self.acc_thresh_label.place(x=338, y=94, height=26, width=191)
        self.acc_thresh_label.configure(activebackground="#f9f9f9")
        self.acc_thresh_label.configure(activeforeground="black")
        self.acc_thresh_label.configure(anchor='nw')
        self.acc_thresh_label.configure(background="#ffffff")
        self.acc_thresh_label.configure(disabledforeground="#a3a3a3")
        self.acc_thresh_label.configure(font="-family {Segoe UI} -size 11")
        self.acc_thresh_label.configure(foreground="#525252")
        self.acc_thresh_label.configure(highlightbackground="#d9d9d9")
        self.acc_thresh_label.configure(highlightcolor="black")
        self.acc_thresh_label.configure(text='''Accuracy Threshold:''')

        self.bert_input_label = tk.Label(top)
        self.bert_input_label.place(x=639, y=94, height=26, width=191)
        self.bert_input_label.configure(activebackground="#f9f9f9")
        self.bert_input_label.configure(activeforeground="black")
        self.bert_input_label.configure(anchor='nw')
        self.bert_input_label.configure(background="#ffffff")
        self.bert_input_label.configure(disabledforeground="#a3a3a3")
        self.bert_input_label.configure(font="-family {Segoe UI} -size 11")
        self.bert_input_label.configure(foreground="#525252")
        self.bert_input_label.configure(highlightbackground="#d9d9d9")
        self.bert_input_label.configure(highlightcolor="black")
        self.bert_input_label.configure(text='''BERT Input Length:''')

        self.f1_label = tk.Label(top)
        self.f1_label.place(x=37, y=154, height=26, width=191)
        self.f1_label.configure(activebackground="#f9f9f9")
        self.f1_label.configure(activeforeground="black")
        self.f1_label.configure(anchor='nw')
        self.f1_label.configure(background="#ffffff")
        self.f1_label.configure(disabledforeground="#a3a3a3")
        self.f1_label.configure(font="-family {Segoe UI} -size 11")
        self.f1_label.configure(foreground="#525252")
        self.f1_label.configure(highlightbackground="#d9d9d9")
        self.f1_label.configure(highlightcolor="black")
        self.f1_label.configure(text='''F1: (undersampling rate)''')

        self.f_label = tk.Label(top)
        self.f_label.place(x=39, y=214, height=26, width=191)
        self.f_label.configure(activebackground="#f9f9f9")
        self.f_label.configure(activeforeground="black")
        self.f_label.configure(anchor='nw')
        self.f_label.configure(background="#ffffff")
        self.f_label.configure(disabledforeground="#a3a3a3")
        self.f_label.configure(font="-family {Segoe UI} -size 11")
        self.f_label.configure(foreground="#525252")
        self.f_label.configure(highlightbackground="#d9d9d9")
        self.f_label.configure(highlightcolor="black")
        self.f_label.configure(text='''F: (oversampling rate)''')

        self.silhouette_label = tk.Label(top)
        self.silhouette_label.place(x=338, y=154, height=26, width=191)
        self.silhouette_label.configure(activebackground="#f9f9f9")
        self.silhouette_label.configure(activeforeground="black")
        self.silhouette_label.configure(anchor='nw')
        self.silhouette_label.configure(background="#ffffff")
        self.silhouette_label.configure(disabledforeground="#a3a3a3")
        self.silhouette_label.configure(font="-family {Segoe UI} -size 11")
        self.silhouette_label.configure(foreground="#525252")
        self.silhouette_label.configure(highlightbackground="#d9d9d9")
        self.silhouette_label.configure(highlightcolor="black")
        self.silhouette_label.configure(text='''Silhouette Threshold:''')

        self.division_method_label = tk.Label(top)
        self.division_method_label.place(x=639, y=154, height=26, width=191)
        self.division_method_label.configure(activebackground="#f9f9f9")
        self.division_method_label.configure(activeforeground="black")
        self.division_method_label.configure(anchor='nw')
        self.division_method_label.configure(background="#ffffff")
        self.division_method_label.configure(disabledforeground="#a3a3a3")
        self.division_method_label.configure(font="-family {Segoe UI} -size 11")
        self.division_method_label.configure(foreground="#525252")
        self.division_method_label.configure(highlightbackground="#d9d9d9")
        self.division_method_label.configure(highlightcolor="black")
        self.division_method_label.configure(text='''Text Division Method:''')

    def reset_entries(self):
        """
        Resets entries to the default view.
        """
        self.niter_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        self.acc_thresh_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        self.f1_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        self.bert_input_length_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        self.silhouette_thresh_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
