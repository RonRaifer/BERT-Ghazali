import sys
import os.path
from GuiFiles import GeneralConfigurations, TrainingStatus, gui_helper
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


def cnn_configurations_start():
    """
       Calls CNNConfigurations_Screen class, updates parameters, and creates mainloop thread.
    """
    global root, top
    root = tk.Tk()
    top = CNNConfigurations_Screen(root)
    read_params_values()
    root.mainloop()


def back_button_click():
    """
    Called on 'Back' button click.

         It destroys the current view, and brings the 'General Configurations' screen.
    """
    global root
    root.destroy()
    GeneralConfigurations.general_configurations_start()
    root = None


def next_button_click():
    """
    Called on 'Next' button click.

         Checks if parameters inserted are valid, raises an error if not. If valid, updates the parameters in utils.
         Then it destroys the current view, and brings the 'TrainingStatus' screen.

    """
    global root
    msg = validate_fields_values()
    if msg != "":
        from tkinter import messagebox as mb
        mb.showerror("Errors", msg)
        return
    update_params()
    root.destroy()
    TrainingStatus.training_status_start()
    root = None


def load_defaults_click():
    """
    Called on 'Load Defaults' button click.

         Populates the entries with default values from utils.
    """
    utils.LoadDefaultCNNConfig()
    top.reset_entries()
    read_params_values()


def read_params_values():
    """
    Populates entries with the parameters (params) from utils.
    """
    global top

    top.kernels_num_value.delete(0, tk.END)
    top.kernels_num_value.insert(0, utils.params['KERNELS'])

    top.batch_size_value.delete(0, tk.END)
    top.batch_size_value.insert(0, utils.params['BATCH_SIZE'])

    conv_kernels = ""
    for val in utils.params['1D_CONV_KERNEL'].values():
        conv_kernels += str(val)
        conv_kernels += ","
    conv_kernels = conv_kernels[:-1]

    top.conv_sizes_value.delete(0, tk.END)
    top.conv_sizes_value.insert(0, conv_kernels)

    top.dropout_value.delete(0, tk.END)
    top.dropout_value.insert(0, utils.params['DROPOUT_RATE'])

    top.epochs_value.delete(0, tk.END)
    top.epochs_value.insert(0, utils.params['NB_EPOCHS'])

    top.learning_rate_value.delete(0, tk.END)
    top.learning_rate_value.insert(0, utils.params['LEARNING_RATE'])

    top.output_size_value.delete(0, tk.END)
    top.output_size_value.insert(0, utils.params['OUTPUT_CLASSES'])

    top.strides_value.delete(0, tk.END)
    top.strides_value.insert(0, utils.params['STRIDES'])

    if utils.params['ACTIVATION_FUNC'] == "Relu":
        top.activation_func_value.current(0)
    else:
        top.activation_func_value.current(1)


def update_params():
    """
    Updates the params in utils, with the parameters from the entries.
    """
    global top
    utils.params['KERNELS'] = int(top.kernels_num_value.get())
    utils.params['LEARNING_RATE'] = float(top.learning_rate_value.get())
    utils.params['NB_EPOCHS'] = int(top.epochs_value.get())
    conv_kernels = str(top.conv_sizes_value.get()).split(",")
    utils.params['1D_CONV_KERNEL'] = {i + 1: int(conv_kernels[i]) for i in range(0, len(conv_kernels))}
    utils.params['DROPOUT_RATE'] = float(top.dropout_value.get())
    utils.params['OUTPUT_CLASSES'] = int(top.output_size_value.get())
    utils.params['STRIDES'] = int(top.strides_value.get())
    utils.params['BATCH_SIZE'] = int(top.batch_size_value.get())
    utils.params['ACTIVATION_FUNC'] = top.activation_func_value.get()


def validate_kernels_size(kernels_string):
    """
    Validates the value entered to 'Kernel Sizes' entry, which must be integers, separated with ','.

    Params:
        - kernels_string(`string`):
          The string to validate.

    Returns:
        True if valid, False if not.
    """
    try:
        splitted_string = kernels_string.split(",")
        numbers_amount = len(splitted_string)
        if numbers_amount < 1:
            return False
        for num in splitted_string:
            res = gui_helper.isint_and_inrange(num, 1, sys.maxsize)
            if not res:
                return res
        return True
    except ValueError:
        return False


def validate_fields_values():
    """
    Validates the inputs from the entries, and returns message indicating which entries are invalid.
    """
    global top
    msg = ""
    if not gui_helper.isint_and_inrange(top.kernels_num_value.get(), 1, sys.maxsize):
        msg += "Number of kernels must be a positive integer\n"
        top.kernels_num_value.configure(highlightbackground="red", highlightcolor="red")
    if not gui_helper.isfloat_and_inrange(top.learning_rate_value.get(), 0, 1):
        msg += "Learning rate must be a float in range (0,1)\n"
        top.learning_rate_value.configure(highlightbackground="red", highlightcolor="red")
    if not gui_helper.isint_and_inrange(top.epochs_value.get(), 1, 100):
        msg += "Number of epoch must be an integer is range [1,99]\n"
        top.epochs_value.configure(highlightbackground="red", highlightcolor="red")
    if not validate_kernels_size(top.conv_sizes_value.get()):
        msg += "Kernel sizes must have one value or more, positive numbers separated by ','\n"
        top.conv_sizes_value.configure(highlightbackground="red", highlightcolor="red")
    if not gui_helper.isfloat_and_inrange(top.dropout_value.get(), 0, 1):
        msg += "Dropout value must be a float in range (0,1)\n"
        top.dropout_value.configure(highlightbackground="red", highlightcolor="red")
    if not gui_helper.isint_and_inrange(top.strides_value.get(), 1, sys.maxsize):
        msg += "Stride size must be a positive integer\n"
        top.strides_value.configure(highlightbackground="red", highlightcolor="red")
    if not gui_helper.isint_and_inrange(top.batch_size_value.get(), 1, sys.maxsize):
        msg += "Batch size must be a positive integer\n"
        top.batch_size_value.configure(highlightbackground="red", highlightcolor="red")
    return msg


class CNNConfigurations_Screen:
    def __init__(self, top=None):
        """
            This class configures and populates the CNN Configurations window.
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

        self.main_ghazali_label = tk.Label(top)
        self.main_ghazali_label.place(x=60, y=20, height=27, width=245)
        self.main_ghazali_label.configure(activebackground="#f9f9f9")
        self.main_ghazali_label.configure(activeforeground="black")
        self.main_ghazali_label.configure(background="#ffffff")
        self.main_ghazali_label.configure(disabledforeground="#a3a3a3")
        self.main_ghazali_label.configure(font="-family {Segoe UI} -size 14 -weight bold")
        self.main_ghazali_label.configure(foreground="#629b1c")
        self.main_ghazali_label.configure(highlightbackground="#d9d9d9")
        self.main_ghazali_label.configure(highlightcolor="black")
        self.main_ghazali_label.configure(text='''CNN Configurations''')

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
        self.Label2.configure(text='''Choose your prefered parameters, or load defaults.''')

        self.start_button = tk.Button(top, command=next_button_click)
        self.start_button.place(x=680, y=315, height=33, width=188)
        self.start_button.configure(activebackground="#ececec")
        self.start_button.configure(activeforeground="#000000")
        self.start_button.configure(background="#629b1c")
        self.start_button.configure(disabledforeground="#a3a3a3")
        self.start_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.start_button.configure(foreground="#ffffff")
        self.start_button.configure(highlightbackground="#d9d9d9")
        self.start_button.configure(highlightcolor="#000000")
        self.start_button.configure(pady="0")
        self.start_button.configure(text='''Next''')

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

        self.kernels_num_value = tk.Entry(top)
        self.kernels_num_value.place(x=40, y=120, height=24, width=150)
        self.gh.entry_widget_defaults(self.kernels_num_value)
        self.gh.tooltip_message(self.kernels_num_value, "Number of kernels. A positive integer")

        self.learning_rate_value = tk.Entry(top)
        self.learning_rate_value.place(x=480, y=120, height=24, width=150)
        self.gh.entry_widget_defaults(self.learning_rate_value)
        self.gh.tooltip_message(self.learning_rate_value, "Learning rate must be a float in range (0,1)")

        self.epochs_value = tk.Entry(top)
        self.epochs_value.place(x=700, y=120, height=24, width=150)
        self.gh.entry_widget_defaults(self.epochs_value)
        self.gh.tooltip_message(self.epochs_value, "Number of epoch must be an integer is range [1,99]")

        self.activation_func_value = ttk.Combobox(top)
        self.activation_func_value['values'] = ('Relu', 'Sigmoid')
        self.activation_func_value.current(0)
        self.activation_func_value.place(x=700, y=246, height=24, width=150)
        self.activation_func_value.configure(font="-family {Segoe UI} -size 11")
        self.activation_func_value.configure(foreground="#525252")
        self.activation_func_value.configure(takefocus="")

        self.Label1 = tk.Label(top)
        self.Label1.place(x=38, y=94, height=26, width=151)
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(activeforeground="black")
        self.Label1.configure(anchor='nw')
        self.Label1.configure(background="#ffffff")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font="-family {Segoe UI} -size 11")
        self.Label1.configure(foreground="#525252")
        self.Label1.configure(highlightbackground="#d9d9d9")
        self.Label1.configure(highlightcolor="black")
        self.Label1.configure(text='''Kernels:''')

        self.Label1_2 = tk.Label(top)
        self.Label1_2.place(x=480, y=94, height=26, width=191)
        self.Label1_2.configure(activebackground="#f9f9f9")
        self.Label1_2.configure(activeforeground="black")
        self.Label1_2.configure(anchor='nw')
        self.Label1_2.configure(background="#ffffff")
        self.Label1_2.configure(disabledforeground="#a3a3a3")
        self.Label1_2.configure(font="-family {Segoe UI} -size 11")
        self.Label1_2.configure(foreground="#525252")
        self.Label1_2.configure(highlightbackground="#d9d9d9")
        self.Label1_2.configure(highlightcolor="black")
        self.Label1_2.configure(text='''Learning Rate:''')

        self.Label1_3 = tk.Label(top)
        self.Label1_3.place(x=699, y=94, height=26, width=191)
        self.Label1_3.configure(activebackground="#f9f9f9")
        self.Label1_3.configure(activeforeground="black")
        self.Label1_3.configure(anchor='nw')
        self.Label1_3.configure(background="#ffffff")
        self.Label1_3.configure(disabledforeground="#a3a3a3")
        self.Label1_3.configure(font="-family {Segoe UI} -size 11")
        self.Label1_3.configure(foreground="#525252")
        self.Label1_3.configure(highlightbackground="#d9d9d9")
        self.Label1_3.configure(highlightcolor="black")
        self.Label1_3.configure(text='''Epochs:''')

        self.Label1_4 = tk.Label(top)
        self.Label1_4.place(x=37, y=160, height=26, width=151)
        self.Label1_4.configure(activebackground="#f9f9f9")
        self.Label1_4.configure(activeforeground="black")
        self.Label1_4.configure(anchor='nw')
        self.Label1_4.configure(background="#ffffff")
        self.Label1_4.configure(disabledforeground="#a3a3a3")
        self.Label1_4.configure(font="-family {Segoe UI} -size 11")
        self.Label1_4.configure(foreground="#525252")
        self.Label1_4.configure(highlightbackground="#d9d9d9")
        self.Label1_4.configure(highlightcolor="black")
        self.Label1_4.configure(text='''1-D Conv Kernels:''')

        self.Label1_6 = tk.Label(top)
        self.Label1_6.place(x=700, y=220, height=26, width=151)
        self.Label1_6.configure(activebackground="#f9f9f9")
        self.Label1_6.configure(activeforeground="black")
        self.Label1_6.configure(anchor='nw')
        self.Label1_6.configure(background="#ffffff")
        self.Label1_6.configure(disabledforeground="#a3a3a3")
        self.Label1_6.configure(font="-family {Segoe UI} -size 11")
        self.Label1_6.configure(foreground="#525252")
        self.Label1_6.configure(highlightbackground="#d9d9d9")
        self.Label1_6.configure(highlightcolor="black")
        self.Label1_6.configure(text='''Activation Func:''')

        self.conv_sizes_value = tk.Entry(top)
        self.conv_sizes_value.place(x=40, y=184, height=24, width=150)
        self.gh.entry_widget_defaults(self.conv_sizes_value)
        self.gh.tooltip_message(self.conv_sizes_value, "Conv sized, shaped: num1,mun2, .. numN.")

        self.dropout_value = tk.Entry(top)
        self.dropout_value.place(x=480, y=184, height=24, width=150)
        self.gh.entry_widget_defaults(self.dropout_value)
        self.gh.tooltip_message(self.dropout_value, "Dropout value. A float in range (0,1)")

        self.Label1_5_1 = tk.Label(top)
        self.Label1_5_1.place(x=479, y=158, height=26, width=141)
        self.Label1_5_1.configure(activebackground="#f9f9f9")
        self.Label1_5_1.configure(activeforeground="black")
        self.Label1_5_1.configure(anchor='nw')
        self.Label1_5_1.configure(background="#ffffff")
        self.Label1_5_1.configure(disabledforeground="#a3a3a3")
        self.Label1_5_1.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5_1.configure(foreground="#525252")
        self.Label1_5_1.configure(highlightbackground="#d9d9d9")
        self.Label1_5_1.configure(highlightcolor="black")
        self.Label1_5_1.configure(text='''Dropout:''')

        self.output_size_value = tk.Entry(top)
        self.output_size_value.insert(0, "1")
        self.output_size_value.configure(state=tk.DISABLED)
        self.output_size_value.place(x=700, y=184, height=24, width=150)
        self.output_size_value.configure(background="white")
        self.output_size_value.configure(font="-family {Segoe UI} -size 11")
        self.output_size_value.configure(foreground="#808080")
        self.output_size_value.configure(highlightbackground="#d9d9d9")
        self.output_size_value.configure(highlightcolor="black")
        self.output_size_value.configure(insertbackground="black")
        self.output_size_value.configure(selectbackground="blue")
        self.output_size_value.configure(selectforeground="white")

        self.Label1_5_1_1 = tk.Label(top)
        self.Label1_5_1_1.place(x=698, y=158, height=26, width=141)
        self.Label1_5_1_1.configure(activebackground="#f9f9f9")
        self.Label1_5_1_1.configure(activeforeground="black")
        self.Label1_5_1_1.configure(anchor='nw')
        self.Label1_5_1_1.configure(background="#ffffff")
        self.Label1_5_1_1.configure(disabledforeground="#a3a3a3")
        self.Label1_5_1_1.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5_1_1.configure(foreground="#525252")
        self.Label1_5_1_1.configure(highlightbackground="#d9d9d9")
        self.Label1_5_1_1.configure(highlightcolor="black")
        self.Label1_5_1_1.configure(text='''Output Size:''')

        self.strides_value = tk.Entry(top)
        self.strides_value.place(x=260, y=120, height=24, width=150)
        self.gh.entry_widget_defaults(self.strides_value)
        self.gh.tooltip_message(self.strides_value, "Number of strides. A positive integer")

        self.batch_size_value = tk.Entry(top)
        self.batch_size_value.place(x=260, y=184, height=24, width=150)
        self.gh.entry_widget_defaults(self.batch_size_value)
        self.gh.tooltip_message(self.batch_size_value, "Batch size of samples. A positive integer")

        self.Label1_5_2 = tk.Label(top)
        self.Label1_5_2.place(x=259, y=94, height=26, width=141)
        self.Label1_5_2.configure(activebackground="#f9f9f9")
        self.Label1_5_2.configure(activeforeground="black")
        self.Label1_5_2.configure(anchor='nw')
        self.Label1_5_2.configure(background="#ffffff")
        self.Label1_5_2.configure(disabledforeground="#a3a3a3")
        self.Label1_5_2.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5_2.configure(foreground="#525252")
        self.Label1_5_2.configure(highlightbackground="#d9d9d9")
        self.Label1_5_2.configure(highlightcolor="black")
        self.Label1_5_2.configure(text='''Strides:''')

        self.Label1_5_3 = tk.Label(top)
        self.Label1_5_3.place(x=259, y=158, height=26, width=141)
        self.Label1_5_3.configure(activebackground="#f9f9f9")
        self.Label1_5_3.configure(activeforeground="black")
        self.Label1_5_3.configure(anchor='nw')
        self.Label1_5_3.configure(background="#ffffff")
        self.Label1_5_3.configure(disabledforeground="#a3a3a3")
        self.Label1_5_3.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5_3.configure(foreground="#525252")
        self.Label1_5_3.configure(highlightbackground="#d9d9d9")
        self.Label1_5_3.configure(highlightcolor="black")
        self.Label1_5_3.configure(text='''Batch Size:''')

    def reset_entries(self):
        """
        Resets entries to the default view.
        """
        self.kernels_num_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        self.learning_rate_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        self.epochs_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        self.conv_sizes_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        self.dropout_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        self.strides_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
        self.batch_size_value.configure(highlightbackground="#d9d9d9", highlightcolor="black")
