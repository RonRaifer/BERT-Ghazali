#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 6.0.1
#  in conjunction with Tcl version 8.6
#    Apr 10, 2021 03:27:02 PM +0300  platform: Windows NT

import sys
import threading

from GuiFiles import GeneralConfigurations, TrainingStatus

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


def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root, top
    root = tk.Tk()
    top = CNNConfigurations_Screen(root)
    read_params_values()
    root.mainloop()


w = None


def create_CNNConfigurations_Screen(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_CNNConfigurations_Screen(root, *args, **kwargs)' .'''
    global w, w_win, root
    # rt = root
    root = rt
    w = tk.Toplevel(root)
    top = CNNConfigurations_Screen(w)
    return (w, top)


def destroy_CNNConfigurations_Screen():
    global w, root
    root.destroy()
    GeneralConfigurations.vp_start_gui()
    root = None


def create_TrainingStatus_Screen():
    global w, root
    update_params()
    root.destroy()
    TrainingStatus.vp_start_gui()
    root = None


def load_defaults_click():
    global top
    from utils import LoadDefaultCNNConfig
    LoadDefaultCNNConfig()
    read_params_values()


def read_params_values():
    global top
    from utils import params

    top.kernels_num_value.delete(0, tk.END)
    top.kernels_num_value.insert(0, params['KERNELS'])

    top.filter_value.delete(0, tk.END)
    top.filter_value.insert(0, params['CNN_FILTERS'])

    top.batch_size_value.delete(0, tk.END)
    top.batch_size_value.insert(0, params['BATCH_SIZE'])

    conv_kernels = str(str(params['1D_CONV_KERNEL'][1]) + ',' +
                       str(params['1D_CONV_KERNEL'][2]) + ',' +
                       str(params['1D_CONV_KERNEL'][3]))
    top.conv_sizes_value.delete(0, tk.END)
    top.conv_sizes_value.insert(0, conv_kernels)

    top.decay_value.delete(0, tk.END)
    top.decay_value.insert(0, params['DECAY'])

    top.epochs_value.delete(0, tk.END)
    top.epochs_value.insert(0, params['NB_EPOCHS'])

    top.learning_rate_value.delete(0, tk.END)
    top.learning_rate_value.insert(0, params['LEARNING_RATE'])

    top.momentum_value.delete(0, tk.END)
    top.momentum_value.insert(0, params['MOMENTUM'])

    top.output_size_value.delete(0, tk.END)
    top.output_size_value.insert(0, params['OUTPUT_CLASSES'])

    top.pooling_size_value.delete(0, tk.END)
    top.pooling_size_value.insert(0, params['POOLING_SIZE'])

    top.strides_value.delete(0, tk.END)
    top.strides_value.insert(0, params['STRIDES'])

    if params['ACTIVATION_FUNC'] == "Relu":
        top.activation_func_value.current(0)
    else:
        top.activation_func_value.current(1)


def update_params():
    global top
    from utils import params
    params['KERNELS'] = int(top.kernels_num_value.get())
    params['CNN_FILTERS'] = int(top.filter_value.get())
    params['LEARNING_RATE'] = float(top.learning_rate_value.get())
    params['NB_EPOCHS'] = int(top.epochs_value.get())
    conv_kernels = str(top.conv_sizes_value.get()).split(",")
    params['1D_CONV_KERNEL'] = {1: int(conv_kernels[0]),
                                2: int(conv_kernels[1]),
                                3: int(conv_kernels[2])}
    params['POOLING_SIZE'] = int(top.pooling_size_value.get())
    params['DECAY'] = int(top.decay_value.get())
    params['OUTPUT_CLASSES'] = int(top.output_size_value.get())
    params['STRIDES'] = int(top.strides_value.get())
    params['BATCH_SIZE'] = int(top.batch_size_value.get())
    params['MOMENTUM'] = float(top.momentum_value.get())
    params['ACTIVATION_FUNC'] = top.activation_func_value.get()


class CNNConfigurations_Screen:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#ececec'  # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.', background=_bgcolor)
        self.style.configure('.', foreground=_fgcolor)
        self.style.configure('.', font="TkDefaultFont")
        self.style.map('.', background=
        [('selected', _compcolor), ('active', _ana2color)])

        w = 886
        h = 363
        ws = top.winfo_screenwidth()
        hs = top.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        top.geometry('%dx%d+%d+%d' % (w, h, x, y))

        # top.geometry("886x363+402+341")
        top.resizable(False, False)
        top.title("Al-Ghazali's Authorship Attribution")
        top.configure(background="#ffffff")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

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

        self.start_button = tk.Button(top, command=create_TrainingStatus_Screen)
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

        self.back_button = tk.Button(top, command=destroy_CNNConfigurations_Screen)
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
        self.kernels_num_value.insert(0, "3")
        self.kernels_num_value.configure(state=tk.DISABLED)
        self.kernels_num_value.place(x=40, y=120, height=24, width=150)
        self.kernels_num_value.configure(background="white")
        self.kernels_num_value.configure(font="-family {Segoe UI} -size 11")
        self.kernels_num_value.configure(foreground="#808080")
        self.kernels_num_value.configure(highlightbackground="#d9d9d9")
        self.kernels_num_value.configure(highlightcolor="black")
        self.kernels_num_value.configure(insertbackground="black")
        self.kernels_num_value.configure(selectbackground="blue")
        self.kernels_num_value.configure(selectforeground="white")

        self.filter_value = tk.Entry(top)
        self.filter_value.place(x=260, y=120, height=24, width=150)
        self.filter_value.configure(background="white")
        self.filter_value.configure(font="-family {Segoe UI} -size 11")
        self.filter_value.configure(foreground="#808080")
        self.filter_value.configure(highlightbackground="#d9d9d9")
        self.filter_value.configure(highlightcolor="black")
        self.filter_value.configure(insertbackground="black")
        self.filter_value.configure(selectbackground="blue")
        self.filter_value.configure(selectforeground="white")

        self.learning_rate_value = tk.Entry(top)
        self.learning_rate_value.place(x=480, y=120, height=24, width=150)
        self.learning_rate_value.configure(background="white")
        self.learning_rate_value.configure(font="-family {Segoe UI} -size 11")
        self.learning_rate_value.configure(foreground="#808080")
        self.learning_rate_value.configure(highlightbackground="#d9d9d9")
        self.learning_rate_value.configure(highlightcolor="black")
        self.learning_rate_value.configure(insertbackground="black")
        self.learning_rate_value.configure(selectbackground="blue")
        self.learning_rate_value.configure(selectforeground="white")

        self.epochs_value = tk.Entry(top)
        self.epochs_value.place(x=700, y=120, height=24, width=150)
        self.epochs_value.configure(background="white")
        self.epochs_value.configure(font="-family {Segoe UI} -size 11")
        self.epochs_value.configure(foreground="#808080")
        self.epochs_value.configure(highlightbackground="#d9d9d9")
        self.epochs_value.configure(highlightcolor="black")
        self.epochs_value.configure(insertbackground="black")
        self.epochs_value.configure(selectbackground="blue")
        self.epochs_value.configure(selectforeground="white")

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

        self.Label1_1 = tk.Label(top)
        self.Label1_1.place(x=258, y=94, height=26, width=151)
        self.Label1_1.configure(activebackground="#f9f9f9")
        self.Label1_1.configure(activeforeground="black")
        self.Label1_1.configure(anchor='nw')
        self.Label1_1.configure(background="#ffffff")
        self.Label1_1.configure(disabledforeground="#a3a3a3")
        self.Label1_1.configure(font="-family {Segoe UI} -size 11")
        self.Label1_1.configure(foreground="#525252")
        self.Label1_1.configure(highlightbackground="#d9d9d9")
        self.Label1_1.configure(highlightcolor="black")
        self.Label1_1.configure(text='''Filters:''')

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

        self.Label1_5 = tk.Label(top)
        self.Label1_5.place(x=258, y=160, height=26, width=141)
        self.Label1_5.configure(activebackground="#f9f9f9")
        self.Label1_5.configure(activeforeground="black")
        self.Label1_5.configure(anchor='nw')
        self.Label1_5.configure(background="#ffffff")
        self.Label1_5.configure(disabledforeground="#a3a3a3")
        self.Label1_5.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5.configure(foreground="#525252")
        self.Label1_5.configure(highlightbackground="#d9d9d9")
        self.Label1_5.configure(highlightcolor="black")
        self.Label1_5.configure(text='''Pooling Size:''')

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
        self.conv_sizes_value.configure(background="white")
        self.conv_sizes_value.configure(font="-family {Segoe UI} -size 11")
        self.conv_sizes_value.configure(foreground="#808080")
        self.conv_sizes_value.configure(highlightbackground="#d9d9d9")
        self.conv_sizes_value.configure(highlightcolor="black")
        self.conv_sizes_value.configure(insertbackground="black")
        self.conv_sizes_value.configure(selectbackground="blue")
        self.conv_sizes_value.configure(selectforeground="white")

        self.pooling_size_value = tk.Entry(top)
        self.pooling_size_value.place(x=260, y=184, height=24, width=150)
        self.pooling_size_value.configure(background="white")
        self.pooling_size_value.configure(font="-family {Segoe UI} -size 11")
        self.pooling_size_value.configure(foreground="#808080")
        self.pooling_size_value.configure(highlightbackground="#d9d9d9")
        self.pooling_size_value.configure(highlightcolor="black")
        self.pooling_size_value.configure(insertbackground="black")
        self.pooling_size_value.configure(selectbackground="blue")
        self.pooling_size_value.configure(selectforeground="white")

        self.decay_value = tk.Entry(top)
        self.decay_value.place(x=480, y=184, height=24, width=150)
        self.decay_value.configure(background="white")
        self.decay_value.configure(font="-family {Segoe UI} -size 11")
        self.decay_value.configure(foreground="#808080")
        self.decay_value.configure(highlightbackground="#d9d9d9")
        self.decay_value.configure(highlightcolor="black")
        self.decay_value.configure(insertbackground="black")
        self.decay_value.configure(selectbackground="blue")
        self.decay_value.configure(selectforeground="white")

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
        self.Label1_5_1.configure(text='''Decay:''')

        self.output_size_value = tk.Entry(top)
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
        self.strides_value.place(x=41, y=247, height=24, width=150)
        self.strides_value.configure(background="white")
        self.strides_value.configure(font="-family {Segoe UI} -size 11")
        self.strides_value.configure(foreground="#808080")
        self.strides_value.configure(highlightbackground="#d9d9d9")
        self.strides_value.configure(highlightcolor="black")
        self.strides_value.configure(insertbackground="black")
        self.strides_value.configure(selectbackground="blue")
        self.strides_value.configure(selectforeground="white")

        self.batch_size_value = tk.Entry(top)
        self.batch_size_value.place(x=260, y=247, height=24, width=150)
        self.batch_size_value.configure(background="white")
        self.batch_size_value.configure(font="-family {Segoe UI} -size 11")
        self.batch_size_value.configure(foreground="#808080")
        self.batch_size_value.configure(highlightbackground="#d9d9d9")
        self.batch_size_value.configure(highlightcolor="black")
        self.batch_size_value.configure(insertbackground="black")
        self.batch_size_value.configure(selectbackground="blue")
        self.batch_size_value.configure(selectforeground="white")

        self.momentum_value = tk.Entry(top)
        self.momentum_value.place(x=480, y=246, height=24, width=150)
        self.momentum_value.configure(background="white")
        self.momentum_value.configure(font="-family {Segoe UI} -size 11")
        self.momentum_value.configure(foreground="#808080")
        self.momentum_value.configure(highlightbackground="#d9d9d9")
        self.momentum_value.configure(highlightcolor="black")
        self.momentum_value.configure(insertbackground="black")
        self.momentum_value.configure(selectbackground="blue")
        self.momentum_value.configure(selectforeground="white")

        self.Label1_5_2 = tk.Label(top)
        self.Label1_5_2.place(x=40, y=221, height=26, width=141)
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
        self.Label1_5_3.place(x=259, y=221, height=26, width=141)
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

        self.Label1_5_4 = tk.Label(top)
        self.Label1_5_4.place(x=479, y=220, height=26, width=141)
        self.Label1_5_4.configure(activebackground="#f9f9f9")
        self.Label1_5_4.configure(activeforeground="black")
        self.Label1_5_4.configure(anchor='nw')
        self.Label1_5_4.configure(background="#ffffff")
        self.Label1_5_4.configure(disabledforeground="#a3a3a3")
        self.Label1_5_4.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5_4.configure(foreground="#525252")
        self.Label1_5_4.configure(highlightbackground="#d9d9d9")
        self.Label1_5_4.configure(highlightcolor="black")
        self.Label1_5_4.configure(text='''Momentum:''')
