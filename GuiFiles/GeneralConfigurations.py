#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 6.0.1
#  in conjunction with Tcl version 8.6
#    Apr 10, 2021 01:30:13 PM +0300  platform: Windows NT
import pathlib
import sys

from GuiFiles import NewGui, CNNConfigurations

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
    top = GeneralConfigurations_Screen(root)
    read_params_values()
    root.mainloop()


w = None


def create_GeneralConfigurations_Screen(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_GeneralConfigurations_Screen(root, *args, **kwargs)' .'''
    global w, w_win, root
    # rt = root
    root = rt
    w = tk.Toplevel(root)
    top = GeneralConfigurations_Screen(w)
    return (w, top)


def destroy_GeneralConfigurations_Screen():
    global w, root, new_window
    root.destroy()
    NewGui.vp_start_gui()
    root = None


def next_button_click():
    from utils import params
    global w, root
    msg = validate_fields_values()
    if msg != "":
        from tkinter import messagebox as mb
        mb.showerror("Errors", msg)
        return

    update_params()
    embeddingsFile = "FS" if params['TEXT_DIVISION_METHOD'] == "Fixed-Size" else "BU"
    embeddingsFile += str(params['BERT_INPUT_LENGTH'])
    if not os.path.exists(pathlib.Path((os.getcwd() + r"\Data\PreviousRuns\Embeddings"
                                        + "\\" + embeddingsFile + ".zip").strip())):  # embedding do not exist
        # raise pop up to user it's going to take a while.
        from tkinter import messagebox as mb
        res = mb.askyesno("Notice", "The word embedding configurations you chose does not exist in the system, "
                                    "producing word embedding might take a while.\nAre you sure you want to continue?")
        if res is True:
            root.destroy()
            CNNConfigurations.vp_start_gui()
            root = None


def load_defaults_click():
    global top
    from utils import LoadDefaultGeneralConfig
    LoadDefaultGeneralConfig()
    read_params_values()


def read_params_values():
    global top
    from utils import params
    top.niter_value.delete(0, tk.END)
    top.niter_value.insert(0, params['Niter'])

    top.acc_thresh_value.delete(0, tk.END)
    top.acc_thresh_value.insert(0, params['ACCURACY_THRESHOLD'])

    top.bert_input_length_value.delete(0, tk.END)
    top.bert_input_length_value.insert(0, params['BERT_INPUT_LENGTH'])

    top.f1_value.delete(0, tk.END)
    top.f1_value.insert(0, params['F1'])

    top.f_value.delete(0, tk.END)
    top.f_value.insert(0, params['F'])
    if params['TEXT_DIVISION_METHOD'] == 'Fixed-Size':
        top.division_method_value.current(0)
    else:
        top.division_method_value.current(1)

    top.silhouette_thresh_value.delete(0, tk.END)
    top.silhouette_thresh_value.insert(0, params['SILHOUETTE_THRESHOLD'])


def update_params():
    global top
    from utils import params
    params['Niter'] = int(top.niter_value.get())
    params['ACCURACY_THRESHOLD'] = float(top.acc_thresh_value.get())
    params['BERT_INPUT_LENGTH'] = int(top.bert_input_length_value.get())
    params['F1'] = float(top.f1_value.get())
    params['SILHOUETTE_THRESHOLD'] = float(top.silhouette_thresh_value.get())
    params['TEXT_DIVISION_METHOD'] = top.division_method_value.get()
    params['F'] = top.f_value.get()


def validate_fields_values():
    global top
    import utils
    msg = ""
    if not utils.isint_and_inrange(top.niter_value.get(), 1, 99):
        msg += "Niter must be an integer in range [1,99]\n"
    if not utils.isfloat_and_inrange(top.acc_thresh_value.get(), 0, 1):
        msg += "Accuracy Threshold must be float in range (0,1)\n"
    if not utils.isfloat_and_inrange(top.f1_value.get(), 0, 1):
        msg += "F1 must be a float in range (0,1)\n"
    if not utils.isint_and_inrange(top.bert_input_length_value.get(), 20, 511):
        msg += "Bert input length must be an integer in range [20,510]\n"
    if not utils.isfloat_and_inrange(top.silhouette_thresh_value.get(), 0, 1):
        msg += "Silhouette threshold must be a float in range (0,1)\n"
    return msg


class GeneralConfigurations_Screen:
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
        # top.geometry("886x363+334+382")
        top.resizable(False, False)
        top.title("Al-Ghazali's Authorship Attribution")
        top.configure(background="#ffffff")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        def validate_input(new_value):
            valid = new_value.isdigit() and len(new_value) <= 2
            print(valid)

        validate = top.register(validate_input)

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
        self.Label2.configure(text='''Choose your prefered parameters, or load defaults.''')

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

        self.back_button = tk.Button(top, command=destroy_GeneralConfigurations_Screen)
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

        self.niter_value = tk.Entry(top) # , validate="focusout", validatecommand=(validate, "% P")
        self.niter_value.place(x=40, y=120, height=24, width=204)
        self.niter_value.configure(background="white")
        self.niter_value.configure(font="-family {Segoe UI} -size 11")
        self.niter_value.configure(foreground="#808080")
        self.niter_value.configure(highlightbackground="#d9d9d9")
        self.niter_value.configure(highlightcolor="black")
        self.niter_value.configure(insertbackground="black")
        self.niter_value.configure(selectbackground="blue")
        self.niter_value.configure(selectforeground="white")
        self.tooltip_font = "-family {Segoe UI} -size 11"
        self.niter_value_tooltip = \
            ToolTip(self.niter_value, self.tooltip_font, "Must")

        self.acc_thresh_value = tk.Entry(top)
        self.acc_thresh_value.place(x=340, y=120, height=24, width=204)
        self.acc_thresh_value.configure(background="white")
        self.acc_thresh_value.configure(font="-family {Segoe UI} -size 11")
        self.acc_thresh_value.configure(foreground="#808080")
        self.acc_thresh_value.configure(highlightbackground="#d9d9d9")
        self.acc_thresh_value.configure(highlightcolor="black")
        self.acc_thresh_value.configure(insertbackground="black")
        self.acc_thresh_value.configure(selectbackground="blue")
        self.acc_thresh_value.configure(selectforeground="white")

        self.bert_input_length_value = tk.Entry(top)
        self.bert_input_length_value.place(x=640, y=120, height=24, width=204)
        self.bert_input_length_value.configure(background="white")
        self.bert_input_length_value.configure(font="-family {Segoe UI} -size 11")
        self.bert_input_length_value.configure(foreground="#808080")
        self.bert_input_length_value.configure(highlightbackground="#d9d9d9")
        self.bert_input_length_value.configure(highlightcolor="black")
        self.bert_input_length_value.configure(insertbackground="black")
        self.bert_input_length_value.configure(selectbackground="blue")
        self.bert_input_length_value.configure(selectforeground="white")

        self.f1_value = tk.Entry(top)
        self.f1_value.place(x=40, y=180, height=24, width=204)
        self.f1_value.configure(background="white")
        self.f1_value.configure(font="-family {Segoe UI} -size 11")
        self.f1_value.configure(foreground="#808080")
        self.f1_value.configure(highlightbackground="#d9d9d9")
        self.f1_value.configure(highlightcolor="black")
        self.f1_value.configure(insertbackground="black")
        self.f1_value.configure(selectbackground="blue")
        self.f1_value.configure(selectforeground="white")

        self.silhouette_thresh_value = tk.Entry(top)
        self.silhouette_thresh_value.place(x=340, y=180, height=24, width=204)
        self.silhouette_thresh_value.configure(background="white")
        self.silhouette_thresh_value.configure(font="-family {Segoe UI} -size 11")
        self.silhouette_thresh_value.configure(foreground="#808080")
        self.silhouette_thresh_value.configure(highlightbackground="#d9d9d9")
        self.silhouette_thresh_value.configure(highlightcolor="black")
        self.silhouette_thresh_value.configure(insertbackground="black")
        self.silhouette_thresh_value.configure(selectbackground="blue")
        self.silhouette_thresh_value.configure(selectforeground="white")

        self.f_value = tk.Entry(top)
        self.f_value.place(x=40, y=240, height=24, width=204)
        self.f_value.configure(background="white")
        self.f_value.configure(font="-family {Segoe UI} -size 11")
        self.f_value.configure(foreground="#808080")
        self.f_value.configure(highlightbackground="#d9d9d9")
        self.f_value.configure(highlightcolor="black")
        self.f_value.configure(insertbackground="black")
        self.f_value.configure(selectbackground="blue")
        self.f_value.configure(selectforeground="white")

        self.division_method_value = ttk.Combobox(top)
        # Adding combobox drop down list
        self.division_method_value['values'] = ('Fixed-Size', 'Bottom-Up')
        self.division_method_value.current(0)
        self.division_method_value.place(x=640, y=180, height=24, width=204)
        self.division_method_value.configure(font="-family {Segoe UI} -size 11")
        # self.division_method_value.configure(textvariable=GeneralConfigurations_support.cmb)
        self.division_method_value.configure(foreground="#525252")
        self.division_method_value.configure(takefocus="")

        self.Label1 = tk.Label(top)
        self.Label1.place(x=37, y=94, height=26, width=191)
        self.Label1.configure(background="#ffffff")
        self.Label1.configure(cursor="fleur")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font="-family {Segoe UI} -size 11")
        self.Label1.configure(foreground="#525252")
        self.Label1.configure(text='''Niter: (number of iterations)''')

        self.Label1_1 = tk.Label(top)
        self.Label1_1.place(x=338, y=94, height=26, width=191)
        self.Label1_1.configure(activebackground="#f9f9f9")
        self.Label1_1.configure(activeforeground="black")
        self.Label1_1.configure(anchor='nw')
        self.Label1_1.configure(background="#ffffff")
        self.Label1_1.configure(disabledforeground="#a3a3a3")
        self.Label1_1.configure(font="-family {Segoe UI} -size 11")
        self.Label1_1.configure(foreground="#525252")
        self.Label1_1.configure(highlightbackground="#d9d9d9")
        self.Label1_1.configure(highlightcolor="black")
        self.Label1_1.configure(text='''Accuracy Threshold:''')

        self.Label1_2 = tk.Label(top)
        self.Label1_2.place(x=639, y=94, height=26, width=191)
        self.Label1_2.configure(activebackground="#f9f9f9")
        self.Label1_2.configure(activeforeground="black")
        self.Label1_2.configure(anchor='nw')
        self.Label1_2.configure(background="#ffffff")
        self.Label1_2.configure(disabledforeground="#a3a3a3")
        self.Label1_2.configure(font="-family {Segoe UI} -size 11")
        self.Label1_2.configure(foreground="#525252")
        self.Label1_2.configure(highlightbackground="#d9d9d9")
        self.Label1_2.configure(highlightcolor="black")
        self.Label1_2.configure(text='''BERT Input Length:''')

        self.Label1_3 = tk.Label(top)
        self.Label1_3.place(x=37, y=154, height=26, width=191)
        self.Label1_3.configure(activebackground="#f9f9f9")
        self.Label1_3.configure(activeforeground="black")
        self.Label1_3.configure(anchor='nw')
        self.Label1_3.configure(background="#ffffff")
        self.Label1_3.configure(disabledforeground="#a3a3a3")
        self.Label1_3.configure(font="-family {Segoe UI} -size 11")
        self.Label1_3.configure(foreground="#525252")
        self.Label1_3.configure(highlightbackground="#d9d9d9")
        self.Label1_3.configure(highlightcolor="black")
        self.Label1_3.configure(text='''F1: (undersampling rate)''')

        self.Label1_4 = tk.Label(top)
        self.Label1_4.place(x=39, y=214, height=26, width=191)
        self.Label1_4.configure(activebackground="#f9f9f9")
        self.Label1_4.configure(activeforeground="black")
        self.Label1_4.configure(anchor='nw')
        self.Label1_4.configure(background="#ffffff")
        self.Label1_4.configure(disabledforeground="#a3a3a3")
        self.Label1_4.configure(font="-family {Segoe UI} -size 11")
        self.Label1_4.configure(foreground="#525252")
        self.Label1_4.configure(highlightbackground="#d9d9d9")
        self.Label1_4.configure(highlightcolor="black")
        self.Label1_4.configure(text='''F: (oversampling rate)''')

        self.Label1_5 = tk.Label(top)
        self.Label1_5.place(x=338, y=154, height=26, width=191)
        self.Label1_5.configure(activebackground="#f9f9f9")
        self.Label1_5.configure(activeforeground="black")
        self.Label1_5.configure(anchor='nw')
        self.Label1_5.configure(background="#ffffff")
        self.Label1_5.configure(disabledforeground="#a3a3a3")
        self.Label1_5.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5.configure(foreground="#525252")
        self.Label1_5.configure(highlightbackground="#d9d9d9")
        self.Label1_5.configure(highlightcolor="black")
        self.Label1_5.configure(text='''Silhouette Threshold:''')

        self.Label1_6 = tk.Label(top)
        self.Label1_6.place(x=639, y=154, height=26, width=191)
        self.Label1_6.configure(activebackground="#f9f9f9")
        self.Label1_6.configure(activeforeground="black")
        self.Label1_6.configure(anchor='nw')
        self.Label1_6.configure(background="#ffffff")
        self.Label1_6.configure(disabledforeground="#a3a3a3")
        self.Label1_6.configure(font="-family {Segoe UI} -size 11")
        self.Label1_6.configure(foreground="#525252")
        self.Label1_6.configure(highlightbackground="#d9d9d9")
        self.Label1_6.configure(highlightcolor="black")
        self.Label1_6.configure(text='''Text Division Method:''')


from time import time, localtime, strftime


class ToolTip(tk.Toplevel):
    """
    Provides a ToolTip widget for Tkinter.
    To apply a ToolTip to any Tkinter widget, simply pass the widget to the
    ToolTip constructor
    """

    def __init__(self, wdgt, tooltip_font, msg=None, msgFunc=None,
                 delay=0.5, follow=True):
        """
        Initialize the ToolTip

        Arguments:
          wdgt: The widget this ToolTip is assigned to
          tooltip_font: Font to be used
          msg:  A static string message assigned to the ToolTip
          msgFunc: A function that retrieves a string to use as the ToolTip text
          delay:   The delay in seconds before the ToolTip appears(may be float)
          follow:  If True, the ToolTip follows motion, otherwise hides
        """
        self.wdgt = wdgt
        # The parent of the ToolTip is the parent of the ToolTips widget
        self.parent = self.wdgt.master
        # Initalise the Toplevel
        tk.Toplevel.__init__(self, self.parent, bg='black', padx=1, pady=1)
        # Hide initially
        self.withdraw()
        # The ToolTip Toplevel should have no frame or title bar
        self.overrideredirect(True)

        # The msgVar will contain the text displayed by the ToolTip
        self.msgVar = tk.StringVar()
        if msg is None:
            self.msgVar.set('No message provided')
        else:
            self.msgVar.set(msg)
        self.msgFunc = msgFunc
        self.delay = delay
        self.follow = follow
        self.visible = 0
        self.lastMotion = 0
        # The text of the ToolTip is displayed in a Message widget
        tk.Message(self, textvariable=self.msgVar, bg='#FFFFDD',
                   font=tooltip_font,
                   aspect=1000).grid()

        # Add bindings to the widget.  This will NOT override
        # bindings that the widget already has
        self.wdgt.bind('<Enter>', self.spawn, '+')
        self.wdgt.bind('<Leave>', self.hide, '+')
        self.wdgt.bind('<Motion>', self.move, '+')

    def spawn(self, event=None):
        """
        Spawn the ToolTip.  This simply makes the ToolTip eligible for display.
        Usually this is caused by entering the widget

        Arguments:
          event: The event that called this funciton
        """
        self.visible = 1
        # The after function takes a time argument in milliseconds
        self.after(int(self.delay * 1000), self.show)

    def show(self):
        """
        Displays the ToolTip if the time delay has been long enough
        """
        if self.visible == 1 and time() - self.lastMotion > self.delay:
            self.visible = 2
        if self.visible == 2:
            self.deiconify()

    def move(self, event):
        """
        Processes motion within the widget.
        Arguments:
          event: The event that called this function
        """
        self.lastMotion = time()
        # If the follow flag is not set, motion within the
        # widget will make the ToolTip disappear
        #
        if self.follow is False:
            self.withdraw()
            self.visible = 1

        # Offset the ToolTip 10x10 pixes southwest of the pointer
        self.geometry('+%i+%i' % (event.x_root + 20, event.y_root - 10))
        try:
            # Try to call the message function.  Will not change
            # the message if the message function is None or
            # the message function fails
            self.msgVar.set(self.msgFunc())
        except:
            pass
        self.after(int(self.delay * 1000), self.show)

    def hide(self, event=None):
        """
        Hides the ToolTip.  Usually this is caused by leaving the widget
        Arguments:
          event: The event that called this function
        """
        self.visible = 0
        self.withdraw()

    def update(self, msg):
        """
        Updates the Tooltip with a new message. Added by Rozen
        """
        self.msgVar.set(msg)

# ===========================================================
#                   End of Class ToolTip
# ===========================================================
