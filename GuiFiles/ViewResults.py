#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 6.0.1
#  in conjunction with Tcl version 8.6
#    Apr 16, 2021 11:43:44 AM +0300  platform: Windows NT

import sys

import utils
from Analyzer import show_results
from GuiFiles import CNNConfigurations

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
    top = view_results_Screen(root)
    aa()
    root.mainloop()


w = None


def create_view_results_Screen(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_view_results_Screen(root, *args, **kwargs)' .'''
    global w, w_win, root
    # rt = root
    root = rt
    w = tk.Toplevel(root)
    top = view_results_Screen(w)
    return (w, top)


def destroy_view_results_Screen():
    global w
    w.destroy()
    w = None


def aa():
    global top
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import numpy as np
    utils.heat_map = np.load(r'C:/Users/Ron/Desktop/BERT-Ghazali/Data/Mat.npy')
    # print(utils.heat_map)
    show_results()
    # utils.kmeans_plot.show()
    top.heatmap_canvas = FigureCanvasTkAgg(utils.kmeans_plot, master=root)
    # top.heatmap_canvas.show()
    top.heatmap_canvas.get_tk_widget().pack()

def back_button_click():
    global w, root
    root.destroy()
    CNNConfigurations.vp_start_gui()
    root = None


class view_results_Screen:
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

        w = 882
        h = 631
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
        self.main_ghazali_label.configure(text='''Results''')

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
        self.Frame1.place(x=-1, y=570, height=65, width=885)
        self.Frame1.configure(background="#eeeeee")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label2 = tk.Label(top)
        self.Label2.place(x=420, y=20, height=21, width=324)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#ffffff")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font="-family {Segoe UI} -size 10")
        self.Label2.configure(foreground="#9d9d9d")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Choose trained model to view results''')

        self.save_button = tk.Button(top)
        self.save_button.place(x=680, y=582, height=33, width=188)
        self.save_button.configure(activebackground="#ececec")
        self.save_button.configure(activeforeground="#000000")
        self.save_button.configure(background="#629b1c")
        self.save_button.configure(disabledforeground="#a3a3a3")
        self.save_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.save_button.configure(foreground="#ffffff")
        self.save_button.configure(highlightbackground="#d9d9d9")
        self.save_button.configure(highlightcolor="#000000")
        self.save_button.configure(pady="0")
        self.save_button.configure(text='''Save''')

        self.back_button = tk.Button(top, command=back_button_click)
        self.back_button.place(x=10, y=583, height=33, width=188)
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

        self.Label1_5_2 = tk.Label(top)
        self.Label1_5_2.place(x=40, y=84, height=26, width=141)
        self.Label1_5_2.configure(activebackground="#f9f9f9")
        self.Label1_5_2.configure(activeforeground="black")
        self.Label1_5_2.configure(anchor='nw')
        self.Label1_5_2.configure(background="#ffffff")
        self.Label1_5_2.configure(disabledforeground="#a3a3a3")
        self.Label1_5_2.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5_2.configure(foreground="#525252")
        self.Label1_5_2.configure(highlightbackground="#d9d9d9")
        self.Label1_5_2.configure(highlightcolor="black")
        self.Label1_5_2.configure(text='''Select Model:''')

        self.heatmap_canvas = tk.Canvas(top)
        self.heatmap_canvas.place(x=40, y=120, height=423, width=393)
        self.heatmap_canvas.configure(background="#d9d9d9")
        self.heatmap_canvas.configure(borderwidth="2")
        self.heatmap_canvas.configure(insertbackground="black")
        self.heatmap_canvas.configure(relief="ridge")
        self.heatmap_canvas.configure(selectbackground="blue")
        self.heatmap_canvas.configure(selectforeground="white")

