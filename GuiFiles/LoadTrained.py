import sys
import utils
import json
import os.path
from Analyzer import read_json
from GuiFiles import HomeScreen, ViewResults, TrainingStatus

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


def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root, top
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    root = tk.Tk()
    top = Load_Trained_Screen(root)
    root.mainloop()


def destroy_Load_Trained_Screen():
    global w, root, new_window
    utils.LoadDefaultCNNConfig()
    utils.LoadDefaultGeneralConfig()
    root.destroy()
    HomeScreen.home_screen_start()
    root = None


def show_results_button_click():
    global w, root
    root.destroy()
    ViewResults.view_results_start("load_trained")
    root = None


def re_run_button_click():
    global w, root
    root.destroy()
    TrainingStatus.vp_start_gui()
    root = None


def update_ui_from_params():
    global top
    from utils import params
    top.general_Text.delete('1.0', tk.END)
    top.cnn_Text.delete('1.0', tk.END)
    top.general_Text.insert(tk.END, "Niter: " + str(params['Niter']))
    top.general_Text.insert(tk.END, "\nAccuracy threshold: " + str(params['ACCURACY_THRESHOLD']))
    top.general_Text.insert(tk.END, "\nBert input length: " + str(params['BERT_INPUT_LENGTH']))
    top.general_Text.insert(tk.END, "\nF1: " + str(params['F1']))
    top.general_Text.insert(tk.END, "\nF: " + str(params['F']))
    top.general_Text.insert(tk.END, "\nSilhouette threshold: " + str(params['SILHOUETTE_THRESHOLD']))
    top.general_Text.insert(tk.END, "\nText division method: " + str(params['TEXT_DIVISION_METHOD']))

    top.cnn_Text.insert(tk.END, "Kernels: " + str(params['KERNELS']))
    # top.cnn_Text.insert(tk.END, "\nCNN filters: " + str(params['CNN_FILTERS']))
    top.cnn_Text.insert(tk.END, "\nLearning rate: " + str(params['LEARNING_RATE']))
    top.cnn_Text.insert(tk.END, "\nNum of epochs: " + str(params['NB_EPOCHS']))

    conv_kernels = ""
    for val in params['1D_CONV_KERNEL'].values():
        conv_kernels += str(val)
        conv_kernels += ","
    conv_kernels = conv_kernels[:-1]
    top.cnn_Text.insert(tk.END, "\n1D conv kernel: " + conv_kernels)
    top.cnn_Text.insert(tk.END, "\nDropout: " + str(params['DROPOUT_RATE']))
    top.cnn_Text.insert(tk.END, "\nOutput classes: " + str(params['OUTPUT_CLASSES']))
    top.cnn_Text.insert(tk.END, "\nStrides: " + str(params['STRIDES']))
    top.cnn_Text.insert(tk.END, "\nBatch size: " + str(params['BATCH_SIZE']))
    # top.cnn_Text.insert(tk.END, "\nMomentum: " + str(params['MOMENTUM']))
    top.cnn_Text.insert(tk.END, "\nActivation function: " + str(params['ACTIVATION_FUNC']))


def disable_button(button):
    button.configure(state=tk.DISABLED)
    button.configure(background="#c0c0c0")


def enable_button(button):
    button.configure(state=tk.NORMAL)
    button.configure(background="#629b1c")


def callback(eventObject):
    import utils
    global top
    data = read_json()
    the_chosen_one = None
    for prevRun in data:
        if prevRun['Name'] == top.model_selection_value.get():
            the_chosen_one = prevRun
            break
    if the_chosen_one is None:
        top.delete_selected_label.place_forget()
        top.view_log_label.place_forget()
        disable_button(top.show_results_button)
        disable_button(top.re_run_button)
        return
    else:
        top.delete_selected_label.place(x=263, y=110, height=21, width=86)
        top.view_log_label.place(x=263, y=170, height=23, width=66)
        enable_button(top.show_results_button)
        enable_button(top.re_run_button)
    # update utils
    utils.params = the_chosen_one
    utils.heat_map = None
    # update UI from utils
    update_ui_from_params()


def view_log_click(model_name):
    from os import startfile
    previous_run_location = os.getcwd() + r"\Data\PreviousRuns\\"
    log_file = previous_run_location + model_name + ".txt"
    startfile(log_file)


def delete_model_click(model_name):
    from tkinter import messagebox as mb
    res = mb.askyesno("Notice", f"You are about to delete {model_name}, "
                                "This action is permanent.\nAre you sure you want to continue?")
    if res is False:
        return
    # If file exists, delete it
    previous_run_location = os.getcwd() + r"\Data\PreviousRuns\\"
    model_file = previous_run_location + model_name + ".npy"
    log_file = previous_run_location + model_name + ".txt"
    try:
        os.remove(model_file)
        os.remove(log_file)
    except FileNotFoundError:
        pass
    data = read_json()
    for p in data:
        if p['Name'] == model_name:
            data.remove(p)
    with open(previous_run_location + 'PreviousRuns.json', 'w') as f:
        json.dump(data, f, indent=4)
    root.destroy()
    vp_start_gui()


class Load_Trained_Screen:
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
        self.main_ghazali_label.configure(text='''Load Trained Model''')

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

        self.show_results_button = tk.Button(self.Frame1, command=show_results_button_click)
        self.show_results_button.place(x=470, y=15, height=33, width=188)
        disable_button(self.show_results_button)
        self.show_results_button.configure(activebackground="#ececec")
        self.show_results_button.configure(activeforeground="#000000")
        self.show_results_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.show_results_button.configure(foreground="#ffffff")
        self.show_results_button.configure(highlightbackground="#d9d9d9")
        self.show_results_button.configure(highlightcolor="#000000")
        self.show_results_button.configure(pady="0")
        self.show_results_button.configure(relief="flat")
        self.show_results_button.configure(text='''Show Results''')

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
        self.Label2.configure(text='''Choose trained model to view results''')

        self.re_run_button = tk.Button(top, command=re_run_button_click)
        self.re_run_button.place(x=680, y=315, height=33, width=188)
        disable_button(self.re_run_button)
        self.re_run_button.configure(activebackground="#ececec")
        self.re_run_button.configure(activeforeground="#000000")
        self.re_run_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.re_run_button.configure(foreground="#ffffff")
        self.re_run_button.configure(highlightbackground="#d9d9d9")
        self.re_run_button.configure(highlightcolor="#000000")
        self.re_run_button.configure(pady="0")
        self.re_run_button.configure(relief="flat")
        self.re_run_button.configure(text='''Re-Run''')

        self.back_button = tk.Button(top, command=destroy_Load_Trained_Screen)
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

        self.model_selection_value = ttk.Combobox(top)
        data = read_json()
        for p in data:
            if p['Name'] not in self.model_selection_value['values']:
                self.model_selection_value['values'] = (*self.model_selection_value['values'], p['Name'])
        self.model_selection_value.place(x=40, y=110, height=24, width=220)
        self.model_selection_value.configure(font="-family {Segoe UI} -size 11")
        self.model_selection_value.configure(foreground="#525252")
        self.model_selection_value.configure(takefocus="")
        self.model_selection_value.bind("<<ComboboxSelected>>", callback)

        self.Label1_2 = tk.Label(top)
        self.Label1_2.place(x=360, y=90, height=26, width=191)
        self.Label1_2.configure(activebackground="#f9f9f9")
        self.Label1_2.configure(activeforeground="black")
        self.Label1_2.configure(anchor='nw')
        self.Label1_2.configure(background="#ffffff")
        self.Label1_2.configure(disabledforeground="#a3a3a3")
        self.Label1_2.configure(font="-family {Segoe UI} -size 11")
        self.Label1_2.configure(foreground="#525252")
        self.Label1_2.configure(highlightbackground="#d9d9d9")
        self.Label1_2.configure(highlightcolor="black")
        self.Label1_2.configure(text='''General Configurations:''')

        self.Label1_5_1_1 = tk.Label(top)
        self.Label1_5_1_1.place(x=630, y=90, height=26, width=141)
        self.Label1_5_1_1.configure(activebackground="#f9f9f9")
        self.Label1_5_1_1.configure(activeforeground="black")
        self.Label1_5_1_1.configure(anchor='nw')
        self.Label1_5_1_1.configure(background="#ffffff")
        self.Label1_5_1_1.configure(disabledforeground="#a3a3a3")
        self.Label1_5_1_1.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5_1_1.configure(foreground="#525252")
        self.Label1_5_1_1.configure(highlightbackground="#d9d9d9")
        self.Label1_5_1_1.configure(highlightcolor="black")
        self.Label1_5_1_1.configure(text='''CNN Configurations:''')

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

        self.TSeparator1_1 = ttk.Separator(top)
        self.TSeparator1_1.place(x=600, y=140, height=120)
        self.TSeparator1_1.configure(orient="vertical")

        self.general_Text = tk.Text(top)
        self.general_Text.place(x=360, y=120, height=164, width=214)
        self.general_Text.configure(background="#ddefc7")
        self.general_Text.configure(borderwidth="2")
        self.general_Text.configure(font="TkTextFont")
        self.general_Text.configure(foreground="black")
        self.general_Text.configure(highlightbackground="#d9d9d9")
        self.general_Text.configure(highlightcolor="black")
        self.general_Text.configure(insertbackground="black")
        self.general_Text.configure(selectbackground="blue")
        self.general_Text.configure(selectforeground="white")
        self.general_Text.configure(wrap="word")

        self.cnn_Text = tk.Text(top)
        self.cnn_Text.place(x=630, y=120, height=164, width=214)
        self.cnn_Text.configure(background="#ddefc7")
        self.cnn_Text.configure(borderwidth="2")
        self.cnn_Text.configure(font="TkTextFont")
        self.cnn_Text.configure(foreground="black")
        self.cnn_Text.configure(highlightbackground="#d9d9d9")
        self.cnn_Text.configure(highlightcolor="black")
        self.cnn_Text.configure(insertbackground="black")
        self.cnn_Text.configure(selectbackground="blue")
        self.cnn_Text.configure(selectforeground="white")
        self.cnn_Text.configure(wrap="word")

        self.delete_selected_label = tk.Label(top)
        # self.delete_selected_label.place(x=263, y=110, height=21, width=86)
        self.delete_selected_label.configure(background="#ffffff")
        self.delete_selected_label.configure(cursor="fleur")
        self.delete_selected_label.configure(disabledforeground="#a3a3a3")
        self.delete_selected_label.configure(font="-family {Segoe UI} -size 10 -underline 1")
        self.delete_selected_label.configure(foreground="#000000")
        self.delete_selected_label.configure(text='''Delete Model''')
        self.delete_selected_label.place_forget()
        self.delete_selected_label.bind("<Button-1>", lambda e: delete_model_click(utils.params['Name']))

        self.view_log_label = tk.Label(top)
        # self.user_help_label.place(x=640, y=220, height=23, width=66)
        self.view_log_label.configure(background="#ffffff")
        self.view_log_label.configure(disabledforeground="#a3a3a3")
        self.view_log_label.configure(font="-family {Segoe UI} -size 10 -weight bold -underline 1")
        self.view_log_label.configure(foreground="#0d25ff")
        self.view_log_label.configure(text='''View Log''')
        self.view_log_label.place_forget()
        self.view_log_label.bind("<Button-1>", lambda e: view_log_click(utils.params['Name']))
