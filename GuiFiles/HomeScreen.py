import os
import webbrowser

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

roo = tk.Tk()


class Splash:
    """
        Shows a loading window, until finished loading heavy modules.
    """
    def __init__(self, top):
        w = 732
        h = 100
        ws = top.winfo_screenwidth()
        hs = top.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        top.geometry('%dx%d+%d+%d' % (w, h, x, y))
        top.resizable(False, False)

        loading_label = ttk.Label(text="Loading, Please wait...")
        loading_label.configure(font="-family {Segoe UI} -size 14 -weight bold")
        loading_label.place(x=w / 2, y=h / 2, anchor="center")


splash = Splash(roo)
roo.overrideredirect(1)
roo.update()
from GuiFiles import GeneralConfigurations, LoadTrained
roo.destroy()

root = None


def home_screen_start():
    """
        Call Home_Screen class, and creates mainloop thread.
    """
    global root
    root = tk.Tk()
    Home_Screen(root)
    root.mainloop()


def new_training_button_click():
    """
        Called when clicking on 'New Training' button.
        The func calls the General Configuration screen function, and destroys current view.
    """
    global root
    root.destroy()
    GeneralConfigurations.general_configurations_start()
    root = None


def load_trained_button_click():
    """
        Called when clicking on 'Load Trained Model' button.
        The func calls the Load Trained screen function, and destroys current view.
    """
    global root
    root.destroy()
    LoadTrained.load_trained_start()
    root = None


def user_help_click(url):
    """
        Called when clicking on 'User Help' label.
        Opens the user-help document, in browser view.
    """
    webbrowser.open_new(url)


class Home_Screen:
    def __init__(self, top=None):
        """
            This class configures and populates the Home Screen window.
            top is the toplevel containing window.

            In this screen, we configure two buttons, each has click event, which redirects to the desired window.
        """

        w = 732
        h = 305
        ws = top.winfo_screenwidth()
        hs = top.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        top.geometry('%dx%d+%d+%d' % (w, h, x, y))
        top.resizable(False, False)
        top.title("Al-Ghazali's Authorship Attribution")
        top.configure(background="#ffffff")
        from GuiFiles.gui_helper import big_button_style as bbs

        self.new_training_button = tk.Button(top, command=new_training_button_click)
        self.new_training_button.place(x=350, y=70, height=54, width=337)
        bbs(self.new_training_button, "New Training")

        self.load_trained_model_button = tk.Button(top, command=load_trained_button_click)
        self.load_trained_model_button.place(x=350, y=150, height=54, width=337)
        bbs(self.load_trained_model_button, "Load Trained Model")

        self.TSeparator1 = ttk.Separator(top)
        self.TSeparator1.place(x=300, y=40, height=200)
        self.TSeparator1.configure(orient="vertical")

        self.main_ghazali_label = tk.Label(top)
        self.main_ghazali_label.place(x=30, y=217, height=27, width=245)
        self.main_ghazali_label.configure(activebackground="#f9f9f9")
        self.main_ghazali_label.configure(activeforeground="black")
        self.main_ghazali_label.configure(background="#ffffff")
        self.main_ghazali_label.configure(disabledforeground="#a3a3a3")
        self.main_ghazali_label.configure(font="-family {Segoe UI} -size 12 -weight bold")
        self.main_ghazali_label.configure(foreground="#629b1c")
        self.main_ghazali_label.configure(highlightbackground="#d9d9d9")
        self.main_ghazali_label.configure(highlightcolor="black")
        self.main_ghazali_label.configure(text='''Al-Ghazali Authorship Analyzer''')

        self.TLabel1 = ttk.Label(top)
        self.TLabel1.place(x=60, y=30, height=180, width=180)
        self.TLabel1.configure(background="#ffffff")
        self.TLabel1.configure(foreground="#000000")
        self.TLabel1.configure(font="TkDefaultFont")
        self.TLabel1.configure(relief="flat")
        self.TLabel1.configure(anchor='center')
        self.TLabel1.configure(justify='center')
        photo_location = os.getcwd() + r"\GuiFiles\Al-Ghazali-Home.png"
        global _img0
        _img0 = tk.PhotoImage(file=photo_location)
        self.TLabel1.configure(image=_img0)

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(x=0, y=260, height=45, width=732)
        self.Frame1.configure(background="#eeeeee")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label1 = tk.Label(self.Frame1)
        self.Label1.place(x=159, y=12, height=21, width=396)
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(activeforeground="black")
        self.Label1.configure(background="#eeeeee")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(foreground="#919191")
        self.Label1.configure(highlightbackground="#d9d9d9")
        self.Label1.configure(highlightcolor="black")
        self.Label1.configure(text='''Machine Learning tool for Authorship Attribution. Ort Braude, Spring 2021''')

        self.Label2 = tk.Label(top)
        self.Label2.place(x=450, y=30, height=21, width=134)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#ffffff")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font="-family {Segoe UI} -size 10")
        self.Label2.configure(foreground="#9d9d9d")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Pick an option below''')

        self.user_help_label = tk.Label(top)
        self.user_help_label.place(x=640, y=220, height=23, width=66)
        self.user_help_label.configure(background="#ffffff")
        self.user_help_label.configure(disabledforeground="#a3a3a3")
        self.user_help_label.configure(font="-family {Segoe UI} -size 10 -weight bold -underline 1")
        self.user_help_label.configure(foreground="#0d25ff")
        self.user_help_label.configure(text='''User Help''')
        user_help_file = os.getcwd() + r"\user_help.html"
        self.user_help_label.bind("<Button-1>", lambda e: user_help_click(user_help_file))
