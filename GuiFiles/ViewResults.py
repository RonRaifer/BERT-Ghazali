import os.path
from Logic.Analyze import Analyzer
from Data import utils
import matplotlib.pyplot as plt
from Logic.Analyze.Analyzer import show_results
from GuiFiles import CNNConfigurations, SaveResults, LoadTrained

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


def view_results_start(bread_crumbs):
    """
        Call ViewResults_Screen class, and creates mainloop thread.
        Also, it overrides the default exit button, in order to fix issues with plots.

        Params:
            bread_crumbs (`str`):
                The name of the screen called View Results. So the Back button will redirects to the desired screen.
    """
    global root, top
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", exit_handler)
    top = view_results_Screen(bread_crumbs, root)
    root.mainloop()


def exit_handler():
    """
        Called on 'exit window' event.

        It closes any plots, and destroys the current view.
    """
    global root
    root.destroy()
    plt.close(utils.kmeans_plot)
    plt.close(utils.heat_map_plot)


def back_button_click_to_CNN():
    """
        Called on 'Back' button click.

        It destroys the current view, and brings the 'CNN Configurations' screen.
        It also closes all the plots, to avoid duplicates.
    """
    global root
    root.destroy()
    plt.close(utils.kmeans_plot)
    plt.close(utils.heat_map_plot)
    CNNConfigurations.cnn_configurations_start()
    root = None


def back_button_click_to_load_trained():
    """
        Called on 'Back' button click.

        It destroys the current view, and brings the 'Load Trained' screen.
        It also closes all the plots, to avoid duplicates.
    """
    global root
    root.destroy()
    plt.close(utils.kmeans_plot)
    plt.close(utils.heat_map_plot)
    LoadTrained.load_trained_start()
    root = None


def save_button_click():
    """
        Called on 'Save Results' button click.

        It withdraw the current view, and brings the 'Save Results' screen.
    """
    global root
    root.withdraw()
    SaveResults.save_results_start(root)


class view_results_Screen:
    def __init__(self, bread_crumbs, top=None):
        """
            This class configures and populates the 'View Results' window.
            top is the toplevel containing window.

            In this screen, we show the results obtained from the training and predictions.

            It shows HeatMap, Clustering, Silhouette, and a table with final classification.
        """
        self.bread_crumbs = bread_crumbs
        w = 882
        h = 631
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
        self.main_ghazali_label.configure(anchor='nw')
        self.main_ghazali_label.configure(background="#ffffff")
        self.main_ghazali_label.configure(font="-family {Segoe UI} -size 14 -weight bold")
        self.main_ghazali_label.configure(foreground="#629b1c")
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

        self.save_button = tk.Button(top, command=save_button_click)
        self.save_button.place(x=680, y=582, height=33, width=188)
        self.save_button.configure(background="#629b1c")
        self.save_button.configure(activebackground="#ececec")
        self.save_button.configure(activeforeground="#000000")
        self.save_button.configure(disabledforeground="#a3a3a3")
        self.save_button.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.save_button.configure(foreground="#ffffff")
        self.save_button.configure(highlightbackground="#d9d9d9")
        self.save_button.configure(highlightcolor="#000000")
        self.save_button.configure(pady="0")
        self.save_button.configure(text='''Save''')

        self.back_button = tk.Button(top,
                                     command=back_button_click_to_CNN if self.bread_crumbs == "CNN"
                                     else back_button_click_to_load_trained)
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
        self.Label1_5_2.place(x=20, y=80, height=26, width=141)
        self.Label1_5_2.configure(activebackground="#f9f9f9")
        self.Label1_5_2.configure(activeforeground="black")
        self.Label1_5_2.configure(anchor='nw')
        self.Label1_5_2.configure(background="#ffffff")
        self.Label1_5_2.configure(disabledforeground="#a3a3a3")
        self.Label1_5_2.configure(font="-family {Segoe UI} -size 11")
        self.Label1_5_2.configure(foreground="#525252")
        self.Label1_5_2.configure(highlightbackground="#d9d9d9")
        self.Label1_5_2.configure(highlightcolor="black")
        self.Label1_5_2.configure(text='''Heat Map:''')

        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        show_results()

        self.flag = True
        self.h = None

        def heat_map_click(event):
            """
                Opens the heat map in a big window.
            """
            if self.flag:
                plt.close(utils.kmeans_plot)
                self.h = plt.gcf()
                self.flag = False
            else:
                Analyzer.produce_heatmap(big_size=True)
            plt.show()

        self.heatmap_canvas = tk.Canvas(top)
        self.heatmap_canvas.place(x=10, y=115, height=400, width=480)
        vsb2 = ttk.Scrollbar(top, orient="vertical", command=self.heatmap_canvas.yview)
        vsb2.place(x=8, y=115, height=400)

        self.heatmap_canvas.configure(yscrollcommand=vsb2.set)

        self.heatmap_canvas = FigureCanvasTkAgg(utils.heat_map_plot, master=self.heatmap_canvas)
        self.heatmap_canvas.draw()
        self.heatmap_canvas.get_tk_widget().bind("<Button-1>", heat_map_click)
        self.heatmap_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.kmeans_canvas = tk.Canvas(top)
        self.kmeans_canvas.place(x=490, y=110, height=225, width=365)
        self.kmeans_canvas = FigureCanvasTkAgg(utils.kmeans_plot, master=self.kmeans_canvas)
        self.kmeans_canvas.draw()
        self.kmeans_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.Label1_5_2_1 = tk.Label(top)
        self.Label1_5_2_1.place(x=490, y=80, height=26, width=161)
        self.Label1_5_2_1.configure(activebackground="#f9f9f9")
        self.Label1_5_2_1.configure(activeforeground="black")
        self.Label1_5_2_1.configure(anchor='nw')
        self.Label1_5_2_1.configure(background="#ffffff")
        self.Label1_5_2_1.configure(disabledforeground="#a3a3a3")
        self.Label1_5_2_1.configure(font="-family {Segoe UI} -size 13")
        self.Label1_5_2_1.configure(foreground="#525252")
        self.Label1_5_2_1.configure(highlightbackground="#d9d9d9")
        self.Label1_5_2_1.configure(highlightcolor="black")
        self.Label1_5_2_1.configure(text='''Cluster Centroids''')

        self.Label1_5_2_1_1 = tk.Label(top)
        self.Label1_5_2_1_1.place(x=490, y=347, height=26, width=141)
        self.Label1_5_2_1_1.configure(activebackground="#f9f9f9")
        self.Label1_5_2_1_1.configure(activeforeground="black")
        self.Label1_5_2_1_1.configure(anchor='nw')
        self.Label1_5_2_1_1.configure(background="#ffffff")
        self.Label1_5_2_1_1.configure(disabledforeground="#a3a3a3")
        self.Label1_5_2_1_1.configure(font="-family {Segoe UI} -size 13")
        self.Label1_5_2_1_1.configure(foreground="#525252")
        self.Label1_5_2_1_1.configure(highlightbackground="#d9d9d9")
        self.Label1_5_2_1_1.configure(highlightcolor="black")
        self.Label1_5_2_1_1.configure(text='''Silhouette Value:''')

        self.silhouette_value_label = tk.Label(top)
        self.silhouette_value_label.place(x=630, y=347, height=26, width=155)
        self.silhouette_value_label.configure(activebackground="#f9f9f9")
        self.silhouette_value_label.configure(activeforeground="black")
        self.silhouette_value_label.configure(anchor='nw')
        self.silhouette_value_label.configure(background="#ffffff")
        self.silhouette_value_label.configure(disabledforeground="#a3a3a3")
        if utils.silhouette_calc < utils.params['SILHOUETTE_THRESHOLD']:
            self.silhouette_value_label.configure(foreground="#E50000")
            self.silhouette_value_label.configure(font="-family {Segoe UI} -size 13 -weight bold")
        else:
            self.silhouette_value_label.configure(foreground="#525252")
            self.silhouette_value_label.configure(font="-family {Segoe UI} -size 13")
        self.silhouette_value_label.configure(highlightbackground="#d9d9d9")
        self.silhouette_value_label.configure(highlightcolor="black")
        self.silhouette_value_label.configure(text=f'''{utils.silhouette_calc}''')

        self.classification_results_table = ttk.Treeview(top, selectmode='browse')
        self.classification_results_table.place(x=490, y=420, height=127, width=365)

        vsb = ttk.Scrollbar(top, orient="vertical", command=self.classification_results_table.yview)
        vsb.place(x=490 + 365, y=420, height=127)

        self.classification_results_table.configure(yscrollcommand=vsb.set)

        self.classification_results_table["columns"] = ("1", "2")
        self.classification_results_table['show'] = 'headings'
        self.classification_results_table.column("1", width=285, anchor='c')
        self.classification_results_table.column("2", width=80, anchor='c')
        self.classification_results_table.heading("1", text="Book Name")
        self.classification_results_table.heading("2", text="Classification")
        self.classification_results_table.insert("", 'end', text="L1",
                                                 values=("Al-mankul min Taliqat al-Usul*",
                                                         "Ghazali" if utils.labels[0] == 0 else 'Not-Ghazali'))
        self.classification_results_table.insert("", 'end', text="L2",
                                                 values=("Al Mustasfa min ilm al-Usul",
                                                         "Ghazali" if utils.labels[1] == 0 else 'Not-Ghazali'))
        self.classification_results_table.insert("", 'end', text="L3",
                                                 values=("Fada’ih al-Batiniyya wa Fada’il al-Mustazhiriyy",
                                                         "Ghazali" if utils.labels[2] == 0 else 'Not-Ghazali'))
        self.classification_results_table.insert("", 'end', text="L4",
                                                 values=("Faysal at-Tafriqa Bayna al-Islam wa al-Zandaqa",
                                                         "Ghazali" if utils.labels[3] == 0 else 'Not-Ghazali'))
        self.classification_results_table.insert("", 'end', text="L5",
                                                 values=("Kitab al-iqtisad fi al-i’tiqad",
                                                         "Ghazali" if utils.labels[4] == 0 else 'Not-Ghazali'))
        self.classification_results_table.insert("", 'end', text="L6",
                                                 values=("Kitab Iljam Al- Awamm an Ilm Al-Kalam",
                                                         "Ghazali" if utils.labels[5] == 0 else 'Not-Ghazali'))
        self.classification_results_table.insert("", 'end', text="L7",
                                                 values=("Tahafut al-Falasifa",
                                                         "Ghazali" if utils.labels[6] == 0 else 'Not-Ghazali'))
        self.classification_results_table.insert("", 'end', text="L8",
                                                 values=("Ahliyi al-Madnun bihi ala ghayri",
                                                         "Ghazali" if utils.labels[7] == 0 else 'Not-Ghazali'))
        self.classification_results_table.insert("", 'end', text="L9",
                                                 values=("Kimiya-yi Sa’ādat*",
                                                         "Ghazali" if utils.labels[8] == 0 else 'Not-Ghazali'))
        self.classification_results_table.insert("", 'end', text="L10",
                                                 values=("Mishakat al-Anwar",
                                                         "Ghazali" if utils.labels[9] == 0 else 'Not-Ghazali'))

        self.Label1_5_2_1_1_1 = tk.Label(top)
        self.Label1_5_2_1_1_1.place(x=490, y=385, height=26, width=171)
        self.Label1_5_2_1_1_1.configure(activebackground="#f9f9f9")
        self.Label1_5_2_1_1_1.configure(activeforeground="black")
        self.Label1_5_2_1_1_1.configure(anchor='nw')
        self.Label1_5_2_1_1_1.configure(background="#ffffff")
        self.Label1_5_2_1_1_1.configure(disabledforeground="#a3a3a3")
        self.Label1_5_2_1_1_1.configure(font="-family {Segoe UI} -size 13")
        self.Label1_5_2_1_1_1.configure(foreground="#525252")
        self.Label1_5_2_1_1_1.configure(highlightbackground="#d9d9d9")
        self.Label1_5_2_1_1_1.configure(highlightcolor="black")
        self.Label1_5_2_1_1_1.configure(text='''Classification Results:''')

        if utils.labels[0] == utils.labels[8]:  # check if anchors belongs to the same cluster
            self.save_button.configure(state=tk.DISABLED)
            self.save_button.configure(background="#c0c0c0")
            from tkinter import messagebox as mb
            mb.showerror("Errors", "The books: Al-mankul min Taliqat al-Usul, Kimiya-yi Sa’ādat, \n"
                                   "are defined as anchors of different clusters, but classified as one.")
