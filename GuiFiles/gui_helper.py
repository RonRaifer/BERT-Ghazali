from time import time, localtime, strftime

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


def on_key_release_apply_defaults(event):
    r"""
    Changes border of widget(Entry) to black.

    :param event: Generated from bind function
    """
    event.widget.configure(highlightbackground="#d9d9d9")
    event.widget.configure(highlightcolor="black")


def entry_widget_defaults(widget):
    r"""
    Configures default parameters for styling and actions for Entry widget.

    :param widget: The Entry widget (tkinter)
    """
    widget.configure(background="white")
    widget.configure(font="-family {Segoe UI} -size 11")
    widget.configure(foreground="#808080")
    widget.configure(highlightbackground="#d9d9d9")
    widget.configure(highlightcolor="black")
    widget.configure(highlightthickness=1)
    widget.configure(insertbackground="black")
    widget.configure(selectbackground="blue")
    widget.configure(selectforeground="white")
    widget.bind("<KeyRelease>", on_key_release_apply_defaults)


def big_button_style(button, text):
    """
    Configures the style and text of the big buttons.
    :param button: A tkinter Button widget.
    :param text: The text to show on button.
    """
    button.configure(activebackground="#ececec")
    button.configure(activeforeground="#000000")
    button.configure(background="#629b1c")
    button.configure(disabledforeground="#a3a3a3")
    button.configure(font="-family {Segoe UI} -size 14 -weight bold")
    button.configure(foreground="#ffffff")
    button.configure(highlightbackground="#d9d9d9")
    button.configure(highlightcolor="black")
    button.configure(pady="0")
    button.configure(text=f'''{text}''')


def tooltip_message(widget, message):
    r"""
    Attaches ToolTip box to the widget (tkinter), with proper message.

    :param widget: Any tkinter widget to attach ToolTip to.
    :param message: The message to be shown while mouse points on widget.
    """
    ToolTip(widget, "-family {Segoe UI} -size 11", message)


def isint_and_inrange(n, start, end):
    """
    Checks if parameter is int and in range (start,end).
    :param n: the checked parameter.
    :param start: the start number of the range.
    :param end: the end number of the range.
    :return: True if parameter is int and in the given range, else False.
    """
    x = False
    try:
        int(n)
        x = True if start <= int(n) <= end else False
        return x
    except ValueError:
        return x


def isfloat_and_inrange(n, start, end):
    """
    Checks if parameter is float and in range (start,end).
    :param n: the checked parameter.
    :param start: the start number of the range.
    :param end: the end number of the range.
    :return: True if parameter is float and in the given range, else False.
    """
    x = False
    try:
        float(n)
        x = True if start < float(n) < end else False
        return x
    except ValueError:
        return x


def isfloat(s):
    """
    Checks if parameter is float.
    :param s: the checked parameter.
    :return: True if value is float, else False.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def isint(s):
    """
        Checks if parameter is int.
        :param s: the checked parameter.
        :return: True if value is int, else False.
        """
    try:
        int(s)
        return True
    except ValueError:
        return False


# ===========================================================
#                   Start Class ToolTip
# ===========================================================

class ToolTip(tk.Toplevel):
    r"""
        Provides a ToolTip widget for Tkinter.
        To apply a ToolTip to any Tkinter widget, simply pass the widget to the
        ToolTip constructor
    """

    def __init__(self, wdgt, tooltip_font, msg=None, msgFunc=None,
                 delay=0.1, follow=True):
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
        tk.Message(self, textvariable=self.msgVar, bg='#E5EBD8',
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
