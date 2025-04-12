import tkinter as tk

class App(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.title("Eng2Span")
        # self.parent.configure(background="red")
        self.parent.minsize(500,500)
        self.parent.geometry("300x300+50+50") # controls where window shows up?

        # horrid component functions inside as components
        def translate_button():
            text = englishEntry.get()
            print("english text entered:", text)
            print("trigger translation...")
            translated_text = text
            
            spanishText.config(state="normal") # unlock text box
            
            spanishText.insert(tk.END, translated_text)
            
            spanishText.config(state="disabled") # lock text box

        # components
        englishLabel = tk.Label(self, text="Enter English to translate:")
        spanishLabel = tk.Label(self, text="Spanish Translation:")

        englishEntry = tk.Entry(self, width=50)
        translateButton = tk.Button(self, text="Translate!", command=translate_button)

        spanishText = tk.Text(self, width=50)
        spanishText.config(state="disabled")

        # setup tags for colored text
        spanishText.tag_config("0", foreground="red") # ???
        
        
        # how stuff will be layed out
        englishLabel.pack()
        englishEntry.pack()
        translateButton.pack()
        spanishLabel.pack()
        spanishText.pack()


if __name__ == "__main__":
    root = tk.Tk()
    App(root).pack(side="top", fill="both", expand=True)
    root.mainloop()