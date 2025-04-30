import tkinter as tk
from tkinter import scrolledtext

import seaborn as sns

from src.translate_repl import translate_with_confidence

# from translate_repl import translate_with_confidence # windows machine setting


class App(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.title("Eng2Span")
        # self.parent.configure(background="red")
        self.parent.minsize(500, 500)
        self.parent.geometry("300x300+50+50")  # controls where window shows up?

        # self.confidence_text = ""
        self.is_displaying_confidence = False

        # horrid component functions inside as components
        def translate_button():
            text = englishEntry.get()
            # print("english text entered:", text)
            print("translating...")
            translated_text, translated_scores = translate_with_confidence(text)

            norm_scores = [
                min(int(score * (len(palette) - 1)), len(palette) - 1)
                for score in translated_scores
            ]

            confidenceText.config(state="normal")
            confidenceText.delete("1.0", tk.END)
            # self.confidence_text = ""
            for word, score in zip(translated_text, translated_scores):
                # print(f"{word} - {score}")
                confidenceText.insert(tk.END, f"{word} - {round(score,3)}\n")
                # self.confidence_text += f"{word} - {score}\n"
            confidenceText.config(state="disabled")

            spanishText.config(
                state="normal"
            )  # unlock text box, must do before able to change things in it (people can type in it when unlocked)
            spanishText.delete("1.0", tk.END)  # delete previous text

            for word, color_index in zip(translated_text, norm_scores):
                spanishText.insert(tk.END, f"{word} ", str(color_index))

            spanishText.config(state="disabled")  # lock text box

        def confidence_button():
            confidenceText.config(state="normal")
            if not self.is_displaying_confidence:
                # confidenceText.insert(tk.END, self.confidence_text)
                confidenceText.pack()
                confidenceButton.config(text="Hide Confidence Scores")
                self.is_displaying_confidence = True
            else:
                # confidenceText.delete("1.0", tk.END)
                confidenceText.pack_forget()
                confidenceButton.config(text="Show Confidence Scores")
                self.is_displaying_confidence = False
            confidenceText.config(state="disabled")

        # components
        text_width = 50
        text_height = 7  # in lines
        englishLabel = tk.Label(self, text="Enter English to translate:")
        spanishLabel = tk.Label(self, text="Spanish Translation:")

        englishEntry = tk.Entry(self, width=text_width)
        translateButton = tk.Button(self, text="Translate!", command=translate_button)

        spanishText = tk.Text(self, width=text_width, height=text_height)
        spanishText.config(state="disabled")

        # setup tags for colored text
        palette = sns.color_palette("viridis_r", 20).as_hex()
        for ii in range(len(palette)):
            spanishText.tag_config(f"{ii}", foreground=palette[ii])
            # confidenceText.tag_config(f"{ii}", foreground=palette[ii])
        # print("num colors:", len(palette))

        # display the gradient
        box_size = 20
        padding = 10
        label_width = 25
        label_height = 10
        canvas_width = len(palette) * box_size + label_width * 2 + padding * 2
        canvas_height = box_size + label_height * 2
        gradient = tk.Canvas(self, width=canvas_width, height=canvas_height)
        gradient.pack(pady=10)

        for ii, color in enumerate(palette):
            x0 = label_width + ii * box_size
            y0 = label_height
            x1 = label_width + (ii + 1) * box_size
            y1 = label_height + box_size
            gradient.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        # Position the "0%" label on the left side
        gradient.create_text(
            label_width / 2 - padding, canvas_height / 2, text="0%", anchor=tk.W
        )

        # Position the "100%" label on the right side
        gradient.create_text(
            2 * label_width + len(palette) * box_size + padding,
            canvas_height / 2,
            text="100%",
            anchor=tk.E,
        )

        # how stuff will be layed out
        englishLabel.pack()
        englishEntry.pack()
        translateButton.pack(pady=10)
        spanishLabel.pack()
        spanishText.pack()

        # show/hide confidence scores
        confidenceButton = tk.Button(
            self, text="Show Confidence Scores", command=confidence_button
        )
        confidenceButton.pack(pady=10)

        confidenceText = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, width=int(text_width * 0.8), height=text_height
        )
        # confidenceText.pack()S


if __name__ == "__main__":
    root = tk.Tk()
    App(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
