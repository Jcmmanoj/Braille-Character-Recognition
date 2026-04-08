from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from glob import glob
from gtts import gTTS
from playsound import playsound
import os

# ===================== GLOBAL VARIABLE =====================
rep = []

# ===================== GUI CLASS =====================
class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.config(bg="pink")

        self.master.title("BRAILLE SCRIPT RECOGNITION")
        self.pack(fill=BOTH, expand=1)

        # Title Label
        title = tk.Label(
            root,
            text=" BRAILLE SCRIPT RECOGNITION",
            fg="#efffff",
            bg="purple",
            font="verdana 20 bold",
            width=30
        )
        title.pack()
        title.place(x=400, y=5)

        # Buttons
        Button(self, command=self.query, text="LOAD IMAGE",
               bg="#006666", fg="#efffff", activebackground="White", width=20).place(x=550, y=160)

        Button(self, command=self.preprocess, text="PREPROCESSING",
               bg="#006666", fg="#efffff", activebackground="White", width=20).place(x=550, y=360)

        Button(self, command=self.feature, text="FEATURE EXTRACTION",
               bg="#006666", fg="#efffff", activebackground="White", width=20).place(x=550, y=560)

        Button(self, command=self.classification, text="PREDICT",
               bg="#efffff", activebackground="White", fg="#2f3737", width=20).place(x=800, y=260)

        # Default logo (if available)
        try:
            img_pil = Image.open("logo.jpeg")
            render = ImageTk.PhotoImage(img_pil)
            for y in [80, 280, 480]:
                lbl = Label(self, image=render, borderwidth=15, highlightthickness=5,
                            height=150, width=150, bg='white')
                lbl.image = render
                lbl.place(x=350, y=y)
        except Exception:
            pass

    # ========== BUTTON FUNCTIONS ==========
    def query(self, event=None):
        global rep
        rep = filedialog.askopenfilenames(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not rep:
            return

        img = cv2.imread(rep[0])
        img = cv2.resize(img, (256, 256))
        self.from_array = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        render = ImageTk.PhotoImage(self.from_array)

        image1 = Label(self, image=render, borderwidth=15, highlightthickness=5,
                       height=150, width=150, bg='white')
        image1.image = render
        image1.place(x=350, y=50)

    def preprocess(self, event=None):
        global rep
        if not rep:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        img = cv2.imread(rep[0])
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Fixed: Use Image.fromarray instead of 'load'
        self.from_array = Image.fromarray(cv2.resize(hsv_img, (200, 200)))
        render = ImageTk.PhotoImage(self.from_array)

        image3 = Label(self, image=render, borderwidth=15, highlightthickness=5,
                       height=150, width=150, bg='white')
        image3.image = render
        image3.place(x=350, y=250)

    def feature(self, event=None):
        global rep
        if not rep:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        img = cv2.imread(rep[0])
        img = cv2.resize(img, (256, 256))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 0, 20])
        upper_green = np.array([100, 255, 255])
        mask = cv2.inRange(hsv_img, lower_green, upper_green)
        result = cv2.bitwise_and(img, img, mask=mask)

        self.from_array = Image.fromarray(cv2.resize(result, (200, 200)))
        render = ImageTk.PhotoImage(self.from_array)

        image3 = Label(self, image=render, borderwidth=15, highlightthickness=5,
                       height=150, width=150, bg='white')
        image3.image = render
        image3.place(x=350, y=450)

    def classification(self, event=None):
        global rep
        if not rep:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        clas1 = [item[5:-1] for item in sorted(glob("./dataa/*/"))]

        def path_to_tensor(img_path, width=224, height=224):
            img = image.load_img(img_path, target_size=(width, height))
            x = image.img_to_array(img)
            return np.expand_dims(x, axis=0)

        MODEL_PATH = r"C:\Users\jcmma\OneDrive\Desktop\Code 1 project\trained_model_CNN.h5"
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"Model file not found:\n{MODEL_PATH}")
            return

        model = load_model(MODEL_PATH)
        test_tensors = path_to_tensor(rep[0]) / 255.0
        pred = model.predict(test_tensors)
        res = clas1[np.argmax(pred)]
        messagebox.showinfo('Braille Recognition', f'Given Script is: {res}')

        # Convert to speech
        try:
            if os.path.exists("alphabet_audio.mp3"):
                os.remove("alphabet_audio.mp3")
            tts = gTTS(res, lang='en')
            tts.save("alphabet_audio.mp3")
            playsound("alphabet_audio.mp3")
        except Exception as e:
            print("Audio error:", e)

# ===================== MAIN =====================
root = Tk()
root.geometry("1400x720")
app = Window(root)
root.mainloop()







