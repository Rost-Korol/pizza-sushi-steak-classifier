import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import random
import torch
from utils import *

# GUI for upload image and makes predict
class ImageApp:
    def __init__(self, master, model):
        self.master = master
        master.title("Pizza steak or sushi")

        # Create Canvas and Label
        self.model = model
        self.canvas = tk.Canvas(master, width=400, height=400, bg='white')
        self.canvas.pack()
        self.label = tk.Label(self.canvas, text="Click the button to upload an image...",
                              fg='#B2B2B2',
                              font=("Helvetica", 14),
                              background='white')

        self.label.place(relx=0.5, rely=0.5, anchor="center")

        # Create Buttons
        self.upload_button = tk.Button(master, text="Upload Image",
                                       command=self.upload_image,
                                       bg="#3A98B9",
                                       activebackground='#6091a6',
                                       fg="white",
                                       activeforeground='white',
                                       font=("Helvetica", 14))

        self.show_button = tk.Button(master, text="Make Predict",
                                     command=self.predict,
                                     bg="#3A98B9",
                                     activebackground='#6091a6',
                                     fg="white",
                                     activeforeground='white',
                                     font=("Helvetica", 14))

        # Add Buttons to Window
        self.upload_button.pack(side="left", padx=10, pady=10)
        self.show_button.pack(side="right", padx=10, pady=10)

    def upload_image(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename()

        # Delete old image if exist
        try:
            self.uploaded_image_label.destroy()
        except AttributeError:
            pass

        # Load selected image
        self.image = Image.open(file_path)
        self.image.thumbnail((400, 400))

        # Convert image to Tkinter PhotoImage format
        self.photo = ImageTk.PhotoImage(self.image)

        # Show uploaded image
        self.uploaded_image_label = tk.Label(self.canvas, image=self.photo)
        self.uploaded_image_label.pack()
        self.label.pack_forget()

    def predict(self):
        try:
            self.model.eval()
            with torch.inference_mode():
                img = image_transform(self.image)
                img = torch.unsqueeze(img, 0)
                pred_logits = self.model(img)
                # pred_proba = torch.softmax(pred_logits.squeeze(), dim=0)
                pred_class = pred_logits.argmax(dim=1)
                class_name = class_names[pred_class.cpu()]
                self.popup_bonus(class_name)

        except (AttributeError, TypeError):
            # Show error message if no image has been uploaded
            self.label.config(text="You didn't upload image...\nPlease, upload one")

    def popup_bonus(self, class_name):

        def update_canvas():
            self.uploaded_image_label.destroy()
            self.canvas.config(width=400, height=400)
            self.label.config(text='Click the button to upload an image...')

        def image_delete():
            self.image = None

        win = tk.Toplevel()
        win.wm_title("Predict")
        user_answer = f"{random.choice(guessing_phrases)} {class_name}"
        l = tk.Label(win, text=user_answer, font=('Helvetica', 14))
        win.geometry('{}x{}'.format(l.winfo_reqwidth() + 40, 45))
        l.pack()
        win.after(3000, lambda: [win.destroy(), update_canvas(), image_delete()])


def main():

    model = load_model()
    root = tk.Tk()
    my_app = ImageApp(root, model)
    root.mainloop()

if __name__ == '__main__':
    main()
