import os
import json
from tkinter import Tk, Button, Label, filedialog, Canvas, Frame, LEFT, RIGHT
from PIL import Image, ImageTk
########

class ImageLabellerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeller")
        self.image_dir = ""
        self.images = []
        self.current_image_index = 0
        self.canvas = None
        self.bboxes = []
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.data = {}

        # Main container
        self.main_frame = Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        # Canvas for displaying images
        self.canvas = Canvas(self.main_frame, cursor="cross")
        self.canvas.pack(side=LEFT, fill="both", expand=True)

        # Sidebar for buttons
        self.sidebar = Frame(self.main_frame, width=100)
        self.sidebar.pack(side=RIGHT, fill="y", padx=5, pady=5)

        # GUI Elements
        self.label = Label(self.sidebar, text="Select a directory of images to start labelling.")
        self.label.pack(pady=10)

        self.open_button = Button(self.sidebar, text="Open Directory", command=self.open_directory)
        self.open_button.pack(pady=5, fill="x")

        self.next_button = Button(self.sidebar, text="Next Image", command=self.next_image, state="disabled")
        self.next_button.pack(pady=5, fill="x")

        self.undo_button = Button(self.sidebar, text="Undo", command=self.undo_last_bbox, state="disabled")
        self.undo_button.pack(pady=5, fill="x")

        self.save_button = Button(self.sidebar, text="Save JSON", command=self.save_json, state="disabled")
        self.save_button.pack(pady=5, fill="x")

        # Bind mouse events for drawing bounding boxes
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def open_directory(self):
        self.image_dir = filedialog.askdirectory()
        if self.image_dir:
            self.images = [f for f in os.listdir(self.image_dir) if f.endswith(".png")]
            if self.images:
                self.current_image_index = 0
                self.load_image()
                self.next_button.config(state="normal")
                self.save_button.config(state="normal")
                self.undo_button.config(state="normal")
            else:
                self.label.config(text="No PNG images found in the directory.")

    def load_image(self):
        if self.images:
            image_path = os.path.join(self.image_dir, self.images[self.current_image_index])
            self.image = Image.open(image_path)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.image.width, height=self.image.height)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.label.config(text=f"Image {self.current_image_index + 1} of {len(self.images)}")
            self.bboxes = []

    def next_image(self):
        if self.images:
            self.save_current_image_data()
            self.current_image_index = (self.current_image_index + 1) % len(self.images)
            self.load_image()

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_mouse_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        end_x, end_y = event.x, event.y
        self.bboxes.append((self.start_x, self.start_y, end_x, end_y))

    def undo_last_bbox(self):
        if self.bboxes:
            self.bboxes.pop()  # Remove the last bounding box
            self.redraw_canvas()

    def redraw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        for bbox in self.bboxes:
            self.canvas.create_rectangle(bbox, outline="red")

    def save_current_image_data(self):
        if self.images and self.bboxes:
            image_name = os.path.splitext(self.images[self.current_image_index])[0]
            self.data[image_name] = self.bboxes

    def save_json(self):
        if self.data:
            save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if save_path:
                with open(save_path, "w") as f:
                    json.dump(self.data, f, indent=4)
                self.label.config(text=f"Data saved to {save_path}")

if __name__ == "__main__":
    root = Tk()
    app = ImageLabellerApp(root)
    root.mainloop()
