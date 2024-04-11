import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
from fall_detection import start_fall_detector_realtime

class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fall Detection App")

        # Load the background image
        self.original_image = Image.open("ui_bg.jpg")  
        self.resized_image = self.original_image.resize((600, 400),resample=Image.LANCZOS)  # Resize image to fit canvas
        self.bg_photo = ImageTk.PhotoImage(self.resized_image)

        # Create a canvas to put the background image
        self.canvas = tk.Canvas(self.root, width=600, height=400)
        self.canvas.pack()

        # Put the background image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_photo)

        self.file_path = ""

        self.create_widgets()

    def create_widgets(self):
        # Create a frame to contain the buttons
        button_frame = tk.Frame(self.root, bg="#F5F5F5")
        button_frame.pack(expand=True)

        self.select_file_button = tk.Button(button_frame, text="Select Video File", command=self.select_file)
        self.select_file_button.pack(pady=10)

        self.file_label = tk.Label(button_frame, text="No file selected")
        self.file_label.pack(pady=5)

        self.run_button = tk.Button(button_frame, text="Run Fall Detection", command=self.run_fall_detection)
        self.run_button.pack(pady=10)

        # Center the button frame in the window
        button_frame.place(relx=0.5, rely=0.35, anchor=tk.CENTER)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Video files", ".mp4;.avi")])
        if self.file_path:
            self.file_label.config(text="File selected")
        else:
            self.file_label.config(text="No file selected")

    def run_fall_detection(self):
        if self.file_path:
            threading.Thread(target=self.start_fall_detection_thread).start()
        else:
            messagebox.showerror("Error", "Please select a video file first.")

    def start_fall_detection_thread(self):
        start_fall_detector_realtime(self.file_path)

def main():
    root = tk.Tk()
    app = FallDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()