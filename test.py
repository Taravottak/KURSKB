import tkinter as tk
from PIL import Image

root = tk.Tk()
root.title("Displaying GIF")

file = "giphy.gif"
info = Image.open(file)

frames = info.n_frames  # number of frames

photoimage_objects = []
for i in range(frames):
    obj = tk.PhotoImage(file=file, format=f"gif -index {i}")
    photoimage_objects.append(obj)

def animation(current_frame=0):
    global loop
    image = photoimage_objects[current_frame]
    gif_label.configure(image=image)

    current_frame = (current_frame + 1) % frames  # Use modulo to loop through frames
    loop = root.after(50, lambda: animation(current_frame))  # Pass current_frame to the function

gif_label = tk.Label(root, image="")
gif_label.pack()

animation()  # Start the animation

root.mainloop()
