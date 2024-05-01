import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox

def on_canvas_click(event):
    # Calculate the position of the click relative to the center of the image
    center_x, center_y = 335, 335  # Image dimensions are 670x670, center is at 335
    delta_x = event.x - center_x
    delta_y = center_y - event.y

    # Normalize valence and arousal to the range 0-9
    valence = ((delta_x / 335 + 1) / 2) * 9
    arousal = ((delta_y / 335 + 1) / 2) * 9

    # Store the values globally and exit
    global user_valence, user_arousal
    user_valence, user_arousal = valence, arousal
    root.destroy()  # Close the window

def get_values():
    global root  # Declare root as global to access it inside on_canvas_click
    root = tk.Tk()
    root.title("Emotion Selector")

    # Load and display the image
    image = Image.open("Linear_RGB_color_wheel_2.png")
    image = image.resize((670, 670), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)

    # Link image to GUI
    emotion_canvas = tk.Canvas(root, width=670, height=670)
    emotion_canvas.pack()
    emotion_canvas.create_image(335, 335, image=photo)  # Center the image on the canvas

    # Link click to GUI
    emotion_canvas.bind("<Button-1>", on_canvas_click)

    root.mainloop()  # Start the GUI event loop

    # After the window is closed, return the values
    try:
        return user_valence, user_arousal
    except NameError:
        messagebox.showerror("Error", "No selection made")
        return None, None

# Optionally test the function
if __name__ == "__main__":
    valence, arousal = get_values()
    if valence is not None and arousal is not None:
        print(f"Valence: {valence:.2f}, Arousal: {arousal:.2f}")
