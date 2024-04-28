import tkinter as tk
from PIL import Image, ImageTk

def on_canvas_click(event):
    # canvas xy
    canvas_x = event.x
    canvas_y = event.y
    
    # center of the image 
    center_x, center_y = 335, 335  # Image dimensions are 670x670, center is at 335
    
    # Delta X and Delta Y, oisition between click and the center 
    delta_x = canvas_x - center_x
    delta_y = center_y - canvas_y  
    
    # normalize
    valence = (delta_x / 335 + 1) / 2
    arousal = ( delta_y / 335+1) / 2
    
   
    save_values(valence, arousal)

def save_values(valence, arousal):
    with open("emotion_values.txt", "a") as file:
        file.write(f"{valence:.2f}, {arousal:.2f}\n")
        print("Values saved:", valence, arousal)  # 

# gui
root = tk.Tk()
root.title("Emotion Selector")

image = Image.open("Linear_RGB_color_wheel_2.png")
image = image.resize((670, 670), Image.Resampling.LANCZOS)
photo = ImageTk.PhotoImage(image)

# link image to gui
emotion_canvas = tk.Canvas(root, width=670, height=670)
emotion_canvas.pack()
emotion_canvas.create_image(335, 335, image=photo)  # Center the image on the canvas

# link click t gui
emotion_canvas.bind("<Button-1>", on_canvas_click)


root.mainloop()
