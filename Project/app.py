# Libraries for building GUI 
import tkinter as tk
import customtkinter as ctk 

import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # Disable symlink creation for Hugging Face

# Machine Learning libraries 
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Libraries for processing image 
from PIL import Image, ImageTk

# private modules 
from authtoken import auth_token

# Create app user interface
app = tk.Tk()
app.geometry("532x632")
app.title("Text to Image Generator")
app.configure(bg='black')
ctk.set_appearance_mode("dark") 

# Create input box on the user interface 
prompt = ctk.CTkEntry(height=40, width=512, text_font=("Arial", 15), text_color="white", fg_color="black") 
prompt.place(x=10, y=10)

# Create a placeholder to show the generated image
img_placeholder = ctk.CTkLabel(height=512, width=512, text="")
img_placeholder.place(x=10, y=110)

# Initialize Stable Diffusion pipeline
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Check for CUDA availability
print(f"Using device: {device}")

stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
    modelid, 
    revision="fp16", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
    use_auth_token=auth_token
) 
stable_diffusion_model.to(device) 

# Generate image from text
def generate(): 
    """Generate an image from text using Stable Diffusion."""
    try:
        input_text = prompt.get()
        if not input_text.strip():
            raise ValueError("Prompt cannot be empty. Please enter a description.")
        
        with autocast(device) if device == "cuda" else torch.no_grad():
            image = stable_diffusion_model(input_text, guidance_scale=8.5).images[0]  # Updated API call
        
        # Save the generated image
        image.save('generatedimage.png')
        
        # Display the generated image on the user interface
        img = ImageTk.PhotoImage(image.resize((512, 512)))  # Resize for UI display
        img_placeholder.configure(image=img)
        img_placeholder.image = img  # Keep a reference to avoid garbage collection
        
        print("Image generated and displayed successfully.")
    except Exception as e:
        print(f"Error generating image: {e}")

# Add Generate button
trigger = ctk.CTkButton(
    height=40, 
    width=120, 
    text_font=("Arial", 15), 
    text_color="black", 
    fg_color="white", 
    command=generate
) 
trigger.configure(text="Generate")
trigger.place(x=206, y=60) 

# Start the GUI application
app.mainloop()
