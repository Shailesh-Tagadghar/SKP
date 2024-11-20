# Install the necessary libraries
# pip install diffusers transformers accelerate torch torchvision

# Import necessary modules
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from IPython.display import display

# Ensure you're using GPU for better performance
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Stable Diffusion pre-trained pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to(device)

# Enable faster generation with optimizations
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  # Faster scheduler
pipe.enable_attention_slicing()  # Reduce memory usage

# Define the text prompt
text_prompt = "A student presenting a Presentation to a class and teacher"

# Generate the image
print("Generating image, please wait...")
with torch.no_grad():
    generated_image = pipe(prompt=text_prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

# Display the generated image
display(generated_image)

# Save the generated image locally
generated_image.save("generated_image.png")
print("Image saved as 'generated_image.png'")
