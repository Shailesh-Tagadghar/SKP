# Install necessary libraries
#pip install diffusers transformers accelerate torch torchvision

# Import necessary modules
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from IPython.display import display

# Ensure you're using GPU for better performance
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Stable Diffusion model with enhanced capabilities
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",  # More advanced Stable Diffusion version
    torch_dtype=torch.float16  # Use 16-bit precision for speed and memory efficiency
)
pipe = pipe.to(device)

# Optimize memory usage for high-resolution images
pipe.enable_attention_slicing()
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Define the text prompt for high-quality image generation
text_prompt = "A hyper-realistic photograph of a futuristic city skyline at sunset with flying cars, 4k resolution, ultra-detailed, cinematic lighting"

# Generate an HD image with high settings
print("Generating high-quality image, please wait...")
with torch.no_grad():
    generated_image = pipe(
        prompt=text_prompt,
        num_inference_steps=50,  # More steps for higher quality
        guidance_scale=8.5,  # Increased adherence to prompt
        height=768,  # Higher resolution (vertical)
        width=768   # Higher resolution (horizontal)
    ).images[0]

# Display the HD image
display(generated_image)

# Save the generated HD image locally
generated_image.save("hd_generated_image.png")
print("HD image saved as 'hd_generated_image.png'")
