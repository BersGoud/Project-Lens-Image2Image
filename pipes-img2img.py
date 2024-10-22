import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# Made by Bers Goudantov, 3ITAI AP Antwerp
class pipes_img2img:
    def __init__(self):
        pass

    # Function to generate image using Image2Image technique for the profile
    def generate_image2image_profile(input_image_path: str, prompt: str, output_image_path: str):
        """
        Generate an image based on an input image and a text prompt using the Stable Diffusion Image2Image pipeline.
        
        Args:
            input_image_path (str): Path to the input image.
            prompt (str): Text prompt to guide the image generation.
            output_image_path (str): Path where the generated image will be saved.
        
        Returns:
            None
        """
        # Load the model pipeline for image2image generation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5")
        pipe = pipe.to(device)

        # Load the input image using Pillow
        input_image = Image.open(input_image_path).convert("RGB")
        input_image = input_image.resize((512, 512))  # Resize if necessary

        # Generate the new image from input image and prompt
        generated_images = pipe(prompt=prompt, image=input_image, strength=0.75, guidance_scale=7.5).images

        # Save the generated image
        generated_image = generated_images[0]
        generated_image.save(output_image_path, format='PNG')
        print(f"Image saved at {output_image_path}")


    # Function to generate image using Image2Image technique for the posts
    def generate_image2image_post(input_image_path: str, prompt: str, output_image_path: str):
        """<
        Generate an image based on an input image and a text prompt using the Stable Diffusion Image2Image pipeline.
        
        Args:
            input_image_path (str): Path to the input image.
            prompt (str): Text prompt to guide the image generation.
            output_image_path (str): Path where the generated image will be saved.
        
        Returns:
            None
        """
        # Load the model pipeline for image2image generation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5")
        pipe = pipe.to(device)

        # Load the input image using Pillow
        input_image = Image.open(input_image_path).convert("RGB")
        input_image = input_image.resize((512, 512))  # Resize if necessary

        # Generate the new image from input image and prompt
        generated_images = pipe(prompt=prompt, image=input_image, strength=0.75, guidance_scale=7.5).images

        # Save the generated image
        generated_image = generated_images[0]
        generated_image.save(output_image_path, format='PNG')
        print(f"Image saved at {output_image_path}")

# Example usage:
if __name__ == "__main__":
    # Path to your input image
    input_img_path = "input_image_prof.png"  # Replace with your image path
    prompt_text = "In the style Van Gogh"  # Example prompt
    output_img_path = "output_image_prof.png"  # Where the generated image will be saved

    # Generate the image
    
    pipes_img2img.generate_image2image_profile(input_img_path, prompt_text, output_img_path)
