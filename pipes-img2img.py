import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# Made by Bers Goudantov, 3ITAI AP Antwerp
class pipes_img2img:
    def __init__(self):
        pass

    # Function to crop the image to the center 512x512
    @staticmethod
    def center_crop_image(input_image, size=(512, 512)):
        """
        Crop the image to the center of the specified size.

        Args:
            input_image (PIL.Image): Input image to crop.
            size (tuple): Desired output size (width, height).

        Returns:
            PIL.Image: Cropped image.
        """
        width, height = input_image.size
        new_width, new_height = size

        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        return input_image.crop((left, top, right, bottom))

    # Function to generate image using Image2Image technique for the profile
    @staticmethod
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

        # Center crop the input image to 512x512 pixels
        input_image = pipes_img2img.center_crop_image(input_image, size=(512, 512))

        # Generate the new image from input image and prompt
        generated_images = pipe(
            prompt=prompt,
            image=input_image,
            strength=0.5,  # Lower strength for faster generation
            guidance_scale=5.0,  # Lower guidance for speed
            num_inference_steps=25  # Reduce inference steps for faster generation
        ).images

        # Save the generated image
        generated_image = generated_images[0]
        generated_image.save(output_image_path, format='PNG')
        print(f"Image saved at {output_image_path}")


    # Function to generate image using Image2Image technique for the posts
    @staticmethod
    def generate_image2image_post(input_image_path: str, prompt: str, output_image_path: str):
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

        # Center crop the input image to 512x512 pixels
        input_image = pipes_img2img.center_crop_image(input_image, size=(512, 512))

        # Generate the new image from input image and prompt
        generated_images = pipe(
            prompt=prompt,
            image=input_image,
            strength=0.5,  # Lower strength for faster generation
            guidance_scale=5.0,  # Lower guidance for speed
            num_inference_steps=25  # Reduce inference steps for faster generation
        ).images

        # Save the generated image
        generated_image = generated_images[0]
        generated_image.save(output_image_path, format='PNG')
        print(f"Image saved at {output_image_path}")

# Example usage:
if __name__ == "__main__":
    # Path to your input image
    input_img_path = "input_image_prof.png"  # Replace with your image path
    prompt_text = "Painted in soft, brushstrokes. The colors are rich and vivid"  # Example prompt
    output_img_path = "output_image_prof.png"  # Where the generated image will be saved

    # Generate the image
    pipes_img2img.generate_image2image_profile(input_img_path, prompt_text, output_img_path)
