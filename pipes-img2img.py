import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# Made by Bers Goudantov, 3ITAI AP Antwerp
class pipes_img2img:
    def __init__(self):
        pass

    # Function to crop the image for profile (center 512x512) or landscape (standard landscape ratio)
    @staticmethod
    def crop_image(input_image, crop_type="profile", size=(512, 512)):
        """
        Crop the image based on the desired crop type (profile or landscape).
        
        Args:
            input_image (PIL.Image): Input image to crop.
            crop_type (str): Type of cropping - 'profile' or 'landscape'.
            size (tuple): Desired output size (width, height). For profile it defaults to 512x512.

        Returns:
            PIL.Image: Cropped image.
        """
        width, height = input_image.size
        
        if crop_type == "profile":
            # For profile images, crop the image to 512x512 (centered)
            new_width, new_height = size
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2
            return input_image.crop((left, top, right, bottom))
        
        elif crop_type == "landscape":
            # For landscape, we crop to a 16:9 ratio
            landscape_ratio = (16, 9)
            new_width = min(width, int(height * landscape_ratio[0] / landscape_ratio[1]))
            new_height = min(height, int(width * landscape_ratio[1] / landscape_ratio[0]))
            
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2
            return input_image.crop((left, top, right, bottom))

    # Function to generate an anime-style image for a character-focused profile image
    @staticmethod
    def generate_image2image_anime_profile(input_image_path: str, output_image_path: str):
        """
        Generate an anime-style profile image (character-focused) using the Stable Diffusion Image2Image pipeline.

        Args:
            input_image_path (str): Path to the input image.
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

        # Center crop the input image to 512x512 pixels (profile image)
        input_image = pipes_img2img.crop_image(input_image, crop_type="profile", size=(512, 512))

        # Define the anime-style prompt with a focus on the character's profile
        anime_profile_prompt = (
            "Anime-style character portrait, highly detailed, sharp line art, "
            "soft lighting, vivid colors, expressive eyes, fine character details, "
            "Studio Ghibli style, dynamic pose, beautiful lighting"
        )

        # Generate the new image from input image and anime-style prompt
        generated_images = pipe(
            prompt=anime_profile_prompt,
            image=input_image,
            strength=0.6,  # Control transformation strength
            guidance_scale=7.5,  # Reduced guidance for smoother transformation
            num_inference_steps=25  # More steps for finer details
        ).images

        # Save the generated image
        generated_image = generated_images[0]
        generated_image.save(output_image_path, format='PNG')
        print(f"Anime-style profile image saved at {output_image_path}")

    # Function to generate an anime-style image for a landscape or character-focused post
    @staticmethod
    def generate_image2image_anime_post(input_image_path: str, output_image_path: str, crop_type="landscape"):
        """
        Generate an anime-style image for posts (either landscape or character-focused) using the Stable Diffusion Image2Image pipeline.

        Args:
            input_image_path (str): Path to the input image.
            output_image_path (str): Path where the generated image will be saved.
            crop_type (str): Crop type - 'landscape' for wide images or 'profile' for character posts.

        Returns:
            None
        """
        # Load the model pipeline for image2image generation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5")
        pipe = pipe.to(device)

        # Load the input image using Pillow
        input_image = Image.open(input_image_path).convert("RGB")

        # Crop the image to either profile (512x512) or landscape (16:9)
        if crop_type == "profile":
            input_image = pipes_img2img.crop_image(input_image, crop_type="profile", size=(512, 512))
        else:
            input_image = pipes_img2img.crop_image(input_image, crop_type="landscape")

        # Define the anime-style prompt with a focus on landscape or dynamic character posts
        anime_post_prompt = (
            "Anime-style scene, dynamic action pose, highly detailed characters, "
            "vivid colors, lush backgrounds, beautiful lighting, artistic brush strokes, "
            "vibrant aesthetics, Studio Ghibli inspired, epic landscape, emotional atmosphere"
        )

        # Generate the new image from input image and anime-style prompt
        generated_images = pipe(
            prompt=anime_post_prompt,
            image=input_image,
            strength=0.6,  # Control transformation strength
            guidance_scale=7.5,  # Reduced guidance for smoother results
            num_inference_steps=25  # More steps for finer details
        ).images

        # Save the generated image
        generated_image = generated_images[0]
        generated_image.save(output_image_path, format='PNG')
        print(f"Anime-style post image saved at {output_image_path}")

# Example usage:
if __name__ == "__main__":
    # Path to your input image
    input_img_path = "img.png"  # Replace with your image path
    output_img_profile = "output_image_anime_profile.png"  # Where the generated profile image will be saved
    output_img_landscape = "output_image_anime_post.png"  # Where the generated landscape image will be saved

    # Generate the anime-style profile (character) image
    pipes_img2img.generate_image2image_anime_profile(input_img_path, output_img_profile)

    # Generate the anime-style landscape or character post image
    pipes_img2img.generate_image2image_anime_post(input_img_path, output_img_landscape, crop_type="landscape")
