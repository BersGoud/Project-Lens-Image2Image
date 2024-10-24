import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import random

# Made by Bers Goudantov, 3ITAI AP Antwerp
class PipesImg2Img:
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

    # "Surprise me" - randomly change the style of the image
    @staticmethod
    def random_style():
        """
        Returns a random style prompt for the "Surprise me" feature.
        """
        styles = [
            # Realistic Styles
            "Oil painting in the style of Rembrandt, detailed brush strokes, vibrant colors",
            "Realistic portrait painting, fine details, soft shadows, photorealistic textures",
            "Baroque oil painting, dramatic lighting, highly detailed textures",

            # Impressionist Styles
            "Impressionist oil painting, soft lighting, blurry strokes, pastel colors",
            "Post-Impressionist painting, vivid colors, emotional expression, dynamic brushwork",

            # Modern & Abstract Styles
            "Modern abstract oil painting, bold colors, artistic and expressive brushwork",
            "Cubist style, fragmented forms, multiple perspectives, muted tones",

            # Anime Styles
            "Anime-style illustration, vibrant colors, exaggerated expressions, whimsical backgrounds",
            "Fantasy anime character design, dynamic poses, ethereal lighting, magical elements",

            # Pixel Art Styles
            "Pixel art style, 8-bit graphics, bright colors, blocky character design",
            "Retro pixel art, nostalgic video game aesthetics, minimalistic details",

            # Cartoon Styles
            "Cartoon style, bold outlines, flat colors, humorous expressions",
            "Comic book style, dynamic action poses, exaggerated features, inking details",

            # Surreal Styles
            "Surrealist painting, dream-like imagery, bizarre combinations, vivid colors",
            "Dali-inspired surrealism, melting clocks, unexpected juxtapositions, detailed textures"
        ]
        return random.choice(styles)

    # Generate an oil painting-style profile image (512x512)
    @staticmethod
    def generate_image2image_profile(input_image_path: str, output_image_path: str, surprise_me=False):
        """
        Generate an oil painting-style profile image using the Stable Diffusion Image2Image pipeline.

        Args:
            input_image_path (str): Path to the input image.
            output_image_path (str): Path where the generated image will be saved.
            surprise_me (bool): If true, randomly selects a style for the image.

        Returns:
            None
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5")
        pipe = pipe.to(device)

        input_image = Image.open(input_image_path).convert("RGB")
        input_image = PipesImg2Img.crop_image(input_image, crop_type="profile", size=(512, 512))

        # Oil painting prompt or "Surprise me" style
        oil_painting_prompt = PipesImg2Img.random_style() if surprise_me else (
            "Oil painting, highly detailed, rich textures, classical portrait, "
            "soft brush strokes, realistic lighting, deep colors"
        )

        generated_images = pipe(
            prompt=oil_painting_prompt,
            image=input_image,
            strength=0.34,
            guidance_scale=7.5,
            num_inference_steps=30
        ).images

        generated_image = generated_images[0]
        generated_image.save(output_image_path, format='PNG')
        print(f"Oil painting-style profile image saved at {output_image_path}")

    # Generate an oil painting-style image for posts (landscape or profile)
    @staticmethod
    def generate_image2image_post(input_image_path: str, output_image_path: str, crop_type="landscape", surprise_me=False):
        """
        Generate an oil painting-style post image using the Stable Diffusion Image2Image pipeline.

        Args:
            input_image_path (str): Path to the input image.
            output_image_path (str): Path where the generated image will be saved.
            crop_type (str): Crop type - 'landscape' for wide images or 'profile' for character posts.
            surprise_me (bool): If true, randomly selects a style for the image.

        Returns:
            None
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5")
        pipe = pipe.to(device)

        input_image = Image.open(input_image_path).convert("RGB")

        if crop_type == "profile":
            input_image = PipesImg2Img.crop_image(input_image, crop_type="profile", size=(512, 512))
        else:
            input_image = PipesImg2Img.crop_image(input_image, crop_type="landscape")

        # Oil painting prompt or "Surprise me" style
        oil_painting_prompt = PipesImg2Img.random_style() if surprise_me else (
            "Oil painting of a scenic landscape, vibrant colors, lush details, dramatic lighting, "
            "detailed textures, realistic brush strokes"
        )

        generated_images = pipe(
            prompt=oil_painting_prompt,
            image=input_image,
            strength=0.34,
            guidance_scale=7.5,
            num_inference_steps=30
        ).images

        generated_image = generated_images[0]
        generated_image.save(output_image_path, format='PNG')
        print(f"Oil painting-style post image saved at {output_image_path}")

# Example usage:
if __name__ == "__main__":
    input_img_path = "img.png"  # Replace with your image path
    output_img_profile = "output_image_oil_painting_profile.png"
    output_img_landscape = "output_image_oil_painting_post.png"

    # Generate an oil painting-style profile image, is standardized to 512px to 512px
    PipesImg2Img.generate_image2image_profile(input_img_path, output_img_profile, surprise_me=True)

    # Generate an oil painting-style landscape or character post image, is standardized to 16:9 scale
    PipesImg2Img.generate_image2image_post(input_img_path, output_img_landscape, crop_type="landscape", surprise_me=True)
