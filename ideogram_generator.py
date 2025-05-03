import os
from dotenv import load_dotenv
from PIL import Image
from api_provider import get_provider

# Load environment variables
load_dotenv()

def generate_image(prompt, aspect_ratio="1:1", magic_prompt_option="Auto", style_type="auto", provider_name=None, progress=None):
    """
    Generate an image using Ideogram v3 model via the configured API provider

    Args:
        prompt (str): Text prompt describing the image to generate
        aspect_ratio (str): Aspect ratio of the generated image (e.g., "1:1", "16:9", "3:2")
        magic_prompt_option (str): Magic prompt option ("Auto", "On", "Off")
        style_type (str): Style type to use ("auto", "general", "realistic", "design", "none")
                         Case-insensitive, will be normalized by provider-specific mappers
        provider_name (str, optional): Name of the preferred provider ("replicate" or "fal")
        progress: Optional progress callback function

    Returns:
        PIL.Image.Image: Generated image or None if generation failed

    Raises:
        ValueError: If no API providers are configured or prompt is empty
        Exception: For other errors during image generation

    Notes:
        This function uses either the Replicate or Fal.ai API for Ideogram v3 model.
        The model generates high-quality images based on text prompts.

        Available aspect ratios:
        - Square: "1:1"
        - Landscape: "16:9", "16:10", "3:2", "4:3", "5:4", "2:1", "3:1"
        - Portrait: "9:16", "10:16", "2:3", "3:4", "4:5", "1:2", "1:3"

        Magic Prompt options:
        - "Auto": Automatically optimize the prompt
        - "On": Always optimize the prompt
        - "Off": Use the prompt as-is

        Style Type options:
        - "auto": Automatically select the style
        - "general": General style
        - "realistic": Realistic style
        - "design": Design style
        - "none": No specific style (maps to auto for some providers)
    """
    # Get the appropriate provider
    provider = get_provider(provider_name)

    # Generate the image using the provider
    return provider.generate_image(prompt, aspect_ratio, magic_prompt_option, style_type, progress)

# Example usage
if __name__ == "__main__":
    try:
        # Prompt for the API provider
        print("Available API providers:")
        print("1. Auto (use first available provider)")
        print("2. Replicate")
        print("3. Fal.ai")

        provider_choice = input("Choose an API provider (1-3, default: 1): ").strip() or "1"

        provider_map = {
            "1": None,  # Auto
            "2": "replicate",
            "3": "fal"
        }

        provider_name = provider_map.get(provider_choice)

        # Try to get the provider to check if it's configured
        try:
            provider = get_provider(provider_name)
            print(f"Using {provider.name} as the API provider")
        except ValueError as e:
            print(f"ERROR: {e}")
            exit(1)

        # Prompt for the image
        prompt = input("Enter a prompt to generate an image: ")

        # Validate input
        if not prompt:
            print("ERROR: No prompt provided.")
            exit(1)

        # Ask for aspect ratio
        print("\nAvailable aspect ratios:")
        print("1. Square (1:1)")
        print("2. Landscape (16:9)")
        print("3. Portrait (9:16)")
        print("4. Landscape (3:2)")
        print("5. Portrait (2:3)")
        print("6. Custom")

        aspect_choice = input("Choose an aspect ratio (1-6, default: 1): ").strip() or "1"

        aspect_ratio_map = {
            "1": "1:1",
            "2": "16:9",
            "3": "9:16",
            "4": "3:2",
            "5": "2:3"
        }

        if aspect_choice == "6":
            aspect_ratio = input("Enter custom aspect ratio (e.g., '4:3', '5:4'): ").strip() or "1:1"
        else:
            aspect_ratio = aspect_ratio_map.get(aspect_choice, "1:1")

        # Ask for magic prompt option
        print("\nMagic Prompt options:")
        print("1. Auto - Automatically optimize the prompt")
        print("2. On - Always optimize the prompt")
        print("3. Off - Use the prompt as-is")

        magic_choice = input("Choose a Magic Prompt option (1-3, default: 1): ").strip() or "1"

        magic_option_map = {
            "1": "Auto",
            "2": "On",
            "3": "Off"
        }

        magic_prompt_option = magic_option_map.get(magic_choice, "Auto")

        # Ask for style type
        print("\nStyle Type options:")
        print("1. Auto")
        print("2. General")
        print("3. Realistic")
        print("4. Design")

        style_choice = input("Choose a Style Type (1-4, default: 1): ").strip() or "1"

        style_type_map = {
            "1": "auto",
            "2": "general",
            "3": "realistic",
            "4": "design"
        }

        style_type = style_type_map.get(style_choice, "auto")

        # Generate image
        print(f"\nGenerating image with Ideogram v3 model...")
        print(f"Prompt: {prompt}")
        print(f"Aspect Ratio: {aspect_ratio}")
        print(f"Magic Prompt: {magic_prompt_option}")
        print(f"Style Type: {style_type}")

        image = generate_image(prompt, aspect_ratio, magic_prompt_option, style_type, provider_name)

        # Save the image
        output_filename = "generated_image.png"
        image.save(output_filename)
        print(f"✅ Image successfully generated and saved to {output_filename}")

        # Provide next steps
        print("\nNext steps:")
        print("1. To vectorize this image, run: python recraft_vectorizer.py")
        print("2. To use the web interface, run: python gradio_app.py")

    except ValueError as e:
        print(f"ERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Image generation failed: {e}")
        exit(1)
