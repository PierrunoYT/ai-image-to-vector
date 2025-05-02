import os
import replicate
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("REPLICATE_API_TOKEN")

def generate_image(prompt, aspect_ratio="1:1", magic_prompt_option="Auto", progress=None):
    """
    Generate an image using Ideogram v3 Quality model via Replicate API

    Args:
        prompt (str): Text prompt describing the image to generate
        aspect_ratio (str): Aspect ratio of the generated image (e.g., "1:1", "16:9", "3:2")
        magic_prompt_option (str): Magic prompt option ("Auto", "On", "Off")
        progress: Optional progress callback function

    Returns:
        PIL.Image.Image: Generated image or None if generation failed

    Raises:
        ValueError: If the API key is not set or prompt is empty
        Exception: For other errors during image generation

    Notes:
        This function uses the Ideogram v3 Quality model from Replicate.
        The model generates high-quality images based on text prompts.

        Available aspect ratios:
        - Square: "1:1"
        - Landscape: "16:9", "16:10", "3:2", "4:3", "5:4", "2:1", "3:1"
        - Portrait: "9:16", "10:16", "2:3", "3:4", "4:5", "1:2", "1:3"

        Magic Prompt options:
        - "Auto": Automatically optimize the prompt
        - "On": Always optimize the prompt
        - "Off": Use the prompt as-is
    """
    # Check if API key is available
    if not api_key:
        raise ValueError("Replicate API token not found. Make sure to set the REPLICATE_API_TOKEN environment variable.")

    # Check if prompt is provided
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty. Please provide a description for the image.")

    # Update progress if provided
    if progress:
        progress(0.2, "Starting image generation...")

    try:
        # Run the Ideogram v3 Quality model
        output = replicate.run(
            "ideogram-ai/ideogram-v3-quality",
            input={
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "magic_prompt_option": magic_prompt_option,
                "seed": 0  # Use 0 for random seed
            }
        )

        # Update progress if provided
        if progress:
            progress(0.7, "Image generated, downloading...")

        # The output is a list with the first item being a FileOutput object
        if output and isinstance(output, list) and len(output) > 0:
            file_output = output[0]

            # Read the file content
            image_data = file_output.read()

            # Convert to PIL Image
            image = Image.open(BytesIO(image_data))

            # Update progress if provided
            if progress:
                progress(1.0, "Image generation complete!")

            return image
        else:
            raise ValueError(f"Unexpected response format from Replicate API: {output}")

    except Exception as e:
        print(f"Error during image generation: {e}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        # Check if API key is available
        if not api_key:
            print("ERROR: Replicate API token not found. Make sure to set the REPLICATE_API_TOKEN environment variable.")
            print("Create a .env file with your API token as REPLICATE_API_TOKEN=your_token")
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

        # Generate image
        print(f"\nGenerating image with Ideogram v3 Quality model...")
        print(f"Prompt: {prompt}")
        print(f"Aspect Ratio: {aspect_ratio}")
        print(f"Magic Prompt: {magic_prompt_option}")

        image = generate_image(prompt, aspect_ratio, magic_prompt_option)

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
