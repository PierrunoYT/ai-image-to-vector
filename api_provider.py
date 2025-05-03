import os
import replicate
import logging
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import requests
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('image_to_vector.api_provider')

# Load environment variables
load_dotenv()

# Get API keys from environment variables
REPLICATE_API_KEY = os.getenv("REPLICATE_API_TOKEN")
FAL_API_KEY = os.getenv("FAL_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class APIProvider:
    """Base class for API providers"""
    def __init__(self):
        self.name = "Base Provider"

    def generate_image(self, prompt, aspect_ratio="1:1", magic_prompt_option="Auto", style_type="auto", progress=None):
        """
        Generate an image using the provider's API

        Args:
            prompt (str): Text prompt describing the image to generate
            aspect_ratio (str): Aspect ratio of the generated image (e.g., "1:1", "16:9", "3:2")
            magic_prompt_option (str): Magic prompt option ("Auto", "On", "Off")
            style_type (str): Style type to use ("auto", "general", "realistic", "design", "none")
                             Case-insensitive, will be normalized by provider-specific mappers
            progress: Optional progress callback function

        Returns:
            PIL.Image.Image: Generated image
        """
        # Check if progress is provided and is a valid callable
        # Avoid accessing progress attributes directly to prevent IndexError
        if progress is not None and hasattr(progress, "__call__"):
            try:
                progress(0.2, "Starting image generation...")
            except Exception as e:
                logger.warning(f"Error updating progress: {e}")
        raise NotImplementedError("This method must be implemented by subclasses")

    def is_configured(self):
        """Check if the provider is properly configured"""
        raise NotImplementedError("This method must be implemented by subclasses")


class ReplicateProvider(APIProvider):
    """Replicate API provider for Ideogram v3 Quality model"""
    def __init__(self):
        super().__init__()
        self.name = "Replicate"
        self.api_key = REPLICATE_API_KEY

    def is_configured(self):
        """Check if Replicate API is properly configured"""
        return self.api_key is not None and len(self.api_key.strip()) > 0

    def _map_style_type_for_replicate(self, style_type):
        """
        Map style type to the format expected by Replicate API

        Args:
            style_type (str): Style type from the UI (case-insensitive)

        Returns:
            str: Style type in the format expected by Replicate API ("None", "Auto", "General", "Realistic", "Design")
        """
        # Map to the format expected by Replicate API (which uses title case)
        style_map = {
            "auto": "Auto",
            "general": "General",
            "realistic": "Realistic",
            "design": "Design",
            "none": "None"
        }

        # Convert to lowercase for case-insensitive matching
        style_key = style_type.lower() if style_type else "auto"
        return style_map.get(style_key, "Auto")

    def generate_image(self, prompt, aspect_ratio="1:1", magic_prompt_option="Auto", style_type="auto", progress=None):
        """
        Generate an image using Ideogram v3 Quality model via Replicate API

        Args:
            prompt (str): Text prompt describing the image to generate
            aspect_ratio (str): Aspect ratio of the generated image (e.g., "1:1", "16:9", "3:2")
            magic_prompt_option (str): Magic prompt option ("Auto", "On", "Off")
            style_type (str): Style type to use ("None", "Auto", "General", "Realistic", "Design")
            progress: Optional progress callback function

        Returns:
            PIL.Image.Image: Generated image or None if generation failed
        """
        # Check if API key is available
        if not self.is_configured():
            raise ValueError("Replicate API token not found. Make sure to set the REPLICATE_API_TOKEN environment variable.")

        # Check if prompt is provided
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty. Please provide a description for the image.")

        # Update progress if provided
        if progress is not None and hasattr(progress, "__call__"):
            try:
                progress(0.2, "Starting image generation with Replicate...")
            except Exception as e:
                logger.warning(f"Error updating progress: {e}")

        try:
            # Map style_type to the format expected by Replicate API
            replicate_style_type = self._map_style_type_for_replicate(style_type)

            # Run the Ideogram v3 Quality model
            # Note: The Replicate API response format can vary depending on the version of the replicate-python client
            # We explicitly set use_file_output=False to get URL strings instead of FileOutput objects
            # This ensures consistent behavior across different versions of the replicate-python client

            # Initialize the Replicate client with a timeout
            replicate_client = replicate.Client(
                api_token=self.api_key,
                timeout=120  # 2 minute timeout for the API call
            )

            output = replicate_client.run(
                "ideogram-ai/ideogram-v3-quality",
                input={
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "magic_prompt_option": magic_prompt_option,
                    "style_type": replicate_style_type,
                    "seed": 0  # Use 0 for random seed
                },
                use_file_output=False  # Force URL string output format
            )

            # Update progress if provided
            if progress is not None and hasattr(progress, "__call__"):
                try:
                    progress(0.7, "Image generated, downloading...")
                except Exception as e:
                    logger.warning(f"Error updating progress: {e}")

            # Process the response from Replicate API
            # With use_file_output=False, the output should be a URL string or a list of URL strings

            # Log debug information
            logger.debug(f"Replicate API response type: {type(output)}")
            logger.debug(f"Replicate API response: {output}")

            # Case 1: Output is a list of URL strings
            if output and isinstance(output, list) and len(output) > 0:
                image_url = output[0]
                logger.debug(f"Processing as list, first item: {image_url}")

                # Download the image from URL
                try:
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()

                    # Convert to PIL Image
                    image = Image.open(BytesIO(response.content))
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error downloading image from URL: {e}")
                    raise ValueError(f"Failed to download image from URL: {image_url}. Error: {e}")
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    raise ValueError(f"Failed to process image from URL: {image_url}. Error: {e}")

                # Update progress if provided
                if progress is not None and hasattr(progress, "__call__"):
                    try:
                        progress(1.0, "Image generation complete!")
                    except Exception as e:
                        logger.warning(f"Error updating progress: {e}")

                return image

            # Case 2: Output is a direct URL string
            elif output and isinstance(output, str):
                logger.debug(f"Processing as string URL: {output}")

                # Download the image from URL
                try:
                    response = requests.get(output, timeout=30)
                    response.raise_for_status()

                    # Convert to PIL Image
                    image = Image.open(BytesIO(response.content))
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error downloading image from URL: {e}")
                    raise ValueError(f"Failed to download image from URL: {output}. Error: {e}")
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    raise ValueError(f"Failed to process image from URL: {output}. Error: {e}")

                # Update progress if provided
                if progress is not None and hasattr(progress, "__call__"):
                    try:
                        progress(1.0, "Image generation complete!")
                    except Exception as e:
                        logger.warning(f"Error updating progress: {e}")

                return image

            # If none of the above formats match, raise an error
            else:
                logger.error(f"Failed to process output, type: {type(output)}, value: {output}")
                raise ValueError(f"Unexpected response format from Replicate API: {output}")

        except Exception as e:
            logger.error(f"Error during image generation with Replicate: {e}")
            raise


class FalProvider(APIProvider):
    """Fal.ai API provider for Ideogram v3 model"""
    def __init__(self):
        super().__init__()
        self.name = "Fal.ai"
        self.api_key = FAL_API_KEY

    def is_configured(self):
        """Check if Fal.ai API is properly configured"""
        return self.api_key is not None and len(self.api_key.strip()) > 0

    def _map_aspect_ratio_to_image_size(self, aspect_ratio):
        """Map aspect ratio to image size for Fal.ai API"""
        # Map common aspect ratios to Fal.ai image size options
        ratio_map = {
            "1:1": "square_hd",
            "16:9": "landscape_16_9",
            "9:16": "portrait_16_9",
            "4:3": "landscape_4_3",
            "3:4": "portrait_4_3"
        }

        # Return the mapped value or default to square_hd
        return ratio_map.get(aspect_ratio, "square_hd")

    def _map_magic_prompt_to_expand_prompt(self, magic_prompt_option):
        """Map magic prompt option to expand_prompt parameter for Fal.ai API"""
        if magic_prompt_option.lower() in ["auto", "on"]:
            return True
        return False

    def _map_style_type_for_fal(self, style_type):
        """
        Map style type to the format expected by Fal.ai API

        Args:
            style_type (str): Style type from the UI (case-insensitive)

        Returns:
            str: Style type in the format expected by Fal.ai API (uppercase)
        """
        # Map to the format expected by Fal.ai API (which uses uppercase)
        style_map = {
            "auto": "AUTO",
            "general": "GENERAL",
            "realistic": "REALISTIC",
            "design": "DESIGN",
            "none": "AUTO"  # Map "none" to "AUTO" for Fal.ai as it doesn't support "None"
        }

        # Convert to lowercase for case-insensitive matching
        style_key = style_type.lower() if style_type else "auto"
        return style_map.get(style_key, "AUTO")

    def generate_image(self, prompt, aspect_ratio="1:1", magic_prompt_option="Auto", style_type="auto", progress=None):
        """
        Generate an image using Ideogram v3 model via Fal.ai API

        Args:
            prompt (str): Text prompt describing the image to generate
            aspect_ratio (str): Aspect ratio of the generated image (e.g., "1:1", "16:9", "3:2")
            magic_prompt_option (str): Magic prompt option ("Auto", "On", "Off")
            style_type (str): Style type to use ("None", "Auto", "General", "Realistic", "Design")
            progress: Optional progress callback function

        Returns:
            PIL.Image.Image: Generated image or None if generation failed
        """
        # Check if API key is available
        if not self.is_configured():
            raise ValueError("Fal.ai API key not found. Make sure to set the FAL_KEY environment variable.")

        # Check if prompt is provided
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty. Please provide a description for the image.")

        # Update progress if provided
        if progress is not None and hasattr(progress, "__call__"):
            try:
                progress(0.2, "Starting image generation with Fal.ai...")
            except Exception as e:
                logger.warning(f"Error updating progress: {e}")

        try:
            # Import fal_client here to avoid import errors if not installed
            try:
                import fal_client
            except ImportError:
                raise ImportError("Fal.ai client not installed. Please install it with 'pip install fal-client'")

            # Use the API key in the client configuration instead of modifying environment variables
            fal_client.configure(api_key=self.api_key)

            # Map parameters to Fal.ai format
            image_size = self._map_aspect_ratio_to_image_size(aspect_ratio)
            expand_prompt = self._map_magic_prompt_to_expand_prompt(magic_prompt_option)
            fal_style_type = self._map_style_type_for_fal(style_type)

            # Define callback for progress updates
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress) and progress is not None and hasattr(progress, "__call__"):
                    for log in update.logs:
                        try:
                            progress(0.5, log["message"])
                        except Exception as e:
                            print(f"Warning: Error updating progress in callback: {e}")

            # Call the Fal.ai API
            result = fal_client.subscribe(
                "fal-ai/ideogram/v3",
                arguments={
                    "prompt": prompt,
                    "rendering_speed": "BALANCED",
                    "style": fal_style_type,
                    "expand_prompt": expand_prompt,
                    "num_images": 1,
                    "image_size": image_size
                },
                with_logs=True,
                on_queue_update=on_queue_update if progress is not None and hasattr(progress, "__call__") else None,
            )

            # Update progress if provided
            if progress is not None and hasattr(progress, "__call__"):
                try:
                    progress(0.7, "Image generated, downloading...")
                except Exception as e:
                    logger.warning(f"Error updating progress: {e}")

            # Process the result
            # Log debug information
            logger.debug(f"Fal.ai API response type: {type(result)}")
            logger.debug(f"Fal.ai API response: {result}")

            if result and "images" in result and len(result["images"]) > 0:
                image_url = result["images"][0]["url"]
                logger.debug(f"Processing Fal.ai image URL: {image_url}")

                # Download the image
                try:
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()

                    # Convert to PIL Image
                    image = Image.open(BytesIO(response.content))
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error downloading image from Fal.ai URL: {e}")
                    raise ValueError(f"Failed to download image from Fal.ai URL: {image_url}. Error: {e}")
                except Exception as e:
                    logger.error(f"Error processing Fal.ai image: {e}")
                    raise ValueError(f"Failed to process image from Fal.ai URL: {image_url}. Error: {e}")

                # Update progress if provided
                if progress is not None and hasattr(progress, "__call__"):
                    try:
                        progress(1.0, "Image generation complete!")
                    except Exception as e:
                        logger.warning(f"Error updating progress: {e}")

                return image
            else:
                logger.error(f"Failed to process Fal.ai result: {result}")
                raise ValueError(f"Unexpected response format from Fal.ai API: {result}")

        except Exception as e:
            logger.error(f"Error during image generation with Fal.ai: {e}")
            raise


class OpenAIProvider(APIProvider):
    """OpenAI API provider for image generation"""
    def __init__(self):
        super().__init__()
        self.name = "OpenAI"
        self.api_key = OPENAI_API_KEY

    def is_configured(self):
        """Check if OpenAI API is properly configured"""
        return self.api_key is not None and len(self.api_key.strip()) > 0

    def _map_aspect_ratio_to_size(self, aspect_ratio):
        """
        Map aspect ratio to size for OpenAI API

        Args:
            aspect_ratio (str): Aspect ratio string (e.g., "1:1", "16:9", "3:2")

        Returns:
            tuple: (width, height) in pixels for the OpenAI API
        """
        # GPT-Image-1 supports 1024x1024, 1536x1024, 1024x1536, or auto
        ratio_map = {
            "1:1": (1024, 1024),  # Square
            "16:9": (1536, 1024),  # Landscape
            "9:16": (1024, 1536),  # Portrait
            "4:3": (1536, 1024),   # Landscape (approximated)
            "3:4": (1024, 1536),   # Portrait (approximated)
            "3:2": (1536, 1024),   # Landscape (approximated)
            "2:3": (1024, 1536),   # Portrait (approximated)
        }

        # Return the mapped value or default to square
        return ratio_map.get(aspect_ratio, (1024, 1024))

    def _get_gpt_image_background(self, style_type):
        """
        Determine the background parameter for GPT-Image-1 based on style type

        Args:
            style_type (str): Style type from the UI (case-insensitive)

        Returns:
            str: Background parameter for GPT-Image-1 ("transparent", "opaque", or "auto")
        """
        # Map style type to background parameter
        # "none" style type maps to transparent background
        if style_type.lower() == "none":
            return "transparent"

        # Default to auto for all other styles
        return "auto"

    def _get_gpt_image_output_format(self, background):
        """
        Determine the output format for GPT-Image-1 based on background setting

        Args:
            background (str): Background parameter ("transparent", "opaque", or "auto")

        Returns:
            str: Output format for GPT-Image-1 ("png", "jpeg", or "webp")
        """
        # If transparent background, use PNG or WebP
        if background == "transparent":
            return "png"

        # Default to PNG for all other cases
        return "png"

    def _map_magic_prompt_to_quality(self, magic_prompt_option):
        """
        Map magic prompt option to quality parameter for OpenAI API

        Args:
            magic_prompt_option (str): Magic prompt option ("Auto", "On", "Off")

        Returns:
            str: Quality parameter for OpenAI API ("auto", "high", "medium", or "low")
        """
        # For GPT-Image-1, quality can be "auto", "high", "medium", or "low"
        if magic_prompt_option.lower() == "on":
            return "high"
        elif magic_prompt_option.lower() == "off":
            return "medium"
        else:  # "Auto"
            return "auto"

    def generate_image(self, prompt, aspect_ratio="1:1", magic_prompt_option="Auto", style_type="auto", progress=None):
        """
        Generate an image using OpenAI's GPT-Image-1 model

        Args:
            prompt (str): Text prompt describing the image to generate
            aspect_ratio (str): Aspect ratio of the generated image (e.g., "1:1", "16:9", "3:2")
            magic_prompt_option (str): Magic prompt option ("Auto", "On", "Off")
            style_type (str): Style type to use ("auto", "general", "realistic", "design", "none")
            progress: Optional progress callback function

        Returns:
            PIL.Image.Image: Generated image or None if generation failed
        """
        # Check if API key is available
        if not self.is_configured():
            raise ValueError("OpenAI API key not found. Make sure to set the OPENAI_API_KEY environment variable.")

        # Check if prompt is provided
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty. Please provide a description for the image.")

        # Update progress if provided
        if progress is not None and hasattr(progress, "__call__"):
            try:
                progress(0.2, "Starting image generation with OpenAI GPT-Image-1...")
            except Exception as e:
                logger.warning(f"Error updating progress: {e}")

        try:
            # Import OpenAI here to avoid import errors if not installed
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("OpenAI client not installed. Please install it with 'pip install openai'")

            # Initialize the OpenAI client
            client = OpenAI(api_key=self.api_key)

            # Update progress if provided
            if progress is not None and hasattr(progress, "__call__"):
                try:
                    progress(0.4, "Sending request to OpenAI...")
                except Exception as e:
                    logger.warning(f"Error updating progress: {e}")

            # Map parameters for GPT-Image-1
            gpt_model = "gpt-image-1"
            width, height = self._map_aspect_ratio_to_size(aspect_ratio)
            gpt_quality = self._map_magic_prompt_to_quality(magic_prompt_option)
            background = self._get_gpt_image_background(style_type)
            output_format = self._get_gpt_image_output_format(background)

            # Log the parameters being used
            logger.info(f"Using GPT-Image-1 with parameters: size={width}x{height}, quality={gpt_quality}, background={background}, output_format={output_format}")

            # Prepare parameters for the API call
            params = {
                "model": gpt_model,
                "prompt": prompt,
                "size": f"{width}x{height}",
                "quality": gpt_quality,
                "background": background,
                "output_format": output_format,
                "n": 1,
                "response_format": "b64_json"  # Get base64 encoded image directly
            }

            # Call the OpenAI API with GPT-Image-1 model
            response = client.images.generate(**params)

            # Log success
            logger.info("Successfully generated image with GPT-Image-1")

            # Update progress if provided
            if progress is not None and hasattr(progress, "__call__"):
                try:
                    progress(0.7, "Image generated, processing...")
                except Exception as e:
                    logger.warning(f"Error updating progress: {e}")

            # Process the result
            # Log debug information
            logger.debug(f"OpenAI API response type: {type(response)}")

            # Extract the image data from the response
            if response and hasattr(response, "data") and len(response.data) > 0:
                # Get the base64 encoded image
                image_data = response.data[0].b64_json

                # Convert base64 to PIL Image
                try:
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes))
                except Exception as e:
                    logger.error(f"Error processing OpenAI image data: {e}")
                    raise ValueError(f"Failed to process image data from OpenAI. Error: {e}")

                # Update progress if provided
                if progress is not None and hasattr(progress, "__call__"):
                    try:
                        progress(1.0, "Image generation complete!")
                    except Exception as e:
                        logger.warning(f"Error updating progress: {e}")

                return image
            else:
                logger.error(f"Failed to process OpenAI result: {response}")
                raise ValueError(f"Unexpected response format from OpenAI API: {response}")

        except Exception as e:
            logger.error(f"Error during image generation with OpenAI: {e}")
            raise


def get_provider(provider_name=None):
    """
    Get the appropriate API provider based on configuration and preference

    Args:
        provider_name (str, optional): Name of the preferred provider ("replicate", "fal", or "openai")
                                      If None, will use the first available configured provider

    Returns:
        APIProvider: An instance of the appropriate provider

    Raises:
        ValueError: If no providers are properly configured
    """
    # Create provider instances
    replicate_provider = ReplicateProvider()
    fal_provider = FalProvider()
    openai_provider = OpenAIProvider()

    # If a specific provider is requested, try to use it
    if provider_name:
        if provider_name.lower() == "replicate" and replicate_provider.is_configured():
            return replicate_provider
        elif provider_name.lower() == "fal" and fal_provider.is_configured():
            return fal_provider
        elif (provider_name.lower() == "openai" or provider_name.lower() == "openai (gpt-image-1)") and openai_provider.is_configured():
            return openai_provider

    # Otherwise, use the first available provider
    if openai_provider.is_configured():
        return openai_provider
    elif replicate_provider.is_configured():
        return replicate_provider
    elif fal_provider.is_configured():
        return fal_provider

    # If no providers are configured, raise an error
    raise ValueError(
        "No API providers are properly configured. "
        "Please set either OPENAI_API_KEY, REPLICATE_API_TOKEN, or FAL_KEY environment variables."
    )


# Example usage
if __name__ == "__main__":
    try:
        # Prompt for the API provider
        print("Available API providers:")
        print("1. Auto (use first available provider)")
        print("2. Replicate")
        print("3. Fal.ai")
        print("4. OpenAI (GPT-Image-1)")

        provider_choice = input("Choose an API provider (1-4, default: 1): ").strip() or "1"

        provider_map = {
            "1": None,  # Auto
            "2": "replicate",
            "3": "fal",
            "4": "openai"
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
        print(f"\nGenerating image with {provider.name} using Ideogram v3 model...")
        print(f"Prompt: {prompt}")
        print(f"Aspect Ratio: {aspect_ratio}")
        print(f"Magic Prompt: {magic_prompt_option}")
        print(f"Style Type: {style_type}")

        image = provider.generate_image(prompt, aspect_ratio, magic_prompt_option, style_type)

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

