import os
import replicate
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import requests

# Load environment variables
load_dotenv()

# Get API keys from environment variables
REPLICATE_API_KEY = os.getenv("REPLICATE_API_TOKEN")
FAL_API_KEY = os.getenv("FAL_KEY")

class APIProvider:
    """Base class for API providers"""
    def __init__(self):
        self.name = "Base Provider"
        
    def generate_image(self, prompt, aspect_ratio="1:1", magic_prompt_option="Auto", style_type="AUTO", progress=None):
        """Generate an image using the provider's API"""
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
    
    def generate_image(self, prompt, aspect_ratio="1:1", magic_prompt_option="Auto", style_type="AUTO", progress=None):
        """
        Generate an image using Ideogram v3 Quality model via Replicate API

        Args:
            prompt (str): Text prompt describing the image to generate
            aspect_ratio (str): Aspect ratio of the generated image (e.g., "1:1", "16:9", "3:2")
            magic_prompt_option (str): Magic prompt option ("Auto", "On", "Off")
            style_type (str): Style type to use ("AUTO", "GENERAL", "REALISTIC", "DESIGN")
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
        if progress:
            progress(0.2, "Starting image generation with Replicate...")

        try:
            # Run the Ideogram v3 Quality model
            output = replicate.run(
                "ideogram-ai/ideogram-v3-quality",
                input={
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "magic_prompt_option": magic_prompt_option,
                    "style_type": style_type,
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
            print(f"Error during image generation with Replicate: {e}")
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
        # Default to square_hd
        image_size = "square_hd"
        
        # Map common aspect ratios to Fal.ai image size options
        ratio_map = {
            "1:1": "square_hd",
            "16:9": "landscape_16_9",
            "9:16": "portrait_16_9",
            "4:3": "landscape_4_3",
            "3:4": "portrait_4_3"
        }
        
        return ratio_map.get(aspect_ratio, "square_hd")
    
    def _map_magic_prompt_to_expand_prompt(self, magic_prompt_option):
        """Map magic prompt option to expand_prompt parameter for Fal.ai API"""
        if magic_prompt_option.lower() in ["auto", "on"]:
            return True
        return False
    
    def generate_image(self, prompt, aspect_ratio="1:1", magic_prompt_option="Auto", style_type="AUTO", progress=None):
        """
        Generate an image using Ideogram v3 model via Fal.ai API

        Args:
            prompt (str): Text prompt describing the image to generate
            aspect_ratio (str): Aspect ratio of the generated image (e.g., "1:1", "16:9", "3:2")
            magic_prompt_option (str): Magic prompt option ("Auto", "On", "Off")
            style_type (str): Style type to use ("AUTO", "GENERAL", "REALISTIC", "DESIGN")
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
        if progress:
            progress(0.2, "Starting image generation with Fal.ai...")

        try:
            # Import fal_client here to avoid import errors if not installed
            try:
                import fal_client
            except ImportError:
                raise ImportError("Fal.ai client not installed. Please install it with 'pip install fal-client'")
            
            # Set the API key
            os.environ["FAL_KEY"] = self.api_key
            
            # Map parameters to Fal.ai format
            image_size = self._map_aspect_ratio_to_image_size(aspect_ratio)
            expand_prompt = self._map_magic_prompt_to_expand_prompt(magic_prompt_option)
            
            # Define callback for progress updates
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress) and progress:
                    for log in update.logs:
                        progress(0.5, log["message"])
            
            # Call the Fal.ai API
            result = fal_client.subscribe(
                "fal-ai/ideogram/v3",
                arguments={
                    "prompt": prompt,
                    "rendering_speed": "BALANCED",
                    "style": style_type,
                    "expand_prompt": expand_prompt,
                    "num_images": 1,
                    "image_size": image_size
                },
                with_logs=True,
                on_queue_update=on_queue_update if progress else None,
            )
            
            # Update progress if provided
            if progress:
                progress(0.7, "Image generated, downloading...")
            
            # Process the result
            if result and "images" in result and len(result["images"]) > 0:
                image_url = result["images"][0]["url"]
                
                # Download the image
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Convert to PIL Image
                image = Image.open(BytesIO(response.content))
                
                # Update progress if provided
                if progress:
                    progress(1.0, "Image generation complete!")
                
                return image
            else:
                raise ValueError(f"Unexpected response format from Fal.ai API: {result}")
                
        except Exception as e:
            print(f"Error during image generation with Fal.ai: {e}")
            raise


def get_provider(provider_name=None):
    """
    Get the appropriate API provider based on configuration and preference
    
    Args:
        provider_name (str, optional): Name of the preferred provider ("replicate" or "fal")
                                      If None, will use the first available configured provider
    
    Returns:
        APIProvider: An instance of the appropriate provider
        
    Raises:
        ValueError: If no providers are properly configured
    """
    # Create provider instances
    replicate_provider = ReplicateProvider()
    fal_provider = FalProvider()
    
    # If a specific provider is requested, try to use it
    if provider_name:
        if provider_name.lower() == "replicate" and replicate_provider.is_configured():
            return replicate_provider
        elif provider_name.lower() == "fal" and fal_provider.is_configured():
            return fal_provider
    
    # Otherwise, use the first available provider
    if replicate_provider.is_configured():
        return replicate_provider
    elif fal_provider.is_configured():
        return fal_provider
    
    # If no providers are configured, raise an error
    raise ValueError(
        "No API providers are properly configured. "
        "Please set either REPLICATE_API_TOKEN or FAL_KEY environment variables."
    )


# Example usage
if __name__ == "__main__":
    try:
        # Get the appropriate provider
        provider = get_provider()
        print(f"Using {provider.name} as the API provider")
        
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
        
        aspect_choice = input("Choose an aspect ratio (1-5, default: 1): ").strip() or "1"
        
        aspect_ratio_map = {
            "1": "1:1",
            "2": "16:9",
            "3": "9:16",
            "4": "3:2",
            "5": "2:3"
        }
        
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
            "1": "AUTO",
            "2": "GENERAL",
            "3": "REALISTIC",
            "4": "DESIGN"
        }
        
        style_type = style_type_map.get(style_choice, "AUTO")
        
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
