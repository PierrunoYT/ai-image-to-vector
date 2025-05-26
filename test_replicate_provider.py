#!/usr/bin/env python
"""
Test script for the Replicate Ideogram v3 Quality integration.
This script tests the ReplicateProvider class to ensure it can generate images correctly.
"""

import os
import logging
from dotenv import load_dotenv
from api_provider import ReplicateProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_replicate_provider')

# Load environment variables
load_dotenv()

def test_replicate_provider():
    """Test the ReplicateProvider class"""
    # Create a provider instance
    provider = ReplicateProvider()
    
    # Check if the provider is configured
    if not provider.is_configured():
        logger.error("Replicate API token not found. Make sure to set the REPLICATE_API_TOKEN environment variable.")
        return False
    
    # Define a simple progress callback
    def progress_callback(value, message=""):
        logger.info(f"Progress: {value:.2f} - {message}")
    
    # Test parameters
    prompt = "The text \"V3 Quality\" in the center middle. A color film-inspired portrait of a young woman with a shallow depth of field that blurs the surrounding elements, drawing attention to the eye."
    aspect_ratio = "3:2"
    magic_prompt_option = "Off"
    style_type = "realistic"
    
    try:
        # Generate an image
        logger.info(f"Generating image with prompt: {prompt}")
        logger.info(f"Aspect ratio: {aspect_ratio}")
        logger.info(f"Magic prompt option: {magic_prompt_option}")
        logger.info(f"Style type: {style_type}")
        
        image = provider.generate_image(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            magic_prompt_option=magic_prompt_option,
            style_type=style_type,
            progress=progress_callback
        )
        
        # Save the image
        if image:
            output_filename = "test_replicate_generated_image.png"
            image.save(output_filename)
            logger.info(f"✅ Image successfully generated and saved to {output_filename}")
            return True
        else:
            logger.error("❌ Failed to generate image")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during image generation: {e}")
        return False

if __name__ == "__main__":
    # Check if REPLICATE_API_TOKEN is set in environment variables
    replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_api_token:
        # Security: Don't accept API tokens via command line input
        print("REPLICATE_API_TOKEN environment variable not found.")
        print("Please set it in your .env file as REPLICATE_API_TOKEN=your_token")
        print("Test cannot proceed without proper API token configuration.")
        exit(1)
    
    logger.info("Testing Replicate Ideogram v3 Quality integration")
    success = test_replicate_provider()
    
    if success:
        logger.info("✅ Test completed successfully!")
    else:
        logger.error("❌ Test failed!")
