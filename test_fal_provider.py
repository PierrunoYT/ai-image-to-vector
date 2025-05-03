#!/usr/bin/env python
"""
Test script for the Fal.ai Ideogram v3 integration.
This script tests the FalProvider class to ensure it can generate images correctly.
"""

import os
import logging
from dotenv import load_dotenv
from api_provider import FalProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_fal_provider')

# Load environment variables
load_dotenv()

def test_fal_provider():
    """Test the FalProvider class"""
    # Create a provider instance
    provider = FalProvider()
    
    # Check if the provider is configured
    if not provider.is_configured():
        logger.error("Fal.ai API key not found. Make sure to set the FAL_KEY environment variable.")
        return False
    
    # Define a simple progress callback
    def progress_callback(value, message=""):
        logger.info(f"Progress: {value:.2f} - {message}")
    
    # Test parameters
    prompt = "A beautiful mountain landscape with a lake and trees"
    aspect_ratio = "16:9"
    magic_prompt_option = "Auto"
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
            output_filename = "test_fal_generated_image.png"
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
    logger.info("Testing Fal.ai Ideogram v3 integration")
    success = test_fal_provider()
    
    if success:
        logger.info("✅ Test completed successfully!")
    else:
        logger.error("❌ Test failed!")
