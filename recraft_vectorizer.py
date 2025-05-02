import os
import requests
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (put your API key in a .env file as RECRAFT_API_TOKEN=your_token)
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("RECRAFT_API_TOKEN")

def vectorize_image(image_path):
    """
    Vectorize an image using Recraft API

    Args:
        image_path (str): Path to the image file to be vectorized

    Returns:
        str: URL to the vectorized SVG image or the base64 encoded SVG data
    """
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Initialize the Recraft client
    client = OpenAI(
        base_url='https://external.api.recraft.ai/v1',
        api_key=api_key,
    )

    # Make the API call to vectorize the image
    try:
        response = client.post(
            path='/images/vectorize',
            cast_to=object,
            options={'headers': {'Content-Type': 'multipart/form-data'}},
            files={'file': open(image_path, 'rb')},
        )

        # Return the URL to the vectorized image
        return response['image']['url']

    except Exception as e:
        print(f"Error during vectorization: {e}")
        raise

def download_svg(url, output_path):
    """
    Download SVG from URL and save it to a file

    Args:
        url (str): URL to the SVG image
        output_path (str): Path where to save the downloaded SVG

    Returns:
        bool: True if download was successful, False otherwise
    """
    response = requests.get(url)
    if response.status_code == 200:
        # Check if the content is SVG
        content_type = response.headers.get('Content-Type', '')
        if 'svg' in content_type.lower() or response.content.startswith(b'<?xml') or response.content.startswith(b'<svg'):
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"SVG file successfully downloaded to {output_path}")
            return True
        else:
            print(f"Warning: Downloaded content may not be an SVG. Content-Type: {content_type}")
            # Save it anyway
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
    else:
        print(f"Failed to download SVG file. Status code: {response.status_code}")
        return False

# Example usage
if __name__ == "__main__":
    # Check if API key is available
    if not api_key:
        print("ERROR: Recraft API token not found. Make sure to set the RECRAFT_API_TOKEN environment variable.")
        exit(1)

    # Path to the image you want to vectorize
    image_path = input("Enter the path to your image file: ")

    try:
        # Get vectorized image URL
        print("Vectorizing image...")
        svg_url = vectorize_image(image_path)
        print(f"Vectorization successful! SVG URL: {svg_url}")

        # Download the SVG file
        output_path = os.path.splitext(os.path.basename(image_path))[0] + "_vectorized.svg"
        print(f"Downloading SVG to {output_path}...")
        download_svg(svg_url, output_path)

    except Exception as e:
        print(f"Vectorization failed: {e}")
