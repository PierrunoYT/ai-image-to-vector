import os
import requests
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

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the API key is not set
        KeyError: If the API response doesn't contain the expected data
        Exception: For other errors during vectorization
    """
    # Check if API key is available
    if not api_key:
        raise ValueError("Recraft API token not found. Make sure to set the RECRAFT_API_TOKEN environment variable.")

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
        # Use context manager to ensure file is properly closed
        with open(image_path, 'rb') as image_file:
            response = client.post(
                path='/images/vectorize',
                cast_to=object,
                options={'headers': {'Content-Type': 'multipart/form-data'}},
                files={'file': image_file},
            )

        # Validate response structure
        if not isinstance(response, dict):
            raise TypeError(f"Unexpected response type: {type(response)}. Expected dictionary.")

        if 'image' not in response:
            raise KeyError(f"Response missing 'image' key. Response keys: {list(response.keys())}")

        if 'url' not in response['image']:
            raise KeyError(f"Response missing 'url' key in 'image'. Image keys: {list(response['image'].keys())}")

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

    Raises:
        ValueError: If the URL is invalid
        requests.RequestException: If there's an error during the request
    """
    if not url or not isinstance(url, str):
        raise ValueError(f"Invalid URL: {url}")

    try:
        # Use a timeout to prevent hanging on slow connections
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses

        # Check if the content is SVG
        content_type = response.headers.get('Content-Type', '')
        is_svg = ('svg' in content_type.lower() or
                 response.content.startswith(b'<?xml') or
                 response.content.startswith(b'<svg'))

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save the file
        with open(output_path, 'wb') as f:
            f.write(response.content)

        if is_svg:
            print(f"SVG file successfully downloaded to {output_path}")
        else:
            print(f"Warning: Downloaded content may not be an SVG. Content-Type: {content_type}")
            print(f"File saved to {output_path} anyway")

        return True

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error occurred: {e}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"Timeout error occurred: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return False
    except IOError as e:
        print(f"I/O error occurred when writing file: {e}")
        return False

# Example usage
if __name__ == "__main__":
    try:
        # Check if API key is available
        if not api_key:
            print("ERROR: Recraft API token not found. Make sure to set the RECRAFT_API_TOKEN environment variable.")
            print("Create a .env file with your API token as RECRAFT_API_TOKEN=your_token")
            exit(1)

        # Path to the image you want to vectorize
        image_path = input("Enter the path to your image file: ")

        # Validate input path
        if not image_path:
            print("ERROR: No image path provided.")
            exit(1)

        if not os.path.exists(image_path):
            print(f"ERROR: Image file not found: {image_path}")
            exit(1)

        # Get vectorized image URL
        print("Vectorizing image...")
        svg_url = vectorize_image(image_path)
        print(f"Vectorization successful! SVG URL: {svg_url}")

        # Generate output path
        output_dir = os.path.dirname(image_path) or "."
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_vectorized.svg"
        output_path = os.path.join(output_dir, output_filename)

        # Download the SVG file
        print(f"Downloading SVG to {output_path}...")
        if download_svg(svg_url, output_path):
            print("✅ Process completed successfully!")
        else:
            print("❌ Failed to download the SVG file.")
            exit(1)

    except ValueError as e:
        print(f"ERROR: {e}")
        exit(1)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Vectorization failed: {e}")
        exit(1)
