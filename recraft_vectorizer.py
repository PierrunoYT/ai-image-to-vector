import os
import requests
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('image_to_vector.recraft_vectorizer')

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

    # Input validation and sanitization
    if not image_path or not isinstance(image_path, str):
        raise ValueError(f"Invalid image path: {image_path}")
    
    # Sanitize path to prevent directory traversal
    image_path = os.path.abspath(image_path)
    
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Validate file extension (basic check)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Unsupported image format. Supported formats: {valid_extensions}")
    
    # Check file size (prevent extremely large files)
    file_size = os.path.getsize(image_path)
    max_size = 50 * 1024 * 1024  # 50MB
    if file_size > max_size:
        raise ValueError(f"Image file too large: {file_size} bytes. Maximum allowed: {max_size} bytes")

    # No need for OpenAI client since we're using requests directly

    # Make the API call to vectorize the image
    try:
        # Use context manager to ensure file is properly closed
        with open(image_path, 'rb') as image_file:
            try:
                # Use requests directly for Recraft API since it's not OpenAI
                files = {'file': image_file}
                headers = {'Authorization': f'Bearer {api_key}'}
                response = requests.post(
                    'https://external.api.recraft.ai/v1/images/vectorize',
                    files=files,
                    headers=headers,
                    timeout=180
                )
                response.raise_for_status()
                response = response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    raise ValueError(f"Invalid API key or authentication error: {e}")
                elif e.response.status_code == 429:
                    raise ValueError(f"Recraft API rate limit exceeded: {e}")
                elif e.response.status_code == 400:
                    raise ValueError(f"Invalid request parameters: {e}")
                else:
                    raise ValueError(f"Recraft API error: {e}")
            except requests.exceptions.ConnectionError as e:
                raise ValueError(f"Network error when connecting to Recraft: {e}")
            except requests.exceptions.Timeout as e:
                raise ValueError(f"Request to Recraft API timed out: {e}")
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Request error: {e}")

        # Validate response structure
        if response is None:
            raise ValueError("Received empty response from Recraft API")
            
        if not isinstance(response, dict):
            raise TypeError(f"Unexpected response type: {type(response)}. Expected dictionary.")

        if 'image' not in response:
            raise KeyError(f"Response missing 'image' key. Response keys: {list(response.keys())}")

        if not isinstance(response['image'], dict):
            raise TypeError(f"Unexpected 'image' type: {type(response['image'])}. Expected dictionary.")

        if 'url' not in response['image']:
            raise KeyError(f"Response missing 'url' key in 'image'. Image keys: {list(response['image'].keys() if isinstance(response['image'], dict) else [])}")

        # Return the URL to the vectorized image
        return response['image']['url']

    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error during vectorization: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during vectorization: {e}")
        raise ValueError(f"Failed to vectorize image: {e}")

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
    # Input validation and sanitization
    if not url or not isinstance(url, str):
        raise ValueError(f"Invalid URL: {url}")
    
    # Basic URL validation
    if not (url.startswith('http://') or url.startswith('https://')):
        raise ValueError(f"URL must start with http:// or https://: {url}")
    
    # Path validation - ensure it's safe
    if not output_path or not isinstance(output_path, str):
        raise ValueError(f"Invalid output path: {output_path}")
    
    # Sanitize output path to prevent directory traversal
    import os.path
    output_path = os.path.abspath(output_path)
    if '..' in output_path or output_path.startswith('/'):
        # Allow absolute paths but prevent traversal
        pass  # os.path.abspath already handles this
    
    # Ensure output directory is valid
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = '.'
    
    # Validate file extension
    if not output_path.lower().endswith('.svg'):
        raise ValueError("Output path must have .svg extension")

    try:
        # Use a timeout to prevent hanging on slow connections
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses

        # Check if the content is SVG with enhanced validation
        content_type = response.headers.get('Content-Type', '')
        content = response.content
        
        # Initial basic checks
        content_check = ('svg' in content_type.lower() or
                         content.startswith(b'<?xml') or
                         content.startswith(b'<svg') or
                         b'<svg' in content[:1000])
        
        # More robust SVG validation
        if content_check:
            try:
                # Check if the content has minimum required SVG elements
                is_svg = (b'<svg' in content and 
                         (b'</svg>' in content or b'/>' in content[-10:]) and 
                         len(content) > 30)  # Arbitrary minimum size for a valid SVG
                
                # Additional check - see if SVG has at least one path or shape
                has_elements = any(tag in content for tag in 
                                  [b'<path', b'<circle', b'<rect', b'<ellipse', 
                                   b'<line', b'<polyline', b'<polygon', b'<g'])
                
                if not has_elements:
                    logger.warning("SVG file doesn't contain any standard shape elements")
                    
            except Exception as e:
                logger.warning(f"Error during SVG validation: {e}")
                is_svg = content_check  # Fall back to basic check
        else:
            is_svg = False

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save the file
        with open(output_path, 'wb') as f:
            f.write(response.content)

        if is_svg:
            logger.info(f"SVG file successfully downloaded to {output_path}")
        else:
            logger.warning(f"Downloaded content may not be an SVG. Content-Type: {content_type}")
            logger.info(f"File saved to {output_path} anyway")

        return True

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error occurred: {e}")
        return False
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error occurred: {e}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during request: {e}")
        return False
    except IOError as e:
        logger.error(f"I/O error occurred when writing file: {e}")
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
            print("❌ Failed to download the SVG file")
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
