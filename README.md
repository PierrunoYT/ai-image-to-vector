# Recraft Image Vectorizer

A tool to vectorize images using the Recraft API. Includes both a command-line interface and a web-based UI powered by Gradio.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the same directory as the script with your Recraft API token:
   ```
   RECRAFT_API_TOKEN=your_token_here
   ```
   You can use the `.env.example` file as a template.

## Usage

### Command Line Interface

Run the script:
```
python recraft_vectorizer.py
```

The script will prompt you to enter the path to your image file. After processing, it will download the vectorized SVG file to the current directory.

### Web UI (Gradio)

Run the Gradio web interface:
```
python gradio_app.py
```

This will start a local web server (typically at http://127.0.0.1:7860) where you can:
1. Upload an image through the browser
2. Click the "Vectorize Image" button
3. View the vectorized SVG directly in the preview area
4. Download the resulting SVG file

All uploaded images are saved to the `output/uploads` directory, and all vectorized SVGs are saved to the `output/vectors` directory.

## Example

```
Enter the path to your image file: path/to/your/image.png
Vectorizing image...
Vectorization successful! SVG URL: https://example.com/vectorized.svg
Downloading SVG to image_vectorized.svg...
SVG file successfully downloaded to image_vectorized.svg
```
