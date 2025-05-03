# Image to Vector

A tool to generate images with OpenAI GPT-Image-1, DALL-E 3, or Ideogram v3 and vectorize them using the Recraft API. Includes both a command-line interface and a web-based UI powered by Gradio.

This tool uses OpenAI's GPT-Image-1 model, DALL-E 3, or the latest Ideogram v3 model from either Replicate or Fal.ai to create high-quality images from text prompts, and then converts them to scalable vector graphics (SVG) using the Recraft API.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the same directory as the script with your API tokens:
   ```
   RECRAFT_API_TOKEN=your_recraft_token_here
   OPENAI_API_KEY=your_openai_api_key_here
   REPLICATE_API_TOKEN=your_replicate_token_here
   FAL_KEY=your_fal_api_key_here
   ```
   You can use the `.env.example` file as a template.

   - Get your Recraft API token from [Recraft](https://recraft.ai)
   - Get your OpenAI API key from [OpenAI](https://platform.openai.com/api-keys)
   - Get your Replicate API token from [Replicate](https://replicate.com/account/api-tokens)
   - Get your Fal.ai API key from [Fal.ai](https://fal.ai/dashboard/keys)

   Note: You only need one of the image generation API keys (OpenAI, Replicate, or Fal.ai) for image generation, but the Recraft API token is required for vectorization.

## Usage

### Command Line Interface

#### Generate an image with AI
```bash
python ideogram_generator.py
```

The script will prompt you to choose an API provider (OpenAI, Replicate, or Fal.ai), enter a text prompt, and select various options. After processing, it will generate an image and save it to the current directory. When using OpenAI, the system will try GPT-Image-1 first and fall back to DALL-E 3 if needed.

#### Vectorize an existing image
```bash
python recraft_vectorizer.py
```

The script will prompt you to enter the path to your image file. After processing, it will download the vectorized SVG file to the current directory.

### Web UI (Gradio)

Run the Gradio web interface:
```bash
python gradio_app.py
```

This will start a local web server (typically at [http://127.0.0.1:7860](http://127.0.0.1:7860)) where you can:

#### Upload Image Tab

1. Upload an image through the browser
2. Click the "Vectorize Image" button
3. View the vectorized SVG directly in the preview area
4. Download the resulting SVG file

#### Generate Image Tab

1. Enter a text prompt describing the image you want to create
2. Choose an aspect ratio for your image (e.g., 1:1, 16:9, 3:2)
3. Select a Magic Prompt option (Auto, On, Off)
4. Choose a Style Type (None, Auto, General, Realistic, Design)
5. Select an API Provider (Auto, OpenAI, Replicate, Fal.ai)
6. Click the "Generate & Vectorize" button
7. Wait for the AI to generate your image and vectorize it
8. View the generated image and vectorized SVG in the preview areas
9. Download the resulting SVG file

All uploaded and generated images are saved to the `output/uploads` directory, and all vectorized SVGs are saved to the `output/vectors` directory.

## Examples

### Generating an image with AI

```bash
Available API providers:
1. Auto (use first available provider)
2. Replicate
3. Fal.ai
4. OpenAI
Choose an API provider (1-4, default: 1): 4
Using OpenAI as the API provider

Enter a prompt to generate an image: A beautiful sunset over a mountain landscape

Available aspect ratios:
1. Square (1:1)
2. Landscape (16:9)
3. Portrait (9:16)
4. Landscape (3:2)
5. Portrait (2:3)
6. Custom
Choose an aspect ratio (1-6, default: 1): 4

Magic Prompt options:
1. Auto - Automatically optimize the prompt
2. On - Always optimize the prompt
3. Off - Use the prompt as-is
Choose a Magic Prompt option (1-3, default: 1): 1

Style Type options:
1. Auto
2. General
3. Realistic
4. Design
Choose a Style Type (1-4, default: 1): 3

Generating image with OpenAI using GPT-Image-1 model...
Prompt: A beautiful sunset over a mountain landscape
Aspect Ratio: 3:2
Magic Prompt: Auto
Style Type: realistic
✅ Image successfully generated and saved to generated_image.png

Next steps:
1. To vectorize this image, run: python recraft_vectorizer.py
2. To use the web interface, run: python gradio_app.py
```

### Vectorizing an image

```bash
Enter the path to your image file: generated_image.png
Vectorizing image...
Vectorization successful! SVG URL: https://example.com/vectorized.svg
Downloading SVG to generated_image_vectorized.svg...
SVG file successfully downloaded to generated_image_vectorized.svg
✅ Process completed successfully!
```

## Features

- Generate high-quality images using OpenAI GPT-Image-1, DALL-E 3, or Ideogram v3 models
- Support for multiple API providers (OpenAI, Replicate, and Fal.ai)
- Vectorize images using Recraft API
- User-friendly web interface with Gradio
- Command-line interface for batch processing
- Multiple aspect ratio options (square, landscape, portrait)
- Magic Prompt optimization for better results
- Various style types (None, Auto, General, Realistic, Design)
- Transparent background support with GPT-Image-1
- Download vectorized SVGs for use in design projects

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

© 2025 PierrunoYT
