import os
import gradio as gr
import traceback
import replicate
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from recraft_vectorizer import vectorize_image, download_svg

# Load environment variables
load_dotenv()

# Get API keys from environment variables
recraft_api_key = os.getenv("RECRAFT_API_TOKEN")
replicate_api_key = os.getenv("REPLICATE_API_TOKEN")

# Create output directories if they don't exist
OUTPUT_DIR = "output"
UPLOADS_DIR = os.path.join(OUTPUT_DIR, "uploads")
VECTORS_DIR = os.path.join(OUTPUT_DIR, "vectors")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VECTORS_DIR, exist_ok=True)

def generate_and_process_image(prompt, aspect_ratio="1:1", magic_prompt_option="Auto", style_type="None", progress=gr.Progress()):
    """
    Generate an image from a text prompt and then vectorize it

    Args:
        prompt: Text prompt for image generation
        aspect_ratio: Aspect ratio of the generated image (e.g., "1:1", "16:9", "3:2")
        magic_prompt_option: Magic prompt option ("Auto", "On", "Off")
        style_type: Style type to use ("None", "Auto", "General", "Realistic", "Design")
        progress: Gradio progress indicator

    Returns:
        tuple: (generated_image, svg_path, svg_html, message)
    """
    if not replicate_api_key:
        return None, None, None, "❌ ERROR: Replicate API token not found!\n\nPlease create a .env file with your REPLICATE_API_TOKEN.\nSee the instructions for more details."

    if not prompt or not prompt.strip():
        return None, None, None, "❌ ERROR: No prompt provided. Please enter a text prompt to generate an image."

    try:
        # Update progress
        progress(0.1, "Starting image generation...")

        # Use the Ideogram v3 Quality model with the appropriate parameters
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

        # Update progress
        progress(0.3, "Image generated, downloading...")

        # Process the output
        if output and isinstance(output, list) and len(output) > 0:
            file_output = output[0]

            # Read the file content
            image_data = file_output.read()

            # Convert to PIL Image
            generated_image = Image.open(BytesIO(image_data))

            # Update progress
            progress(0.4, "Image generation complete, starting vectorization...")

            # Process the generated image
            svg_path, svg_html, message = process_image_internal(generated_image, progress, start_progress=0.4)

            # Return the results
            return generated_image, svg_path, svg_html, message
        else:
            raise ValueError(f"Unexpected response format from Replicate API: {output}")

    except ValueError as e:
        # Handle specific errors
        traceback.print_exc()
        error_message = f"❌ ERROR: {str(e)}"
        return None, None, None, error_message
    except Exception as e:
        # Print the full traceback to the console for debugging
        traceback.print_exc()

        # Create a user-friendly error message
        error_message = f"❌ Image generation failed!\n\n⚠️ Error: {str(e)}\n\nPlease check the console for more details."
        return None, None, None, error_message

def process_image(image, progress=gr.Progress()):
    """
    Process the uploaded image and return the vectorized SVG

    Args:
        image: The uploaded image file
        progress: Gradio progress indicator

    Returns:
        tuple: (svg_path, svg_html, message)
    """
    return process_image_internal(image, progress)

def process_image_internal(image, progress=gr.Progress(), start_progress=0.0):
    """
    Internal function to process an image and return the vectorized SVG

    Args:
        image: The image to process (PIL Image)
        progress: Gradio progress indicator
        start_progress: Starting progress value (0.0 to 1.0)

    Returns:
        tuple: (svg_path, svg_html, message)
    """
    if not recraft_api_key:
        return None, None, "❌ ERROR: Recraft API token not found!\n\nPlease create a .env file with your RECRAFT_API_TOKEN.\nSee the instructions for more details."

    if image is None:
        return None, None, "❌ ERROR: No image uploaded. Please upload an image first."

    try:
        # Generate unique filenames based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the uploaded image to the uploads directory
        input_filename = f"image_{timestamp}.png"
        input_path = os.path.join(UPLOADS_DIR, input_filename)

        # Ensure the image is saved
        try:
            image.save(input_path)
        except Exception as e:
            return None, None, f"❌ ERROR: Failed to save uploaded image: {str(e)}"

        # Validate the image file
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            return None, None, "❌ ERROR: Failed to save the image or the image is empty."

        # Check if the file is a valid image
        try:
            with Image.open(input_path) as img:
                img.verify()  # Verify it's a valid image
        except Exception:
            os.remove(input_path)  # Clean up invalid file
            return None, None, "❌ ERROR: The uploaded file is not a valid image."

        # Calculate adjusted progress values
        progress_start = start_progress
        progress_mid = start_progress + (1.0 - start_progress) * 0.5
        progress_end = start_progress + (1.0 - start_progress) * 0.9

        # Update progress
        progress(progress_start, "Image ready, starting vectorization...")

        # Vectorize the image
        svg_url = vectorize_image(input_path)

        # Update progress
        progress(progress_mid, "Vectorization complete, downloading SVG...")

        # Generate output path in the vectors directory
        output_filename = f"vector_{timestamp}.svg"
        output_path = os.path.join(VECTORS_DIR, output_filename)

        # Download the SVG
        success = download_svg(svg_url, output_path)
        if not success:
            return None, None, f"❌ ERROR: Failed to download SVG from URL: {svg_url}"

        # For debugging
        file_size = os.path.getsize(output_path)
        print(f"SVG file size: {file_size} bytes")

        # Create HTML to display the SVG properly
        svg_html = create_svg_preview_html(output_path)

        # Update progress
        progress(progress_end, "Process complete!")

        # Create a more informative success message with better formatting for the UI
        success_message = (
            f"✅ Vectorization successful!\n\n"
            f"🔗 SVG URL:\n{svg_url}\n\n"
            f"💾 Files saved to:\n"
            f"📄 Input: {os.path.basename(input_path)}\n"
            f"📄 Output: {os.path.basename(output_path)}"
        )

        return output_path, svg_html, success_message

    except ValueError as e:
        # Handle specific errors
        traceback.print_exc()
        error_message = f"❌ ERROR: {str(e)}"
        return None, None, error_message
    except FileNotFoundError as e:
        traceback.print_exc()
        error_message = f"❌ ERROR: {str(e)}"
        return None, None, error_message
    except Exception as e:
        # Print the full traceback to the console for debugging
        traceback.print_exc()

        # Create a user-friendly error message
        error_message = f"❌ Vectorization failed!\n\n⚠️ Error: {str(e)}\n\nPlease check the console for more details."
        return None, None, error_message

def create_svg_preview_html(svg_path):
    """
    Create HTML to display an SVG file

    Args:
        svg_path: Path to the SVG file

    Returns:
        str: HTML code to display the SVG
    """
    try:
        # Read the SVG file
        with open(svg_path, 'r') as f:
            svg_content = f.read()

        # Create HTML with the SVG embedded
        html = f"""
        <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; background-color:#f5f5f5; border-radius:8px; padding:10px;">
            {svg_content}
        </div>
        """
        return html
    except Exception as e:
        print(f"Error creating SVG preview: {e}")
        return f"""
        <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; background-color:#f5f5f5; border-radius:8px; padding:10px;">
            <p style="color:red;">Error loading SVG: {str(e)}</p>
        </div>
        """

# Create the Gradio interface with a nicer design
with gr.Blocks(title="Ideogram to Vector", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🖼️ Ideogram to Vector
        ### Generate images with Ideogram v3 Quality and convert them to scalable vector graphics
        """
    )

    # Create tabs for different input methods
    with gr.Tabs():
        # Tab 1: Upload an existing image
        with gr.TabItem("Upload Image"):
            with gr.Row():
                # Left column - Input
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload Image")
                    input_image = gr.Image(
                        type="pil",
                        label="",
                        elem_id="input-image",
                        height=300,
                        image_mode="RGB"
                    )
                    vectorize_button = gr.Button(
                        "🔄 Vectorize Image",
                        variant="primary",
                        size="lg"
                    )

                # Right column - Output
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Vectorized Result")
                    output_preview = gr.HTML(
                        label="",
                        elem_id="output-preview"
                    )

        # Tab 2: Generate an image with AI
        with gr.TabItem("Generate Image"):
            with gr.Row():
                # Left column - Input
                with gr.Column(scale=1):
                    gr.Markdown("### 🤖 Generate Image with AI")
                    text_prompt = gr.Textbox(
                        label="Text Prompt",
                        placeholder="Describe the image you want to generate...",
                        lines=3,
                        max_lines=5,
                        elem_id="text-prompt"
                    )

                    # Add aspect ratio dropdown
                    aspect_ratio = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=[
                            "1:1", # Square
                            "16:9", "16:10", "3:2", "4:3", "5:4", "2:1", "3:1", # Landscape
                            "9:16", "10:16", "2:3", "3:4", "4:5", "1:2", "1:3"  # Portrait
                        ],
                        value="1:1",
                        elem_id="aspect-ratio"
                    )

                    # Add magic prompt option
                    magic_prompt_option = gr.Dropdown(
                        label="Magic Prompt",
                        choices=["Auto", "On", "Off"],
                        value="Auto",
                        elem_id="magic-prompt",
                        info="Magic Prompt will interpret your prompt and optimize it to maximize variety and quality"
                    )

                    # Add style type dropdown
                    style_type = gr.Dropdown(
                        label="Style Type",
                        choices=["None", "Auto", "General", "Realistic", "Design"],
                        value="None",
                        elem_id="style-type",
                        info="The style helps define the specific aesthetic of the image"
                    )

                    generate_button = gr.Button(
                        "✨ Generate & Vectorize",
                        variant="primary",
                        size="lg"
                    )

                # Right column - Output
                with gr.Column(scale=1):
                    gr.Markdown("### 🎨 Generated Image")
                    generated_image = gr.Image(
                        type="pil",
                        label="",
                        elem_id="generated-image",
                        height=300,
                        image_mode="RGB"
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Vectorized Result")
                    generated_output_preview = gr.HTML(
                        label="",
                        elem_id="generated-output-preview"
                    )

    # Common output components
    with gr.Row():
        with gr.Column():
            output_file = gr.File(
                label="Download SVG",
                elem_id="output-file",
                height="auto",
                type="filepath"
            )
            output_message = gr.Textbox(
                label="Status",
                elem_id="status-message",
                lines=4,
                max_lines=6,
                show_copy_button=True
            )

    # Add a separator
    gr.Markdown("---")

    # Instructions section
    with gr.Accordion("Instructions", open=False):
        gr.Markdown(
            """
            ## Upload Image Tab
            1. Upload an image using the upload area
            2. Click the 'Vectorize Image' button
            3. Wait for the processing to complete
            4. View the vectorized SVG in the preview area
            5. Download the resulting SVG file using the download button

            ## Generate Image Tab
            1. Enter a text prompt describing the image you want to create
            2. Choose an aspect ratio for your image (e.g., 1:1, 16:9, 3:2)
            3. Select a Magic Prompt option:
               - Auto: Automatically optimize the prompt
               - On: Always optimize the prompt
               - Off: Use the prompt as-is
            4. Choose a Style Type (None, Auto, General, Realistic, Design)
            5. Click the 'Generate & Vectorize' button
            6. Wait for the AI to generate your image and vectorize it
            7. View the generated image and vectorized SVG in the preview areas
            8. Download the resulting SVG file using the download button

            **Note:**
            - This tool requires valid API tokens to be set in the .env file:
              - Recraft API token for vectorization
              - Replicate API token for image generation
            - All images are saved to the 'output/uploads' directory
            - All vectorized SVGs are saved to the 'output/vectors' directory
            """
        )

    # Connect the buttons to their respective processing functions
    vectorize_button.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_file, output_preview, output_message],
        show_progress=True
    )

    generate_button.click(
        fn=generate_and_process_image,
        inputs=[text_prompt, aspect_ratio, magic_prompt_option, style_type],
        outputs=[generated_image, output_file, generated_output_preview, output_message],
        show_progress=True
    )

def check_environment():
    """
    Check if the environment is properly set up

    Returns:
        tuple: (is_ready, message)
    """
    issues = []

    # Check if API keys are available
    if not recraft_api_key:
        issues.append("⚠️ Recraft API token not found! Please create a .env file with your RECRAFT_API_TOKEN.")

    if not replicate_api_key:
        issues.append("⚠️ Replicate API token not found! Please create a .env file with your REPLICATE_API_TOKEN.")

    # Check if output directories exist and are writable
    for directory in [UPLOADS_DIR, VECTORS_DIR]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                issues.append(f"⚠️ Could not create directory {directory}: {str(e)}")
        elif not os.access(directory, os.W_OK):
            issues.append(f"⚠️ Directory {directory} is not writable")

    # Return results
    if issues:
        return False, "\n".join(issues)
    else:
        return True, "✅ Environment is properly set up. Ready to vectorize images!"

# Launch the app
if __name__ == "__main__":
    # Check environment
    is_ready, message = check_environment()

    if is_ready:
        print(f"\n{message}\n")
    else:
        print(f"\n{message}")
        print("The app will start, but vectorization may fail without proper setup.\n")

    # Launch with a nice message
    print("🚀 Starting the Image Vectorizer web interface...")
    app.launch(share=False, inbrowser=True)
