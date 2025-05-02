import os
import gradio as gr
import traceback
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from recraft_vectorizer import vectorize_image, download_svg

# Load environment variables
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("RECRAFT_API_TOKEN")

# Create output directories if they don't exist
OUTPUT_DIR = "output"
UPLOADS_DIR = os.path.join(OUTPUT_DIR, "uploads")
VECTORS_DIR = os.path.join(OUTPUT_DIR, "vectors")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VECTORS_DIR, exist_ok=True)

def process_image(image, progress=gr.Progress()):
    """
    Process the uploaded image and return the vectorized SVG

    Args:
        image: The uploaded image file
        progress: Gradio progress indicator

    Returns:
        tuple: (svg_path, svg_html, message)
    """
    if not api_key:
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

        # Update progress
        progress(0.3, "Image uploaded, starting vectorization...")

        # Vectorize the image
        svg_url = vectorize_image(input_path)

        # Update progress
        progress(0.7, "Vectorization complete, downloading SVG...")

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
        progress(1.0, "Process complete!")

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
with gr.Blocks(title="Image Vectorizer", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🖼️ Image Vectorizer
        ### Convert raster images to scalable vector graphics using the Recraft API
        """
    )

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
            1. Upload an image using the upload area above
            2. Click the 'Vectorize Image' button
            3. Wait for the processing to complete
            4. View the vectorized SVG in the preview area
            5. Download the resulting SVG file using the download button

            **Note:**
            - This tool requires a valid Recraft API token to be set in the .env file
            - Uploaded images are saved to the 'output/uploads' directory
            - Vectorized SVGs are saved to the 'output/vectors' directory
            """
        )

    # Connect the button to the processing function
    vectorize_button.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_file, output_preview, output_message],
        show_progress=True
    )

def check_environment():
    """
    Check if the environment is properly set up

    Returns:
        tuple: (is_ready, message)
    """
    issues = []

    # Check if API key is available
    if not api_key:
        issues.append("⚠️ Recraft API token not found! Please create a .env file with your RECRAFT_API_TOKEN.")

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
