import os
import tempfile
import gradio as gr
import base64
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv
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
        tuple: (svg_path, svg_content, message)
    """
    if not api_key:
        return None, None, "❌ ERROR: Recraft API token not found!\n\nPlease create a .env file with your RECRAFT_API_TOKEN.\nSee the instructions for more details."

    try:
        # Generate unique filenames based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the uploaded image to the uploads directory
        input_filename = f"image_{timestamp}.png"
        input_path = os.path.join(UPLOADS_DIR, input_filename)
        image.save(input_path)

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
            return None, None, f"Failed to download SVG from URL: {svg_url}"

        # For debugging
        file_size = os.path.getsize(output_path)
        print(f"SVG file size: {file_size} bytes")

        # Update progress
        progress(1.0, "Process complete!")

        # Create a more informative success message with better formatting for the UI
        success_message = f"✅ Vectorization successful!\n\n🔗 SVG URL:\n{svg_url}\n\n💾 Files saved to:\n📄 Input: {os.path.basename(input_path)}\n📄 Output: {os.path.basename(output_path)}"

        return output_path, output_path, success_message

    except Exception as e:
        # Print the full traceback to the console for debugging
        traceback.print_exc()

        # Create a user-friendly error message
        error_message = f"❌ Vectorization failed!\n\n⚠️ Error: {str(e)}\n\nPlease check the console for more details."
        return None, None, error_message

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
            output_preview = gr.Image(
                label="",
                elem_id="output-preview",
                height=300,
                show_download_button=False
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

# Launch the app
if __name__ == "__main__":
    # Check if API key is available
    if not api_key:
        print("\n⚠️  WARNING: Recraft API token not found!")
        print("   Please create a .env file with your RECRAFT_API_TOKEN.")
        print("   The app will start, but vectorization will fail without a valid API token.\n")
    else:
        print("\n✅ Recraft API token found. Ready to vectorize images!\n")

    # Launch with a nice message
    print("🚀 Starting the Image Vectorizer web interface...")
    app.launch(share=False, inbrowser=True)
