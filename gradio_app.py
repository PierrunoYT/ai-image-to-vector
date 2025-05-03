import os
import gradio as gr
import traceback
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from recraft_vectorizer import vectorize_image, download_svg
from ideogram_generator import generate_image
from api_provider import get_provider

# Load environment variables
load_dotenv()

# Get API keys from environment variables
recraft_api_key = os.getenv("RECRAFT_API_TOKEN")
replicate_api_key = os.getenv("REPLICATE_API_TOKEN")
fal_api_key = os.getenv("FAL_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create output directories if they don't exist
OUTPUT_DIR = "output"
UPLOADS_DIR = os.path.join(OUTPUT_DIR, "uploads")
VECTORS_DIR = os.path.join(OUTPUT_DIR, "vectors")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VECTORS_DIR, exist_ok=True)

def generate_and_process_image(prompt, aspect_ratio="1:1", magic_prompt_option="Auto", style_type="auto", provider_name=None, progress=gr.Progress()):
    """
    Generate an image from a text prompt and then vectorize it

    Args:
        prompt: Text prompt for image generation
        aspect_ratio: Aspect ratio of the generated image (e.g., "1:1", "16:9", "3:2")
        magic_prompt_option: Magic prompt option ("Auto", "On", "Off")
        style_type: Style type to use ("auto", "general", "realistic", "design", "none")
        provider_name: Name of the preferred provider ("openai", "replicate", or "fal")
        progress: Gradio progress indicator

    Returns:
        tuple: (generated_image, svg_path, svg_html, message)
    """
    # Check if any API provider is available
    if not openai_api_key and not replicate_api_key and not fal_api_key:
        return None, None, None, "❌ ERROR: No API provider configured!\n\nPlease create a .env file with either OPENAI_API_KEY, REPLICATE_API_TOKEN, or FAL_KEY.\nSee the instructions for more details."

    if not prompt or not prompt.strip():
        return None, None, None, "❌ ERROR: No prompt provided. Please enter a text prompt to generate an image."

    try:
        # Define progress stages and their relative weights for the entire process
        stages = {
            "prepare": {"weight": 0.05, "message": "Preparing to generate image..."},
            "generate": {"weight": 0.35, "message": "Generating image..."},
            "vectorize": {"weight": 0.6, "message": "Vectorizing image..."}
        }

        # Function to calculate weighted progress
        def update_progress(stage, stage_progress=1.0):
            # Calculate the beginning and end progress values for this stage
            stage_keys = list(stages.keys())
            stage_idx = stage_keys.index(stage)

            # Sum of weights up to this stage
            previous_weight = sum(stages[stage_keys[i]]["weight"] for i in range(stage_idx))

            # Weight of the current stage
            current_weight = stages[stage]["weight"]

            # Calculate the absolute progress (0.0 to 1.0)
            absolute_progress = (previous_weight + current_weight * stage_progress) / sum(s["weight"] for s in stages.values())

            # Update the progress bar
            progress(absolute_progress, stages[stage]["message"])

        # Start preparation stage
        update_progress("prepare", 1.0)

        # Start generation stage
        update_progress("generate", 0.0)

        # Generate the image using the appropriate provider
        # Create a progress wrapper to map image generation progress (0.0-1.0) to our generation stage (0.0-1.0)
        class ProgressWrapper:
            def __call__(self, value, message=""):
                # Map the value from the image generation (0.0-1.0) to our generation stage progress
                update_progress("generate", value)

        # Map provider name from UI to the format expected by the API
        provider_map = {
            "Auto": None,  # Auto selection
            "OpenAI (GPT-Image-1)": "openai",
            "Replicate": "replicate",
            "Fal.ai": "fal"
        }

        # Get the mapped provider name or None for Auto
        mapped_provider = provider_map.get(provider_name, None)

        # Generate the image using the appropriate provider
        generated_image = generate_image(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            magic_prompt_option=magic_prompt_option,
            style_type=style_type,
            provider_name=mapped_provider,
            progress=ProgressWrapper()
        )

        # Ensure we're at 100% for the generation stage
        update_progress("generate", 1.0)

        # Start vectorization stage
        update_progress("vectorize", 0.0)

        # Create a progress wrapper for the vectorization stage
        class VectorizationProgressWrapper:
            def __call__(self, value, message=""):
                # Map the value from vectorization progress (0.0-1.0) to our vectorization stage (0.0-1.0)
                update_progress("vectorize", value)

        # Process the generated image with our custom progress wrapper
        svg_path, svg_html, message = process_image_internal(generated_image, VectorizationProgressWrapper())

        # Return the results
        return generated_image, svg_path, svg_html, message

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
    # Check if the image is too large
    if image is not None:
        # Check dimensions
        MAX_DIMENSION = 4000  # Limit image to 4000x4000 pixels
        width, height = image.size
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            return None, None, f"❌ ERROR: Image is too large. Maximum dimensions allowed are {MAX_DIMENSION}x{MAX_DIMENSION} pixels. Your image is {width}x{height} pixels."

        # Check file size (calculate approximate size in memory)
        estimated_size_mb = (width * height * 3) / (1024 * 1024)  # 3 bytes per pixel (RGB)
        MAX_SIZE_MB = 10  # 10MB limit
        if estimated_size_mb > MAX_SIZE_MB:
            return None, None, f"❌ ERROR: Image is too large. Maximum file size allowed is {MAX_SIZE_MB}MB. Your image is approximately {estimated_size_mb:.2f}MB."

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
        # Define progress stages and their relative weights
        stages = {
            "save": {"weight": 0.1, "message": "Saving and validating image..."},
            "vectorize": {"weight": 0.5, "message": "Vectorizing image..."},
            "download": {"weight": 0.3, "message": "Downloading SVG..."},
            "finalize": {"weight": 0.1, "message": "Finalizing output..."}
        }

        # Function to calculate weighted progress
        def update_progress(stage, stage_progress=1.0):
            # Calculate the beginning and end progress values for this stage
            stage_keys = list(stages.keys())
            stage_idx = stage_keys.index(stage)

            # Sum of weights up to this stage
            previous_weight = sum(stages[stage_keys[i]]["weight"] for i in range(stage_idx))

            # Weight of the current stage
            current_weight = stages[stage]["weight"]

            # Calculate the absolute progress (0.0 to 1.0)
            absolute_progress = start_progress + (
                (previous_weight + current_weight * stage_progress) /
                sum(s["weight"] for s in stages.values())
            ) * (1.0 - start_progress)

            # Update the progress bar
            progress(absolute_progress, stages[stage]["message"])

        # Start the first stage - save and validate
        update_progress("save", 0.0)

        # Generate unique filenames based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the uploaded image to the uploads directory
        input_filename = f"image_{timestamp}.png"
        input_path = os.path.join(UPLOADS_DIR, input_filename)

        # Ensure the image is saved
        try:
            image.save(input_path)
            update_progress("save", 0.5)
        except Exception as e:
            return None, None, f"❌ ERROR: Failed to save uploaded image: {str(e)}"

        # Validate the image file
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            return None, None, "❌ ERROR: Failed to save the image or the image is empty."

        # Check if the file is a valid image
        try:
            with Image.open(input_path) as img:
                img.verify()  # Verify it's a valid image
            update_progress("save", 1.0)
        except Exception:
            os.remove(input_path)  # Clean up invalid file
            return None, None, "❌ ERROR: The uploaded file is not a valid image."

        # Start vectorization stage
        update_progress("vectorize", 0.0)

        # Vectorize the image
        svg_url = vectorize_image(input_path)
        update_progress("vectorize", 1.0)

        # Start download stage
        update_progress("download", 0.0)

        # Generate output path in the vectors directory
        output_filename = f"vector_{timestamp}.svg"
        output_path = os.path.join(VECTORS_DIR, output_filename)

        # Download the SVG
        success = download_svg(svg_url, output_path)
        if not success:
            return None, None, f"❌ ERROR: Failed to download SVG from URL: {svg_url}"
        update_progress("download", 1.0)

        # Start final stage
        update_progress("finalize", 0.0)

        # For debugging
        file_size = os.path.getsize(output_path)
        print(f"SVG file size: {file_size} bytes")

        # Create HTML to display the SVG properly
        svg_html = create_svg_preview_html(output_path)
        update_progress("finalize", 1.0)

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
        # Check if the file exists
        if not os.path.exists(svg_path):
            return f"""
            <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; background-color:#f5f5f5; border-radius:8px; padding:10px;">
                <p style="color:red;">Error: SVG file not found at path: {svg_path}</p>
            </div>
            """

        # Check if the file is readable and not empty
        if not os.access(svg_path, os.R_OK):
            return f"""
            <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; background-color:#f5f5f5; border-radius:8px; padding:10px;">
                <p style="color:red;">Error: SVG file is not readable: {svg_path}</p>
            </div>
            """

        if os.path.getsize(svg_path) == 0:
            return f"""
            <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; background-color:#f5f5f5; border-radius:8px; padding:10px;">
                <p style="color:red;">Error: SVG file is empty: {svg_path}</p>
            </div>
            """

        # Read the SVG file
        with open(svg_path, 'r') as f:
            svg_content = f.read()

        # Basic validation of SVG content
        if not svg_content.strip().startswith('<svg') and not '<?xml' in svg_content[:100]:
            return f"""
            <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; background-color:#f5f5f5; border-radius:8px; padding:10px;">
                <p style="color:red;">Error: File does not appear to be a valid SVG: {svg_path}</p>
            </div>
            """

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
with gr.Blocks(title="Image to Vector", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🖼️ Image to Vector
        ### Generate images with OpenAI GPT-Image-1 or Ideogram v3 and convert them to scalable vector graphics
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
                        label="Maximum dimensions: 4000x4000 pixels, Maximum size: 10MB",
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
                            "1:1",    # Square - supported by all
                            "3:2",    # Landscape - supported by GPT-Image-1
                            "2:3",    # Portrait - supported by GPT-Image-1
                            "16:9",   # Landscape - not supported by GPT-Image-1, will use auto
                            "9:16",   # Portrait - not supported by GPT-Image-1, will use auto
                            "4:3",    # Landscape - not supported by GPT-Image-1, will use auto
                            "3:4",    # Portrait - not supported by GPT-Image-1, will use auto
                            "auto"    # Auto - let API decide
                        ],
                        value="1:1",
                        elem_id="aspect-ratio"
                    )

                    # Add magic prompt option
                    magic_prompt_option = gr.Dropdown(
                        label="Magic Prompt (optimizes prompt for better results)",
                        choices=["Auto", "On", "Off"],
                        value="Auto",
                        elem_id="magic-prompt"
                    )

                    # Add style type dropdown
                    style_type = gr.Dropdown(
                        label="Style Type (defines aesthetic of the image)",
                        choices=["auto", "general", "realistic", "design", "none"],
                        value="auto",
                        elem_id="style-type"
                    )

                    # Add provider selection dropdown
                    provider_name = gr.Dropdown(
                        label="API Provider (image generation service)",
                        choices=["Auto", "OpenAI (GPT-Image-1)", "Replicate", "Fal.ai"],
                        value="Auto",
                        elem_id="provider-name"
                    )

                    # Add a warning message area for compatibility info
                    compatibility_warning = gr.Markdown(
                        visible=False,
                        value="",
                        elem_id="compatibility-warning"
                    )

                    # Function to update UI components based on provider
                    def update_ui_for_provider(provider, current_aspect_ratio):
                        # Initialize outputs
                        warning_message = ""
                        warning_visible = False

                        # For OpenAI GPT-Image-1
                        if provider == "OpenAI (GPT-Image-1)":
                            # 1. Update style type choices to background options for GPT-Image-1
                            style_update = gr.update(
                                choices=["auto", "transparent", "opaque"],  # All supported options
                                value="auto",
                                label="Background Type (auto, transparent, opaque)"
                            )

                            # 2. Update magic prompt options to quality options for GPT-Image-1
                            magic_prompt_update = gr.update(
                                choices=["auto", "high", "medium", "low"],
                                value="auto",
                                label="Quality (auto, high, medium, low)"
                            )

                            # 3. Check aspect ratio compatibility
                            gpt_supported_ratios = ["1:1", "3:2", "2:3", "auto"]
                            if current_aspect_ratio not in gpt_supported_ratios:
                                warning_message = """
                                ⚠️ **Aspect Ratio Warning**: GPT-Image-1 only supports 1:1, 3:2, 2:3, or auto.
                                Your selected ratio will default to "auto" size.

                                ℹ️ **GPT-Image-1 Parameters**:
                                - "Background Type": Controls background (auto, transparent, opaque)
                                - "Quality": Controls image quality (auto, high, medium, low)
                                """
                                warning_visible = True
                            else:
                                warning_message = """
                                ℹ️ **GPT-Image-1 Parameters**:
                                - "Background Type": Controls background (auto, transparent, opaque)
                                - "Quality": Controls image quality (auto, high, medium, low)
                                """
                                warning_visible = True
                        elif provider == "Auto":
                            # For Auto, we need to check if OpenAI is configured
                            # If it is, we should show the same warnings as for OpenAI
                            openai_api_key = os.getenv("OPENAI_API_KEY")
                            if openai_api_key and len(openai_api_key.strip()) > 0:
                                # OpenAI is configured and might be used, so show the same warnings
                                style_update = gr.update(
                                    choices=["auto", "transparent", "opaque"],  # All supported options
                                    value="auto",
                                    label="Background Type (auto, transparent, opaque)"
                                )

                                # Update magic prompt options
                                magic_prompt_update = gr.update(
                                    choices=["auto", "high", "medium", "low"],
                                    value="auto",
                                    label="Quality/Magic Prompt (depends on provider)"
                                )

                                gpt_supported_ratios = ["1:1", "3:2", "2:3", "auto"]
                                if current_aspect_ratio not in gpt_supported_ratios:
                                    warning_message = """
                                    ⚠️ **Aspect Ratio Warning**: If GPT-Image-1 is selected, it only supports 1:1, 3:2, 2:3, or auto.
                                    Your selected ratio will default to "auto" size when using GPT-Image-1.

                                    ℹ️ **Parameter Info**:
                                    - For GPT-Image-1: "Background Type" controls background (auto, transparent, opaque) and "Quality" sets quality level (auto, high, medium, low)
                                    - For Ideogram v3: "Style Type" controls aesthetic and "Magic Prompt" optimizes the prompt
                                    """
                                    warning_visible = True
                                else:
                                    warning_message = """
                                    ℹ️ **Parameter Info**:
                                    - For GPT-Image-1: "Background Type" controls background (auto, transparent, opaque) and "Quality" sets quality level (auto, high, medium, low)
                                    - For Ideogram v3: "Style Type" controls aesthetic and "Magic Prompt" optimizes the prompt
                                    """
                                    warning_visible = True
                            else:
                                # OpenAI is not configured, so show all style options
                                style_update = gr.update(
                                    choices=["auto", "general", "realistic", "design", "none"],
                                    value="auto",
                                    label="Style Type (defines aesthetic of the image)"
                                )

                                # Reset magic prompt options for Ideogram
                                magic_prompt_update = gr.update(
                                    choices=["Auto", "On", "Off"],
                                    value="Auto",
                                    label="Magic Prompt (optimizes prompt for better results)"
                                )
                        else:
                            # For other providers, all style options are available
                            style_update = gr.update(
                                choices=["auto", "general", "realistic", "design", "none"],
                                value="auto",
                                label="Style Type (defines aesthetic of the image)"
                            )

                            # Reset magic prompt options for Ideogram
                            magic_prompt_update = gr.update(
                                choices=["Auto", "On", "Off"],
                                value="Auto",
                                label="Magic Prompt (optimizes prompt for better results)"
                            )

                        return [style_update, gr.update(visible=warning_visible, value=warning_message), magic_prompt_update]

                    # Connect provider and aspect ratio changes to UI updates
                    provider_name.change(
                        fn=update_ui_for_provider,
                        inputs=[provider_name, aspect_ratio],
                        outputs=[style_type, compatibility_warning, magic_prompt_option]
                    )

                    aspect_ratio.change(
                        fn=update_ui_for_provider,
                        inputs=[provider_name, aspect_ratio],
                        outputs=[style_type, compatibility_warning, magic_prompt_option]
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
               - Note: GPT-Image-1 only supports 1:1, 3:2, and 2:3 ratios
            3. Select a Magic Prompt option:
               - Auto: Automatically optimize the prompt (default)
               - On: Always optimize the prompt
               - Off: Use the prompt as-is
            4. Choose a Style Type or Background Type (depending on provider)
            5. Select an API Provider:
               - Auto: Use the first available provider
               - OpenAI (GPT-Image-1): Uses OpenAI's image generation model
               - Replicate: Uses Ideogram v3 via Replicate
               - Fal.ai: Uses Ideogram v3 via Fal.ai
            6. Click the 'Generate & Vectorize' button
            7. Wait for the AI to generate your image and vectorize it
            8. View the generated image and vectorized SVG in the preview areas
            9. Download the resulting SVG file using the download button

            ## Provider-Specific Parameters

            ### OpenAI (GPT-Image-1)
            - **Background Type**: Controls background
              - auto: Let OpenAI decide (default)
              - transparent: Transparent background (requires PNG format)
              - opaque: Solid background
            - **Quality**: Controls image quality
              - auto: Let OpenAI decide (default)
              - high: High quality
              - medium: Medium quality
              - low: Low quality
            - **Aspect Ratio**: Only supports 1:1, 3:2, 2:3, or auto

            ### Ideogram v3 (Replicate and Fal.ai)
            - **Style Type**: Controls the aesthetic style
              - auto: Automatically select style (default)
              - general: General style
              - realistic: Realistic style
              - design: Design style
              - none: No specific style
            - **Magic Prompt**: Controls prompt optimization
              - Auto: Automatically optimize (default)
              - On: Always optimize
              - Off: Use prompt as-is
            - **Aspect Ratio**: Supports various ratios (1:1, 16:9, 9:16, etc.)

            **Note:**
            - This tool requires valid API tokens to be set in the .env file:
              - Recraft API token for vectorization (required)
              - At least one of the following for image generation:
                - OpenAI API key (OPENAI_API_KEY) for GPT-Image-1
                - Replicate API token (REPLICATE_API_TOKEN) for Ideogram v3
                - Fal.ai API key (FAL_KEY) for Ideogram v3
            - You can choose which API provider to use in the dropdown menu
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
        inputs=[text_prompt, aspect_ratio, magic_prompt_option, style_type, provider_name],
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

    # Check if at least one image generation API is available
    if not openai_api_key and not replicate_api_key and not fal_api_key:
        issues.append("⚠️ No image generation API configured! Please create a .env file with either OPENAI_API_KEY, REPLICATE_API_TOKEN, or FAL_KEY.")
    else:
        if not openai_api_key:
            print("ℹ️ OpenAI API key not found. GPT-Image-1 will not be available. Another provider will be used for image generation.")
        else:
            print("✅ OpenAI API key found. GPT-Image-1 will be available for image generation.")

        if not replicate_api_key:
            print("ℹ️ Replicate API token not found. Ideogram v3 via Replicate will not be available.")
        else:
            print("✅ Replicate API token found. Ideogram v3 via Replicate will be available for image generation.")

        if not fal_api_key:
            print("ℹ️ Fal.ai API key not found. Ideogram v3 via Fal.ai will not be available.")
        else:
            print("✅ Fal.ai API key found. Ideogram v3 via Fal.ai will be available for image generation.")

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
