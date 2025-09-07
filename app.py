"""
Gradio interface for Virtual Fashion Try-On deployment on Hugging Face Spaces
"""

import gradio as gr
import sys
import os
from PIL import Image
import torch
import numpy as np

# Add webapp to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'webapp'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'virtual_tryon.settings.production')

try:
    import django
    django.setup()
    from tryon_app.views import process_virtual_tryon
    DJANGO_AVAILABLE = True
except ImportError as e:
    print(f"Django setup failed: {e}")
    DJANGO_AVAILABLE = False

def virtual_tryon_interface(image, clothing_prompt):
    """
    Main interface function for virtual try-on
    """
    try:
        if image is None:
            return None, "Please upload an image"
        
        if not clothing_prompt or clothing_prompt.strip() == "":
            return None, "Please enter a clothing description"
        
        # Validate image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            return None, "Invalid image format"
        
        # Process the virtual try-on
        if DJANGO_AVAILABLE:
            try:
                result_image = process_virtual_tryon(image, clothing_prompt)
                return result_image, "Try-on completed successfully!"
            except Exception as e:
                return None, f"Processing error: {str(e)}"
        else:
            # Fallback: return original image with overlay text
            return image, "Django not available - showing original image"
            
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .image-container {
        max-height: 600px;
    }
    """
    
    # Create the interface
    interface = gr.Interface(
        fn=virtual_tryon_interface,
        inputs=[
            gr.Image(
                type="pil", 
                label="📸 Upload Your Photo",
                info="Upload a clear photo of yourself (preferably upper body)"
            ),
            gr.Textbox(
                label="👕 Describe the Clothing",
                placeholder="e.g., 'red summer dress', 'blue denim jacket', 'white cotton t-shirt'",
                info="Be specific about color, style, and type of clothing"
            )
        ],
        outputs=[
            gr.Image(label="✨ Try-On Result", type="pil"),
            gr.Textbox(label="📝 Status", interactive=False)
        ],
        title="🌟 Virtual Fashion Try-On",
        description="""
        **AI-Powered Virtual Clothing Try-On System**
        
        Upload your photo and describe the clothing you want to try on. 
        Our AI will generate a realistic preview of how it would look on you!
        
        💡 **Tips for best results:**
        - Use clear, well-lit photos
        - Face the camera directly
        - Describe clothing with specific details
        - Try different styles and colors!
        """,
        article="""
        ### 🔧 How it works:
        1. **Human Parsing**: AI identifies your body structure
        2. **Garment Generation**: Creates clothing from your description
        3. **Virtual Synthesis**: Blends the garment onto your photo
        
        ### 🎯 Best Practices:
        - **Photo Quality**: Good lighting, clear background
        - **Descriptions**: Be specific (e.g., "red silk blouse" vs "shirt")
        - **Patience**: Processing may take 30-60 seconds
        
        ---
        *Made with ❤️ using PyTorch, Stable Diffusion, and Gradio*
        """,
        examples=[
            [None, "red summer dress"],
            [None, "blue denim jacket"],
            [None, "white cotton t-shirt"],
            [None, "black leather jacket"],
            [None, "floral print blouse"]
        ],
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple"
        ),
        allow_flagging="never"
    )
    
    return interface

if __name__ == "__main__":
    # Check system resources
    print("🚀 Starting Virtual Fashion Try-On...")
    print(f"📦 Python version: {sys.version}")
    print(f"🔥 PyTorch available: {torch.__version__ if 'torch' in sys.modules else 'Not installed'}")
    print(f"🖥️  CUDA available: {torch.cuda.is_available() if 'torch' in sys.modules else 'Unknown'}")
    print(f"🌐 Django available: {DJANGO_AVAILABLE}")
    
    # Create and launch the interface
    interface = create_interface()
    
    # Launch with appropriate settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        enable_queue=True,
        max_threads=10
    )