is this app.py?
Edit

Yes, this file appears to be an app.py file for a Gradio interface that implements a custom "Lightning Painter" application for SDXL (Stable Diffusion XL) models. The application provides both inpainting and outpainting functionality using custom pipelines and models.

The code sets up a Gradio web interface with several tabs:

    Inpaint - For filling in masked areas of an image
    Outpaint - For extending images beyond their boundaries
    Misc - For utility functions like clearing CUDA cache

Key features of this application include:

    Support for custom SDXL models including NSFW ones
    A model selection dropdown that can be customized with Hugging Face model IDs
    ControlNet integration for better inpainting/outpainting results
    Various settings for controlling the generation process (alignment, overlap, resize options)
    Image history and preview functionality
    Memory management tools for GPU usage

This file is designed to run in an environment with GPU support, such as Google Colab Pro/Pro+ or Vast.ai, and requires several companion files (controlnet_union.py, pipeline_fill_sd_xl.py, and requirements.txt) to be present in the same directory.

Givme the a optimized file with impromivnet sn and bugfixes
Edit

I'll provide an optimized version of the app.py file with improvements and bugfixes. This version includes better memory management, cleaner code structure, improved error handling, and fixes for some potential issues in the original code.

Si that the entrie folder?
1 message remaining until 1:00 AM
Est. messages: 3.1
Reset in: 3h 47m
# Lightning Painter (In n' Out) - Custom SDXL Notebook - Optimized Version
# 
# This notebook sets up and runs an optimized version of the Lightning Painter application,
# allowing you to use custom SDXL models (including NSFW ones) for outpainting and inpainting
# on powerful GPU instances like those provided by Colab Pro/Pro+ or Vast.ai.

import os
import gc
import torch
import spaces
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL, TCDScheduler, UNet2DConditionModel
from diffusers.models.model_loading_utils import load_state_dict
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from gradio_imageslider import ImageSlider

# Import custom modules with proper error handling
try:
    from controlnet_union import ControlNetModel_Union
    CONTROLNET_AVAILABLE = True
except ImportError:
    print("Error: controlnet_union.py not found. Please ensure it's in the same directory.")
    CONTROLNET_AVAILABLE = False

try:
    from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    print("Error: pipeline_fill_sd_xl.py not found. Please ensure it's in the same directory.")
    PIPELINE_AVAILABLE = False

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Define the models you want to use
MODELS = {
    "RealVisXL V5.0 Lightning": "SG161222/RealVisXL_V5.0_Lightning",
    "Lustify Lightning": "GraydientPlatformAPI/lustify-lightning",
    "Juggernaut XL Lightning": "RunDiffusion/Juggernaut-XL-Lightning",
    "Juggernaut-XL-V9-GE-RDPhoto2": "AiWise/Juggernaut-XL-V9-GE-RDPhoto2-Lightning_4S",
    "SatPony-Lightning": "John6666/satpony-lightning-v2-sdxl",
    # Add your custom NSFW SDXL models here
    # "CustomModel1": "username/model-id",
    # "CustomModel2": "username/model-id",
}

# Global variables
pipe = None
vae = None
controlnet_instance = None
current_model_id = None


def load_vae():
    """Load the fixed VAE with proper error handling"""
    global vae
    
    if vae is not None:
        return vae
        
    try:
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=TORCH_DTYPE
        ).to(DEVICE)
        print("VAE loaded successfully.")
        return vae
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return None


def load_controlnet():
    """Load the custom ControlNet with proper error handling"""
    global controlnet_instance
    
    if not CONTROLNET_AVAILABLE:
        return None
        
    if controlnet_instance is not None:
        return controlnet_instance
        
    try:
        print("Loading ControlNet...")
        controlnet_repo_id = "xinsir/controlnet-union-sdxl-1.0"
        
        # Download and load config
        config_file_path = hf_hub_download(
            controlnet_repo_id,
            filename="config_promax.json",
        )
        config = ControlNetModel_Union.load_config(config_file_path)
        controlnet_model = ControlNetModel_Union.from_config(config)
        
        # Download and load model weights
        model_file_path = hf_hub_download(
            controlnet_repo_id,
            filename="diffusion_pytorch_model_promax.safetensors",
        )
        state_dict = load_state_dict(model_file_path)
        
        # Load the pretrained model
        controlnet_instance, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
            controlnet_model, state_dict, model_file_path, controlnet_repo_id
        )
        controlnet_instance.to(device=DEVICE, dtype=TORCH_DTYPE)
        print("ControlNet loaded successfully.")
        return controlnet_instance
    except Exception as e:
        print(f"Error loading ControlNet: {e}")
        return None


def unload_pipeline():
    """Safely unload pipeline and free GPU memory"""
    global pipe
    
    if pipe is not None:
        try:
            print("Offloading existing pipeline...")
            # Move components to CPU first
            pipe.to("cpu")
            # Delete the pipeline
            del pipe
            pipe = None
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("Pipeline offloaded successfully.")
        except Exception as e:
            print(f"Error during pipeline offloading: {e}")


def load_pipeline_components(model_id):
    """Load UNet and text encoders for a specific model"""
    try:
        print(f"Loading model components for {model_id}...")
        
        # Load UNet
        unet = UNet2DConditionModel.from_pretrained(
            model_id, 
            subfolder="unet", 
            torch_dtype=TORCH_DTYPE
        ).to(DEVICE)
        
        # Load text encoders
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, 
            subfolder="text_encoder", 
            torch_dtype=TORCH_DTYPE
        ).to(DEVICE)
        
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id, 
            subfolder="text_encoder_2", 
            torch_dtype=TORCH_DTYPE
        ).to(DEVICE)
        
        # Load tokenizers
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        
        print(f"Components for {model_id} loaded successfully.")
        return unet, text_encoder, text_encoder_2, tokenizer, tokenizer_2
    except Exception as e:
        print(f"Error loading model components: {e}")
        return None, None, None, None, None


def load_pipeline_for_model(model_id):
    """Load the pipeline for a given model ID"""
    global pipe, current_model_id
    
    # Skip loading if all required components aren't available
    if not PIPELINE_AVAILABLE or not CONTROLNET_AVAILABLE:
        print("Required modules missing. Cannot load pipeline.")
        return None
    
    # Load VAE and ControlNet if not already loaded
    vae_instance = load_vae()
    controlnet = load_controlnet()
    
    if vae_instance is None or controlnet is None:
        print("Failed to load VAE or ControlNet. Cannot proceed.")
        return None
    
    # Check if the requested model is already loaded
    if pipe is not None and current_model_id == model_id:
        print(f"Pipeline for {model_id} is already loaded.")
        return pipe
    
    # Unload existing pipeline to free memory
    unload_pipeline()
    
    # Load model components
    unet, text_encoder, text_encoder_2, tokenizer, tokenizer_2 = load_pipeline_components(model_id)
    
    if unet is None:
        print(f"Failed to load components for {model_id}")
        return None
    
    try:
        # Create the pipeline
        print("Assembling pipeline...")
        pipeline = StableDiffusionXLFillPipeline(
            vae=vae_instance,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=TCDScheduler.from_config(unet.config.scheduler_config)
        )
        
        # Set the scheduler and move to device
        pipeline.scheduler = TCDScheduler.from_config(pipeline.scheduler.config)
        pipeline.to(DEVICE)
        
        # Store the current model ID
        current_model_id = model_id
        pipe = pipeline
        
        print(f"Pipeline for {model_id} loaded successfully")
        return pipe
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        current_model_id = None
        return None


# Initialize default pipeline on module load if all components are available
def init_default_pipeline():
    """Initialize the default pipeline on module load"""
    if PIPELINE_AVAILABLE and CONTROLNET_AVAILABLE:
        default_model = list(MODELS.values())[0]
        return load_pipeline_for_model(default_model)
    else:
        print("Skipping initial pipeline load because required components are missing.")
        return None


# --- Inpainting Functions ---

def fill_image(prompt, image, model_selection_inpaint, paste_back):
    """Generate inpainted image based on mask"""
    global pipe
    
    # Load selected model if needed
    model_id = MODELS[model_selection_inpaint]
    if pipe is None or current_model_id != model_id:
        pipe = load_pipeline_for_model(model_id)
    
    if pipe is None or image is None:
        yield None, None
        return
    
    try:
        # Encode prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt, DEVICE, True
        )
        
        # Process image and mask
        source = image["background"]
        mask = image["layers"][0]
        
        # Ensure mask has alpha channel
        if mask.mode != 'RGBA':
            print("Warning: Mask does not have an alpha channel")
            yield None, None
            return
            
        alpha_channel = mask.split()[3]
        binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
        
        # Create image for ControlNet input
        cnet_image = source.copy()
        cnet_image.paste(0, (0, 0), binary_mask)
        
        # Generate inpainted image
        for intermediate_image in pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
        ):
            # Create masked version of original for comparison
            original_masked = source.copy()
            original_masked.paste(Image.new('RGB', source.size, (0,0,0)), (0,0), binary_mask)
            
            yield original_masked, intermediate_image
        
        # Final result processing
        final_image = intermediate_image
        
        if paste_back and final_image is not None:
            # Paste generated content back into original
            final_image_rgb = final_image.convert("RGB")
            inverted_mask = Image.fromarray(255 - np.array(binary_mask))
            source_copy = source.copy()
            source_copy.paste(final_image_rgb, (0, 0), inverted_mask)
            final_output_image = source_copy
        elif final_image is not None:
            final_output_image = final_image
        else:
            final_output_image = source
            
        yield source, final_output_image
        
    except Exception as e:
        print(f"Error during inpainting: {e}")
        yield None, None


# --- Outpainting Functions ---

def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Check if expansion is possible with given parameters"""
    if alignment == "Left" and target_width <= source_width: return False
    if alignment == "Right" and target_width <= source_width: return False
    if alignment == "Top" and target_height <= source_height: return False
    if alignment == "Bottom" and target_height <= source_height: return False
    if alignment == "Middle" and (target_width <= source_width and target_height <= source_height): return False
    return True


def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, 
                          alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    """Prepare background image and mask for outpainting"""
    if image is None:
        return None, None
    
    target_size = (width, height)
    source_width_original, source_height_original = image.size
    
    # Calculate resize dimensions based on selected option
    if resize_option == "Full":
        scale_factor = min(target_size[0] / source_width_original, target_size[1] / source_height_original)
        source_width_resized = int(source_width_original * scale_factor)
        source_height_resized = int(source_height_original * scale_factor)
    elif resize_option == "Custom":
        resize_factor = custom_resize_percentage / 100.0
        source_width_resized = int(source_width_original * resize_factor)
        source_height_resized = int(source_height_original * resize_factor)
    else:  # Percentage options
        resize_percentage = int(resize_option.replace("%", ""))
        resize_factor = resize_percentage / 100.0
        source_width_resized = int(source_width_original * resize_factor)
        source_height_resized = int(source_height_original * resize_factor)
    
    # Ensure minimum dimensions
    source_width_resized = max(source_width_resized, 64)
    source_height_resized = max(source_height_resized, 64)
    
    # Resize source image
    source = image.resize((source_width_resized, source_height_resized), Image.LANCZOS)
    
    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - source_width_resized) // 2
        margin_y = (target_size[1] - source_height_resized) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - source_height_resized) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - source_width_resized
        margin_y = (target_size[1] - source_height_resized) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - source_width_resized) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - source_width_resized) // 2
        margin_y = target_size[1] - source_height_resized
    else:
        margin_x = 0
        margin_y = 0
    
    # Ensure margins are within bounds
    margin_x = max(0, min(margin_x, target_size[0] - source_width_resized))
    margin_y = max(0, min(margin_y, target_size[1] - source_height_resized))
    
    # Create background and paste resized source
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))
    
    # Create mask (255=masked area to be generated, 0=keep original)
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)
    
    # Small patch to ensure minimal unmasked area
    white_gaps_patch = 2
    
    # Calculate overlap dimensions
    overlap_x = max(int(source_width_resized * (overlap_percentage / 100.0)), 1)
    overlap_y = max(int(source_height_resized * (overlap_percentage / 100.0)), 1)
    
    # Calculate unmasked area coordinates
    unmasked_left = margin_x - overlap_x if overlap_left else margin_x + white_gaps_patch
    unmasked_top = margin_y - overlap_y if overlap_top else margin_y + white_gaps_patch
    unmasked_right = margin_x + source_width_resized + overlap_x if overlap_right else margin_x + source_width_resized - white_gaps_patch
    unmasked_bottom = margin_y + source_height_resized + overlap_y if overlap_bottom else margin_y + source_height_resized - white_gaps_patch
    
    # Adjust for alignment at edges
    if alignment == "Left":
        unmasked_left = margin_x
    elif alignment == "Right":
        unmasked_right = margin_x + source_width_resized
    elif alignment == "Top":
        unmasked_top = margin_y
    elif alignment == "Bottom":
        unmasked_bottom = margin_y + source_height_resized
    
    # Ensure coordinates are within bounds
    unmasked_left = max(0, unmasked_left)
    unmasked_top = max(0, unmasked_top)
    unmasked_right = min(target_size[0], unmasked_right)
    unmasked_bottom = min(target_size[1], unmasked_bottom)
    
    # Check for valid rectangle
    if unmasked_left >= unmasked_right or unmasked_top >= unmasked_bottom:
        print("Warning: Invalid unmasked area. Using fallback.")
        # Fallback to small central area
        center_x, center_y = target_size[0] // 2, target_size[1] // 2
        mask_draw.rectangle([(center_x - 10, center_y - 10), (center_x + 10, center_y + 10)], fill=0)
    else:
        # Draw unmasked area (black=0=unmasked)
        mask_draw.rectangle([
            (unmasked_left, unmasked_top),
            (unmasked_right, unmasked_bottom)
        ], fill=0)
    
    return background, mask


def preview_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, 
                          alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    """Generate preview with masked area highlighted in red"""
    if image is None:
        return None
        
    background, mask = prepare_image_and_mask(
        image, width, height, overlap_percentage, resize_option, custom_resize_percentage,
        alignment, overlap_left, overlap_right, overlap_top, overlap_bottom
    )
    
    if background is None or mask is None:
        return None
        
    # Create visualization with red overlay for masked areas
    preview = background.copy().convert('RGBA')
    red_overlay = Image.new('RGBA', background.size, (255, 0, 0, 64))
    red_mask = Image.new('RGBA', background.size, (0, 0, 0, 0))
    red_mask.paste(red_overlay, (0, 0), mask)
    preview = Image.alpha_composite(preview, red_mask)
    
    return preview


@spaces.GPU(duration=12)
def infer(image, width, height, overlap_percentage, num_inference_steps, resize_option, custom_resize_percentage,
         prompt_input, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom, model_selection_outpaint):
    """Generate outpainted image"""
    global pipe
    
    # Load selected model if needed
    model_id = MODELS[model_selection_outpaint]
    if pipe is None or current_model_id != model_id:
        pipe = load_pipeline_for_model(model_id)
    
    if pipe is None or image is None:
        yield None, None
        return
    
    try:
        # Prepare background and mask
        background, mask = prepare_image_and_mask(
            image, width, height, overlap_percentage, resize_option, custom_resize_percentage,
            alignment, overlap_left, overlap_right, overlap_top, overlap_bottom
        )
        
        if background is None or mask is None:
            print("Failed to prepare image and mask.")
            yield None, None
            return
        
        # Check if any masked area exists
        if np.max(np.array(mask)) < 255:
            print("No masked area detected.")
            yield background, background
            return
        
        # Check if expansion is actually possible
        if not can_expand(image.width, image.height, width, height, alignment):
            print(f"Warning: Selected configuration may not result in expansion.")
        
        # Create ControlNet input image
        cnet_image = background.copy()
        black_image = Image.new('RGB', background.size, (0, 0, 0))
        cnet_image.paste(black_image, (0, 0), mask)
        
        # Prepare prompt
        final_prompt = f"{prompt_input} , high quality, 4k"
        
        # Encode prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            final_prompt, DEVICE, True
        )
        
        # Generate outpainted image
        for intermediate_image in pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
            num_inference_steps=num_inference_steps
        ):
            yield background, intermediate_image
        
        # Final result processing
        final_image = intermediate_image
        
        if final_image is not None:
            # Paste generated content back into background
            final_output = background.copy()
            final_image_rgb = final_image.convert("RGB")
            final_output.paste(final_image_rgb, (0, 0), mask)
            yield background, final_output
        else:
            print("No final image generated.")
            yield background, background
            
    except Exception as e:
        print(f"Error during outpainting: {e}")
        if background is not None:
            yield background, background
        else:
            yield None, None


# --- UI Helper Functions ---

def clear_result():
    """Clear result display"""
    return gr.update(value=None)


def use_output_as_input(output_image_tuple):
    """Set the output image as the new input"""
    if output_image_tuple is None or not isinstance(output_image_tuple, tuple) or len(output_image_tuple) < 2:
        return gr.update(value=None)
    
    generated_image = output_image_tuple[1]
    return gr.update(value=generated_image)


def preload_presets(target_ratio, ui_width, ui_height):
    """Set dimensions based on selected ratio preset"""
    if target_ratio == "9:16":
        return 768, 1280, gr.update()
    elif target_ratio == "2:3":
        return 1024, 1536, gr.update()
    elif target_ratio == "16:9":
        return 1280, 768, gr.update()
    elif target_ratio == "1:1":
        return 1024, 1024, gr.update()
    elif target_ratio == "Custom":
        return ui_width, ui_height, gr.update(open=True)
    else:
        return ui_width, ui_height, gr.update()


def select_the_right_preset(user_width, user_height):
    """Identify which preset matches the current dimensions"""
    if user_width == 768 and user_height == 1280:
        return "9:16"
    elif user_width == 1024 and user_height == 1536:
        return "2:3"
    elif user_width == 1280 and user_height == 768:
        return "16:9"
    elif user_width == 1024 and user_height == 1024:
        return "1:1"
    else:
        return "Custom"


def toggle_custom_resize_slider(resize_option):
    """Show/hide custom resize slider based on selection"""
    return gr.update(visible=(resize_option == "Custom"))


def update_history(new_image_tuple, history):
    """Add new image to history gallery"""
    if (isinstance(new_image_tuple, tuple) and 
        len(new_image_tuple) == 2 and 
        new_image_tuple[1] is not None):
        
        new_image = new_image_tuple[1]
        if history is None:
            history = []
            
        # Add new image to beginning of history
        history.insert(0, new_image)
        
        # Limit history size
        return history[:20]
        
    return history


def clear_cache():
    """Clear CUDA cache and unload pipeline"""
    unload_pipeline()
    return gr.update(value="Cache cleared!")


def load_default_pipeline_ui():
    """Load default pipeline from UI button"""
    try:
        result = load_pipeline_for_model(list(MODELS.values())[0])
        if result is not None:
            return gr.update(value="Default pipeline loaded!")
        else:
            return gr.update(value="Failed to load default pipeline.")
    except Exception as e:
        return gr.update(value=f"Error loading default pipeline: {e}")


# --- Functions for Gradio Interface ---

def fill_image(prompt, image, model_selection_inpaint, paste_back):
    global pipe
    # Reload pipeline only if the selected model is different from the current one
    model_id = MODELS[model_selection_inpaint]
    if pipe is None or (hasattr(pipe, 'pretrained_model_name_or_path') and pipe.pretrained_model_name_or_path != model_id):
         pipe = load_pipeline_for_model(model_id)

    if pipe is None:
        print("Pipeline not loaded. Cannot generate.")
        yield None, None
        return

    print(f"Received image: {image}")
    if image is None:
        yield None, None
        return

    # The rest of the fill_image function remains largely the same,
    # as it now uses the globally updated 'pipe'
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(prompt, "cuda", True)
    source = image["background"]
    mask = image["layers"][0]
    alpha_channel = mask.split()[3]
    binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
    cnet_image = source.copy()
    cnet_image.paste(0, (0, 0), binary_mask)

    intermediate_image = None # Initialize to None
    # Use the pipeline's __call__ method which is a generator
    # The pipeline is expected to yield generated images
    for intermediate_image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
    ):
        # Yield a tuple for ImageSlider: original masked area image and the generated intermediate image
        # For inpainting, the "control image" passed to the pipeline is the original image with the inpaint area masked black.
        # Let's show this as the left image in the slider.
        original_masked_for_inpainting = source.copy()
        original_masked_for_inpainting.paste(Image.new('RGB', source.size, (0,0,0)), (0,0), binary_mask) # Mask the inpaint area black
        yield original_masked_for_inpainting, intermediate_image # Yield for ImageSlider
        # intermediate_image is already updated by the loop


    # After the loop, the last yielded image is the final result
    final_image = intermediate_image # Assuming the loop yields the final image last

    print(f"{model_selection_inpaint=}")
    print(f"{paste_back=}")
    if paste_back and final_image is not None:
        final_image_rgb = final_image.convert("RGB") # Ensure RGB before pasting
        # Paste the generated area back into the original image based on the binary mask
        # The paste mask (binary_mask) is 255 where generated, 0 where original should be kept.
        # This is reverse of the inpainting mask logic, so we invert the mask for pasting.
        inverted_mask = Image.fromarray(255 - np.array(binary_mask))
        source.paste(final_image_rgb, (0, 0), inverted_mask)
        final_output_image = source # The image with the generated part pasted back

    elif final_image is not None:
         final_output_image = final_image # Just the generated image

    else:
        final_output_image = source # If no image was generated, return original source


    # Yield the final result tuple (original source, final output image) for ImageSlider
    yield source, final_output_image


def clear_result():
    return gr.update(value=None)

def can_expand(source_width, source_height, target_width, target_height, alignment):
    # Check if the target size is actually larger in the direction of alignment
    if alignment == "Left" and target_width <= source_width: return False
    if alignment == "Right" and target_width <= source_width: return False
    if alignment == "Top" and target_height <= source_height: return False
    if alignment == "Bottom" and target_height <= source_height: return False
    # For Middle, check if target is larger in at least one dimension (assuming expansion is desired if any dimension is larger)
    if alignment == "Middle" and (target_width <= source_width and target_height <= source_height): return False

    return True


def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    if image is None:
        return None, None

    target_size = (width, height)

    # Determine the size of the source image within the target frame based on resize_option and alignment
    source_width_original, source_height_original = image.size

    if resize_option == "Full":
        resize_percentage = 100
        # When 'Full', the image is scaled to fit the smaller dimension of the target,
        # maintaining aspect ratio.
        scale_factor = min(target_size[0] / source_width_original, target_size[1] / source_height_original)
        source_width_resized = int(source_width_original * scale_factor)
        source_height_resized = int(source_height_original * scale_factor)

    elif resize_option == "Custom":
         resize_percentage = custom_resize_percentage
         resize_factor = resize_percentage / 100.0
         source_width_resized = int(source_width_original * resize_factor)
         source_height_resized = int(source_height_original * resize_factor)

    else: # "80%", "66%", "50%", "33%", "25%"
        resize_percentage = int(resize_option.replace("%", ""))
        resize_factor = resize_percentage / 100.0
        source_width_resized = int(source_width_original * resize_factor)
        source_height_resized = int(source_height_original * resize_factor)

    # Ensure minimum size
    source_width_resized = max(source_width_resized, 64)
    source_height_resized = max(source_height_resized, 64)


    # Resize the source image
    source = image.resize((source_width_resized, source_height_resized), Image.LANCZOS)

    # Calculate margins based on the resized source image and alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - source_width_resized) // 2
        margin_y = (target_size[1] - source_height_resized) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - source_height_resized) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - source_width_resized
        margin_y = (target_size[1] - source_height_resized) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - source_width_resized) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - source_width_resized) // 2
        margin_y = target_size[1] - source_height_resized
    else: # Should not happen with current choices, but as a fallback
        margin_x = 0
        margin_y = 0

    # Ensure margins are not negative (image is larger than target in a dimension)
    margin_x = max(0, margin_x)
    margin_y = max(0, margin_y)

    # Ensure the pasted area fits within the target size
    margin_x = min(margin_x, target_size[0] - source_width_resized)
    margin_y = min(margin_y, target_size[1] - source_height_resized)


    # Create background and paste the resized source image
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new('L', target_size, 255) # Start with a white mask (unmasked)
    mask_draw = ImageDraw.Draw(mask)

    white_gaps_patch = 2 # Small patch to ensure a minimal unmasked area around the source

    # Calculate the overlap area coordinates for masking
    overlap_x = int(source_width_resized * (overlap_percentage / 100.0))
    overlap_y = int(source_height_resized * (overlap_percentage / 100.0))
    overlap_x = max(overlap_x, 1) # Minimum overlap of 1 pixel
    overlap_y = max(overlap_y, 1) # Minimum overlap of 1 pixel


    # Determine the coordinates of the rectangle to draw on the mask (the unmasked area)
    # This rectangle corresponds to the source image area plus overlaps.
    unmasked_left = margin_x - overlap_x if overlap_left else margin_x + white_gaps_patch # Adjust for overlap or small patch
    unmasked_top = margin_y - overlap_y if overlap_top else margin_y + white_gaps_patch # Adjust for overlap or small patch
    unmasked_right = margin_x + source_width_resized + overlap_x if overlap_right else margin_x + source_width_resized - white_gaps_patch # Adjust for overlap or small patch
    unmasked_bottom = margin_y + source_height_resized + overlap_y if overlap_bottom else margin_y + source_height_resized - white_gaps_patch # Adjust for overlap or small patch


    # Adjust unmasked area based on alignment for edge cases at the target boundaries
    # If aligned to an edge, the unmasked area boundary should match the target boundary on that side,
    # overriding the overlap or patch.
    if alignment == "Left":
         unmasked_left = margin_x
         # On the opposite side, still use overlap or patch if not going to the very edge
         if not overlap_right and (margin_x + source_width_resized - white_gaps_patch) > margin_x:
             unmasked_right = margin_x + source_width_resized - white_gaps_patch
         else:
             unmasked_right = margin_x + source_width_resized + overlap_x if overlap_right else margin_x + source_width_resized


    elif alignment == "Right":
         unmasked_right = margin_x + source_width_resized
         if not overlap_left and (margin_x + white_gaps_patch) < (margin_x + source_width_resized):
             unmasked_left = margin_x + white_gaps_patch
         else:
              unmasked_left = margin_x - overlap_x if overlap_left else margin_x

    elif alignment == "Top":
         unmasked_top = margin_y
         if not overlap_bottom and (margin_y + source_height_resized - white_gaps_patch) > margin_y:
             unmasked_bottom = margin_y + source_height_resized - white_gaps_patch
         else:
              unmasked_bottom = margin_y + source_height_resized + overlap_y if overlap_bottom else margin_y + source_height_resized

    elif alignment == "Bottom":
         unmasked_bottom = margin_y + source_height_resized
         if not overlap_top and (margin_y + white_gaps_patch) < (margin_y + source_height_resized):
             unmasked_top = margin_y + white_gaps_patch
         else:
              unmasked_top = margin_y - overlap_y if overlap_top else margin_y

    # Ensure unmasked coordinates are within target size boundaries (0 to width/height)
    unmasked_left = max(0, unmasked_left)
    unmasked_top = max(0, unmasked_top)
    unmasked_right = min(target_size[0], unmasked_right)
    unmasked_bottom = min(target_size[1], unmasked_bottom)

    # Ensure the rectangle is valid (left < right, top < bottom)
    if unmasked_left >= unmasked_right or unmasked_top >= unmasked_bottom:
        print("Warning: Invalid unmasked rectangle coordinates. Masking might be incorrect.")
        # As a fallback, just mask the whole image except a tiny central area
        mask = Image.new('L', target_size, 255)
        center_x = target_size[0] // 2
        center_y = target_size[1] // 2
        mask_draw.rectangle([(center_x - 10, center_y - 10), (center_x + 10, center_y + 10)], fill=0) # Unmask a small central area
        return background, mask



    # Draw a black rectangle for the unmasked area
    # Mask value of 0 means unmasked, 255 means masked.
    mask_draw.rectangle([
        (unmasked_left, unmasked_top),
        (unmasked_right, unmasked_bottom)
    ], fill=0) # Fill the calculated area with black (0) to UNMASK it.


    return background, mask


def preview_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    if image is None:
        return None
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    if background is None or mask is None:
        return None
    preview = background.copy().convert('RGBA')
    # Create a red overlay where the mask is NOT 0 (i.e., where the mask is 255 - the masked area)
    # Inverted mask: areas with mask value 255 become red overlay (opacity 64/255), areas with mask 0 remain transparent
    red_overlay = Image.new('RGBA', background.size, (255, 0, 0, 64))
    # Create a full transparent layer
    red_mask = Image.new('RGBA', background.size, (0, 0, 0, 0))
    # Paste the red overlay onto the transparent layer, using the mask itself as the alpha channel.
    # This is slightly counter-intuitive. The mask values (0 or 255) are used as the alpha for the red overlay.
    # Where the mask is 255 (white - masked area), the red overlay is fully opaque.
    # Where the mask is 0 (black - unmasked area), the red overlay is fully transparent.
    # So, the red overlay is applied *only* to the masked areas.
    red_mask.paste(red_overlay, (0, 0), mask)
    preview = Image.alpha_composite(preview, red_mask)
    return preview


@spaces.GPU(duration=12)
def infer(image, width, height, overlap_percentage, num_inference_steps, resize_option, custom_resize_percentage, prompt_input, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom, model_selection_outpaint):
    global pipe
    # Reload pipeline only if the selected model is different from the current one
    model_id = MODELS[model_selection_outpaint]
    if pipe is None or (hasattr(pipe, 'pretrained_model_name_or_path') and pipe.pretrained_model_name_or_path != model_id):
         pipe = load_pipeline_for_model(model_id)

    if pipe is None:
        print("Pipeline not loaded. Cannot generate.")
        yield None, None
        return


    if image is None:
        yield None, None
        return

    # Prepare background and mask
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    if background is None or mask is None:
        print("Failed to prepare image and mask.")
        yield None, None
        return

    # Check if any masked area exists. If not, no outpainting is needed.
    if np.max(np.array(mask)) < 255:
         print("No masked area detected. Cannot outpaint.")
         # Yield the background image as the "result" (left and right of slider will be the same)
         yield background, background
         return

    # Check if expansion is possible with the selected alignment (simplified check after mask is prepared)
    # If the prepared background is not larger than the original image in the direction of alignment,
    # and there is a masked area, it means the original image was likely too large in that direction
    # or the resize/alignment settings prevent expansion.
    # We could potentially try a different alignment automatically here as a fallback, but for now,
    # let's just indicate that outpainting might not result in expansion as expected if can_expand is False.
    if not can_expand(image.width, image.height, width, height, alignment):
         print(f"Warning: Selected target size and alignment '{alignment}' may not result in outward expansion.")


    cnet_image = background.copy()
    # Create the input image for ControlNet - original image with black in the masked area
    # Paste black where the mask is 255 (masked)
    black_image = Image.new('RGB', background.size, (0, 0, 0))
    cnet_image.paste(black_image, (0, 0), mask) # Paste black using the mask as alpha

    final_prompt = f"{prompt_input} , high quality, 4k"
    print(f"Outpainting using SDXL model: {pipe.pretrained_model_name_or_path}")

    # Encode prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(final_prompt, "cuda", True)

    intermediate_image = None # Initialize to hold the last yielded image
    # Use the pipeline's __call__ method which is a generator
    # The pipe is expected to return generated images at intermediate steps and the final image.
    for intermediate_generated_image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image, # Pass the ControlNet conditioning image (original with masked area black)
        num_inference_steps=num_inference_steps
    ):
         if intermediate_generated_image is not None:
              # Yield a tuple for ImageSlider: original masked area image and the generated intermediate image
              # The infer function in app.py should yield the background on the left of the slider
              # and the generated/processed image on the right.
              yield background, intermediate_generated_image
              intermediate_image = intermediate_generated_image # Keep track of the last generated image


    # After the loop, combine the final generated image with the original background
    final_generated_image_area = intermediate_image # The last yielded image is the final generated area

    final_output_image = None
    if final_generated_image_area is not None:
        # Paste the generated part back into the original background based on the mask
        final_output_image = background.copy()
        # Convert generated image to RGB (ControlNet pipeline outputs might be PIL RGB already, but safer)
        final_generated_image_rgb = final_generated_image_area.convert("RGB")
        # Paste the generated part where the mask is 255 (masked area)
        # Need to use the mask directly for pasting the generated part onto the background.
        final_output_image.paste(final_generated_image_rgb, (0, 0), mask)


        yield background, final_output_image # Yield the final result tuple for ImageSlider
    else:
        print("No final image generated by the pipeline after the loop.")
        yield background, background # Yield original background if no final image was generated


def use_output_as_input(output_image_tuple):
    if output_image_tuple is None or not isinstance(output_image_tuple, tuple) or len(output_image_tuple) < 2:
        return gr.update(value=None)
    # Assuming output_image_tuple is a tuple (background, generated_image)
    # We want the generated_image to become the new input_image for outpainting.
    generated_image = output_image_tuple[1]
    return gr.update(value=generated_image)


def preload_presets(target_ratio, ui_width, ui_height):
    if target_ratio == "9:16":
        changed_width = 768
        changed_height = 1280
        return changed_width, changed_height, gr.update()
    elif target_ratio == "2:3":
        changed_width = 1024
        changed_height = 1536
        return changed_width, changed_height, gr.update()
    elif target_ratio == "16:9":
        changed_width = 1280
        changed_height = 768
        return changed_width, changed_height, gr.update()
    elif target_ratio == "1:1":
        changed_width = 1024
        changed_height = 1024
        return changed_width, changed_height, gr.update()
    elif target_ratio == "Custom":
        return ui_width, ui_height, gr.update(open=True)
    else:
        return ui_width, ui_height, gr.update()

def select_the_right_preset(user_width, user_height):
    if user_width == 768 and user_height == 1280:
        return "9:16"
    elif user_width == 1024 and user_height == 1536:
        return "2:3"
    elif user_width == 1280 and user_height == 768:
        return "16:9"
    elif user_width == 1024 and user_height == 1024:
        return "1:1"
    else:
        return "Custom"

def toggle_custom_resize_slider(resize_option):
    return gr.update(visible=(resize_option == "Custom"))

def update_history(new_image_tuple, history):
    # This function is triggered when infer yields. We only want to add the final image
    # to history, not intermediate ones. The infer function now yields (background, generated_image).
    # Let's check if the first image in the tuple is the background to identify the final yield.
    # A more robust check could involve a flag in the yielded tuple.
    if new_image_tuple is None or not isinstance(new_image_tuple, tuple) or len(new_image_tuple) < 2:
        return history # Return existing history if the yield is not a valid tuple

    # Check if the first element of the tuple is likely the background image
    # This is a heuristic based on the infer function's final yield structure.
    # It assumes the first element is the background and the second is the final generated image.
    # For intermediate yields, the first element is the cnet_image.
    # Since both background and cnet_image have the same size, this check might not be perfect.
    # A better way would be for infer to explicitly signal the final image yield.
    # However, let's use this heuristic for now. If the first element is the *original* input image (background before masking),
    # that strongly suggests it's the final yield tuple.
    # The `background` variable *inside* `infer` is the correctly sized white background with original pasted.
    # Let's compare the first element of the yielded tuple to the global `pipe.background_image` if we were to store it,
    # but that's getting complex.

    # A simpler approach: trust that the *last* yield from infer is the final result, and add it.
    # Gradio's .then() seems to process all yields from the generator. The history update needs
    # to only capture the very last one. The current structure adding all yields is likely not intended.
    # Let's just modify this to append the LAST received image tuple's second element. This requires
    # handling the sequence of yields in the history update.

    # Alternative: Modify the infer function to *only* yield the final result, not intermediates,
    # if we want history to only show final results. But user asked for intermediates.
    # The current update_history appends *every* generated image.

    # Let's revise update_history to store a list of generated images, triggered *after* the infer
    # function is complete. The infer function needs to return the final result *instead* of yielding.
    # Or, the history update needs to look at the *final* value in the result output component,
    # not the yielded values during generation.

    # Let's change infer to return the final tuple instead of yielding only the final.
    # Then update_history gets the final tuple.

    # RETHINKING REQUIRED: Gradio's generator function output behavior with .then() and history.
    # When a generator function yields, the output component updates. A `.then()` attached
    # to a generator will trigger *after* the generator finishes, receiving the final value
    # assigned to the output component.
    # So, update_history is getting the FINAL value of `result_outpaint` which *should* be
    # the last yielded tuple (background, final_image). This makes the current update_history logic correct.

    # Let's just add a simple check for tuple format.
    if isinstance(new_image_tuple, tuple) and len(new_image_tuple) == 2 and new_image_tuple[1] is not None:
        new_image = new_image_tuple[1]
        if history is None:
            history = []
        # Add the new image to the beginning of the history list
        history.insert(0, new_image)
        # Keep history size manageable, e.g., last 20 images
        history = history[:20]
        return history

    return history # Return existing history if the yield is not a valid final tuple


def clear_cache():
    global pipe
    print("Clearing CUDA cache and offloading pipeline...")
    # Attempt to free memory more aggressively
    if pipe is not None:
        try:
            pipe.to("cpu") # Move to CPU
            del pipe # Delete the pipeline object
        except Exception as e:
             print(f"Error during pipeline deletion: {e}")
    pipe = None
    torch.cuda.empty_cache()
    gc.collect() # Explicitly call garbage collector
    print("Cache cleared and pipeline offloaded.")
    return gr.update(value="Cache cleared!")

def load_default_pipeline_ui():
    global pipe
    print("Attempting to load default pipeline via UI button.")
    try:
        pipe = load_pipeline_for_model(list(MODELS.values())[0])
        if pipe is not None:
             print("Default pipeline loaded successfully.")
             return gr.update(value="Default pipeline loaded!")
        else:
             print("Failed to load default pipeline.")
             return gr.update(value="Failed to load default pipeline.")
    except Exception as e:
        print(f"Error loading default pipeline from UI: {e}")
        return gr.update(value=f"Error loading default pipeline: {e}")


css = """
.nulgradio-container {
    width: 86vw !important;
}
.nulcontain {
    overflow-y: scroll !important;
    padding: 10px 40px !important;
}
div#component-17 {
    height: auto !important;
}


@media screen and (max-width: 600px) {
    .img-row{
        display: block !important;
        margin-bottom: 20px !important;
    }
    div#component-16 {
        display: block !important;
    }
}

"""

title = """<h1 align="center">Diffusers Image Outpaint</h1>
<div align="center">Drop an image you would like to extend, pick your expected ratio and hit Generate.</div>
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
    <p style="display: flex;gap: 6px;">
         <a href="https://huggingface.co/spaces/fffiloni/diffusers-image-outpout?duplicate=true">
            <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-md.svg" alt="Duplicate this Space">
        </a> to skip the queue and enjoy faster inference on the GPU of your choice
    </p>
</div>
"""

# Check if required modules were imported successfully before creating the Gradio app
if StableDiffusionXLFillPipeline is None or ControlNetModel_Union is None:
    print("Skipping Gradio app creation due to missing modules.")
else:
    with gr.Blocks(css=css, fill_height=True) as demo:
        gr.Markdown("# Diffusers Inpaint and Outpaint")
        with gr.Tabs():
            with gr.TabItem("Inpaint"):
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            prompt = gr.Textbox(
                                label="Prompt",
                                info="Describe what to inpaint the mask with",
                                lines=3,
                            )
                        with gr.Column():
                            model_selection_inpaint = gr.Dropdown( # Renamed for clarity
                                choices=list(MODELS.keys()),
                                value=list(MODELS.keys())[0], # Default to the first model
                                label="Model",
                            )
                            with gr.Row():
                                run_button = gr.Button("Generate")
                                paste_back = gr.Checkbox(True, label="Paste back original")
                    with gr.Row(equal_height=False):
                        input_image = gr.ImageMask(
                            type="pil", label="Input Image", layers=True, elem_classes="img-row"
                        )
                        result = ImageSlider(
                            interactive=False,
                            label="Generated Image",
                            elem_classes="img-row"
                        )
                    use_as_input_button = gr.Button("Use as Input Image", visible=False)
                    # Pass the full tuple output of infer to use_output_as_input
                    use_as_input_button.click(
                        fn=use_output_as_input, inputs=[result], outputs=[input_image]
                    )
                    # Model change handler for Inpaint tab
                    model_selection_inpaint.change(
                         fn=lambda selected_model_name: load_pipeline_for_model(MODELS[selected_model_name]),
                         inputs=[model_selection_inpaint],
                         outputs=[], # No direct UI output, just loads the model
                         queue=False # Load model immediately without waiting for other events
                    )
                    run_button.click(
                        fn=clear_result,
                        inputs=None,
                        outputs=result,
                    ).then(
                        fn=lambda: gr.update(visible=False),
                        inputs=None,
                        outputs=use_as_input_button,
                    ).then(
                        # fill_image is a generator, Gradio handles yielding intermediate results to 'result'
                        fn=fill_image,
                        inputs=[prompt, input_image, model_selection_inpaint, paste_back],
                        outputs=[result],
                    ).then(
                        fn=lambda: gr.update(visible=True),
                        inputs=None,
                        outputs=use_as_input_button,
                    )
                    prompt.submit(
                        fn=clear_result,
                        inputs=None,
                        outputs=result,
                    ).then(
                        fn=lambda: gr.update(visible=False),
                        inputs=None,
                        outputs=use_as_input_button,
                    ).then(
                        # fill_image is a generator
                        fn=fill_image,
                        inputs=[prompt, input_image, model_selection_inpaint, paste_back],
                        outputs=[result],
                    ).then(
                        fn=lambda: gr.update(visible=True),
                        inputs=None,
                        outputs=use_as_input_button,
                    )
            with gr.TabItem("Outpaint"):
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            input_image_outpaint = gr.Image(
                                type="pil",
                                label="Input Image"
                            )
                            with gr.Row():
                                with gr.Column(scale=2):
                                    prompt_input = gr.Textbox(label="Prompt (Optional)")
                                with gr.Column(scale=1):
                                    runout_button = gr.Button("Generate")
                            with gr.Row():
                                target_ratio = gr.Radio(
                                    label="Expected Ratio",
                                    choices=["2:3", "9:16", "16:9", "1:1", "Custom"],
                                    value="1:1",
                                    scale=2
                                )
                                alignment_dropdown = gr.Dropdown(
                                    choices=["Middle", "Left", "Right", "Top", "Bottom"],
                                    value="Middle",
                                    label="Alignment"
                                )
                            # Add model selection dropdown for Outpaint tab
                            model_selection_outpaint = gr.Dropdown(
                                choices=list(MODELS.keys()),
                                value=list(MODELS.keys())[0], # Default to the first model
                                label="Model",
                            )
                            with gr.Accordion(label="Advanced settings", open=False) as settings_panel:
                                with gr.Column():
                                    with gr.Row():
                                        width_slider = gr.Slider(
                                            label="Target Width",
                                            minimum=720,
                                            maximum=1536,
                                            step=8,
                                            value=1024,
                                        )
                                        height_slider = gr.Slider(
                                            label="Target Height",
                                            minimum=720,
                                            maximum=1536,
                                            step=8,
                                            value=1024,
                                        )
                                    num_inference_steps = gr.Slider(label="Steps", minimum=4, maximum=12, step=1, value=8)
                                    with gr.Group():
                                        overlap_percentage = gr.Slider(
                                            label="Mask overlap (%)",
                                            minimum=1,
                                            maximum=80,
                                            value=10,
                                            step=1
                                        )
                                        with gr.Row():
                                            overlap_top = gr.Checkbox(label="Overlap Top", value=True)
                                            overlap_right = gr.Checkbox(label="Overlap Right", value=True)
                                        with gr.Row():
                                            overlap_left = gr.Checkbox(label="Overlap Left", value=True)
                                            overlap_bottom = gr.Checkbox(label="Overlap Bottom", value=True)
                                    with gr.Row():
                                        resize_option = gr.Radio(
                                            label="Resize input image",
                                            choices=["Full", "80%", "66%", "50%", "33%", "25%", "Custom"],
                                            value="Full"
                                        )
                                        custom_resize_percentage = gr.Slider(
                                            label="Custom resize (%)",
                                            minimum=1,
                                            maximum=100,
                                            step=1,
                                            value=50,
                                            visible=False
                                        )
                                    with gr.Column():
                                        preview_button = gr.Button("Preview alignment and mask")
                            # gr.Examples(...) # Removed examples for brevity
                        with gr.Column():
                            result_outpaint = ImageSlider(
                                interactive=False,
                                label="Generated Image",
                            )
                            use_as_input_button_outpaint = gr.Button("Use as Input Image", visible=False)
                            # Pass the full tuple output of infer to use_output_as_input
                            use_as_input_button_outpaint.click(
                                fn=use_output_as_input,
                                inputs=[result_outpaint],
                                outputs=[input_image_outpaint]
                            )
                            history_gallery = gr.Gallery(label="History", columns=6, object_fit="contain", interactive=False)


                            preview_image = gr.Image(label="Preview")

            with gr.TabItem("Misc"):
                with gr.Column():
                    clear_cache_button = gr.Button("Clear CUDA Cache")
                    clear_cache_message = gr.Markdown("")
                    clear_cache_button.click(
                        fn=clear_cache,
                        inputs=None,
                        outputs=clear_cache_message,
                    )
                    load_default_button = gr.Button("Load Default Pipeline")
                    load_default_message = gr.Markdown("")
                    load_default_button.click(
                        fn=load_default_pipeline_ui, # Use the UI wrapper function
                        inputs=None,
                        outputs=load_default_message,
                    )

        target_ratio.change(
            fn=preload_presets,
            inputs=[target_ratio, width_slider, height_slider],
            outputs=[width_slider, height_slider, settings_panel],
            queue=False
        )

        width_slider.change(
            fn=select_the_right_preset,
            inputs=[width_slider, height_slider],
            outputs=[target_ratio],
            queue=False
        )

        height_slider.change(
            fn=select_the_right_preset,
            inputs=[width_slider, height_slider],
            outputs=[target_ratio],
            queue=False
        )

        resize_option.change(
            fn=toggle_custom_resize_slider,
            inputs=[resize_option],
            outputs=[custom_resize_percentage],
            queue=False
        )


        # Model change handler for Outpaint tab
        model_selection_outpaint.change(
             fn=lambda selected_model_name: load_pipeline_for_model(MODELS[selected_model_name]),
             inputs=[model_selection_outpaint],
             outputs=[], # No direct UI output, just loads the model
             queue=False # Load model immediately without waiting for other events
        )

        runout_button.click(
            fn=clear_result,
            inputs=None,
            outputs=result_outpaint,
        ).then(
            # infer is a generator, Gradio handles yielding intermediate results to 'result_outpaint'
            fn=infer,
            inputs=[input_image_outpaint, width_slider, height_slider, overlap_percentage, num_inference_steps,
                    resize_option, custom_resize_percentage, prompt_input, alignment_dropdown,
                    overlap_left, overlap_right, overlap_top, overlap_bottom, model_selection_outpaint], # Pass model selection
            outputs=[result_outpaint], # The ImageSlider
        ).then(
            # Update history after infer generator finishes and result_outpaint has the final value
            fn=lambda x, history: update_history(x, history),
            inputs=[result_outpaint, history_gallery],
            outputs=history_gallery,
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[use_as_input_button_outpaint],
        )

        prompt_input.submit(
            fn=clear_result,
            inputs=None,
            outputs=result_outpaint,
        ).then(
            # infer is a generator
            fn=infer,
            inputs=[input_image_outpaint, width_slider, height_slider, overlap_percentage, num_inference_steps,
                    resize_option, custom_resize_percentage, prompt_input, alignment_dropdown,
                    overlap_left, overlap_right, overlap_top, overlap_bottom, model_selection_outpaint], # Pass model selection
            outputs=[result_outpaint], # The ImageSlider
        ).then(
             # Update history after infer generator finishes
            fn=lambda x, history: update_history(x, history),
            inputs=[result_outpaint, history_gallery],
            outputs=history_gallery,
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[use_as_input_button_outpaint],
        )

        preview_button.click(
            fn=preview_image_and_mask,
            inputs=[input_image_outpaint, width_slider, height_slider, overlap_percentage, resize_option, custom_resize_percentage, alignment_dropdown,
                    overlap_left, overlap_right, overlap_top, overlap_bottom],
            outputs=[preview_image],
            queue=False
        )


    # Ensure the Gradio app is launched when the script is run
    # The notebook structure means this is automatically run when the cell is executed
    # So __main__ check is not strictly necessary in notebook but harmless.
    if __name__ == "__main__":
        demo.launch(share=True, debug=True) # Set share=True to get a public URL

# --- End of app.py code ---