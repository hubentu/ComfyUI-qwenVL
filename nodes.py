import os
import uuid
import folder_paths
import numpy as np

from PIL import Image
from pathlib import Path

# Model directories
text_encoders_directory = os.path.join(folder_paths.models_dir, "text_encoders")
os.makedirs(text_encoders_directory, exist_ok=True)

# Legacy support - also check VLM_GGUF directory
vlm_gguf_directory = os.path.join(folder_paths.models_dir, "VLM_GGUF")
os.makedirs(vlm_gguf_directory, exist_ok=True)

# Register text_encoders folder with ComfyUI's folder_paths system
# This enables support for extra_model_paths.yaml configuration
try:
    # Check if text_encoders is already registered (e.g., from extra_model_paths.yaml)
    existing_paths = folder_paths.folder_names_and_paths.get("text_encoders")
    if existing_paths:
        # Already registered, possibly from extra_model_paths.yaml
        # Ensure our default directory is included
        if text_encoders_directory not in existing_paths[0]:
            folder_paths.add_model_folder_path("text_encoders", text_encoders_directory)
            print(f"✅ Added default text_encoders path: {text_encoders_directory}")
        print(f"📂 text_encoders paths configured: {existing_paths[0]}")
    else:
        # Not registered yet, register it
        folder_paths.add_model_folder_path("text_encoders", text_encoders_directory)
        print(f"✅ Registered text_encoders folder: {text_encoders_directory}")
except Exception as e:
    print(f"⚠️  Error registering text_encoders folder: {e}")


def get_available_gguf_models():
    """Scan for available GGUF models in text_encoders directory and extra model paths"""
    models = []
    model_paths_map = {}  # Map model name to full path
    
    # Get all possible paths for text_encoders (includes extra_model_paths.yaml)
    try:
        # Try to get all configured paths for text_encoders
        all_text_encoder_paths = folder_paths.get_folder_paths("text_encoders")
        print(f"🔍 Scanning for GGUF models in text_encoders paths:")
        for path in all_text_encoder_paths:
            print(f"   - {path} (exists: {os.path.exists(path)})")
    except Exception as e:
        # Fallback if folder type not registered
        print(f"⚠️  Could not get folder_paths for text_encoders: {e}")
        all_text_encoder_paths = [text_encoders_directory]
        print(f"   Using fallback: {text_encoders_directory}")
    
    # Search in all text_encoders directories (including extra paths)
    for base_dir in all_text_encoder_paths:
        if os.path.exists(base_dir):
            print(f"📁 Searching in: {base_dir}")
            found_count = 0
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('.gguf') or file.endswith('.GGUF'):
                        # Get relative path from base directory
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, base_dir)
                        
                        # Use just filename if in root, otherwise use relative path
                        if os.path.dirname(rel_path) == "":
                            model_key = file
                        else:
                            model_key = rel_path
                        
                        # Store the mapping and add to list if not duplicate
                        if model_key not in model_paths_map:
                            model_paths_map[model_key] = full_path
                            models.append(model_key)
                            found_count += 1
                            print(f"   ✓ Found: {model_key}")
            
            if found_count == 0:
                print(f"   (no GGUF files found)")
        else:
            print(f"⚠️  Path does not exist: {base_dir}")
    
    # Also search VLM_GGUF for backward compatibility
    if os.path.exists(vlm_gguf_directory):
        print(f"📁 Searching legacy VLM_GGUF: {vlm_gguf_directory}")
        for root, dirs, files in os.walk(vlm_gguf_directory):
            for file in files:
                if file.endswith('.gguf') or file.endswith('.GGUF'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, vlm_gguf_directory)
                    model_key = f"VLM_GGUF/{rel_path}"
                    if model_key not in model_paths_map:
                        model_paths_map[model_key] = full_path
                        models.append(model_key)
                        print(f"   ✓ Found: {model_key}")
    
    if not models:
        print("❌ No GGUF models found in any configured path")
        models = ["No GGUF models found - place in models/text_encoders/"]
    else:
        print(f"✅ Total GGUF models found: {len(models)}")
    
    # Store the mapping globally for use in load_model
    get_available_gguf_models.model_paths = model_paths_map
    
    return sorted(models)


class LoadQwen2_5_VL_GGUF:
    """
    ComfyUI Node to load GGUF quantized Qwen2.5-VL models using llama.cpp
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (get_available_gguf_models(), {
                    "tooltip": "Select GGUF model from models/text_encoders/ directory"
                }),
                "mmproj_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional: Path to mmproj file. Leave empty for auto-detection."
                }),
                "n_ctx": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 32768,
                    "step": 512,
                    "tooltip": "Context window size"
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 200,
                    "tooltip": "Number of layers to offload to GPU (-1 for all)"
                }),
            },
        }

    RETURN_TYPES = ("QWEN2_5_VL_GGUF_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen2.5-VL-GGUF"

    def load_model(self, model, mmproj_path, n_ctx, n_gpu_layers):
        """Load and configure GGUF model (lazy loading)"""
        # Resolve model path - check the stored mapping first
        model_path = None
        
        # Try to get path from the mapping created during model scanning
        if hasattr(get_available_gguf_models, 'model_paths') and model in get_available_gguf_models.model_paths:
            model_path = get_available_gguf_models.model_paths[model]
        else:
            # Fallback to old behavior for compatibility
            if model.startswith("VLM_GGUF/"):
                # Legacy VLM_GGUF directory
                rel_path = model.replace("VLM_GGUF/", "")
                model_path = os.path.join(vlm_gguf_directory, rel_path)
            else:
                # Primary text_encoders directory
                model_path = os.path.join(text_encoders_directory, model)
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ GGUF model file not found: {model}\n"
                f"Searched path: {model_path if model_path else 'N/A'}\n"
                f"Please ensure the model is in one of the configured text_encoders directories"
            )
        
        # Store configuration for lazy loading
        model_config = {
            "model_path": model_path,
            "mmproj_path": mmproj_path if mmproj_path else None,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
        }
        
        print(f"✅ GGUF model configured: {model_path}")
        if mmproj_path:
            print(f"📁 Using specified mmproj: {mmproj_path}")
        else:
            print(f"🔍 mmproj will be auto-detected from model directory")
        
        return (model_config,)


class Qwen2_5_VL_GGUF_Describe_Image:
    """
    ComfyUI Node to describe images using GGUF quantized Qwen2.5-VL models
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN2_5_VL_GGUF_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "tooltip": "Text prompt for image description"
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Maximum number of tokens to generate"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature (higher = more random)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling probability"
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Top-k sampling"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for reproducibility"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "describe_image"
    CATEGORY = "Qwen2.5-VL-GGUF"

    def describe_image(self, model, image, prompt, max_tokens, temperature, top_p, top_k, seed):
        """Generate image description using GGUF model"""
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler

            # Lazy load model if not already loaded
            if "llm_instance" not in model:
                model_path = model["model_path"]
                
                # Get or auto-detect mmproj path
                if model.get("mmproj_path"):
                    mmproj_path = model["mmproj_path"]
                else:
                    mmproj_path = self._find_mmproj_path(model_path)
                
                n_ctx = model.get("n_ctx", 4096)
                n_gpu_layers = model.get("n_gpu_layers", -1)

                print(f"🚀 Loading GGUF model: {model_path}")
                print(f"📁 Using mmproj: {mmproj_path}")
                print(f"⚙️  Config: n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")

                chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
                llm = Llama(
                    model_path=model_path,
                    chat_handler=chat_handler,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                    seed=seed
                )
                model["llm_instance"] = llm
            else:
                llm = model["llm_instance"]

            # Save image to temporary file
            image_path = self._save_temp_image(image, seed)
            
            # Prepare message content for llama.cpp
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_path}"}
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            messages = [{"role": "user", "content": content}]

            print(f"📤 Generating description with prompt: {prompt[:50]}...")

            # Generate response
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream=False
            )

            output_text = response["choices"][0]["message"]["content"]
            
            # Clean up temporary file
            try:
                os.remove(image_path)
            except:
                pass
            
            print(f"✅ Generated description: {output_text[:100]}...")
            return (str(output_text),)

        except ImportError as e:
            error_msg = "❌ llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            print(error_msg)
            return (error_msg,)
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}"
            print(f"❌ Detailed error:\n{traceback.format_exc()}")
            return (error_msg,)

    def _save_temp_image(self, image, seed):
        """Save tensor image to temporary file"""
        unique_id = uuid.uuid4().hex
        image_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}_{unique_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensor to PIL Image
        img_array = image.cpu().numpy()
        if img_array.ndim == 4:
            img_array = img_array[0]  # Take first image if batch
        img_array = np.clip(255.0 * img_array, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        img.save(str(image_path))
        
        return str(image_path.resolve())

    def _find_mmproj_path(self, model_path):
        """Auto-detect mmproj file - enhanced search"""
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        
        # Remove extension and quantization for pattern matching
        base_name = model_name.replace(".gguf", "").replace(".GGUF", "")
        # Remove common quantization patterns (Q4_K_M, Q5_K_S, etc.)
        import re
        base_name_clean = re.sub(r'-Q\d+_[KM](_[MS])?$', '', base_name)
        
        print(f"🔍 Searching for mmproj file...")
        print(f"   Model: {model_name}")
        print(f"   Base name: {base_name_clean}")
        
        # Strategy 1: Check same directory as model
        possible_names = [
            f"{base_name}.mmproj.gguf",
            f"{base_name}.mmproj-f16.gguf",
            f"{base_name}.mmproj-F16.gguf",
            f"{base_name_clean}.mmproj.gguf",
            f"{base_name_clean}.mmproj-f16.gguf",
            f"{base_name_clean}.mmproj-F16.gguf",
            f"{base_name}-mmproj.gguf",
            f"{base_name_clean}-mmproj.gguf",
            "mmproj.gguf",
            "mmproj-f16.gguf",
            "mmproj-F16.gguf",
        ]
        
        for name in possible_names:
            mmproj_path = os.path.join(model_dir, name)
            if os.path.isfile(mmproj_path):
                print(f"✅ Found mmproj in same directory: {name}")
                return mmproj_path
            
            # Check for symlinks
            if os.path.islink(mmproj_path):
                real_path = os.path.realpath(mmproj_path)
                if os.path.exists(real_path):
                    print(f"✅ Found mmproj symlink: {name} -> {real_path}")
                    return real_path
        
        # Strategy 2: Search recursively in parent directory (text_encoders)
        # This handles cases where model and mmproj are at same level
        parent_dir = os.path.dirname(model_dir) if model_dir != text_encoders_directory else model_dir
        
        # If model is directly in text_encoders, search there
        if model_dir == text_encoders_directory or model_dir == vlm_gguf_directory:
            search_dir = model_dir
        else:
            search_dir = parent_dir
        
        print(f"🔍 Searching recursively in: {search_dir}")
        
        # Look for any mmproj file in the directory tree
        mmproj_candidates = []
        try:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if 'mmproj' in file.lower() and file.endswith('.gguf'):
                        full_path = os.path.join(root, file)
                        mmproj_candidates.append(full_path)
                        print(f"   Found candidate: {file}")
        except Exception as e:
            print(f"   Search error: {e}")
        
        # Strategy 3: Use the first mmproj found
        if mmproj_candidates:
            # Prefer mmproj files with similar base names
            for candidate in mmproj_candidates:
                candidate_name = os.path.basename(candidate).lower()
                if base_name_clean.lower() in candidate_name:
                    print(f"✅ Found matching mmproj: {candidate}")
                    return candidate
            
            # Otherwise use the first one found
            print(f"✅ Using first available mmproj: {mmproj_candidates[0]}")
            return mmproj_candidates[0]
        
        # Strategy 4: Show all available GGUF files for debugging
        print(f"❌ No mmproj file found!")
        try:
            all_gguf = []
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.gguf'):
                        rel_path = os.path.relpath(os.path.join(root, file), search_dir)
                        all_gguf.append(rel_path)
            
            if all_gguf:
                print(f"🔍 Available GGUF files in {search_dir}:")
                for f in sorted(all_gguf):
                    print(f"   - {f}")
        except Exception as e:
            print(f"   Could not list files: {e}")
        
        raise FileNotFoundError(
            f"Could not find mmproj file for {model_path}.\n"
            f"Searched in: {search_dir}\n"
            f"Please either:\n"
            f"1. Place mmproj file in the same directory as the model\n"
            f"2. Specify mmproj_path manually in the loader node\n"
            f"3. Ensure mmproj file contains 'mmproj' in its name and ends with .gguf"
        )


class Qwen2_5_VL_GGUF_Batch_Describe:
    """
    ComfyUI Node to describe multiple images in batch using GGUF models
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN2_5_VL_GGUF_MODEL",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "Describe this image.",
                    "multiline": True,
                }),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("descriptions",)
    FUNCTION = "describe_batch"
    CATEGORY = "Qwen2.5-VL-GGUF"

    def describe_batch(self, model, images, prompt, max_tokens, temperature, seed):
        """Process multiple images and return concatenated descriptions"""
        try:
            # Get number of images in batch
            num_images = images.shape[0]
            descriptions = []
            
            print(f"📦 Processing batch of {num_images} images...")
            
            # Create a single-image describe instance
            describer = Qwen2_5_VL_GGUF_Describe_Image()
            
            for i in range(num_images):
                # Extract single image from batch
                single_image = images[i:i+1]
                
                # Generate description
                description = describer.describe_image(
                    model=model,
                    image=single_image,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    seed=seed + i  # Vary seed for each image
                )[0]
                
                descriptions.append(f"Image {i+1}: {description}")
                print(f"✅ Processed image {i+1}/{num_images}")
            
            # Join all descriptions
            result = "\n\n".join(descriptions)
            return (result,)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Batch processing error: {str(e)}"
            print(f"❌ Detailed error:\n{traceback.format_exc()}")
            return (error_msg,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadQwen2_5_VL_GGUF": LoadQwen2_5_VL_GGUF,
    "Qwen2_5_VL_GGUF_Describe_Image": Qwen2_5_VL_GGUF_Describe_Image,
    "Qwen2_5_VL_GGUF_Batch_Describe": Qwen2_5_VL_GGUF_Batch_Describe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadQwen2_5_VL_GGUF": "Load Qwen2.5-VL GGUF Model",
    "Qwen2_5_VL_GGUF_Describe_Image": "Qwen2.5-VL GGUF Describe Image",
    "Qwen2_5_VL_GGUF_Batch_Describe": "Qwen2.5-VL GGUF Batch Describe",
}

