# ComfyUI Qwen2.5-VL GGUF Nodes

ComfyUI custom nodes for running **GGUF quantized Qwen2.5-VL** models using llama.cpp. This allows efficient image description and vision-language tasks with reduced memory usage.

## Features

- 🚀 **GGUF Model Support**: Run quantized Qwen2.5-VL models efficiently
- 💾 **Low Memory Usage**: Use quantized models (Q4, Q5, Q6, Q8, etc.)
- 🖼️ **Image Description**: Generate detailed descriptions of images
- 📦 **Batch Processing**: Process multiple images at once
- ⚙️ **Configurable**: Control context size, GPU layers, temperature, etc.
- 🔄 **Auto-detection**: Automatically finds mmproj files

## Installation

### 1. Install llama-cpp-python

**For GPU support (CUDA):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python
```

**For CPU only:**
```bash
uv pip install llama-cpp-python
```

**For Metal (macOS):**
```bash
CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python
```

### 2. Install Dependencies

```bash
cd /home/qhu/Workspace/ComfyUI-qwenVL
uv pip install -r requirements.txt
```

## Getting GGUF Models

### Option 1: Download Pre-quantized Models

Search for Qwen2.5-VL GGUF models on Hugging Face. Example:
- [bartowski/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-VL-7B-Instruct-GGUF)

Download both:
- The GGUF model file (e.g., `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`)
- The mmproj file (e.g., `Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf`)

### Option 2: Quantize Your Own

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release

# Convert and quantize
python convert_hf_to_gguf.py /path/to/Qwen2.5-VL-7B-Instruct
./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

## Usage

### Node 1: Load Qwen2.5-VL GGUF Model

Loads the GGUF model configuration. Automatically scans `models/text_encoders/` directory.

**Parameters:**
- `model`: Dropdown selection of available GGUF models (scanned from `models/text_encoders/`)
- `mmproj_path`: (Optional) Path to mmproj file. Leave empty for auto-detection
- `n_ctx`: Context window size (default: 4096)
- `n_gpu_layers`: GPU layers to offload (-1 = all, 0 = CPU only)

**Output:**
- `model`: GGUF model configuration object

### Node 2: Qwen2.5-VL GGUF Describe Image

Generates a description for a single image.

**Parameters:**
- `model`: Model from loader node
- `image`: Input image (IMAGE type from ComfyUI)
- `prompt`: Text prompt for description (default: "Describe this image in detail.")
- `max_tokens`: Maximum tokens to generate (default: 512)
- `temperature`: Sampling temperature (0.0-2.0, default: 0.7)
- `top_p`: Nucleus sampling (0.0-1.0, default: 0.9)
- `top_k`: Top-k sampling (default: 50)
- `seed`: Random seed for reproducibility

**Output:**
- `description`: Generated text description

### Node 3: Qwen2.5-VL GGUF Batch Describe

Process multiple images and return all descriptions.

**Parameters:**
- `model`: Model from loader node
- `images`: Batch of images (IMAGE type)
- `prompt`: Text prompt for all images
- `max_tokens`: Maximum tokens per image
- `temperature`: Sampling temperature
- `seed`: Base random seed

**Output:**
- `descriptions`: Concatenated descriptions for all images

## Example Workflow

```
[Load Image] → [Qwen2.5-VL GGUF Describe Image] → [Save Text]
                        ↑
[Load Qwen2.5-VL GGUF Model]
```

1. **Load Qwen2.5-VL GGUF Model**
   - model_path: `/path/to/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
   - mmproj_path: (leave empty for auto-detection)
   - n_ctx: 4096
   - n_gpu_layers: -1

2. **Load Image**
   - Load your image

3. **Qwen2.5-VL GGUF Describe Image**
   - Connect model and image
   - prompt: "Describe this image in detail."
   - max_tokens: 512

4. **Save or Display**
   - Use the description output

## Quantization Formats

Common GGUF quantization formats (ordered by size/quality):

| Format | Size | Quality | Use Case |
|--------|------|---------|----------|
| Q2_K | Smallest | Lower | Extreme compression |
| Q3_K_M | Small | Good | Mobile/edge devices |
| Q4_K_M | Medium | Very Good | **Recommended balance** |
| Q5_K_M | Larger | Excellent | High quality |
| Q6_K | Large | Near-original | Maximum quality |
| Q8_0 | Very Large | ~Original | Minimal loss |

## Troubleshooting

### Error: mmproj file not found

Either:
1. Specify `mmproj_path` manually in the loader node
2. Ensure the mmproj file is in the same directory as the model with a matching name pattern

### Error: llama-cpp-python not installed

Install with GPU support:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python
```

### Out of Memory Error

- Reduce `n_ctx` (context size)
- Use a more aggressive quantization (Q4 instead of Q6)
- Reduce `n_gpu_layers` to offload fewer layers to GPU

### Slow Performance

- Increase `n_gpu_layers` to offload more to GPU
- Use a smaller quantization (but may reduce quality)
- Ensure CUDA/GPU support is properly compiled

## Model Storage

**Primary directory:** `ComfyUI/models/text_encoders/`

Place your GGUF models here for automatic discovery:
```
ComfyUI/models/text_encoders/
├── qwen2.5-vl-7b-q4/
│   ├── model.gguf
│   └── mmproj-f16.gguf
└── qwen2.5-vl-3b-q5/
    ├── model.gguf
    └── mmproj-f16.gguf
```

**Legacy support:** Models in `ComfyUI/models/VLM_GGUF/` are also detected for backward compatibility.

## Resources

- [Qwen2.5-VL GitHub](https://github.com/QwenLM/Qwen2.5-VL)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## License

MIT License - See LICENSE file for details

## Credits

- Qwen2.5-VL by Alibaba Cloud
- llama.cpp by Georgi Gerganov
- ComfyUI by comfyanonymous

