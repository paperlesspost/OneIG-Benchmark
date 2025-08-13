# Style Embedding Converter - Usage Guide

This Modal script generates style embeddings for new art styles to be used with the OneIG-Benchmark evaluation framework.

## 🚀 Quick Start

### 1. Prerequisites

- Install Modal: `pip install modal`
- Set up Modal account: `modal setup`
- Have your style reference images ready

### 2. Prepare Your Style Images

Upload your reference images to the Modal volume in this structure:
```
/vol/
├── style-imgs/
│   ├── art_deco/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── image3.jpg
│   ├── cyberpunk/
│   │   ├── cyber1.jpg
│   │   └── cyber2.png
│   └── impressionism/
│       ├── monet1.jpg
│       └── renoir1.jpg
└── embeddings/
    ├── CSD_embed.pt
    └── SE_embed.pt
```

To upload images to the Modal volume:
```bash
# Create and mount the volume locally
modal volume create style-embeddings
modal volume put style-embeddings local/path/to/your/art_deco_images /vol/style-imgs/art_deco
```

### 3. Run the Converter

```bash
# Basic usage
modal run style_embedding_converter.py --style-name "art_deco"

# With parameters
modal run style_embedding_converter.py --style-name "cyberpunk" --update-existing true
```

## 📋 What the Script Does

1. **🔧 Model Setup** (30s-60s):
   - Downloads CLIP ViT-L/14 model (~1.7GB)
   - Downloads CSD checkpoint (~500MB) 
   - Loads both CSD and SE encoders on H200 GPU

2. **📸 Image Processing** (1-3 min):
   - Finds all images in `/style-imgs/{style_name}/`
   - Processes each image through both encoders
   - Generates embeddings for style similarity comparison

3. **💾 Saving Results**:
   - Updates or creates `CSD_embed.pt` and `SE_embed.pt` files
   - Saves to Modal persistent volume `style-embeddings`
   - Commits changes for persistence

## 🔧 Configuration Details

### GPU & Timeout
- **GPU**: H200 (as requested)
- **Timeout**: 30 minutes (1800 seconds)
- **Memory**: Automatically allocated for H200

### File Structure
```
Modal Volume: style-embeddings
└── /vol/
    ├── style-imgs/        # Input: Your reference images
    │   └── {style_name}/
    └── embeddings/        # Output: Generated embedding files
        ├── CSD_embed.pt   # Content-Style Disentanglement embeddings
        └── SE_embed.pt    # Style Encoder embeddings
```

## 📊 Output Files

The script generates two embedding files compatible with OneIG-Benchmark:

### `CSD_embed.pt`
```python
{
    'baroque': tensor([[0.1, 0.2, ...], [0.3, 0.4, ...], ...]),     # N x D tensor
    'cubism': tensor([[0.5, 0.6, ...], [0.7, 0.8, ...], ...]),     
    'art_deco': tensor([[0.9, 1.0, ...], [1.1, 1.2, ...], ...]),   # Your new style
    # ... other styles
}
```

### `SE_embed.pt`
```python
{
    'baroque': tensor([[0.1, 0.2, ...], [0.3, 0.4, ...], ...]),     # N x D tensor  
    'cubism': tensor([[0.5, 0.6, ...], [0.7, 0.8, ...], ...]),
    'art_deco': tensor([[0.9, 1.0, ...], [1.1, 1.2, ...], ...]),   # Your new style
    # ... other styles
}
```

## 📥 Downloading Results

```bash
# Download the generated embedding files
modal volume get style-embeddings /vol/embeddings/CSD_embed.pt ./CSD_embed.pt
modal volume get style-embeddings /vol/embeddings/SE_embed.pt ./SE_embed.pt
```

## 🔍 Integration with OneIG-Benchmark

1. **Copy embedding files** to your OneIG-Benchmark directory:
   ```bash
   cp CSD_embed.pt benchmarks/OneIG-Benchmark/scripts/style/
   cp SE_embed.pt benchmarks/OneIG-Benchmark/scripts/style/
   ```

2. **Update the style list** in `scripts/style/style_score.py`:
   ```python
   style_list = ['abstract_expressionism', 'art_nouveau', 'baroque', 
                 # ... existing styles ...
                 'art_deco']  # Add your new style
   ```

3. **Add test prompts** to your dataset with the new style:
   ```csv
   Anime_Stylization,XXX,"A futuristic cityscape with geometric patterns and metallic elements, rendered in art deco style.",NP,middle,art_deco
   ```

## ⚠️ Important Notes

### Image Requirements
- **Format**: JPG, PNG, BMP, TIFF, WEBP
- **Quality**: High-resolution, clear examples of the style
- **Quantity**: 15-50 images recommended for robust embeddings
- **Diversity**: Varied subjects but consistent style

### Model Limitations
- **CSD Model**: Uses simplified version (full model requires specific checkpoint)
- **SE Model**: Uses standard CLIP as placeholder (fine-tuned model preferred)
- **Performance**: Results may vary based on reference image quality

### Cost Estimation
- **H200 GPU**: ~$10/hour on Modal
- **Typical Runtime**: 3-5 minutes for 20-30 images
- **Total Cost**: ~$1.50-$2.50 per style

## 🐛 Troubleshooting

### Common Issues

1. **"Style directory does not exist"**
   ```bash
   # Check volume contents
   modal volume ls style-embeddings
   # Upload images if missing
   modal volume put style-embeddings local/images /vol/style-imgs/your_style
   ```

2. **"No embeddings generated"**
   - Check image file formats (must be valid image files)
   - Verify images aren't corrupted
   - Check file permissions

3. **CUDA out of memory**
   - Reduce batch size (script processes one image at a time)
   - Use smaller images (script auto-resizes to 224x224)

4. **Model download fails**
   - Check internet connectivity in Modal container
   - Verify Google Drive link for CSD model is accessible

### Getting Help

- Check Modal logs: `modal logs style-embedding-converter`
- Monitor GPU usage: Modal dashboard shows real-time metrics
- Debug mode: Add `print()` statements and redeploy

## 🔄 Advanced Usage

### Batch Processing Multiple Styles
```bash
# Process multiple styles
for style in art_deco cyberpunk steampunk; do
    modal run style_embedding_converter.py --style-name "$style"
done
```

### Custom Model Paths
Edit the script to use your own trained models by updating the model loading paths in the `setup_models()` method.

---

**📧 Questions?** Check the OneIG-Benchmark documentation or Modal docs for more details! 