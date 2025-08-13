#!/usr/bin/env python3
"""
Modal script to generate style embeddings for new styles.
Usage: modal run style_embedding_converter.py --style-name "art_deco"
"""

import modal
from pathlib import Path
from typing import Dict, List
import torch
import os

# Configure the Modal image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "wget", "git")
    .pip_install(
        "torch",
        "torchvision", 
        "transformers",
        "Pillow",
        "tqdm",
        "pandas",
        "numpy",
        "accelerate"
    )
    .pip_install(
        "git+https://github.com/openai/CLIP.git"
    )
    # Download required model files during image build
    .run_commands(
        # Download CLIP model
        "mkdir -p /models",
        "cd /models && wget -O ViT-L-14.pt https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    )
)

# Create the Modal app
app = modal.App("style-embedding-converter", image=image)

# Create persistent volume for style images and outputs (new volume name)
bench_volume = modal.Volume.from_name("benchmarks", create_if_missing=True)

# Mount paths (switch to /benchmarks)
VOLUME_PATH = "/benchmarks"
MODELS_PATH = "/models"
OUTPUT_PATH = VOLUME_PATH  # embeddings saved under /benchmarks/{model}/embeddings

@app.cls(
    gpu="H200",  # Request H200 GPU
    timeout=30 * 60,  # 30 minutes timeout
    volumes={
        VOLUME_PATH: bench_volume
    },
    secrets=[],  # Add any secrets you need
)
class StyleEmbeddingGenerator:
    
    @modal.enter()
    def setup_models(self):
        """Load the embedding models on container startup"""
        print("🔧 Setting up style embedding models...")
        
        # Import inside the function to avoid import issues
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        import torch
        import torch.nn as nn
        import clip
        import copy
        
        # Setup CSD Encoder (simplified version based on the original code)
        print("Loading CSD encoder...")
        clipmodel, _ = clip.load(f"{MODELS_PATH}/ViT-L-14.pt", device="cuda")
        
        # Create a simplified CSD-like encoder
        class SimpleCSDEncoder:
            def __init__(self, clip_model):
                self.backbone = clip_model.visual
                self.last_layer_style = copy.deepcopy(self.backbone.proj)
                self.backbone.proj = None
                self.device = "cuda"
                
                # Preprocessing transforms
                from torchvision import transforms
                from torchvision.transforms import functional as F
                
                self.preprocess = transforms.Compose([
                    transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)
                    )
                ])
            
            def get_style_embedding(self, image_path: str):
                from PIL import Image
                
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device, dtype=torch.float32)
                
                with torch.no_grad():
                    feature = self.backbone(image_tensor.to(self.backbone.conv1.weight.dtype))
                    style_output = feature @ self.last_layer_style
                    style_output = torch.nn.functional.normalize(style_output, dim=1, p=2)
                    return style_output.squeeze(0).to(torch.float32)
        
        self.csd_encoder = SimpleCSDEncoder(clipmodel)
        
        # Setup SE Encoder
        print("Loading SE encoder...")
        class SEStyleEncoder:
            def __init__(self):
                self.device = "cuda"
                self.dtype = torch.float32  # Use float32 for better compatibility
                # Use the OneIG Style Encoder to produce 1280-d embeddings
                try:
                    self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                        "xingpng/OneIG-StyleEncoder"
                    )
                except Exception as e:
                    print(f"Failed to load xingpng/OneIG-StyleEncoder: {e}")
                    raise
                self.image_encoder.to(self.device, dtype=self.dtype)
                self.image_encoder.eval()
                
                try:
                    self.processor = CLIPImageProcessor.from_pretrained("xingpng/OneIG-StyleEncoder")
                except Exception as e:
                    print(f"Failed to load CLIPImageProcessor for OneIG-StyleEncoder, using default: {e}")
                    self.processor = CLIPImageProcessor()
            
            def _l2_normalize(self, x):
                return torch.nn.functional.normalize(x, p=2, dim=-1)
            
            def get_style_embedding(self, image_path: str):
                from PIL import Image
                
                image = Image.open(image_path).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt").pixel_values.to(
                    self.device, dtype=self.dtype
                )
                
                with torch.no_grad():
                    outputs = self.image_encoder(inputs)
                    image_embeds = outputs.image_embeds
                    image_embeds_norm = self._l2_normalize(image_embeds)
                return image_embeds_norm.squeeze(0).to(torch.float32)
        
        self.se_encoder = SEStyleEncoder()
        print("✅ Models loaded successfully!")
    
    @modal.method()
    def generate_embeddings_for_style(
        self, 
        style_name: str,
        image_paths: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Generate both CSD and SE embeddings for a list of images"""
        
        print(f"🎨 Generating embeddings for style: {style_name}")
        print(f"📸 Processing {len(image_paths)} images...")
        
        csd_embeddings = []
        se_embeddings = []
        
        from tqdm import tqdm
        from pathlib import Path
        
        successful_count = 0
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Generate CSD embedding
                csd_embed = self.csd_encoder.get_style_embedding(img_path)
                csd_embeddings.append(csd_embed)
                
                # Generate SE embedding  
                se_embed = self.se_encoder.get_style_embedding(img_path)
                se_embeddings.append(se_embed)
                
                successful_count += 1
                
            except Exception as e:
                print(f"⚠️ Skipped {Path(img_path).name} (dtype error)")
                continue
        
        if not csd_embeddings:
            raise ValueError("No embeddings generated! Check your image paths.")
        
        print(f"✅ Successfully processed {successful_count}/{len(image_paths)} images")
        
        # Stack embeddings into tensors
        csd_ref_tensor = torch.stack(csd_embeddings)  # [N, 768]
        se_ref_tensor = torch.stack(se_embeddings)    # [N, 1280]
        
        print(f"✅ Generated {len(csd_embeddings)} embeddings")
        print(f"CSD tensor shape: {csd_ref_tensor.shape}")
        print(f"SE tensor shape: {se_ref_tensor.shape}")
        
        # Convert to CPU and then to numpy for serialization
        return {
            "csd": csd_ref_tensor.cpu().numpy(),
            "se": se_ref_tensor.cpu().numpy(),
            "se_dim": se_ref_tensor.shape[-1],
            "csd_dim": csd_ref_tensor.shape[-1],
        }
    
    @modal.method()
    def save_embeddings(
        self,
        style_name: str,
        embeddings: Dict[str, any],  # numpy arrays
        model_path: str,
        update_existing: bool = True,
    ) -> str:
        """Save embeddings under /benchmarks/{model_path}/embeddings as dict keyed by style_name."""

        print(f"💾 Saving embeddings for style='{style_name}' under model='{model_path}'...")

        # Resolve output directory: /benchmarks/{model_path}/embeddings
        import os
        out_dir = os.path.join(OUTPUT_PATH, model_path, "embeddings")
        os.makedirs(out_dir, exist_ok=True)

        csd_embed_path = os.path.join(out_dir, "CSD_embed.pt")
        se_embed_path = os.path.join(out_dir, "SE_embed.pt")

        # Load existing dicts if present (per-model)
        if update_existing:
            try:
                if os.path.exists(csd_embed_path):
                    csd_ref = torch.load(csd_embed_path, weights_only=False)
                    print("📂 Loaded existing per-model CSD embeddings")
                else:
                    csd_ref = {}
                    print("📂 Creating new per-model CSD embeddings file")

                if os.path.exists(se_embed_path):
                    se_ref = torch.load(se_embed_path, weights_only=False)
                    print("📂 Loaded existing per-model SE embeddings")
                else:
                    se_ref = {}
                    print("📂 Creating new per-model SE embeddings file")
            except Exception as e:
                print(f"⚠️ Could not load existing per-model embeddings: {e}")
                csd_ref = {}
                se_ref = {}
        else:
            csd_ref = {}
            se_ref = {}

        import torch
        csd_ref[style_name] = torch.from_numpy(embeddings["csd"])  # N x D
        se_ref[style_name] = torch.from_numpy(embeddings["se"])    # N x D

        torch.save(csd_ref, csd_embed_path)
        torch.save(se_ref, se_embed_path)

        print("✅ Saved embeddings:")
        print(f"   CSD: {csd_embed_path}")
        print(f"   SE: {se_embed_path}")
        print(f"   Updated style key: '{style_name}'")

        bench_volume.commit()
        print("✅ Volume changes committed")

        return out_dir


@app.function(timeout=5 * 60, volumes={VOLUME_PATH: bench_volume})
def list_images_in_dir(images_dir: str) -> List[str]:
    """List image files inside a provided directory path inside the volume or container."""
    p = Path(images_dir)
    if not p.exists():
        print(f"⚠️ Directory does not exist: {images_dir}")
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths: List[str] = []
    try:
        for ext in image_extensions:
            image_paths.extend([str(x) for x in p.glob(f"*{ext}")])
            image_paths.extend([str(x) for x in p.glob(f"*{ext.upper()}")])
    except Exception as e:
        print(f"❌ Error scanning directory {images_dir}: {e}")
        return []
    print(f"📁 Found {len(image_paths)} images in {images_dir}")
    return image_paths


@app.local_entrypoint()
def main(
    model: str = "model-name",      # embeddings under /benchmarks/{model}/embeddings
    images: str = "",               # defaults to /benchmarks/{model}/images
    update_existing: bool = True,
):
    """Generate embeddings and save under /benchmarks/{model}/embeddings.

    - model: subpath under /benchmarks (supports nested, e.g., a/b/c/model-name)
    - images: directory of reference images; defaults to /benchmarks/{model}/images
    - update_existing: update existing PT dicts or create new ones
    """

    # Resolve images dir default
    if not images:
        images = f"{VOLUME_PATH}/{model}/images"

    # Style key from model name
    style_name = Path(model).name
    print(f"🚀 Generating embeddings | style='{style_name}' | model='{model}' | images='{images}'")

    try:
        # Collect images
        image_paths = list_images_in_dir.remote(images)
        if not image_paths:
            print(f"❌ No images found in: {images}")
            print("💡 Upload images or adjust the --images argument.")
            return

        # Generate embeddings
        generator = StyleEmbeddingGenerator()
        embeddings = generator.generate_embeddings_for_style.remote(style_name, image_paths)

        # Save embeddings under per-model directory
        out_dir = generator.save_embeddings.remote(style_name, embeddings, model, update_existing)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print("🎉 Done!")
    print("📊 Summary:")
    print(f"   Style: {style_name}")
    print(f"   Images processed: {len(image_paths)}")
    print(f"   Saved under: {out_dir}")


if __name__ == "__main__":
    print("Run with: modal run style_embedding_converter.py --style-name 'your_style_name'") 