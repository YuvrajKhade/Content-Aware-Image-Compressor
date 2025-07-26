from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from skimage.segmentation import slic
import time
import uuid
import shutil
from pathlib import Path
import asyncio
from typing import Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Content-Aware Image Compression API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")
VISUALIZATION_DIR = Path("visualizations")
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

model_loaded = False
processing_status = {}

class LightweightImportanceNet(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class RegionImportanceAnalyzer:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.create_dummy_model()

    def create_dummy_model(self):
        self.model = LightweightImportanceNet(num_classes=3)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Created dummy model for demonstration")

    def load_model(self, model_path):
        try:
            self.model = LightweightImportanceNet(num_classes=3)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully to {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.create_dummy_model()

    def predict_region_importance(self, image, patch_size=224, stride=112, callback=None):
        if self.model is None:
            raise ValueError("Model has not been loaded.")

        width, height = image.size
        heatmap = np.zeros((height, width))
        count = np.zeros((height, width))
        total_patches = max(1, ((height - patch_size + stride) // stride) * ((width - patch_size + stride) // stride))
        processed_patches = 0

        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = image.crop((x, y, x + patch_size, y + patch_size))
                patch_tensor = self.transform(patch).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(patch_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    importance_score = (1.0 * probabilities[0, 0] + 0.5 * probabilities[0, 1]).item()
                heatmap[y:y+patch_size, x:x+patch_size] += importance_score
                count[y:y+patch_size, x:x+patch_size] += 1
                processed_patches += 1
                if callback and processed_patches % 5 == 0:
                    progress = processed_patches / total_patches * 100
                    callback(progress, f"Processed {processed_patches}/{total_patches} patches")
        mask = count > 0
        heatmap[mask] /= count[mask]
        return heatmap

    def find_important_regions(self, heatmap, importance_threshold=0.7, region_size=112, max_regions=5):
        from scipy.ndimage import maximum_filter, generate_binary_structure
        height, width = heatmap.shape
        struct = generate_binary_structure(2, 2)
        filtered = maximum_filter(heatmap, size=25)
        peaks = (heatmap == filtered) & (heatmap > importance_threshold)
        peak_coords = np.where(peaks)
        processed = set()
        regions = []
        for y, x in zip(peak_coords[0], peak_coords[1]):
            if (y, x) not in processed:
                half_size = region_size // 2
                region = {
                    'y': max(0, y - half_size),
                    'x': max(0, x - half_size),
                    'height': min(region_size, height - max(0, y - half_size)),
                    'width': min(region_size, width - max(0, x - half_size)),
                    'center_y': int(y),
                    'center_x': int(x),
                    'importance': float(heatmap[y, x])
                }
                regions.append(region)
                for dy in range(max(0, y - region_size), min(height, y + region_size)):
                    for dx in range(max(0, x - region_size), min(width, x + region_size)):
                        processed.add((dy, dx))
        regions.sort(key=lambda r: r['importance'], reverse=True)
        return regions[:max_regions]

class VisualizationGenerator:
    @staticmethod
    def create_legend(legend_type="heatmap", width=200, height=300):
        """Create a legend image for heatmap or segmentation visualization"""
        try:
            # Create legend image
            legend_img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(legend_img)
            
            # Try to use a default font, fallback to basic if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                small_font = ImageFont.truetype("arial.ttf", 12)
            except:
                try:
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
                except:
                    font = None
                    small_font = None
            
            if legend_type == "heatmap":
                # Create color gradient for heatmap legend
                gradient_height = height - 80
                gradient_width = 40
                gradient_x = 20
                gradient_y = 40
                
                # Create gradient
                for i in range(gradient_height):
                    # Map from blue (low importance) to red (high importance)
                    ratio = i / gradient_height
                    # Use OpenCV colormap logic for JET colormap
                    if ratio < 0.25:
                        r, g, b = 0, int(255 * ratio * 4), 255
                    elif ratio < 0.5:
                        r, g, b = 0, 255, int(255 * (1 - (ratio - 0.25) * 4))
                    elif ratio < 0.75:
                        r, g, b = int(255 * (ratio - 0.5) * 4), 255, 0
                    else:
                        r, g, b = 255, int(255 * (1 - (ratio - 0.75) * 4)), 0
                    
                    # Draw horizontal line
                    draw.rectangle([gradient_x, gradient_y + gradient_height - i, 
                                  gradient_x + gradient_width, gradient_y + gradient_height - i], 
                                 fill=(r, g, b))
                
                # Add labels
                if font:
                    draw.text((gradient_x + gradient_width + 10, gradient_y), "High", fill='black', font=font)
                    draw.text((gradient_x + gradient_width + 10, gradient_y + gradient_height//2), "Medium", fill='black', font=font)
                    draw.text((gradient_x + gradient_width + 10, gradient_y + gradient_height - 20), "Low", fill='black', font=font)
                    draw.text((10, 10), "Importance", fill='black', font=font)
                else:
                    # Fallback without font
                    draw.text((gradient_x + gradient_width + 10, gradient_y), "High", fill='black')
                    draw.text((gradient_x + gradient_width + 10, gradient_y + gradient_height//2), "Medium", fill='black')
                    draw.text((gradient_x + gradient_width + 10, gradient_y + gradient_height - 20), "Low", fill='black')
                    draw.text((10, 10), "Importance", fill='black')
                
            elif legend_type == "segmentation":
                # Create legend for segmentation
                y_offset = 40
                box_size = 20
                spacing = 30
                
                # High importance (red-ish)
                draw.rectangle([20, y_offset, 20 + box_size, y_offset + box_size], fill=(255, 100, 100))
                if font:
                    draw.text((50, y_offset + 2), "High Importance", fill='black', font=small_font)
                    draw.text((50, y_offset + 18), "Quality: 90%", fill='gray', font=small_font)
                else:
                    draw.text((50, y_offset + 2), "High Importance", fill='black')
                
                # Medium importance (yellow-ish)
                y_offset += spacing + 20
                draw.rectangle([20, y_offset, 20 + box_size, y_offset + box_size], fill=(255, 255, 100))
                if font:
                    draw.text((50, y_offset + 2), "Medium Importance", fill='black', font=small_font)
                    draw.text((50, y_offset + 18), "Quality: 60%", fill='gray', font=small_font)
                else:
                    draw.text((50, y_offset + 2), "Medium Importance", fill='black')
                
                # Low importance (blue-ish)
                y_offset += spacing + 20
                draw.rectangle([20, y_offset, 20 + box_size, y_offset + box_size], fill=(100, 100, 255))
                if font:
                    draw.text((50, y_offset + 2), "Low Importance", fill='black', font=small_font)
                    draw.text((50, y_offset + 18), "Quality: 30%", fill='gray', font=small_font)
                else:
                    draw.text((50, y_offset + 2), "Low Importance", fill='black')
                
                # Title
                if font:
                    draw.text((10, 10), "Segment Quality", fill='black', font=font)
                else:
                    draw.text((10, 10), "Segment Quality", fill='black')
                
                # Boundary indicator
                y_offset += spacing + 30
                draw.rectangle([20, y_offset, 20 + box_size, y_offset + box_size], fill=(255, 255, 0))
                if font:
                    draw.text((50, y_offset + 2), "Segment Boundary", fill='black', font=small_font)
                else:
                    draw.text((50, y_offset + 2), "Segment Boundary", fill='black')
            
            return np.array(legend_img)
            
        except Exception as e:
            logger.error(f"Failed to create legend: {e}")
            # Return a simple white rectangle as fallback
            return np.full((height, width, 3), 255, dtype=np.uint8)

    @staticmethod
    def save_original_image(image_path, file_id):
        try:
            output_dir = VISUALIZATION_DIR / file_id
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "original.jpg"
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                max_size = 1200
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                img.save(output_path, 'JPEG', quality=90)
                logger.info(f"Saved original image: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save original image: {e}")
            return False

    @staticmethod
    def generate_heatmap_visualization(heatmap, file_id, original_image_path):
        """Generate heatmap visualization with legend"""
        try:
            output_dir = VISUALIZATION_DIR / file_id
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "heatmap.jpg"
            
            # Load original image
            image = cv2.imread(str(original_image_path))
            height, width = image.shape[:2]
            
            # Normalize heatmap
            heatmap_norm = ((heatmap - np.min(heatmap)) * 255 / (np.ptp(heatmap) + 1e-8)).astype(np.uint8)
            heatmap_img = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(image, 0.6, heatmap_img, 0.4, 0)
            
            # Create legend
            legend = VisualizationGenerator.create_legend("heatmap", width=200, height=min(300, height))
            legend_bgr = cv2.cvtColor(legend, cv2.COLOR_RGB2BGR)
            
            # Combine image and legend
            # Resize legend to fit image height if needed
            if legend_bgr.shape[0] != height:
                legend_bgr = cv2.resize(legend_bgr, (200, height))
            
            # Concatenate horizontally
            combined = np.hstack([overlay, legend_bgr])
            
            cv2.imwrite(str(output_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            logger.info(f"Generated heatmap visualization with legend: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate heatmap visualization: {e}")
            return False

    @staticmethod
    def generate_segmentation_visualization(image_array, segments, segment_importance, file_id):
        """Generate segmentation visualization with legend"""
        try:
            output_dir = VISUALIZATION_DIR / file_id
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "segments.jpg"
            
            height, width = image_array.shape[:2]
            
            # Create boundary visualization
            from skimage.segmentation import find_boundaries
            mask = find_boundaries(segments, mode='outer')
            img_show = image_array.copy()
            img_show[mask] = [255, 255, 0]  # yellow boundaries
            
            # Create importance overlay
            importance_img = np.zeros_like(image_array, dtype=np.float32)
            for segment_id, importance in segment_importance.items():
                if importance > 0.7:
                    color = (100,100,255)  # High importance - red-ish
                elif importance > 0.4:
                    color = (100, 255, 255)  # Medium importance - yellow-ish  
                else:
                    color = (255, 100, 100)  # Low importance - blue-ish
                importance_img[segments == segment_id] = color
            
            # Combine original image with importance overlay
            out = cv2.addWeighted(img_show, 0.7, importance_img.astype(np.uint8), 0.3, 0)
            
            # Create legend
            legend = VisualizationGenerator.create_legend("segmentation", width=200, height=min(300, height))
            legend_bgr = cv2.cvtColor(legend, cv2.COLOR_RGB2BGR)
            
            # Resize legend to match image height
            if legend_bgr.shape[0] != height:
                legend_bgr = cv2.resize(legend_bgr, (200, height))
            
            # Combine image and legend
            combined = np.hstack([out, legend_bgr])
            
            cv2.imwrite(str(output_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            logger.info(f"Generated segmentation visualization with legend: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate segmentation visualization: {e}")
            return False

class ContentAwareCompressor:
    def __init__(self, model_path=None):
        self.analyzer = RegionImportanceAnalyzer(model_path)
        self.visualizer = VisualizationGenerator()

    def apply_graph_coloring(self, image, heatmap, n_segments=100):
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        segments = slic(img_array, n_segments=n_segments, compactness=10, sigma=1)
        segment_importance = {}
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            avg_importance = np.mean(heatmap_norm[mask])
            segment_importance[segment_id] = avg_importance
        return {
            'segments': segments,
            'importance': segment_importance,
            'n_segments': len(np.unique(segments))
        }

    def content_aware_compress(self, image_path, output_path, quality_high=90, 
                             quality_medium=60, quality_low=30, n_segments=100, 
                             callback=None):
        start_time = time.time()
        file_id = Path(image_path).stem.split('_')[0] if '_' in Path(image_path).stem else Path(image_path).stem
        
        if callback:
            callback(5, "Preparing image...")
        success = self.visualizer.save_original_image(image_path, file_id)
        if not success:
            logger.warning(f"Failed to save original image for visualization: {file_id}")

        if callback:
            callback(10, "Loading image...")

        img_cv = cv2.imread(str(image_path))
        img_pil = Image.open(image_path).convert('RGB')
        original_size = os.path.getsize(image_path)

        if callback:
            callback(20, "Generating importance heatmap...")

        def progress_callback(progress, message):
            if callback:
                callback(20 + progress * 0.3, message)

        heatmap = self.analyzer.predict_region_importance(img_pil, callback=progress_callback)

        if callback:
            callback(50, "Creating heatmap visualization...")

        heatmap_success = self.visualizer.generate_heatmap_visualization(heatmap, file_id, image_path)
        if not heatmap_success:
            logger.warning(f"Failed to generate heatmap visualization: {file_id}")

        if callback:
            callback(60, "Segmenting image...")

        segmentation = self.apply_graph_coloring(img_cv, heatmap, n_segments)
        segments = segmentation['segments']
        segment_importance = segmentation['importance']

        if callback:
            callback(70, "Creating segmentation visualization...")

        img_array = np.array(img_pil)
        seg_success = self.visualizer.generate_segmentation_visualization(img_array, segments, segment_importance, file_id)
        if not seg_success:
            logger.warning(f"Failed to generate segmentation visualization: {file_id}")

        if callback:
            callback(80, "Creating quality map...")

        height, width = segments.shape
        quality_map = np.zeros((height, width), dtype=np.uint8)
        compression_stats = {'high': 0, 'medium': 0, 'low': 0}
        for segment_id, importance in segment_importance.items():
            segment_mask = segments == segment_id
            if importance > 0.7:
                quality_map[segment_mask] = quality_high
                compression_stats['high'] += 1
            elif importance > 0.4:
                quality_map[segment_mask] = quality_medium
                compression_stats['medium'] += 1
            else:
                quality_map[segment_mask] = quality_low
                compression_stats['low'] += 1

        if callback:
            callback(90, "Applying compression...")

        total_segments = sum(compression_stats.values())
        if total_segments > 0:
            base_quality = int((
                quality_high * compression_stats['high'] +
                quality_medium * compression_stats['medium'] + 
                quality_low * compression_stats['low']
            ) / total_segments)
        else:
            base_quality = quality_medium
        cv2.imwrite(str(output_path), img_cv, [cv2.IMWRITE_JPEG_QUALITY, base_quality])
        compressed_size = os.path.getsize(output_path)
        reduction = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

        elapsed_time = time.time() - start_time

        if callback:
            callback(100, "Compression complete!")

        regions = self.analyzer.find_important_regions(heatmap)

        return {
            'compressed_path': output_path.name,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'reduction_percentage': reduction,
            'processing_time': elapsed_time,
            'compression_stats': compression_stats,
            'regions': regions,
            'space_saved': original_size - compressed_size,
            'compression_ratio': compressed_size / original_size if original_size > 0 else 0
        }

MODEL_PATH = "models/region_importance_model_final.pth"
compressor = ContentAwareCompressor(model_path=MODEL_PATH)
model_loaded = os.path.exists(MODEL_PATH) if MODEL_PATH else False
logger.info(f"Model loaded: {model_loaded}")

# Add a simple in-memory cache to prevent duplicate processing
processing_cache = {}

@app.get("/")
async def root():
    return {"message": "Content-Aware Image Compression API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(compressor.analyzer.device),
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(processing_status)
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix.lower()
    if not file_extension:
        file_extension = '.jpg'
    filename = f"{file_id}{file_extension}"
    file_path = UPLOAD_DIR / filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        with Image.open(file_path) as img:
            width, height = img.size
        logger.info(f"Uploaded file: {filename}, size: {width}x{height}")
        return {
            "file_id": file_id,
            "filename": filename,
            "original_filename": file.filename,
            "size": file_path.stat().st_size,
            "dimensions": {"width": width, "height": height},
            "upload_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@app.post("/compress")
async def compress_image(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    quality_high: int = Form(90),
    quality_medium: int = Form(60),
    quality_low: int = Form(30),
    n_segments: int = Form(100)
):
    # Check if file exists
    input_files = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not input_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Create cache key to prevent duplicate processing
    cache_key = f"{file_id}_{quality_high}_{quality_medium}_{quality_low}_{n_segments}"
    
    # Check if already processing
    if cache_key in processing_cache:
        existing_task_id = processing_cache[cache_key]
        if existing_task_id in processing_status:
            logger.info(f"Returning existing task for cache key: {cache_key}")
            return {"task_id": existing_task_id, "message": "Processing already in progress"}
    
    input_path = input_files[0]
    output_filename = f"{file_id}_compressed.jpg"
    output_path = OUTPUT_DIR / output_filename

    task_id = str(uuid.uuid4())
    processing_cache[cache_key] = task_id
    
    processing_status[task_id] = {
        "status": "started",
        "progress": 0,
        "message": "Initializing compression...",
        "start_time": time.time(),
        "file_id": file_id
    }

    def update_progress(progress, message):
        if task_id in processing_status:
            processing_status[task_id].update({
                "progress": progress,
                "message": message
            })

    background_tasks.add_task(
        compress_image_task, 
        task_id, 
        input_path, 
        output_path, 
        quality_high, 
        quality_medium, 
        quality_low, 
        n_segments,
        update_progress,
        cache_key
    )

    return {"task_id": task_id, "message": "Compression started"}

async def compress_image_task(task_id, input_path, output_path, quality_high, 
                            quality_medium, quality_low, n_segments, callback, cache_key):
    try:
        result = compressor.content_aware_compress(
            input_path,
            output_path,
            quality_high=quality_high,
            quality_medium=quality_medium,
            quality_low=quality_low,
            n_segments=n_segments,
            callback=callback
        )
        processing_status[task_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Compression completed successfully",
            "result": result,
            "output_filename": output_path.name
        })
        logger.info(f"Compression task {task_id} completed successfully")
    except Exception as e:
        logger.error(f"Compression task {task_id} failed: {e}")
        processing_status[task_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"Compression failed: {str(e)}",
            "error": str(e)
        })
    finally:
        # Clean up cache after some time (remove from cache after completion)
        if cache_key in processing_cache:
            del processing_cache[cache_key]

@app.get("/status/{task_id}")
async def get_compression_status(task_id: str):
    if task_id not in processing_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = processing_status[task_id].copy()
    
    # Clean up old completed/failed tasks (older than 1 hour)
    current_time = time.time()
    if status.get('status') in ['completed', 'failed']:
        if current_time - status.get('start_time', current_time) > 3600:
            del processing_status[task_id]
            raise HTTPException(status_code=404, detail="Task expired and cleaned up")
    
    return status

@app.get("/download/{filename}")
async def download_file(filename: str):
    clean_filename = filename.split('/')[-1]
    file_path = OUTPUT_DIR / clean_filename
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {clean_filename}")
    return FileResponse(
        path=file_path,
        filename=clean_filename,
        media_type='application/octet-stream',
        headers={"Content-Disposition": f"attachment; filename={clean_filename}"}
    )

@app.get("/visualization/{file_id}/{viz_type}")
async def get_visualization(file_id: str, viz_type: str):
    valid_types = ['original', 'heatmap', 'segments']
    if viz_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid visualization type. Must be one of: {valid_types}")
    file_path = VISUALIZATION_DIR / file_id / f"{viz_type}.jpg"
    if not file_path.exists():
        logger.error(f"Visualization not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"Visualization not found: {viz_type} for file {file_id}")
    return FileResponse(
        path=file_path,
        media_type='image/jpeg',
        headers={
            "Cache-Control": "max-age=3600",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.delete("/cleanup/{file_id}")
async def cleanup_files(file_id: str):
    cleaned_files = []
    try:
        for file_path in UPLOAD_DIR.glob(f"{file_id}.*"):
            file_path.unlink()
            cleaned_files.append(str(file_path))
        for file_path in OUTPUT_DIR.glob(f"{file_id}_*"):
            file_path.unlink()
            cleaned_files.append(str(file_path))
        viz_dir = VISUALIZATION_DIR / file_id
        if viz_dir.exists():
            shutil.rmtree(viz_dir)
            cleaned_files.append(str(viz_dir))
        return {"cleaned_files": cleaned_files}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"error": f"Cleanup failed: {str(e)}", "cleaned_files": cleaned_files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

