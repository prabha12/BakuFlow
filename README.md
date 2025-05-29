# BakuFlow - Professional Annotation Made Simple

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-Custom-red.svg)](LICENSE)

**BakuFlow** is a modern, efficient, and user-friendly image annotation tool for object detection datasets. Developed by BakuAI AS (2024), it combines traditional manual labeling with AI-powered automatic annotation to make professional annotation simple and efficient.

---

## 🚀 Key Features

### 🤖 AI-Powered Automation
- **Visual Prompt Auto-Labeling**: Use annotated examples as visual prompts to automatically label similar objects
- **YOLO-E Integration**: Leverages [YOLOE models](https://github.com/THU-MIG/yoloe) for intelligent object detection
- **Batch Auto-Labeling**: Process multiple images automatically with consistent quality
- **Smart Learning**: Improves accuracy with quality prompt examples

### 📦 Multi-Format Support
- **YOLO Format**: `.txt` per image, normalized coordinates (primary format)
- **Pascal VOC**: `.xml` per image, Pascal VOC format (basic export)
- **COCO Format**: Single `coco_annotations.json` per folder (basic export)
- **Format Switching**: Change output format anytime during annotation

### 🔍 Precision Tools
- **Advanced Bounding Boxes**: Draw, move, resize, and delete with mouse and keyboard
- **Multi-Selection**: Hold Ctrl+Click to select multiple boxes for batch operations
- **Built-in Magnifier**: Pixel-accurate magnifier for precise annotation (2x-4x zoom)
- **Undo/Redo System**: Per-image history for all labeling actions (Ctrl+Z/Ctrl+Y)

### 🎨 Professional Workflow
- **Multi-Class Management**: Easily manage and switch between multiple classes
- **Color-Coded Legend**: Each class assigned unique color with live-updating legend
- **Label Propagation**: Copy selected or all bounding boxes to similar images
- **Auto-Save**: Automatically saves annotations after each edit
- **Custom Classes**: Each image folder can have its own `classes.txt`

### 🌐 User Experience
- **Keyboard Shortcuts**: Fast navigation and editing with comprehensive hotkeys
- **Multi-Language Support**: 8 languages supported with auto-detection
- **Flexible UI**: Resizable panels optimized for productivity
- **Data Augmentation**: Batch processing with rotation, brightness, contrast adjustments

---

## 🏗️ Technical Architecture

```
BakuFlow/
├── labelimg/                 # Core application module
│   ├── gui/                 # User interface components
│   ├── core/                # Core functionality
│   ├── inference/           # AI inference engine
│   ├── io/                  # File I/O operations
│   ├── controller/          # Business logic controllers
│   └── yoloe/              # YOLO-E integration (THU-MIG/yoloe)
├── pretrain/               # Pre-trained models
├── resources/              # Application resources
└── bakuai-labelimg.py     # Main entry point
```

**Note**: This project integrates [YOLOE](https://github.com/THU-MIG/yoloe) by THU-MIG for AI-powered object detection capabilities.

---

## 🛠️ Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/bakuai-as/BakuFlow.git
cd BakuFlow

# Install dependencies
pip install -r requirements.txt

# Run the application
python bakuai-labelimg.py
```

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, Linux Ubuntu 18.04+
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **GPU**: Optional - CUDA-compatible GPU will improve AI feature performance
- **Storage**: 2GB free space for installation

### Dependencies
```
PyQt5>=5.15.0          # GUI framework
opencv-python>=4.5.0   # Computer vision operations
numpy>=1.19.0          # Numerical computing
torch>=1.8.0           # Deep learning framework
torchvision>=0.9.0     # Computer vision models
Pillow>=8.0.0          # Image processing
ultralytics>=8.0.0     # YOLO implementation
```

---

## 🖥️ Usage Guide

### 1. Basic Workflow
1. **Launch**: Run `python bakuai-labelimg.py`
2. **Open Dataset**: Use `File > Open Directory` or press `O`
3. **Load Classes**: Create or select `classes.txt` file when prompted
4. **Select Format**: Choose output format (YOLO/VOC/COCO) at top left
5. **Start Labeling**: Draw bounding boxes and assign classes
6. **Save**: Press `S` to save or enable Auto Save

### 2. AI-Assisted Labeling
1. **Create Examples**: Label several representative images manually
2. **Select Prompts**: Choose your best examples as visual prompts
3. **Auto-Label**: Use `Auto Label > Visual Prompt Auto Labeling`
4. **Review Results**: Check and adjust automatically generated annotations
5. **Batch Process**: Apply to remaining unlabeled images

### 3. Advanced Operations
- **Multi-Selection**: Hold Ctrl+Click to select multiple boxes
- **Batch Delete**: Select multiple boxes and press Delete/E
- **Copy Labels**: Use 'C' to copy annotations between similar images
- **Magnifier**: Press 'M' to toggle pixel-accurate magnification
- **Data Augmentation**: Access via `Data Augmentation > Batch Data Augmentation`

---

## ⌨️ Keyboard Shortcuts

| Key | Action | Description |
|-----|--------|-------------|
| `O` | Open Directory | Load image folder |
| `S` | Save | Save current annotations |
| `F` | Next Image | Navigate to next image |
| `D` | Previous Image | Navigate to previous image |
| `E` / `Delete` | Delete | Remove selected bounding box(es) |
| `C` | Label Propagation | Toggle annotation copying |
| `M` | Magnifier | Toggle magnifier tool |
| `Q` | Quit | Exit application |
| `↑` / `↓` | Navigate | Previous/Next image |
| `Ctrl+Click` | Multi-Select | Select multiple bounding boxes |
| `Ctrl+Z` | Undo | Undo last action (per image) |
| `Ctrl+Y` | Redo | Redo last undone action |

---

## 🔧 Configuration

### Language Configuration
Create `language_config.py` in the root directory:
```python
# Language options: 'en', 'zh-tw', 'zh-cn', 'ja', 'it', 'de', 'no', 'es', 'fr'
LANGUAGE = 'en'
```

### Model Configuration
Pre-trained models are automatically downloaded to `pretrain/` folder:
- `yoloe-11l-seg.pt` - YOLO-E segmentation model
- Custom models can be added to this directory

## 🤖 YOLOE Setup & Usage

**BakuFlow includes custom YOLOE modifications for enhanced performance.** This section shows how to set up YOLOE with BakuFlow customizations.

### 🚀 Quick Setup (Recommended)

Run the automated setup script:

```bash
# Navigate to your BakuFlow directory
cd /path/to/your/BakuFlow

# Run the automated setup script
python setup_yoloe.py
```

This script will:
1. ✅ Clone YOLOE repository
2. ✅ Install all dependencies
3. ✅ Apply BakuFlow customizations
4. ✅ Download pre-trained models
5. ✅ Verify installation

### 🔧 Manual Setup (Advanced Users)

If you prefer manual setup or need to troubleshoot:

### 1. Install YOLOE

#### Option 1: Clone Original Repository (Recommended)
```bash
# Navigate to your BakuFlow directory
cd /path/to/your/BakuFlow

# Clone YOLOE into the correct location
git clone https://github.com/THU-MIG/yoloe.git labelimg/yoloe

# Install YOLOE dependencies
cd labelimg/yoloe
pip install -r requirements.txt
pip install -e .
```

#### Option 2: Download and Extract
1. Go to [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)
2. Download as ZIP and extract to `labelimg/yoloe/`
3. Install dependencies as above

### 2. Download Pre-trained Models

#### Automatic Download (Recommended)
```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub==0.26.3

# Download YOLOE models
huggingface-cli download jameslahm/yoloe yoloe-v8l-seg.pt --local-dir pretrain
huggingface-cli download jameslahm/yoloe yoloe-11l-seg.pt --local-dir pretrain
```

#### Manual Download
1. Visit [jameslahm/yoloe on Hugging Face](https://huggingface.co/jameslahm/yoloe)
2. Download the following models to your `pretrain/` folder:
   - `yoloe-v8l-seg.pt` (Primary model)
   - `yoloe-11l-seg.pt` (Alternative model)

### 3. Verify Installation

Run this test to verify YOLOE is properly installed:

```python
# test_yoloe.py
import sys
import os

# Add YOLOE to path (same as BakuFlow does)
yoloe_path = os.path.abspath('labelimg/yoloe')
if yoloe_path not in sys.path:
    sys.path.insert(0, yoloe_path)

try:
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
    
    # Test model loading
    model = YOLOE('pretrain/yoloe-v8l-seg.pt')
    print("✅ YOLOE installation successful!")
    print(f"Model loaded: {model}")
    
except ImportError as e:
    print(f"❌ YOLOE import failed: {e}")
    print("Please check your installation steps.")
except FileNotFoundError as e:
    print(f"❌ Model file not found: {e}")
    print("Please download the pre-trained models.")
```

### 4. Using AI Features in BakuFlow

Once YOLOE is properly installed:

#### 4.1 Visual Prompt Auto-Labeling
1. **Prepare Prompt Images**: Manually label 3-5 representative images with good examples
2. **Access AI Menu**: Go to `Auto Label > Visual Prompt Auto Labeling`
3. **Select Prompts**: Choose your manually labeled images as visual prompts (they will appear green)
4. **Select Target**: Choose unlabeled images to auto-label
5. **Run Auto-Label**: The AI will use your examples to find similar objects

#### 4.2 Current Image Auto-Labeling
1. **Label Current Image**: Add some annotations to the current image
2. **Use as Prompts**: Go to `Auto Label > Auto Label Current Image`
3. **Confirm**: The AI will use current annotations to find more similar objects in the same image

#### 4.3 Batch Auto-Labeling
1. **Prepare Prompts**: Ensure you have good examples labeled
2. **Select All Unlabeled**: Go to `Auto Label > Auto Label All Unlabeled Images`  
3. **Choose Strategy**: Decide whether to overwrite existing labels
4. **Monitor Progress**: Watch the progress dialog for completion

### 5. Troubleshooting YOLOE

#### Common Issues:

**Q: "YOLOE not found" error**
```bash
# Solution: Check YOLOE installation
cd labelimg/yoloe
ls -la  # Should show YOLOE files
pip list | grep ultralytics  # Should show ultralytics package
```

**Q: "Model file not found" error**
```bash
# Solution: Check model files
ls -la pretrain/
# Should show .pt files like yoloe-v8l-seg.pt
```

**Q: "CUDA out of memory" error**
```python
# Solution: The code automatically falls back to CPU
# Check device selection in console output:
device='cpu' if not torch.cuda.is_available() else 'cuda'
```

**Q: AI features not appearing in menu**
```python
# Solution: Check if classes.txt is loaded
# The YOLOEWrapper is only initialized after classes are loaded
```

### 6. Model Variants

BakuFlow supports different YOLOE model variants:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yoloe-v8s-seg.pt` | Small | Fast | Good | Quick annotation |
| `yoloe-v8m-seg.pt` | Medium | Medium | Better | Balanced usage |
| `yoloe-v8l-seg.pt` | Large | Slow | Best | High accuracy needed |
| `yoloe-11l-seg.pt` | Large | Slow | Best | Latest model |

**Note**: The model path is configured in the code. To use a different model, update the path in `labelimg/inference/yoloe_vp_discard.py`.

### 7. Performance Tips

- **GPU Usage**: YOLOE will automatically use GPU if available, otherwise CPU
- **Memory**: Larger models require more GPU memory (4GB+ recommended)
- **Prompt Quality**: Use diverse, high-quality examples for better results
- **Batch Size**: Process in smaller batches if memory is limited

---

## 🎨 Advanced Features

### Visual Prompt Auto-Labeling
- **Concept**: Use manually labeled images as "visual prompts" to teach the AI
- **Process**: AI learns object appearance patterns from your examples  
- **Application**: Automatically finds similar objects in new images
- **Accuracy**: Depends on quality and diversity of prompt images

### Data Augmentation
- **Batch Processing**: Apply transformations to entire datasets
- **Transformations**: Rotation, brightness, contrast, saturation adjustments
- **Geometric**: Horizontal/vertical flipping with coordinate preservation
- **Output**: Customizable augmentation parameters and output directories

---

## 📝 License

This software is released under a custom license with the following terms:

### Usage Permissions

**Academic Use** (Free with Registration)
- ✅ Allowed for research, educational, and non-commercial purposes
- ℹ️ Acknowledgment/citation required in publications
- 📧 Register your academic institution with post@bakuai.no

**Commercial Use**
- ❌ Requires separate commercial license
- 💼 Contact post@bakuai.no for commercial licensing details
- 🏢 Enterprise solutions available

### Academic Acknowledgment

For academic use, please include this acknowledgment:
```
BakuFlow developed by BakuAI AS (2024) - "Professional Annotation Made Simple"
```

For publications using this software, please cite:
```
BakuFlow: AI-Powered Image Annotation Tool for Object Detection
BakuAI AS, Norway (2024)
```

---

## 🤝 Contributing & Support

**Technical Support**
- 📧 Email: post@bakuai.no
- 🐛 Issues: [GitHub Issues](https://github.com/bakuai-as/BakuFlow/issues)
- 💬 Community: [Discord Server](https://discord.gg/bakuai)

**Contributing**
- Pull requests and issues are welcome!
- Submit bug reports, feature requests, or improvements via GitHub
- Check our contributing guidelines before submitting

**Commercial Inquiries**
- Commercial Licensing: post@bakuai.no
- Enterprise Solutions: post@bakuai.no
- Academic Partnerships: post@bakuai.no

---

## 👨‍💻 Development Team

**BakuFlow** is originally developed by **Dr. Jerry Chun-Wei Lin**, together with **Patrick Chen** and modern AI-assisted development tools (Cursor, VSCode + Deepseek/Gemini/Claude), and maintained by the Innovation Team at **BakuAI AS**, Norway.

**Core Contributors:**
- Dr. Jerry Chun-Wei Lin - Lead Developer & AI Architecture
- Patrick Chen - Frontend Development & User Experience  
- BakuAI Innovation Team - Ongoing Development & Maintenance

---

*Made with ❤️ by BakuAI Team - Making Professional Annotation Simple*

## 🙏 Acknowledgments

**BakuFlow** leverages several outstanding open-source projects:

- **[YOLOE](https://github.com/THU-MIG/yoloe)**: Real-Time Seeing Anything by THU-MIG (AGPL-3.0)
- **[PyQt5](https://pypi.org/project/PyQt5/)**: Cross-platform GUI framework
- **[OpenCV](https://opencv.org/)**: Computer vision library
- **[PyTorch](https://pytorch.org/)**: Deep learning framework

**Citation for YOLOE**:
```bibtex
@misc{wang2025yoloerealtimeseeing,
      title={YOLOE: Real-Time Seeing Anything}, 
      author={Ao Wang and Lihao Liu and Hui Chen and Zijia Lin and Jungong Han and Guiguang Ding},
      year={2025},
      eprint={2503.07465},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.07465}, 
}
```

## ❓ FAQ

**Q: Why can't I see my classes?**  
A: Make sure `classes.txt` exists in your image folder. You can create or import it via the UI prompt.

**Q: How do I export to different formats?**  
A: Select the desired format (YOLO/VOC/COCO) at the top left before saving. The tool will output the correct file type.

**Q: How do I use AI auto-labeling?**  
A: First manually label a few representative images, then use them as visual prompts via `Auto Label > Visual Prompt Auto Labeling`. See the YOLOE Setup section above for detailed instructions.

**Q: AI auto-labeling not working?**  
A: Check: (1) YOLOE is properly installed in `labelimg/yoloe/`, (2) Classes.txt file is loaded, (3) Prompt images are properly labeled, (4) Pre-trained models are downloaded to `pretrain/` folder.

**Q: How do I batch operations?**  
A: Hold Ctrl and click to select multiple boxes, then use Delete, Copy Labels, or other batch operations.

**Q: Application crashes on startup?**  
A: Update PyQt5: `pip install --upgrade PyQt5`, ensure Python 3.8+, check all dependencies are installed.

**Q: Where do I get the YOLOE models?**  
A: Models are automatically downloaded from Hugging Face. See the "Download Pre-trained Models" section above for manual download instructions.

**Q: Can I use my own YOLOE models?**  
A: Yes! Place your custom `.pt` model files in the `pretrain/` folder and update the model path in the code.
