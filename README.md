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

---

## ❓ FAQ

**Q: Why can't I see my classes?**  
A: Make sure `classes.txt` exists in your image folder. You can create or import it via the UI prompt.

**Q: How do I export to different formats?**  
A: Select the desired format (YOLO/VOC/COCO) at the top left before saving. The tool will output the correct file type.

**Q: How do I use AI auto-labeling?**  
A: First manually label a few representative images, then use them as visual prompts via `Auto Label > Visual Prompt Auto Labeling`.

**Q: AI auto-labeling not working?**  
A: Check: (1) Classes.txt file is loaded, (2) Prompt images are properly labeled, (3) Pre-trained models are downloaded.

**Q: How do I batch operations?**  
A: Hold Ctrl and click to select multiple boxes, then use Delete, Copy Labels, or other batch operations.

**Q: Application crashes on startup?**  
A: Update PyQt5: `pip install --upgrade PyQt5`, ensure Python 3.8+, check all dependencies are installed.

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
