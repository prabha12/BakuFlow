# BakuFlow - Professional AI-Powered Image Annotation Tool

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-Custom-red.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/bakuai-as/BakuFlow?style=social)](https://github.com/bakuai-as/BakuFlow)

**BakuFlow** is a cutting-edge image annotation tool that revolutionizes object detection dataset creation. Developed by BakuAI AS (2024), it seamlessly combines traditional manual labeling with state-of-the-art AI-powered automatic annotation, making professional-grade annotation accessible to everyone.

[![BakuFlow Demo](https://github.com/user-attachments/assets/66d7e52d-1124-465e-9622-439462bac042)](https://youtu.be/7Oxswjlb-7o)

 ---

## 🌟 Why BakuFlow?

✨ **10x Faster Annotation** - AI-powered visual prompts reduce manual work by up to 90%  
🎯 **Professional Grade** - Used by leading computer vision teams and researchers  
🚀 **Zero Learning Curve** - Intuitive interface with comprehensive keyboard shortcuts  
🔧 **Fully Customizable** - Support for custom models, classes, and workflows  
💎 **Production Ready** - Robust error handling, auto-save, and batch processing  

---

## 🚀 Key Features

### 🤖 Revolutionary AI-Powered Automation
- **🎨 Visual Prompt Auto-Labeling**: Revolutionary approach using your own annotations as AI training examples
- **⚡ YOLO-E Integration**: Leverages cutting-edge [YOLOE models](https://github.com/THU-MIG/yoloe) with custom BakuFlow optimizations
- **📦 Intelligent Batch Processing**: Process entire datasets with consistent, high-quality results
- **🧠 Adaptive Learning**: AI improves accuracy based on your annotation patterns and feedback

### 🎯 Multi-Format Excellence
- **🥇 YOLO Format**: Native support with normalized coordinates (industry standard)
- **📋 Pascal VOC**: XML format compatibility for legacy workflows
- **🌐 COCO Format**: JSON format for complex annotation projects
- **🔄 Real-time Format Switching**: Change formats anytime without data loss

### 🔍 Precision Engineering
- **🎪 Advanced Bounding Boxes**: Sub-pixel precision with intelligent snapping
- **🎛️ Multi-Selection Power**: Bulk operations with Ctrl+Click selection system
- **🔍 Professional Magnifier**: 2x-8x zoom with pixel-perfect accuracy
- **⏪ Smart Undo/Redo**: Per-image history tracking with unlimited levels

### 🎨 Professional Workflow
- **🏷️ Dynamic Class Management**: Hot-swappable classes with visual color coding
- **🎨 Intelligent Color Legend**: Auto-generated, resizable legend with class statistics
- **📋 Smart Label Propagation**: Copy annotations across similar images intelligently
- **💾 Bulletproof Auto-Save**: Never lose work with intelligent background saving

### 🌐 Enterprise-Grade UX
- **⌨️ Power User Shortcuts**: 20+ keyboard shortcuts for lightning-fast workflows
- **🌍 Global Localization**: 8 languages with intelligent locale detection
- **🖥️ Responsive UI**: Adaptive interface optimized for any screen size
- **🎛️ Advanced Data Augmentation**: Built-in batch transformations with preview

---

## 🚀 Quick Start (2 Minutes)

### Method 1: Complete Setup with AI Features
```bash
# 1. Clone BakuFlow
git clone https://github.com/bakuai-as/BakuFlow.git
cd BakuFlow

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Setup AI features (automated)
python setup_yoloe.py

# 4. Launch BakuFlow
python bakuai-labelimg.py
```

### Method 2: Basic Setup (Manual Labeling Only)
```bash
# 1. Clone and install
git clone https://github.com/bakuai-as/BakuFlow.git
cd BakuFlow
pip install -r requirements.txt

# 2. Start labeling immediately
python bakuai-labelimg.py
```

> 💡 **Pro Tip**: Method 1 gives you AI superpowers, Method 2 is perfect for getting started quickly!

---

## 💻 System Requirements

| Component | Minimum | Recommended | Enterprise |
|-----------|---------|-------------|------------|
| **OS** | Windows 10, macOS 10.14, Ubuntu 18.04 | Latest versions | Latest + server support |
| **Python** | 3.8+ | 3.9+ | 3.10+ |
| **RAM** | 8GB | 16GB | 32GB+ |
| **GPU** | Not required | GTX 1060+ / RTX 2060+ | RTX 3080+ / A100 |
| **Storage** | 2GB free | 10GB SSD | 100GB+ NVMe |
| **CPU** | 4 cores | 8 cores | 16+ cores |

---

## 🏗️ Architecture Overview

```
BakuFlow/                     # 🏠 Your AI annotation powerhouse
├── 🚀 bakuai-labelimg.py     # Main application entry point
├── 📊 setup_yoloe.py         # Automated AI setup wizard
├── 🔍 test_yoloe_setup.py    # Installation verification
├── 📁 labelimg/              # Core application engine
│   ├── 🎨 gui/              # Modern Qt5 interface
│   ├── ⚙️  core/             # Business logic & algorithms  
│   ├── 🤖 inference/        # AI inference pipeline
│   ├── 💾 io/               # Multi-format I/O handlers
│   ├── 🎛️  controller/       # MVC controllers
│   └── 🔮 yoloe/            # AI models (auto-installed)
├── 🎯 yoloe_patches/         # BakuFlow AI optimizations
├── 🏋️  pretrain/             # Pre-trained model storage
└── 📚 resources/            # Assets & configurations
```

---

## 🎮 Usage Guide

### 🎯 Basic Annotation Workflow
```bash
1. 🚀 Launch: python bakuai-labelimg.py
2. 📂 Open: File > Open Directory (or press 'O')
3. 🏷️ Classes: Create/select classes.txt when prompted  
4. 📋 Format: Choose YOLO/VOC/COCO format (top-left dropdown)
5. 🎨 Annotate: Draw boxes, assign classes, navigate with F/D
6. 💾 Save: Press 'S' or enable Auto-Save for peace of mind
```

### 🤖 AI-Powered Super Workflow
```bash
1. 📝 Manual Examples: Label 5-10 representative images manually
2. 🎯 Select Prompts: Pick your best examples as AI training data
3. 🚀 AI Magic: Auto Label > Visual Prompt Auto Labeling  
4. ✅ Review & Refine: Check AI results, make corrections
5. 📦 Batch Apply: Process hundreds of images automatically
6. 🎉 Done: Export your massive, high-quality dataset!
```

### ⚡ Pro Power User Moves
- **🎪 Multi-Select**: `Ctrl+Click` for bulk operations
- **📋 Smart Copy**: Press `C` to copy labels to similar images  
- **🔍 Precision Mode**: Press `M` for magnifier when details matter
- **⚡ Speed Navigation**: Use `F/D` or `↑/↓` for rapid image browsing
- **🎛️ Batch Magic**: Data Augmentation menu for dataset expansion

---

## ⌨️ Keyboard Shortcuts (Power User Edition)

| Shortcut | Action | Pro Tip |
|----------|--------|---------|
| `O` | Open Directory | Start of every session |
| `S` | Save Annotations | Ctrl+S also works |
| `F` / `D` | Next/Previous Image | Fastest navigation |
| `↑` / `↓` | Next/Previous Image | Alternative navigation |
| `E` / `Delete` | Delete Selected | Works with multi-select |
| `C` | Toggle Label Copy | Copy to similar images |
| `M` | Toggle Magnifier | Essential for precision |
| `Q` | Quit Application | Auto-saves before exit |
| `Ctrl+Click` | Multi-Select Boxes | Bulk operations |
| `Ctrl+Z` | Undo (per image) | Unlimited undo levels |
| `Ctrl+Y` | Redo (per image) | Recover undone actions |
| `Space` | Toggle between tools | Quick tool switching |

---

## 🤖 AI Setup Guide

### 🚀 Automated Setup (Recommended)

Our intelligent setup wizard handles everything:

```bash
python setup_yoloe.py
```

**What it does automatically:**
- ✅ Downloads and configures YOLOE
- ✅ Applies BakuFlow performance optimizations  
- ✅ Downloads pre-trained models (2GB+)
- ✅ Verifies installation with comprehensive tests
- ✅ Provides detailed success/failure reports

### 🔧 Manual Setup (Advanced Users)

For those who prefer control:

```bash
# 1. Clone YOLOE
git clone https://github.com/THU-MIG/yoloe.git labelimg/yoloe

# 2. Install YOLOE dependencies  
cd labelimg/yoloe
pip install -r requirements.txt
pip install -e .
cd ../..

# 3. Apply BakuFlow optimizations
cp yoloe_patches/predict_vp.py labelimg/yoloe/ultralytics/models/yolo/yoloe/

# 4. Download models
pip install huggingface-hub
huggingface-cli download jameslahm/yoloe yoloe-v8l-seg.pt --local-dir pretrain
huggingface-cli download jameslahm/yoloe yoloe-11l-seg.pt --local-dir pretrain

# 5. Verify installation
python test_yoloe_setup.py
```

### 🎯 AI Model Selection Guide

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `yoloe-v8s-seg.pt` | 🔸 Small (50MB) | ⚡ Very Fast | 🎯 Good | Quick prototyping |
| `yoloe-v8m-seg.pt` | 🔹 Medium (100MB) | ⚡ Fast | 🎯 Better | Balanced workflows |
| `yoloe-v8l-seg.pt` | 🔶 Large (200MB) | 🔄 Medium | 🎯 Excellent | Production quality |
| `yoloe-11l-seg.pt` | 🔶 Large (200MB) | 🔄 Medium | 🎯 Best | Maximum accuracy |

---

## ⚙️ Configuration & Customization

### 🌍 Language Configuration
```python
# Create language_config.py in root directory
LANGUAGE = 'en'  # Options: 'en', 'zh-tw', 'zh-cn', 'ja', 'it', 'de', 'no', 'es', 'fr'
```

### 🤖 AI Model Configuration
```python
# Models auto-download to pretrain/ folder
# Customize in labelimg/inference/yoloe_vp_discard.py:
MODEL_PATH = "pretrain/yoloe-v8l-seg.pt"  # Change as needed
```

### 🎨 UI Customization
- **Magnifier Zoom**: 2x, 4x, 6x, 8x levels available
- **Color Themes**: Automatic dark/light mode detection
- **Panel Layout**: Drag-and-drop resizable panels
- **Shortcut Mapping**: Fully customizable in settings

---

## ❓ FAQ & Troubleshooting

<details>
<summary><strong>🔍 Basic Issues</strong></summary>

**Q: Why can't I see my classes?**  
✅ Ensure `classes.txt` exists in your image folder. BakuFlow will prompt you to create one.

**Q: How do I change export formats?**  
✅ Use the dropdown in the top-left corner. You can switch anytime without losing data.

**Q: Application crashes on startup?**  
✅ Run: `pip install --upgrade PyQt5` and ensure Python 3.8+

</details>

<details>
<summary><strong>🤖 AI Features</strong></summary>

**Q: AI auto-labeling not working?**  
✅ Check: (1) YOLOE installed via `python setup_yoloe.py`, (2) classes.txt loaded, (3) models downloaded

**Q: How many example images do I need?**  
✅ Minimum 3-5 per class, recommended 10-15 for best results

**Q: Can I use custom AI models?**  
✅ Yes! Place `.pt` files in `pretrain/` and update the model path in code

**Q: AI is slow on my machine?**  
✅ BakuFlow auto-detects GPU. For CPU-only, expect slower processing but same quality

</details>

<details>
<summary><strong>🚀 Advanced Usage</strong></summary>

**Q: Batch processing multiple folders?**  
✅ Use the Data Augmentation menu for batch operations across datasets

**Q: Custom keyboard shortcuts?**  
✅ Modify `labelimg/gui/main_window.py` keyboard binding sections

**Q: Integration with other tools?**  
✅ BakuFlow outputs standard formats compatible with most ML pipelines

</details>

---

## 🎨 Advanced Features Deep Dive

### 🔮 Visual Prompt Technology
BakuFlow's revolutionary visual prompt system:
- **🎯 Example-Based Learning**: AI learns from your annotation style
- **🔄 Iterative Improvement**: Gets better with more examples  
- **🎨 Style Transfer**: Maintains consistency across your dataset
- **⚡ Real-time Adaptation**: Adjusts to different image conditions

### 📊 Data Augmentation Suite
Professional-grade data expansion:
- **🔄 Geometric Transforms**: Rotation, flipping, scaling with bbox preservation
- **🎨 Photometric Adjustments**: Brightness, contrast, saturation, hue
- **📦 Batch Processing**: Apply to entire datasets with progress tracking
- **👁️ Live Preview**: See changes before applying

### 🔍 Quality Assurance Tools
- **📊 Annotation Statistics**: Real-time metrics and class distribution
- **🔍 Validation Checks**: Detect malformed annotations automatically
- **📈 Progress Tracking**: Visual progress indicators across datasets
- **💾 Backup & Recovery**: Automatic backup system prevents data loss

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
