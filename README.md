# BakuFlow developed by BakuAI AS (2024) - "Professional Annotation Made Simple"

A modern, efficient, and user-friendly image annotation tool for object detection datasets. Supports YOLO, VOC, and COCO formats, multi-class labeling, auto-labeling model based on YOLOE, , multi-bbox copy/selection, and more.

---

## 🚀 Features
- **Bounding Box Annotation**: Draw, move, resize, and delete bounding boxes with mouse and keyboard.
- **Multi-class Support**: Easily manage and switch between multiple classes. Supports custom class files per image folder.
- **Color Legend**: Each class is assigned a unique color, with a live-updating legend panel.
- **Auto Save**: Automatically saves annotations after each edit.
- **Multi-format Export**: Supports YOLO (.txt), VOC (.xml), and COCO (.json) formats. Switch format anytime.
- **Multi-bbox Selection**: Hold Ctrl and click to select multiple boxes for batch delete/copy.
- **Copy Labels**: Copy selected or all bounding boxes to the next image.
- **Magnifier**: Pixel-accurate magnifier for precise annotation.
- **Undo/Redo**: Per-image undo/redo for all labeling actions (Ctrl+Z / Ctrl+Y).
- **Keyboard Shortcuts**: Fast navigation and editing with hotkeys.
- **Internationalization Ready**: UI and messages in English, easy to localize.

---

## 🛠️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/bakuai-labelimg.git
   cd bakuai-labelimg
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage
1. Run the tool:
   ```bash
   python bakuai-labelimg.py
   ```
2. Open an image folder. If no `classes.txt` is found, you will be prompted to select or create one.
3. Select output format (YOLO/VOC/COCO) at the top left.
4. Draw bounding boxes, select class, and annotate images.
5. Use Ctrl+Click to select multiple boxes for batch operations.
6. Save annotations or enable Auto Save.

---

## ⌨️ Hotkeys
| Key           | Action                        |
|---------------|------------------------------|
| F             | Next Image                   |
| D             | Previous Image               |
| S             | Save Annotations             |
| O             | Open Directory               |
| E/Delete      | Delete Selected Box(es)      |
| Q             | Quit                         |
| ↑ / ↓         | Previous/Next Image          |
| C             | Switch Copy Labels           |
| Ctrl+Click    | Multi-select bounding boxes  |
| Ctrl+Z        | Undo (per image labeling)    |
| Ctrl+Y        | Redo (per image labeling)    |

---

## 📦 Supported Formats
- **YOLO**: `.txt` per image, normalized coordinates.
- **VOC**: `.xml` per image, Pascal VOC format.
- **COCO**: Single `coco_annotations.json` per folder, standard COCO format.

---

## 💡 Advanced Features
- **Custom Classes**: Each image folder can have its own `classes.txt`. Add new classes via the UI or by editing the file.
- **Batch Copy/Delete**: Select multiple boxes and perform batch operations.
- **Magnifier**: Toggle magnifier for pixel-level accuracy.
- **Flexible UI**: Panels can be resized, and the layout is optimized for productivity.
- **Undo/Redo**: Per-image undo/redo for all labeling actions.

---

## ❓ FAQ
**Q: Why can't I see my classes?**  
A: Make sure `classes.txt` exists in your image folder. You can create or import it via the UI prompt.

**Q: How do I export to COCO/VOC?**  
A: Select the desired format at the top left before saving. The tool will output the correct file type.

**Q: How do I batch delete or copy boxes?**  
A: Hold Ctrl and click to select multiple boxes, then use Delete or Copy Labels.

**Q: How do I undo/redo?**  
A: Use Ctrl+Z (undo) and Ctrl+Y (redo) for per-image labeling history.

---

## 🤝 Contribution
Pull requests and issues are welcome! Please submit bug reports, feature requests, or improvements via GitHub Issues or PRs.

---

## 📧 Contact
For business or support, contact: [post@bakuai.no](mailto:post@bakuai.no)

---

## 📝 License
This software is released under a custom license that includes the following key points:

### Usage Permissions

1. Personal Use in Academia (register by eduation)
   - ✅ Allowed for research, educational and non-commerical purposes
   - ℹ️ Acknowledgment/citation required
   - ❌ No commercial use permitted

3. Commercial Use
   - ❌ Requires commercial license
   - 📧 Contact post@bakuai.no for the details of [commercial licensing contact]

### Academic Acknowledgment Requirements

For academic use, please include the following acknowledgment:

```
[Software Name] developed by [Author/Organization Name] ([Year])
```

For publications using this software, please cite:
[Citation format or relevant paper]

### Contact Information

- Commercial Licensing: post@bakuai.no
- Academic Collaboration: post@bakuai.no
- Technical Support: post@bakuai.no

---

**BakuFlow ** is originally developed by Dr. Jerry Chun-Wei Lin, together with Patrick Chen and vibe coding tools (e.g., cursor, vscode + Deepseek/Gemini/Claud) and maintained by the Innovation Team in BakuAI AS, Norway.
