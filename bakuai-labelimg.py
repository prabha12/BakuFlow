#### bounding box is moviable and adjustable

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QListWidget, QCheckBox, QComboBox, QMenu, QInputDialog,
                            QSplitter, QFrame, QDialog, QSlider, QAction,
                            QGroupBox, QSpinBox, QDoubleSpinBox, QLineEdit, QProgressDialog)
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QCursor, QPainter, QPen, QColor
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtWidgets import QMessageBox  # 添加这个导入
import random
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import QScrollArea
import locale

# 只在 Windows 系統上導入 windll
if sys.platform == 'win32':
    from ctypes import windll

TRANSLATIONS = {
    "en": {
        "open": "Open",
        "save": "Save",
        "next_image": "Next Image",
        "prev_image": "Previous Image",
        "delete": "Delete Selected Box(es)",
        "auto_save": "Auto Save",
        "copy_labels": "Copy Labels",
        "output_format": "Output Format",
        "classes": "Classes",
        "ready": "Ready",
        "no_classes_file": "No classes.txt found in this folder.\nWould you like to select an existing txt file as the class file?\nChoose Yes to select a file, or No to manually enter classes.",
        "yes": "Yes",
        "no": "No",
        "hotkeys_help": "Hotkeys Help",
        "quit": "Quit",
        "file_menu": "File(&F)",
        "load_label_file": "Load Label File (e.g., .txt)",
        "help_menu": "Help(&H)",
        "hotkeys_menu": "Hotkeys(&K)",
        "about_menu": "About BakuAI AS(&A)",
        "hotkeys_title": "Hotkeys Help",
        "hotkeys_list": "Hotkeys list：",
        "next_image_hotkey": "Next Image",
        "prev_image_hotkey": "Previous Image",
        "save_hotkey": "Save Annotations",
        "open_hotkey": "Open Directory",
        "delete_hotkey": "Delete Selected Box(es)",
        "quit_hotkey": "Quit",
        "copy_labels_hotkey": "Switch Copy Labels",
        "multi_select_hotkey": "Multi-select bounding boxes",
        "undo_hotkey": "Undo (per image labeling)",
        "redo_hotkey": "Redo (per image labeling)",
        "about_title": "About BakuAI",
        "version": "Version",
        "copyright": "Copyright Reserved",
        "website": "Official Website",
        "software_description": "This software is used for image labeling, supported for YOLO/VOC/COCO formats",
        "debug_info": "Debug Information",
        "current_working_dir": "Current Working Directory",
        "dist_dir_contents": "Dist Directory Contents",
        "debug_info_end": "Debug Information End",
        "class_save_failed": "Class Save Failed",
        "failed_to_write_classes": "Failed to write classes.txt: {0}",
        "save_error": "Save Error",
        "failed_to_save_yolo": "Failed to save YOLO annotations:\n{0}",
        "edit_label": "Edit Label",
        "enter_new_label": "Enter new label:",
        "data_aug_menu": "Data Augmentation(&D)",
        "data_aug_title": "Batch Data Augmentation",
        "saturation": "Saturation",
        "contrast": "Contrast",
        "brightness": "Brightness",
        "rotation": "Rotation (degrees)",
        "flip_ud": "Flip Up-Down",
        "flip_lr": "Flip Left-Right",
        "apply": "Apply",
        "cancel": "Cancel",
        "reset": "Reset",
        "preview": "Preview",
        "aug_per_image": "Augmentations per image",
        "output_folder": "Output Folder",
        "select_folder": "Select Folder",
        "processing": "Processing...",
        "aug_complete": "Augmentation Complete",
        "aug_progress": "Processing {0}/{1} images",
        "aug_success": "Successfully augmented {0} images",
        "aug_error": "Error during augmentation",
        "preserve_original": "Preserve Original Files",
        "aug_settings": "Augmentation Settings",
        "random_rotation": "Random Rotation Range",
        "random_brightness": "Random Brightness Range",
        "random_contrast": "Random Contrast Range",
        "random_saturation": "Random Saturation Range",
        "flip_options": "Flip Options",
        "warning": "Warning",
        "no_image": "No image loaded. Please open an image first.",
    },
    "zh-tw": {
        "open": "開啟",
        "save": "儲存",
        "next_image": "下一張圖片",
        "prev_image": "上一張圖片",
        "delete": "刪除選取框",
        "auto_save": "自動儲存",
        "copy_labels": "複製標註",
        "output_format": "輸出格式",
        "classes": "類別",
        "ready": "就緒",
        "no_classes_file": "此資料夾下找不到 classes.txt，是否要選擇現有的 txt 檔作為類別檔？\n選是可選檔，選否可自行輸入類別。",
        "yes": "是",
        "no": "否",
        "hotkeys_help": "快捷鍵說明",
        "quit": "離開",
        "file_menu": "檔案(&F)",
        "load_label_file": "載入標籤檔 (例如：.txt)",
        "help_menu": "說明(&H)",
        "hotkeys_menu": "快捷鍵(&K)",
        "about_menu": "關於 BakuAI AS(&A)",
        "hotkeys_title": "快捷鍵說明",
        "hotkeys_list": "快捷鍵列表：",
        "next_image_hotkey": "下一張圖片",
        "prev_image_hotkey": "上一張圖片",
        "save_hotkey": "儲存標註",
        "open_hotkey": "開啟資料夾",
        "delete_hotkey": "刪除選取框",
        "quit_hotkey": "離開",
        "copy_labels_hotkey": "切換複製標註",
        "multi_select_hotkey": "多選標註框",
        "undo_hotkey": "復原 (每張圖片獨立)",
        "redo_hotkey": "重做 (每張圖片獨立)",
        "about_title": "關於 BakuAI",
        "version": "版本",
        "copyright": "版權所有",
        "website": "官方網站",
        "software_description": "本軟體用於影像標註，支援 YOLO/VOC/COCO 格式",
        "debug_info": "除錯資訊",
        "current_working_dir": "目前工作目錄",
        "dist_dir_contents": "發佈目錄內容",
        "debug_info_end": "除錯資訊結束",
        "class_save_failed": "類別儲存失敗",
        "failed_to_write_classes": "無法寫入 classes.txt：{0}",
        "save_error": "儲存錯誤",
        "failed_to_save_yolo": "儲存 YOLO 標註失敗：\n{0}",
        "edit_label": "編輯標籤",
        "enter_new_label": "輸入新標籤：",
        "data_aug_menu": "資料增強(&D)",
        "data_aug_title": "批次資料增強",
        "saturation": "飽和度",
        "contrast": "對比度",
        "brightness": "亮度",
        "rotation": "旋轉角度",
        "flip_ud": "上下翻轉",
        "flip_lr": "左右翻轉",
        "apply": "套用",
        "cancel": "取消",
        "reset": "重設",
        "preview": "預覽",
        "aug_per_image": "每張圖片的增強數量",
        "output_folder": "輸出資料夾",
        "select_folder": "選擇資料夾",
        "processing": "處理中...",
        "aug_complete": "資料增強完成",
        "aug_progress": "處理第 {0}/{1} 張圖片",
        "aug_success": "成功增強 {0} 張圖片",
        "aug_error": "資料增強過程發生錯誤",
        "preserve_original": "保留原始檔案",
        "aug_settings": "增強設定",
        "random_rotation": "隨機旋轉範圍",
        "random_brightness": "隨機亮度範圍",
        "random_contrast": "隨機對比度範圍",
        "random_saturation": "隨機飽和度範圍",
        "flip_options": "翻轉選項",
        "warning": "警告",
        "no_image": "尚未載入圖片，請先開啟圖片。",
    },
    "zh-cn": {
        "open": "打开",
        "save": "保存",
        "next_image": "下一张图片",
        "prev_image": "上一张图片",
        "delete": "删除选中框",
        "auto_save": "自动保存",
        "copy_labels": "复制标注",
        "output_format": "输出格式",
        "classes": "类别",
        "ready": "就绪",
        "no_classes_file": "此文件夹下找不到 classes.txt，是否选择已有 txt 文件作为类别文件？\n选择是可选文件，选择否可手动输入类别。",
        "yes": "是",
        "no": "否",
        "hotkeys_help": "快捷键说明",
        "quit": "退出",
        "file_menu": "文件(&F)",
        "load_label_file": "加载标签文件 (例如：.txt)",
        "help_menu": "帮助(&H)",
        "hotkeys_menu": "快捷键(&K)",
        "about_menu": "关于 BakuAI AS(&A)",
        "hotkeys_title": "快捷键说明",
        "hotkeys_list": "快捷键列表：",
        "next_image_hotkey": "下一张图片",
        "prev_image_hotkey": "上一张图片",
        "save_hotkey": "保存标注",
        "open_hotkey": "打开文件夹",
        "delete_hotkey": "删除选中框",
        "quit_hotkey": "退出",
        "copy_labels_hotkey": "切换复制标注",
        "multi_select_hotkey": "多选标注框",
        "undo_hotkey": "撤销 (每张图片独立)",
        "redo_hotkey": "重做 (每张图片独立)",
        "about_title": "关于 BakuAI",
        "version": "版本",
        "copyright": "版权所有",
        "website": "官方网站",
        "software_description": "本软件用于图像标注，支持 YOLO/VOC/COCO 格式",
        "debug_info": "调试信息",
        "current_working_dir": "当前工作目录",
        "dist_dir_contents": "Dist 目录内容",
        "debug_info_end": "调试信息结束",
        "class_save_failed": "类别保存失败",
        "failed_to_write_classes": "无法写入 classes.txt：{0}",
        "save_error": "保存错误",
        "failed_to_save_yolo": "保存 YOLO 标注失败：\n{0}",
        "edit_label": "编辑标签",
        "enter_new_label": "输入新标签：",
        "data_aug_menu": "数据增强(&D)",
        "data_aug_title": "数据增强设置",
        "saturation": "饱和度",
        "contrast": "对比度",
        "brightness": "亮度",
        "rotation": "旋转角度",
        "flip_ud": "上下翻转",
        "flip_lr": "左右翻转",
        "apply": "应用",
        "cancel": "取消",
        "reset": "重置",
        "preview": "预览",
        "flip_options": "翻转选项",
        "warning": "警告",
        "no_image": "尚未加载图片，请先打开图片。",
    },
    "ja": {
        "open": "開く",
        "save": "保存",
        "next_image": "次の画像",
        "prev_image": "前の画像",
        "delete": "選択したボックスを削除",
        "auto_save": "自動保存",
        "copy_labels": "ラベルをコピー",
        "output_format": "出力形式",
        "classes": "クラス",
        "ready": "準備完了",
        "no_classes_file": "このフォルダに classes.txt が見つかりません。\n既存の txt ファイルをクラスファイルとして選択しますか？\nはいを選択するとファイルを選択、いいえで手動入力。",
        "yes": "はい",
        "no": "いいえ",
        "hotkeys_help": "ショートカット一覧",
        "quit": "終了",
        "software_description": "このソフトウェアは画像ラベリング用で、YOLO/VOC/COCO 形式をサポートしています",
        "debug_info": "デバッグ情報",
        "current_working_dir": "現在の作業ディレクトリ",
        "dist_dir_contents": "Dist ディレクトリの内容",
        "debug_info_end": "デバッグ情報終了",
        "class_save_failed": "クラス保存失敗",
        "failed_to_write_classes": "classes.txt の書き込みに失敗：{0}",
        "save_error": "保存エラー",
        "failed_to_save_yolo": "YOLO アノテーションの保存に失敗：\n{0}",
        "edit_label": "ラベルを編集",
        "enter_new_label": "新しいラベルを入力：",
    },
    "it": {
        "open": "Apri",
        "save": "Salva",
        "next_image": "Immagine successiva",
        "prev_image": "Immagine precedente",
        "delete": "Elimina box selezionati",
        "auto_save": "Salvataggio automatico",
        "copy_labels": "Copia etichette",
        "output_format": "Formato di output",
        "classes": "Classi",
        "ready": "Pronto",
        "no_classes_file": "Nessun classes.txt trovato in questa cartella.\nVuoi selezionare un file txt esistente come file delle classi?\nSì per selezionare, No per inserire manualmente.",
        "yes": "Sì",
        "no": "No",
        "hotkeys_help": "Scorciatoie da tastiera",
        "quit": "Esci",
        "file_menu": "File(&F)",
        "load_label_file": "Carica file etichette (es. .txt)",
        "help_menu": "Aiuto(&H)",
        "hotkeys_menu": "Scorciatoie(&K)",
        "about_menu": "Informazioni su BakuAI AS(&A)",
        "hotkeys_title": "Scorciatoie da tastiera",
        "hotkeys_list": "Elenco scorciatoie：",
        "next_image_hotkey": "Immagine successiva",
        "prev_image_hotkey": "Immagine precedente",
        "save_hotkey": "Salva annotazioni",
        "open_hotkey": "Apri directory",
        "delete_hotkey": "Elimina box selezionati",
        "quit_hotkey": "Esci",
        "copy_labels_hotkey": "Cambia copia etichette",
        "multi_select_hotkey": "Selezione multipla box",
        "undo_hotkey": "Annulla (per immagine)",
        "redo_hotkey": "Ripeti (per immagine)",
        "about_title": "Informazioni su BakuAI",
        "version": "Versione",
        "copyright": "Tutti i diritti riservati",
        "website": "Sito ufficiale",
        "software_description": "Questo software è utilizzato per l'etichettatura delle immagini, supporta i formati YOLO/VOC/COCO",
        "debug_info": "Informazioni di debug",
        "current_working_dir": "Directory di lavoro corrente",
        "dist_dir_contents": "Contenuto directory Dist",
        "debug_info_end": "Fine informazioni di debug",
        "class_save_failed": "Salvataggio classe fallito",
        "failed_to_write_classes": "Impossibile scrivere classes.txt: {0}",
        "save_error": "Errore di salvataggio",
        "failed_to_save_yolo": "Impossibile salvare le annotazioni YOLO:\n{0}",
        "edit_label": "Modifica etichetta",
        "enter_new_label": "Inserisci nuova etichetta:",
        "data_aug_menu": "Aumento Dati(&D)",
        "data_aug_title": "Impostazioni Batch di Data Augmentation",
        "saturation": "Saturazione",
        "contrast": "Contrasto",
        "brightness": "Luminosità",
        "rotation": "Rotazione (gradi)",
        "flip_ud": "Ribalta Su-Giù",
        "flip_lr": "Ribalta Sinistra-Destra",
        "apply": "Applica",
        "cancel": "Annulla",
        "reset": "Resetta",
        "preview": "Anteprima",
        "aug_per_image": "Augmentazioni per immagine",
        "output_folder": "Cartella di Output",
        "select_folder": "Seleziona Cartella",
        "processing": "Elaborazione in corso...",
        "aug_complete": "Augmentazione completata",
        "aug_progress": "Elaborazione {0}/{1} immagini",
        "aug_success": "{0} immagini aumentate con successo",
        "aug_error": "Errore durante l'augmentazione",
        "preserve_original": "Conserva i file originali",
        "aug_settings": "Impostazioni di Augmentazione",
        "random_rotation": "Intervallo Rotazione Casuale",
        "random_brightness": "Intervallo Luminosità Casuale",
        "random_contrast": "Intervallo Contrasto Casuale",
        "random_saturation": "Intervallo Saturazione Casuale",
        "flip_options": "Opzioni di Ribaltamento",
        "warning": "Avviso",
        "no_image": "Nessuna immagine caricata. Apri prima un'immagine.",
    },
    "de": {
        "open": "Öffnen",
        "save": "Speichern",
        "next_image": "Nächstes Bild",
        "prev_image": "Vorheriges Bild",
        "delete": "Ausgewählte Box(en) löschen",
        "auto_save": "Automatisch speichern",
        "copy_labels": "Labels kopieren",
        "output_format": "Ausgabeformat",
        "classes": "Klassen",
        "ready": "Bereit",
        "no_classes_file": "Keine classes.txt in diesem Ordner gefunden.\nMöchten Sie eine vorhandene txt-Datei als Klassen-Datei auswählen?\nJa zum Auswählen, Nein zum manuellen Eingeben.",
        "yes": "Ja",
        "no": "Nein",
        "hotkeys_help": "Tastenkombinationen",
        "quit": "Beenden",
        "file_menu": "Datei(&F)",
        "load_label_file": "Label-Datei laden (z.B. .txt)",
        "help_menu": "Hilfe(&H)",
        "hotkeys_menu": "Tastenkombinationen(&K)",
        "about_menu": "Über BakuAI AS(&A)",
        "hotkeys_title": "Tastenkombinationen",
        "hotkeys_list": "Tastenkombinationen Liste：",
        "next_image_hotkey": "Nächstes Bild",
        "prev_image_hotkey": "Vorheriges Bild",
        "save_hotkey": "Anmerkungen speichern",
        "open_hotkey": "Verzeichnis öffnen",
        "delete_hotkey": "Ausgewählte Box(en) löschen",
        "quit_hotkey": "Beenden",
        "copy_labels_hotkey": "Labels kopieren umschalten",
        "multi_select_hotkey": "Mehrfachauswahl Boxen",
        "undo_hotkey": "Rückgängig (pro Bild)",
        "redo_hotkey": "Wiederholen (pro Bild)",
        "about_title": "Über BakuAI",
        "version": "Version",
        "copyright": "Alle Rechte vorbehalten",
        "website": "Offizielle Website",
        "software_description": "Diese Software wird für die Bildbeschriftung verwendet und unterstützt YOLO/VOC/COCO-Formate",
        "debug_info": "Debug-Informationen",
        "current_working_dir": "Aktuelles Arbeitsverzeichnis",
        "dist_dir_contents": "Dist-Verzeichnisinhalt",
        "debug_info_end": "Debug-Informationen Ende",
        "class_save_failed": "Klassenspeicherung fehlgeschlagen",
        "failed_to_write_classes": "Konnte classes.txt nicht schreiben: {0}",
        "save_error": "Speicherfehler",
        "failed_to_save_yolo": "YOLO-Annotationen konnten nicht gespeichert werden:\n{0}",
        "edit_label": "Beschriftung bearbeiten",
        "enter_new_label": "Neue Beschriftung eingeben:",
        "data_aug_menu": "Datenaugmentation(&D)",
        "data_aug_title": "Batch-Datenaugmentation",
        "saturation": "Sättigung",
        "contrast": "Kontrast",
        "brightness": "Helligkeit",
        "rotation": "Rotation (Grad)",
        "flip_ud": "Vertikal spiegeln",
        "flip_lr": "Horizontal spiegeln",
        "apply": "Anwenden",
        "cancel": "Abbrechen",
        "reset": "Zurücksetzen",
        "preview": "Vorschau",
        "aug_per_image": "Augmentierungen pro Bild",
        "output_folder": "Ausgabeordner",
        "select_folder": "Ordner auswählen",
        "processing": "Verarbeitung...",
        "aug_complete": "Augmentation abgeschlossen",
        "aug_progress": "Verarbeite {0}/{1} Bilder",
        "aug_success": "{0} Bilder erfolgreich augmentiert",
        "aug_error": "Fehler bei der Augmentation",
        "preserve_original": "Originaldateien behalten",
        "aug_settings": "Augmentierungseinstellungen",
        "random_rotation": "Zufälliger Rotationsbereich",
        "random_brightness": "Zufälliger Helligkeitsbereich",
        "random_contrast": "Zufälliger Kontrastbereich",
        "random_saturation": "Zufälliger Sättigungsbereich",
        "flip_options": "Spiegelungsoptionen",
        "warning": "Warnung",
        "no_image": "Kein Bild geladen. Bitte zuerst ein Bild öffnen.",
    },
    "no": {
        "open": "Åpne",
        "save": "Lagre",
        "next_image": "Neste bilde",
        "prev_image": "Forrige bilde",
        "delete": "Slett valgte bokser",
        "auto_save": "Auto-lagring",
        "copy_labels": "Kopier etiketter",
        "output_format": "Utdataformat",
        "classes": "Klasser",
        "ready": "Klar",
        "no_classes_file": "Fant ikke classes.txt i denne mappen.\nVil du velge en eksisterende txt-fil som klassefil?\nJa for å velge, Nei for å skrive inn manuelt.",
        "yes": "Ja",
        "no": "Nei",
        "hotkeys_help": "Hurtigtaster",
        "quit": "Avslutt",
        "file_menu": "Fil(&F)",
        "load_label_file": "Last inn merkelappfil (f.eks. .txt)",
        "help_menu": "Hjelp(&H)",
        "hotkeys_menu": "Hurtigtaster(&K)",
        "about_menu": "Om BakuAI AS(&A)",
        "hotkeys_title": "Hurtigtaster",
        "hotkeys_list": "Hurtigtastliste：",
        "next_image_hotkey": "Neste bilde",
        "prev_image_hotkey": "Forrige bilde",
        "save_hotkey": "Lagre merknader",
        "open_hotkey": "Åpne mappe",
        "delete_hotkey": "Slett valgte bokser",
        "quit_hotkey": "Avslutt",
        "copy_labels_hotkey": "Bytt kopiering av etiketter",
        "multi_select_hotkey": "Flervalg av bokser",
        "undo_hotkey": "Angre (per bilde)",
        "redo_hotkey": "Gjør om (per bilde)",
        "about_title": "Om BakuAI",
        "version": "Versjon",
        "copyright": "Alle rettigheter reservert",
        "website": "Offisiell nettside",
        "software_description": "Dette programvaren brukes til bildemerkering, støtter YOLO/VOC/COCO-formater",
        "debug_info": "Feilsøkingsinformasjon",
        "current_working_dir": "Nåværende arbeidsmappe",
        "dist_dir_contents": "Dist-mappeinnhold",
        "debug_info_end": "Feilsøkingsinformasjon slutt",
        "class_save_failed": "Klasselagring mislyktes",
        "failed_to_write_classes": "Kunne ikke skrive classes.txt: {0}",
        "save_error": "Lagringsfeil",
        "failed_to_save_yolo": "Kunne ikke lagre YOLO-annotasjoner:\n{0}",
        "edit_label": "Rediger etikett",
        "enter_new_label": "Skriv inn ny etikett:",
        "data_aug_menu": "Dataforsterkning(&D)",
        "data_aug_title": "Batch Dataforsterkning",
        "saturation": "Metning",
        "contrast": "Kontrast",
        "brightness": "Lysstyrke",
        "rotation": "Rotasjon (grader)",
        "flip_ud": "Vend opp-ned",
        "flip_lr": "Vend venstre-høyre",
        "apply": "Bruk",
        "cancel": "Avbryt",
        "reset": "Tilbakestill",
        "preview": "Forhåndsvisning",
        "aug_per_image": "Forsterkninger per bilde",
        "output_folder": "Utdata-mappe",
        "select_folder": "Velg mappe",
        "processing": "Behandler...",
        "aug_complete": "Forsterkning fullført",
        "aug_progress": "Behandler {0}/{1} bilder",
        "aug_success": "{0} bilder forsterket",
        "aug_error": "Feil under forsterkning",
        "preserve_original": "Behold originalfiler",
        "aug_settings": "Forsterkningsinnstillinger",
        "random_rotation": "Tilfeldig rotasjonsområde",
        "random_brightness": "Tilfeldig lysstyrkeområde",
        "random_contrast": "Tilfeldig kontrastområde",
        "random_saturation": "Tilfeldig metningsområde",
        "flip_options": "Vendingsvalg",
        "warning": "Advarsel",
        "no_image": "Ingen bilde lastet. Vennligst åpne et bilde først.",
    },
    "es": {
        "open": "Abrir",
        "save": "Guardar",
        "next_image": "Siguiente imagen",
        "prev_image": "Imagen anterior",
        "delete": "Eliminar cajas seleccionadas",
        "auto_save": "Guardado automático",
        "copy_labels": "Copiar etiquetas",
        "output_format": "Formato de salida",
        "classes": "Clases",
        "ready": "Listo",
        "no_classes_file": "No se encontró classes.txt en esta carpeta.\n¿Desea seleccionar un archivo txt existente como archivo de clases?\nSí para seleccionar, No para ingresar manualmente.",
        "yes": "Sí",
        "no": "No",
        "hotkeys_help": "Ayuda de atajos",
        "quit": "Salir",
        "file_menu": "Archivo(&F)",
        "load_label_file": "Cargar archivo de etiquetas (ej. .txt)",
        "help_menu": "Ayuda(&H)",
        "hotkeys_menu": "Atajos(&K)",
        "about_menu": "Acerca de BakuAI AS(&A)",
        "hotkeys_title": "Ayuda de atajos",
        "hotkeys_list": "Lista de atajos：",
        "next_image_hotkey": "Siguiente imagen",
        "prev_image_hotkey": "Imagen anterior",
        "save_hotkey": "Guardar anotaciones",
        "open_hotkey": "Abrir directorio",
        "delete_hotkey": "Eliminar cajas seleccionadas",
        "quit_hotkey": "Salir",
        "copy_labels_hotkey": "Cambiar copia de etiquetas",
        "multi_select_hotkey": "Selección múltiple de cajas",
        "undo_hotkey": "Deshacer (por imagen)",
        "redo_hotkey": "Rehacer (por imagen)",
        "about_title": "Acerca de BakuAI",
        "version": "Versión",
        "copyright": "Todos los derechos reservados",
        "website": "Sitio web oficial",
        "software_description": "Este software se utiliza para el etiquetado de imágenes, compatible con formatos YOLO/VOC/COCO",
        "debug_info": "Información de depuración",
        "current_working_dir": "Directorio de trabajo actual",
        "dist_dir_contents": "Contenido del directorio Dist",
        "debug_info_end": "Fin de información de depuración",
        "class_save_failed": "Error al guardar clase",
        "failed_to_write_classes": "No se pudo escribir classes.txt: {0}",
        "save_error": "Error de guardado",
        "failed_to_save_yolo": "Error al guardar anotaciones YOLO:\n{0}",
        "edit_label": "Editar etiqueta",
        "enter_new_label": "Ingrese nueva etiqueta:",
        "data_aug_menu": "Aumento de Datos(&D)",
        "data_aug_title": "Ajustes de Aumento por Lote",
        "saturation": "Saturación",
        "contrast": "Contraste",
        "brightness": "Brillo",
        "rotation": "Rotación (grados)",
        "flip_ud": "Voltear Arriba-Abajo",
        "flip_lr": "Voltear Izquierda-Derecha",
        "apply": "Aplicar",
        "cancel": "Cancelar",
        "reset": "Restablecer",
        "preview": "Vista previa",
        "aug_per_image": "Aumentos por imagen",
        "output_folder": "Carpeta de salida",
        "select_folder": "Seleccionar carpeta",
        "processing": "Procesando...",
        "aug_complete": "Aumento completado",
        "aug_progress": "Procesando {0}/{1} imágenes",
        "aug_success": "{0} imágenes aumentadas correctamente",
        "aug_error": "Error durante el aumento",
        "preserve_original": "Conservar archivos originales",
        "aug_settings": "Configuración de aumento",
        "random_rotation": "Rango de rotación aleatoria",
        "random_brightness": "Rango de brillo aleatorio",
        "random_contrast": "Rango de contraste aleatorio",
        "random_saturation": "Rango de saturación aleatoria",
        "flip_options": "Opciones de volteo",
        "warning": "Advertencia",
        "no_image": "No se ha cargado ninguna imagen. Por favor, abra una imagen primero.",
    },
    "fr": {
        "open": "Ouvrir",
        "save": "Enregistrer",
        "next_image": "Image suivante",
        "prev_image": "Image précédente",
        "delete": "Supprimer les boîtes sélectionnées",
        "auto_save": "Enregistrement auto",
        "copy_labels": "Copier les étiquettes",
        "output_format": "Format de sortie",
        "classes": "Classes",
        "ready": "Prêt",
        "no_classes_file": "Aucun classes.txt trouvé dans ce dossier.\nVoulez-vous sélectionner un fichier txt existant comme fichier de classes ?\nOui pour sélectionner, Non pour saisir manuellement.",
        "yes": "Oui",
        "no": "Non",
        "hotkeys_help": "Raccourcis clavier",
        "quit": "Quitter",
        "file_menu": "Fichier(&F)",
        "load_label_file": "Charger fichier d'étiquettes (ex. .txt)",
        "help_menu": "Aide(&H)",
        "hotkeys_menu": "Raccourcis(&K)",
        "about_menu": "À propos de BakuAI AS(&A)",
        "hotkeys_title": "Raccourcis clavier",
        "hotkeys_list": "Liste des raccourcis：",
        "next_image_hotkey": "Image suivante",
        "prev_image_hotkey": "Image précédente",
        "save_hotkey": "Enregistrer les annotations",
        "open_hotkey": "Ouvrir le répertoire",
        "delete_hotkey": "Supprimer les boîtes sélectionnées",
        "quit_hotkey": "Quitter",
        "copy_labels_hotkey": "Basculer la copie des étiquettes",
        "multi_select_hotkey": "Sélection multiple de boîtes",
        "undo_hotkey": "Annuler (par image)",
        "redo_hotkey": "Rétablir (par image)",
        "about_title": "À propos de BakuAI",
        "version": "Version",
        "copyright": "Tous droits réservés",
        "website": "Site web officiel",
        "software_description": "Ce logiciel est utilisé pour l'étiquetage d'images, prend en charge les formats YOLO/VOC/COCO",
        "debug_info": "Informations de débogage",
        "current_working_dir": "Répertoire de travail actuel",
        "dist_dir_contents": "Contenu du répertoire Dist",
        "debug_info_end": "Fin des informations de débogage",
        "class_save_failed": "Échec de l'enregistrement de la classe",
        "failed_to_write_classes": "Impossible d'écrire classes.txt : {0}",
        "save_error": "Erreur de sauvegarde",
        "failed_to_save_yolo": "Échec de l'enregistrement des annotations YOLO :\n{0}",
        "edit_label": "Modifier l'étiquette",
        "enter_new_label": "Entrez une nouvelle étiquette :",
        "data_aug_menu": "Augmentation de Données(&D)",
        "data_aug_title": "Paramètres d'Augmentation par Lot",
        "saturation": "Saturation",
        "contrast": "Contraste",
        "brightness": "Luminosité",
        "rotation": "Rotation (degrés)",
        "flip_ud": "Retourner Haut-Bas",
        "flip_lr": "Retourner Gauche-Droite",
        "apply": "Appliquer",
        "cancel": "Annuler",
        "reset": "Réinitialiser",
        "preview": "Aperçu",
        "aug_per_image": "Augmentations par image",
        "output_folder": "Dossier de sortie",
        "select_folder": "Sélectionner le dossier",
        "processing": "Traitement...",
        "aug_complete": "Augmentation terminée",
        "aug_progress": "Traitement de {0}/{1} images",
        "aug_success": "{0} images augmentées avec succès",
        "aug_error": "Erreur lors de l'augmentation",
        "preserve_original": "Conserver les fichiers originaux",
        "aug_settings": "Paramètres d'augmentation",
        "random_rotation": "Plage de rotation aléatoire",
        "random_brightness": "Plage de luminosité aléatoire",
        "random_contrast": "Plage de contraste aléatoire",
        "random_saturation": "Plage de saturation aléatoire",
        "flip_options": "Options de retournement",
        "warning": "Avertissement",
        "no_image": "Aucune image chargée. Veuillez d'abord ouvrir une image.",
    }
}

# 自動偵測系統語言
def get_system_language():
    try:
        # 嘗試多種方法獲取系統語言
        if sys.platform == 'darwin':  # macOS
            import subprocess
            try:
                lang = subprocess.check_output(['defaults', 'read', 'NSGlobalDomain', 'AppleLanguages']).decode('utf-8')
                if 'zh-Hant' in lang or 'zh_TW' in lang:
                    return 'zh-tw'
                elif 'zh-Hans' in lang or 'zh_CN' in lang:
                    return 'zh-cn'
            except:
                pass
        elif sys.platform == 'win32':  # Windows
            import ctypes
            windll = ctypes.windll.kernel32
            try:
                lang = locale.windows_locale[windll.GetUserDefaultUILanguage()]
                if lang.startswith('zh_TW'):
                    return 'zh-tw'
                elif lang.startswith('zh_CN'):
                    return 'zh-cn'
            except:
                pass
        
        # 如果上述方法都失敗，使用 locale
        try:
            lang = locale.getdefaultlocale()[0]
            if lang:
                if lang.startswith('zh_TW'):
                    return 'zh-tw'
                elif lang.startswith('zh_CN'):
                    return 'zh-cn'
                elif lang.startswith('ja'):
                    return 'ja'
                elif lang.startswith('it'):
                    return 'it'
                elif lang.startswith('de'):
                    return 'de'
                elif lang.startswith('no') or lang.startswith('nb') or lang.startswith('nn'):
                    return 'no'
                elif lang.startswith('ko'):
                    return 'ko'
                elif lang.startswith('es'):
                    return 'es'
                elif lang.startswith('fr'):
                    return 'fr'
        except:
            pass
            
        # 如果所有方法都失敗，檢查環境變數
        try:
            lang = os.environ.get('LANG', '').split('.')[0]
            if lang.startswith('zh_TW'):
                return 'zh-tw'
            elif lang.startswith('zh_CN'):
                return 'zh-cn'
        except:
            pass
            
    except Exception as e:
        print(f"Language detection error: {str(e)}")
    
    return 'en'  # 預設使用英文

# 使用新的語言偵測函數
LANG = get_system_language()

def tr(key):
    return TRANSLATIONS.get(LANG, TRANSLATIONS['en']).get(key, key)

# 之後所有 UI 文字都用 tr("key") 取代

    # 在文件顶部添加虚线绘制函数
def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_length=10):
    """绘制虚线矩形"""
    x1, y1 = pt1
    x2, y2 = pt2
    # 顶部水平线
    for x in range(x1, x2, dash_length * 2):
        end_x = min(x + dash_length, x2)
        cv2.line(img, (x, y1), (end_x, y1), color, thickness)

    # 底部水平线
    for x in range(x1, x2, dash_length * 2):
        end_x = min(x + dash_length, x2)
        cv2.line(img, (x, y2), (end_x, y2), color, thickness)

    # 左侧垂直线
    for y in range(y1, y2, dash_length * 2):
        end_y = min(y + dash_length, y2)
        cv2.line(img, (x1, y), (x1, end_y), color, thickness)

    # 右侧垂直线
    for y in range(y1, y2, dash_length * 2):
        end_y = min(y + dash_length, y2)
        cv2.line(img, (x2, y), (x2, end_y), color, thickness)


class BoundingBox:
    def __init__(self, x, y, w, h, label=""):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.label = label

    def get_resize_handle(self, point, threshold=10):
        """Enhanced handle detection with visual size consideration"""
        handles = {
            'top-left': (self.x, self.y),
            'top-right': (self.x + self.w, self.y),
            'bottom-left': (self.x, self.y + self.h),
            'bottom-right': (self.x + self.w, self.y + self.h)
        }
        for handle_name, (hx, hy) in handles.items():
            if (abs(point.x() - hx) < threshold and
                abs(point.y() - hy) < threshold):
                return handle_name
        return None
        
    def contains(self, point, buffer=5):
        """Check if point is inside bbox with optional buffer zone"""
        # First check if point is fully inside the box
        if (self.x + buffer <= point.x() <= self.x + self.w - buffer and
            self.y + buffer <= point.y() <= self.y + self.h - buffer):
            return True
        # Then check if point is in buffer zone near edges
        return (self.x - buffer <= point.x() <= self.x + self.w + buffer and
                self.y - buffer <= point.y() <= self.y + self.h + buffer)
    
    def get_resize_handle(self, point, threshold=8):
        handles = {
            'top-left': (self.x, self.y),
            'top-right': (self.x + self.w, self.y),
            'bottom-left': (self.x, self.y + self.h),
            'bottom-right': (self.x + self.w, self.y + self.h)
        }
        for handle_name, (hx, hy) in handles.items():
            if abs(point.x() - hx) < threshold and abs(point.y() - hy) < threshold:
                return handle_name
        return None

class MagnifierWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.magnifier_size = 200
        self.zoom_factor = 2.0
        self.setFixedSize(self.magnifier_size, self.magnifier_size)
        
        # 設置半透明背景
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 180);
                border: 2px solid #666;
                border-radius: 100px;
            }
        """)
        
        # 初始化計時器用於更新位置
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.updatePosition)
        self.update_timer.start(16)  # 約60fps
        
    def updatePosition(self):
        if self.parent():
            cursor_pos = QCursor.pos()
            # 計算放大鏡視窗的位置，使其跟隨滑鼠但不會超出螢幕
            screen = QApplication.primaryScreen().geometry()
            x = cursor_pos.x() + 20
            y = cursor_pos.y() + 20
            
            # 確保放大鏡不會超出螢幕邊界
            if x + self.magnifier_size > screen.right():
                x = cursor_pos.x() - self.magnifier_size - 20
            if y + self.magnifier_size > screen.bottom():
                y = cursor_pos.y() - self.magnifier_size - 20
                
            self.move(x, y)
            self.update()  # 強制重繪
            
    def paintEvent(self, event):
        if not self.parent():
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 獲取父視窗
        parent = self.parent()
        if not parent.current_image is None:
            # 獲取滑鼠在圖片上的實際位置
            cursor_pos = QCursor.pos()
            image_pos = parent.image_label.mapFromGlobal(cursor_pos)
            scaled_pos = parent.getScaledPoint(image_pos)
            
            # 計算源圖像的區域
            source_size = int(self.magnifier_size / self.zoom_factor)
            source_x = max(0, int(scaled_pos.x() - source_size // 2))
            source_y = max(0, int(scaled_pos.y() - source_size // 2))
            
            # 確保不超出圖像邊界
            h, w = parent.current_image.shape[:2]
            source_x = min(source_x, w - source_size)
            source_y = min(source_y, h - source_size)
            
            # 截取並放大圖像區域
            roi = parent.current_image[source_y:source_y + source_size, 
                                     source_x:source_x + source_size]
            if roi.size > 0:
                # 轉換為RGB並創建QImage
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                q_img = QImage(roi_rgb.data, roi_rgb.shape[1], roi_rgb.shape[0],
                             roi_rgb.strides[0], QImage.Format_RGB888)
                
                # 繪製放大後的圖像
                painter.drawImage(self.rect(), q_img)
                
                # 繪製十字準線
                painter.setPen(QPen(QColor(255, 0, 0), 1))
                center = self.rect().center()
                painter.drawLine(center.x(), 0, center.x(), self.height())
                painter.drawLine(0, center.y(), self.width(), center.y())
                
                # 繪製當前正在繪製的框
                if parent.drawing and parent.bbox_start and parent.bbox_end:
                    # 計算框在放大區域中的相對位置
                    x1 = int((parent.bbox_start.x() - source_x) * self.zoom_factor)
                    y1 = int((parent.bbox_start.y() - source_y) * self.zoom_factor)
                    x2 = int((parent.bbox_end.x() - source_x) * self.zoom_factor)
                    y2 = int((parent.bbox_end.y() - source_y) * self.zoom_factor)
                    
                    # 繪製虛線框
                    pen = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                
                # 繪製現有的框
                for bbox in parent.bboxes:
                    # 計算框在放大區域中的相對位置
                    x = int((bbox.x - source_x) * self.zoom_factor)
                    y = int((bbox.y - source_y) * self.zoom_factor)
                    w = int(bbox.w * self.zoom_factor)
                    h = int(bbox.h * self.zoom_factor)
                    
                    # 如果框在放大區域內
                    if (x + w > 0 and x < self.magnifier_size and 
                        y + h > 0 and y < self.magnifier_size):
                        # 使用框的標籤顏色
                        color = parent.get_label_color(bbox.label)
                        pen = QPen(QColor(*color), 2)
                        if bbox == parent.selected_bbox:
                            pen.setStyle(Qt.DashLine)
                        painter.setPen(pen)
                        painter.drawRect(x, y, w, h)
                        
                        # 如果是選中的框，繪製調整手柄
                        if bbox == parent.selected_bbox:
                            handle_size = int(parent.handle_visual_size * self.zoom_factor)
                            for handle in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
                                painter.fillRect(
                                    int(handle[0] - handle_size/2),
                                    int(handle[1] - handle_size/2),
                                    handle_size,
                                    handle_size,
                                    QColor(0, 255, 255)
                                )

class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化放大鏡相關屬性
        self.magnifier_enabled = True  # 預設啟用
        self.magnifier = None  # 將在 initUI 中初始化
        self.magnifier_active = False  # 控制放大鏡是否應該顯示
        self.min_box_size = 30  # Adjust this value as needed
        self.handle_visual_size = 6  # 新增配置變量
        self.handle_detection_threshold = 8  # 新增檢測閾值
        self.label_font_scale = 0.4  # 控制bounding box文字大小
        self.cursor_pos = None
        # Add a class color mapping dictionary
        self.label_colors = {}
        self.autosave = True  # This line should exist in your __init__
        self.image_list = []
        self.current_index = -1
        self.current_image = None
        self.bboxes = []
        self.selected_bbox = None
        self.drawing = False
        self.dragging = False
        self.resize_handle = None
        self.bbox_start = None
        self.bbox_end = None
        self.scale_factor = 1.0
        self.autosave = True
        self.classes = []  # 初始化為空列表
        self.label_cache = {}
        self.viewed_indices = set()
        self.total_images = 0
        self.handle_visual_size = 6
        self.handle_detection_threshold = 10
        self.min_box_size = 15
        self.drag_threshold = 5
        self.has_dragged = False
        self.original_bbox = None
        self.selected_bboxes = set()  # 多框選取集合
        # Undo/Redo stacks
        self.undo_stack = {}  # {image_index: [bboxes_list, ...]}
        self.redo_stack = {}  # {image_index: [bboxes_list, ...]}

        self.initUI()
        self.loadLogo()

    def initUI(self):
        self.setWindowTitle('BakuLabel Tool')
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 初始化放大鏡視窗
        self.magnifier = MagnifierWindow(self)
        self.magnifier.hide()
        
        # 先初始化 image_label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.showContextMenu)
        self.image_label.setMinimumSize(100, 100)
        # 將滑鼠事件掛載到 image_label
        self.image_label.mousePressEvent = self.mousePressEvent
        self.image_label.mouseMoveEvent = self.mouseMoveEvent
        self.image_label.mouseReleaseEvent = self.mouseReleaseEvent
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)  # 更明顯的分隔條
        splitter.setChildrenCollapsible(False)  # 防止子部件被完全收縮
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #eee, stop:1 #ccc);
                border: 1px solid #777;
                width: 10px;
                height: 10px;
            }
        """)

        # 创建主菜单
        self.createMainMenu()

        # 左侧面板
        left_panel = QWidget()
        control_layout = QVBoxLayout(left_panel)

        # Output Format
        format_label = QLabel(tr("output_format"))
        format_label.setStyleSheet("font-weight: bold; font-size: 13px; margin-bottom: 2px;")
        control_layout.addWidget(format_label)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["YOLO", "VOC", "COCO"])
        self.format_combo.setCurrentText("YOLO")
        control_layout.addWidget(self.format_combo)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.onFileSelected)
        control_layout.addWidget(self.file_list)

        btn_layout = QHBoxLayout()
        open_btn = QPushButton(tr("open"))
        open_btn.clicked.connect(self.openDirectory)
        btn_layout.addWidget(open_btn)
        save_btn = QPushButton(tr("save"))
        save_btn.clicked.connect(self.saveAnnotations)
        btn_layout.addWidget(save_btn)
        control_layout.addLayout(btn_layout)

        # Classes（移到Open/Save下方）
        class_label = QLabel(tr("classes"))
        class_label.setStyleSheet("font-weight: bold; font-size: 13px; margin-bottom: 2px;")
        control_layout.addWidget(class_label)
        class_combo_layout = QHBoxLayout()
        self.class_combo = QComboBox()
        self.class_combo.setEditable(True)
        # 移除這裡的 loadClasses() 呼叫
        class_combo_layout.addWidget(self.class_combo)
        class_load_btn = QPushButton("...")
        class_load_btn.setFixedWidth(28)
        class_load_btn.setToolTip("Load custom classes.txt")
        class_load_btn.clicked.connect(self.loadClassesFromFile)
        class_combo_layout.addWidget(class_load_btn)
        control_layout.addLayout(class_combo_layout)
        self.class_combo.lineEdit().editingFinished.connect(self.syncClassComboToClasses)

        # 狀態列與 Auto Save/Copy Labels 放在同一個 QFrame 框中
        status_options_frame = QFrame()
        status_options_frame.setFrameShape(QFrame.StyledPanel)
        status_options_frame.setStyleSheet("QFrame { border: 1px solid #bbb; border-radius: 4px; margin-top: 8px; margin-bottom: 8px; }")
        status_options_layout = QVBoxLayout(status_options_frame)
        status_options_layout.setContentsMargins(8, 4, 8, 4)
        status_options_layout.setSpacing(6)
        self.status_label = QLabel(tr("ready"))
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 2px 8px; background: transparent; border: none;")
        status_options_layout.addWidget(self.status_label)
        options_layout = QHBoxLayout()
        options_layout.setSpacing(12)
        autosave_checkbox = QCheckBox(tr("auto_save"))
        autosave_checkbox.setChecked(self.autosave)
        autosave_checkbox.stateChanged.connect(self.toggleAutosave)
        options_layout.addWidget(autosave_checkbox)
        self.copy_checkbox = QCheckBox(tr("copy_labels"))
        self.copy_checkbox.setChecked(False)
        options_layout.addWidget(self.copy_checkbox)
        status_options_layout.addLayout(options_layout)
        control_layout.addWidget(status_options_frame)

        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.image_label, stretch=1)

        # Add a color legend panel
        legend_frame = QFrame()
        legend_frame.setFrameShape(QFrame.StyledPanel)
        legend_layout = QVBoxLayout(legend_frame)
        legend_layout.setContentsMargins(5, 5, 5, 5)
        legend_layout.setSpacing(2)
        legend_title = QLabel("Color Legend")
        legend_title.setMaximumHeight(20)
        legend_layout.addWidget(legend_title)
        self.legend_list = QListWidget()
        self.legend_list.setMaximumHeight(200)
        self.legend_list.status_level = 1
        self.legend_list.setStyleSheet("QListWidget::item { height: 18px; }")
        legend_layout.addWidget(self.legend_list)
        control_layout.addWidget(legend_frame)

        self.statusBar = self.statusBar()
        self.statusBar.showMessage('Ready')

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        quit_btn = QPushButton(tr("quit"))
        quit_btn.clicked.connect(self.close)
        quit_btn.setFixedSize(80, 30)
        button_layout.addWidget(quit_btn)
        right_layout.addLayout(button_layout)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 900])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        left_panel.setMinimumWidth(200)
        right_panel.setMinimumWidth(400)
        splitter.splitterMoved.connect(self.handleSplitterMoved)

        main_layout = QHBoxLayout(main_widget)
        main_layout.addWidget(splitter)

        # 快捷键设置
        shortcuts = [
            (QKeySequence("F"), self.nextImage),
            (QKeySequence("D"), self.prevImage),
            (QKeySequence("S"), self.saveAnnotations),
            (QKeySequence("O"), self.openDirectory),
            (QKeySequence("E"), self.deleteSelectedBox),
            (QKeySequence("Q"), self.close),
            (QKeySequence(Qt.Key_Delete), self.deleteSelectedBox),
            (QKeySequence(Qt.Key_Up), self.prevImage),
            (QKeySequence(Qt.Key_Down), self.nextImage),
            (QKeySequence("C"), self.toggle_copy_label),
            (QKeySequence("Ctrl+Z"), self.undo),
            (QKeySequence("Ctrl+Y"), self.redo)
        ]
        for seq, func in shortcuts:
            QShortcut(seq, self, activated=func)

    def createMainMenu(self):
        """創建主菜單欄，新增載入標籤檔選項"""
        main_menu = self.menuBar()
        # 載入標籤檔
        file_menu = main_menu.addMenu(tr("file_menu"))
        load_classes_action = file_menu.addAction(tr("load_label_file"))
        load_classes_action.triggered.connect(self.loadClassesFromFile)
        # 幫助菜單
        help_menu = main_menu.addMenu(tr("help_menu"))
        shortcut_action = help_menu.addAction(tr("hotkeys_menu"))
        shortcut_action.triggered.connect(self.showShortcutHelp)
        about_action = help_menu.addAction(tr("about_menu"))
        about_action.triggered.connect(self.showAboutDialog)

        # 在選單欄中添加資料增強選項
        data_aug_menu = main_menu.addMenu(tr("data_aug_menu"))
        data_aug_action = QAction(tr("data_aug_menu"),self)
        data_aug_action.triggered.connect(self.show_data_augmentation_dialog)
        data_aug_menu.addAction(data_aug_action)

    def showShortcutHelp(self):
        """顯示快捷鍵幫助對話框"""
        shortcuts = [
            ("F", tr("next_image_hotkey")),
            ("D", tr("prev_image_hotkey")),
            ("S", tr("save_hotkey")),
            ("O", tr("open_hotkey")),
            ("E/Delete", tr("delete_hotkey")),
            ("Q", tr("quit_hotkey")),
            ("↑", tr("prev_image_hotkey")),
            ("↓", tr("next_image_hotkey")),
            ("C", tr("copy_labels_hotkey")),
            ("Ctrl+Click", tr("multi_select_hotkey")),
            ("Ctrl+Z", tr("undo_hotkey")),
            ("Ctrl+Y", tr("redo_hotkey"))
        ]
        QShortcut(QKeySequence("Ctrl+H"), self, activated=self.showShortcutHelp)
        QShortcut(QKeySequence("Ctrl+A"), self, activated=self.showAboutDialog)
        text = f"<b>{tr('hotkeys_list')}</b><br><br>"
        text += "<table>"
        for key, desc in shortcuts:
            text += f"<tr><td><code>{key}</code></td><td> - </td><td>{desc}</td></tr>"
        text += "</table>"
        msg = QMessageBox(self)
        msg.setWindowTitle(tr("hotkeys_title"))
        msg.setTextFormat(Qt.RichText)
        msg.setText(text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def showAboutDialog(self):
        """顯示關於對話框"""
        about_text = f"""
        <b>BakuAI Image Labeling Tool</b><br><br>
        {tr('version')} 1.1.0<br>
        © 2024 BakuAI AS, Norway. {tr('copyright')} <br><br>
        {tr('software_description')} <br><br>
        {tr('website')}：<a href='https://bakuai.no'>bakuai.no</a>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle(tr("about_title"))
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.setIcon(QMessageBox.Information)
        # 調整視窗大小
        msg.setMinimumWidth(320)  # 增加寬度
        msg.setMinimumHeight(200)  # 增加高度
        msg.setStyleSheet("""
            QMessageBox {
                font-size: 14px;
            }
            QMessageBox QLabel {
                min-width: 320px;
                min-height: 200px;
            }
        """)
        msg.exec_()

    def loadLogo(self):
        """新增：加载程序启动时的logo"""
        # 獲取應用程式的基礎路徑
        if getattr(sys, 'frozen', False):
            # 如果是打包後的執行檔
            base_path = sys._MEIPASS
        else:
            # 如果是直接執行 Python 腳本
            base_path = os.path.abspath(os.path.dirname(__file__))
        
        logo_path = os.path.join(base_path, "logo.png")
        
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
        else:
            # 創建默認 logo
            pixmap = QPixmap(400, 300)
            pixmap.fill(QColor(240, 240, 240))
            painter = QPainter(pixmap)
            painter.setPen(Qt.blue)
            painter.setFont(self.font())
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "BakuLabel Tool\n\nOpen image folder to start")
            painter.end()
        
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def toggle_copy_label(self):
        """切换复制标签复选框状态"""
        current_state = self.copy_checkbox.isChecked()
        self.copy_checkbox.setChecked(not current_state)
        self.status_label.setText(f"Copy label {'actived' if not current_state else 'Disabled'}")

    def update_class_stats(self):
        """新增：更新類別統計數據"""
        pass  # 移除此方法的內容

    def updateAnnotationStats(self):
        """Update class-based annotation statistics"""
        pass  # 移除此方法的內容

    def update_stats_display(self):
        """Update statistics display without processed indices"""
        # Navigation statistics
        nav_stats = [
            f"Viewed: {len(self.viewed_indices)}",
            f"Index: {self.current_index + 1}",
            f"Total: {self.total_images}"
        ]
        
        # Update UI elements
        self.status_label.setText(" | ".join(nav_stats))

    def calculate_annotation_stats(self):
        """Calculate class-based annotation statistics"""
        pass  # 移除此方法的內容

    def onFileSelected(self, item):
        """处理文件列表点击事件（新增关键方法）"""
        if not item or not self.image_list:
            return
            
        filename = item.text()
        dir_path = os.path.dirname(self.image_list[0])
        full_path = os.path.join(dir_path, filename)
        
        try:
            idx = self.image_list.index(full_path)
        except ValueError:
            return
            
        if idx != self.current_index:
            if self.autosave:
                self.saveAnnotations()
            self.current_index = idx
            self.viewed_indices.add(idx)  # 添加當前索引到已查看集合
            self.loadImage(self.image_list[idx])
            self.file_list.setCurrentRow(idx)
            self.updateStatusDisplay()  # 更新狀態顯示
            self.updateFileListColors()  # 更新檔案列表顏色

    def updateFileListColors(self):
        """更新檔案列表的顏色顯示"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if i in self.viewed_indices:
                item.setForeground(QColor(0, 128, 0))  # 綠色
            else:
                item.setForeground(QColor(0, 0, 0))  # 黑色

    def loadClasses(self):
        # 優先讀取目前資料夾下的 classes.txt
        class_path = os.path.join(getattr(self, 'current_dir', ''), 'classes.txt') if hasattr(self, 'current_dir') else 'classes.txt'
        import shutil
        if not os.path.exists(class_path):
            # 彈窗詢問
            reply = QMessageBox.question(self, "Classes File Not Found", tr("no_classes_file"), QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                file_path, _ = QFileDialog.getOpenFileName(self, "Select Class File", self.current_dir if hasattr(self, 'current_dir') else "", "Text Files (*.txt)")
                if file_path:
                    shutil.copy(file_path, class_path)
                    with open(class_path, 'r', encoding='utf-8') as f:
                        self.classes = [line.strip() for line in f if line.strip()]
                else:
                    self.classes = []
            else:
                # 讓使用者手動輸入類別
                from PyQt5.QtWidgets import QInputDialog
                text, ok = QInputDialog.getMultiLineText(self, "Enter Classes", "Please enter all classes (one per line):")
                if ok and text.strip():
                    self.classes = [line.strip() for line in text.splitlines() if line.strip()]
                    with open(class_path, 'w', encoding='utf-8') as f:
                        f.write("\n".join(self.classes))
                else:
                    self.classes = []
        else:
            with open(class_path, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f if line.strip()]
        self.class_combo.clear()
        self.class_combo.addItems(self.classes)

    def showContextMenu(self, position):
        if self.current_image is None:
            return
            
        scaled_pos = self.getScaledPoint(position)
        for bbox in self.bboxes:
            if bbox.contains(scaled_pos):
                menu = QMenu()
                edit_action = menu.addAction(tr("edit_label"))
                delete_action = menu.addAction(tr("delete"))
                
                action = menu.exec_(self.image_label.mapToGlobal(position))
                
                if action == delete_action:
                    self.bboxes.remove(bbox)
                    if bbox == self.selected_bbox:
                        self.selected_bbox = None
                    self.updateColorLegend()
                elif action == edit_action:
                    text, ok = QInputDialog.getText(self, tr("edit_label"), 
                                                  tr("enter_new_label"), 
                                                  text=bbox.label)
                    if ok and text:
                        if text not in self.classes:
                            self.classes.append(text)
                            self.class_combo.addItem(text)
                            class_path = os.path.join(getattr(self, 'current_dir', ''), 'classes.txt') if hasattr(self, 'current_dir') else 'classes.txt'
                            try:
                                with open(class_path, 'w', encoding='utf-8') as f:
                                    f.write("\n".join(self.classes))
                            except Exception as e:
                                QMessageBox.warning(self, tr("class_save_failed"), tr("failed_to_write_classes").format(str(e)))
                        bbox.label = text
                        self.updateColorLegend()
                self.updateDisplay()
                break

    def openDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.current_dir = dir_path  # 記錄目前資料夾
            self.image_list = []
            self.viewed_indices = set()  # Remove processed_indices initialization
            valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
            # Simplified directory scanning
            for filename in sorted(os.listdir(dir_path)):
                if filename.lower().endswith(valid_extensions):
                    img_path = os.path.join(dir_path, filename)
                    self.image_list.append(img_path)
            self.total_images = len(self.image_list)
            self.file_list.clear()
            self.file_list.addItems([os.path.basename(f) for f in self.image_list])
            if self.image_list:
                self.current_index = 0
                self.viewed_indices.add(0)
                self.loadClasses()  # 在打開資料夾時載入 classes.txt
                self.loadImage(self.image_list[0])
                self.file_list.setCurrentRow(0)
                self.updateStatusDisplay()
                self.updateFileListColors()  # 更新檔案列表顏色

    def updateStatusDisplay(self):
        """Simplified status without processed count"""
        status_text = f"Viewed: {len(self.viewed_indices)} | Index: {self.current_index + 1} | Total: {self.total_images}"
        self.status_label.setText(status_text)


    def loadImage(self, image_path):
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                QMessageBox.critical(self, tr("load_error"), tr("failed_to_load_image").format(image_path))
                return False

            # 清空当前标注
            self.bboxes = []
            self.selected_bbox = None
            self.selected_bboxes = set()

            # 加载标注
            annotation_path = os.path.splitext(image_path)[0] + '.txt'
            if os.path.exists(annotation_path):
                self.loadAnnotations(annotation_path)
            elif self.current_index in self.label_cache:
                # 从缓存加载时保持标注顺序
                self.bboxes = [BoundingBox(b.x, b.y, b.w, b.h, b.label) for b in self.label_cache[self.current_index]]

            # 更新显示
            self.updateDisplay()
            self.updateColorLegend()
            return True
        except Exception as e:
            QMessageBox.critical(self, tr("load_error"), tr("failed_to_load_image").format(str(e)))
            return False

    def nextImage(self):
        """Next image navigation with proper status updates"""
        if self.current_index < len(self.image_list) - 1:
            try:
                # Auto-save if enabled
                if self.autosave:
                    save_result = self.saveAnnotations()
                    if not save_result:
                        self.status_label.clear()

                # Update navigation
                self.current_index += 1
                self.viewed_indices.add(self.current_index)
                
                # Load new image
                self.loadImage(self.image_list[self.current_index])
                self.file_list.setCurrentRow(self.current_index)
                
                # Update displays
                self.updateStatusDisplay()
                self.updateFileListColors()  # 更新檔案列表顏色

            except Exception as e:
                self.status_label.setText(f"Navigation error: {str(e)}")
                QMessageBox.critical(self, "Navigation Error", f"Failed to load next image:\n{str(e)}")

    def prevImage(self):
        """Previous image navigation with proper status updates"""
        if self.current_index > 0:
            try:
                # Auto-save if enabled
                if self.autosave:
                    save_result = self.saveAnnotations()
                    if not save_result:
                        self.status_label.clear()

                # Update navigation
                self.current_index -= 1
                self.viewed_indices.add(self.current_index)
                
                # Load previous image
                self.loadImage(self.image_list[self.current_index])
                self.file_list.setCurrentRow(self.current_index)
                
                # Update displays
                self.updateStatusDisplay()
                self.updateFileListColors()  # 更新檔案列表顏色

            except Exception as e:
                self.status_label.setText(f"Navigation error: {str(e)}")
                QMessageBox.critical(self, "Navigation Error", f"Failed to load previous image:\n{str(e)}")
    
    def saveAnnotations(self):
        if self.current_image is None or self.current_index < 0:
            return False
        # 先自動存 YOLO
        self.saveYOLOAnnotation()
        fmt = self.format_combo.currentText()
        if fmt == "YOLO":
            return True  # 已經存過 YOLO
        elif fmt == "VOC":
            return self.saveVOCAnnotation()
        elif fmt == "COCO":
            return self.saveCOCOAnnotation()
        return False

    def saveYOLOAnnotation(self):
        img_path = self.image_list[self.current_index]
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        try:
            height, width = self.current_image.shape[:2]
            valid_boxes = 0
            with open(txt_path, 'w') as f:
                # 按照标注顺序保存
                for bbox in self.bboxes:
                    if (bbox.w < self.min_box_size or bbox.h < self.min_box_size or bbox.x < 0 or bbox.y < 0 or bbox.x + bbox.w > width or bbox.y + bbox.h > height):
                        continue
                    try:
                        class_idx = self.classes.index(bbox.label)
                    except ValueError:
                        continue
                    x_center = (bbox.x + bbox.w/2) / width
                    y_center = (bbox.y + bbox.h/2) / height
                    w_norm = bbox.w / width
                    h_norm = bbox.h / height
                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    valid_boxes += 1
            if valid_boxes > 0:
                # 保存到缓存时也保持标注顺序
                self.label_cache[self.current_index] = [BoundingBox(b.x, b.y, b.w, b.h, b.label) for b in self.bboxes]
            else:
                if os.path.exists(txt_path):
                    os.remove(txt_path)
            self.updateStatusDisplay()
            return True
        except Exception as e:
            QMessageBox.critical(self, tr("save_error"), tr("failed_to_save_yolo").format(str(e)))
            return False

    def saveVOCAnnotation(self):
        # VOC格式：每張圖一個xml
        import xml.etree.ElementTree as ET
        img_path = self.image_list[self.current_index]
        xml_path = os.path.splitext(img_path)[0] + '.xml'
        try:
            height, width = self.current_image.shape[:2]
            annotation = ET.Element('annotation')
            ET.SubElement(annotation, 'folder').text = os.path.basename(os.path.dirname(img_path))
            ET.SubElement(annotation, 'filename').text = os.path.basename(img_path)
            size = ET.SubElement(annotation, 'size')
            ET.SubElement(size, 'width').text = str(width)
            ET.SubElement(size, 'height').text = str(height)
            ET.SubElement(size, 'depth').text = '3'
            for bbox in self.bboxes:
                obj = ET.SubElement(annotation, 'object')
                ET.SubElement(obj, 'name').text = bbox.label
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(bbox.x)
                ET.SubElement(bndbox, 'ymin').text = str(bbox.y)
                ET.SubElement(bndbox, 'xmax').text = str(bbox.x + bbox.w)
                ET.SubElement(bndbox, 'ymax').text = str(bbox.y + bbox.h)
            tree = ET.ElementTree(annotation)
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            self.updateStatusDisplay()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save VOC annotation:\n{str(e)}")
            return False

    def saveCOCOAnnotation(self):
        # COCO格式：整個資料夾一個json
        import json
        img_path = self.image_list[self.current_index]
        folder = os.path.dirname(img_path)
        json_path = os.path.join(folder, 'coco_annotations.json')
        try:
            images = []
            annotations = []
            categories = []
            cat_map = {name: i+1 for i, name in enumerate(self.classes)}
            ann_id = 1
            for i, img_file in enumerate(self.image_list):
                img = cv2.imread(img_file)
                if img is None:
                    continue
                h, w = img.shape[:2]
                images.append({
                    'id': i+1,
                    'file_name': os.path.basename(img_file),
                    'height': h,
                    'width': w
                })
                # 取得標註框（優先用 label_cache，否則嘗試即時載入）
                bboxes = self.label_cache.get(i)
                if bboxes is None:
                    # 嘗試載入 YOLO 標註
                    txt_path = os.path.splitext(img_file)[0] + '.txt'
                    bboxes = []
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    continue
                                try:
                                    class_idx = int(parts[0])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    w = float(parts[3])
                                    h = float(parts[4])
                                    x = (x_center - w/2) * w
                                    y = (y_center - h/2) * h
                                    width_box = w_norm * w
                                    height_box = h_norm * h
                                    label = self.classes[class_idx] if class_idx < len(self.classes) else 'unknown'
                                    bboxes.append(BoundingBox(x, y, width_box, height_box, label))
                                except Exception:
                                    continue
                    # 若還是沒有，略過
                    if not bboxes:
                        continue
                for bbox in bboxes:
                    if bbox.label not in cat_map:
                        continue
                    category_id = cat_map[bbox.label]
                    annotations.append({
                        'id': ann_id,
                        'image_id': i+1,
                        'category_id': category_id,
                        'bbox': [float(bbox.x), float(bbox.y), float(bbox.w), float(bbox.h)],
                        'area': float(bbox.w) * float(bbox.h),
                        'iscrowd': 0
                    })
                    ann_id += 1
            for name, cid in cat_map.items():
                categories.append({'id': cid, 'name': name})
            coco_dict = {
                'images': images,
                'annotations': annotations,
                'categories': categories
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(coco_dict, f, ensure_ascii=False, indent=2)
            self.updateStatusDisplay()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save COCO annotation:\n{str(e)}")
            return False

    def toggleAutosave(self, state):
        """Handle autosave checkbox state changes"""
        self.autosave = state == Qt.Checked
        self.status_label.setText(f"Autosave {'Enabled' if self.autosave else 'Disabled'}")

    def loadAnnotations(self, annotation_path):
        """Load annotations from YOLO, VOC, or COCO format"""
        self.bboxes = []
        if not os.path.exists(annotation_path):
            return
            
        height, width = self.current_image.shape[:2]
        
        # 根據副檔名決定格式
        ext = os.path.splitext(annotation_path)[1].lower()
        
        try:
            if ext == '.txt':  # YOLO format
                with open(annotation_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                            
                        try:
                            class_idx = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            # Convert from YOLO format to pixel coordinates
                            x = int((x_center - w/2) * width)
                            y = int((y_center - h/2) * height)
                            w_pixels = int(w * width)
                            h_pixels = int(h * height)
                            
                            # Get label from classes list
                            try:
                                label = self.classes[class_idx]
                            except IndexError:
                                label = "unknown"
                                
                            self.bboxes.append(BoundingBox(x, y, w_pixels, h_pixels, label))
                            
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing YOLO line: {line} - {str(e)}")
                            
            elif ext == '.xml':  # VOC format
                import xml.etree.ElementTree as ET
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    bbox = obj.find('bndbox')
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))
                    
                    # Convert to width/height format
                    w = xmax - xmin
                    h = ymax - ymin
                    
                    self.bboxes.append(BoundingBox(xmin, ymin, w, h, label))
                    
            elif ext == '.json':  # COCO format
                import json
                with open(annotation_path, 'r') as f:
                    coco_data = json.load(f)
                # 建立圖片ID到檔名的映射
                image_map = {img['id']: img['file_name'] for img in coco_data['images']}
                # 建立類別ID到名稱的映射
                category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
                # 取得當前圖片檔名（忽略大小寫與副檔名）
                current_filename = os.path.basename(self.image_list[self.current_index])
                current_filename_noext = os.path.splitext(current_filename)[0].lower()
                # 找到當前圖片的ID（忽略副檔名與大小寫）
                current_image_id = None
                for img_id, filename in image_map.items():
                    fname_noext = os.path.splitext(filename)[0].lower()
                    if fname_noext == current_filename_noext:
                        current_image_id = img_id
                        break
                if current_image_id is None:
                    QMessageBox.warning(self, "COCO 標註錯誤", f"找不到對應圖片檔名於 COCO 標註檔案中：{current_filename}\n請檢查 file_name 欄位與實際圖片檔名是否一致。")
                    return
                found = False
                for ann in coco_data['annotations']:
                    if ann['image_id'] == current_image_id:
                        bbox = ann['bbox']  # COCO格式：[x, y, width, height]
                        category_id = ann['category_id']
                        label = category_map.get(category_id, "unknown")
                        # 確保標籤在類別列表中
                        if label not in self.classes:
                            self.classes.append(label)
                            self.class_combo.addItem(label)
                        x = int(bbox[0])
                        y = int(bbox[1])
                        w = int(bbox[2])
                        h = int(bbox[3])
                        # 確保座標在圖片範圍內
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))
                        w = max(1, min(w, width - x))
                        h = max(1, min(h, height - y))
                        self.bboxes.append(BoundingBox(x, y, w, h, label))
                        found = True
                if not found:
                    QMessageBox.information(self, "COCO 標註", f"此圖片在 COCO 標註檔案中沒有標註框。\n圖片檔名：{current_filename}")
                    return
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load annotations:\n{str(e)}")
            return
            
        # After loading annotations, update the color legend
        self.updateColorLegend()

    def toggleMagnifier(self, state):
        """切換放大鏡功能"""
        self.magnifier_enabled = state == Qt.Checked
        if self.magnifier_enabled:
            self.magnifier.show()
        else:
            self.magnifier.hide()

    def get_label_color(self, label):
        """Return a consistent and high-contrast color for each label"""
        if label not in self.label_colors:
            # Use HSL color space with fixed saturation and lightness
            # and evenly distributed hues for maximum contrast
            hue = (hash(label) % 360)  # Use hash for consistent colors
            saturation = 0.8  # High saturation for vivid colors
            lightness = 0.6   # Medium lightness for good visibility
            
            # Convert HSL to RGB
            c = (1 - abs(2 * lightness - 1)) * saturation
            x = c * (1 - abs((hue / 60) % 2 - 1))
            m = lightness - c / 2
            
            if hue < 60:
                r, g, b = c, x, 0
            elif hue < 120:
                r, g, b = x, c, 0
            elif hue < 180:
                r, g, b = 0, c, x
            elif hue < 240:
                r, g, b = 0, x, c
            elif hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
                
            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)
            
            self.label_colors[label] = (r, g, b)
        return self.label_colors[label]

    def deleteSelectedBox(self):
        """刪除所有已選中的標註框"""
        if self.selected_bboxes:
            self.push_undo()  # 刪除前先記錄
            for bbox in list(self.selected_bboxes):
                if bbox in self.bboxes:
                    self.bboxes.remove(bbox)
            self.selected_bboxes.clear()
            self.selected_bbox = None
            self.updateDisplay()
            self.updateColorLegend()  # 刪除標籤時更新
        elif self.selected_bbox in self.bboxes:
            self.push_undo()  # 刪除前先記錄
            self.bboxes.remove(self.selected_bbox)
            self.selected_bbox = None
            self.updateDisplay()
            self.updateColorLegend()

    def keyPressEvent(self, event):
        """Enhanced key press handler with better delete key support"""
        if event.key() == Qt.Key_Delete:
            if self.selected_bbox:
                self.deleteSelectedBox()
                self.updateDisplay()
                if self.autosave:
                    self.saveAnnotations()
            else:
                QMessageBox.information(self, "No Selection",
                                      "Please select a bounding box first")
        else:
            super().keyPressEvent(event)

    def updateDisplay(self):
        """更新圖片顯示與標註框，annotation 文字半透明、細、字體小"""
        if self.current_image is None:
            return

        display_image = cv2.cvtColor(self.current_image.copy(), cv2.COLOR_BGR2RGB)
        height, width, _ = display_image.shape
        
        # 計算最佳縮放比例並保持圖片比例
        label_size = self.image_label.size()
        if width > 0 and height > 0:
            width_ratio = label_size.width() / width
            height_ratio = label_size.height() / height
            self.scale_factor = min(width_ratio, height_ratio)

        font_scale = 0.4
        thickness = 1
        for bbox in self.bboxes:
            color = self.get_label_color(bbox.label)
            if bbox in getattr(self, 'selected_bboxes', set()):
                # 多選框：加粗紅色虛線
                overlay = display_image.copy()
                cv2.rectangle(overlay, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), (255,0,0), -1)
                cv2.addWeighted(overlay, 0.12, display_image, 0.88, 0, display_image)
                draw_dashed_rect(display_image, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), (255,0,0), 3, 8)
                handle_size = self.handle_visual_size
                for handle in [(bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y), (bbox.x, bbox.y + bbox.h), (bbox.x + bbox.w, bbox.y + bbox.h)]:
                    cv2.rectangle(display_image, (handle[0] - handle_size, handle[1] - handle_size), (handle[0] + handle_size, handle[1] + handle_size), (0, 255, 255), -1)
            elif bbox == self.selected_bbox:
                overlay = display_image.copy()
                cv2.rectangle(overlay, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), color, -1)
                cv2.addWeighted(overlay, 0.18, display_image, 0.82, 0, display_image)
                draw_dashed_rect(display_image, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), color, 2, 10)
                handle_size = self.handle_visual_size
                for handle in [(bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y), (bbox.x, bbox.y + bbox.h), (bbox.x + bbox.w, bbox.y + bbox.h)]:
                    cv2.rectangle(display_image, (handle[0] - handle_size, handle[1] - handle_size), (handle[0] + handle_size, handle[1] + handle_size), (0, 255, 255), -1)
            else:
                cv2.rectangle(display_image, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), color, 1)
            # 標籤文字（半透明、細、字體小）
            text = bbox.label
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # 建立透明圖層
            text_layer = np.zeros_like(display_image, dtype=np.uint8)
            cv2.putText(text_layer, text, (bbox.x, bbox.y - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            # 疊加半透明
            alpha = 0.6  # 半透明程度
            mask = (text_layer > 0)
            display_image[mask] = (display_image[mask] * (1 - alpha) + text_layer[mask] * alpha).astype(np.uint8)

        # 畫正在繪製的虛線框
        if self.drawing and self.bbox_start and self.bbox_end:
            x1 = min(self.bbox_start.x(), self.bbox_end.x())
            y1 = min(self.bbox_start.y(), self.bbox_end.y())
            x2 = max(self.bbox_start.x(), self.bbox_end.x())
            y2 = max(self.bbox_start.y(), self.bbox_end.y())
            draw_dashed_rect(display_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 轉換為QPixmap
        bytes_per_line = display_image.strides[0]
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        scaled_width = int(width * self.scale_factor)
        scaled_height = int(height * self.scale_factor)
        pixmap = QPixmap.fromImage(q_image).scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def handleSplitterMoved(self, pos, index):
        if self.current_image is not None:
            self.updateDisplay()

    def getScaledPoint(self, point):
        """將視窗座標轉換為圖片座標，支援縮放與置中"""
        if not self.image_label.pixmap() or self.scale_factor == 0:
            return QPoint(0, 0)
        
        pixmap = self.image_label.pixmap()
        label_size = self.image_label.size()
        pixmap_size = pixmap.size()
        
        x_offset = max(0, (label_size.width() - pixmap_size.width()) // 2)
        y_offset = max(0, (label_size.height() - pixmap_size.height()) // 2)
        
        # Clamp coordinates to valid pixmap area
        local_x = max(x_offset, min(point.x(), x_offset + pixmap_size.width() - 1))
        local_y = max(y_offset, min(point.y(), y_offset + pixmap_size.height() - 1))
        
        return QPoint(
            int((local_x - x_offset) / self.scale_factor),
            int((local_y - y_offset) / self.scale_factor)
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            scaled_pos = self.getScaledPoint(pos)
            box_clicked = False
            ctrl_pressed = event.modifiers() & Qt.ControlModifier
            for bbox in reversed(self.bboxes):
                handle = bbox.get_resize_handle(scaled_pos, self.handle_detection_threshold)
                if handle:
                    self.selected_bbox = bbox
                    self.resize_handle = handle
                    self.drag_start = scaled_pos
                    self.original_bbox = BoundingBox(bbox.x, bbox.y, bbox.w, bbox.h, bbox.label)
                    box_clicked = True
                    break
                if bbox.contains(scaled_pos):
                    if ctrl_pressed:
                        # 多選：Ctrl+點擊切換選取狀態
                        if bbox in self.selected_bboxes:
                            self.selected_bboxes.remove(bbox)
                        else:
                            self.selected_bboxes.add(bbox)
                        self.selected_bbox = None
                    else:
                        # 單選：只選這一個
                        self.selected_bboxes = set([bbox])
                        self.selected_bbox = bbox
                    self.dragging = True
                    self.drag_start = scaled_pos
                    self.original_bbox = BoundingBox(bbox.x, bbox.y, bbox.w, bbox.h, bbox.label)
                    box_clicked = True
                    break
            if not box_clicked:
                self.drawing = True
                self.bbox_start = scaled_pos
                self.bbox_end = scaled_pos
                self.selected_bbox = None
                self.selected_bboxes = set()
            # 放大鏡
            if self.magnifier_enabled:
                self.magnifier_active = True
                self.magnifier.show()
                self.magnifier.update()
            self.updateDisplay()

    def mouseMoveEvent(self, event):
        if self.image_label.underMouse():
            pos = event.pos()
            scaled_pos = self.getScaledPoint(pos)
            if self.drawing and self.bbox_start:
                self.bbox_end = scaled_pos
                self.updateDisplay()
            elif self.resize_handle and self.selected_bbox:
                dx = scaled_pos.x() - self.drag_start.x()
                dy = scaled_pos.y() - self.drag_start.y()
                new_w = max(self.min_box_size, self.original_bbox.w + (dx if 'right' in self.resize_handle else -dx))
                new_h = max(self.min_box_size, self.original_bbox.h + (dy if 'bottom' in self.resize_handle else -dy))
                if 'left' in self.resize_handle:
                    self.selected_bbox.x = self.original_bbox.x + (self.original_bbox.w - new_w)
                if 'top' in self.resize_handle:
                    self.selected_bbox.y = self.original_bbox.y + (self.original_bbox.h - new_h)
                self.selected_bbox.w = new_w
                self.selected_bbox.h = new_h
                self.updateDisplay()
            elif self.dragging and self.selected_bbox:
                dx = scaled_pos.x() - self.drag_start.x()
                dy = scaled_pos.y() - self.drag_start.y()
                self.selected_bbox.x = self.original_bbox.x + dx
                self.selected_bbox.y = self.original_bbox.y + dy
                self.updateDisplay()
            # 放大鏡
            if self.magnifier_enabled and self.magnifier_active:
                self.magnifier.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.magnifier_enabled:
                self.magnifier_active = False
                self.magnifier.hide()
            if self.drawing and self.bbox_start and self.bbox_end:
                x1 = min(self.bbox_start.x(), self.bbox_end.x())
                y1 = min(self.bbox_start.y(), self.bbox_end.y())
                x2 = max(self.bbox_start.x(), self.bbox_end.x())
                y2 = max(self.bbox_start.y(), self.bbox_end.y())
                w = x2 - x1
                h = y2 - y1
                if w >= self.min_box_size and h >= self.min_box_size:
                    label, ok = QInputDialog.getItem(
                        self, 'Annotation Label', 'Select or enter label:',
                        self.classes, 0, True
                    )
                    if ok and label:
                        self.push_undo()  # 新增前先記錄
                        if label not in self.classes:
                            self.classes.append(label)
                            self.class_combo.addItem(label)
                            class_path = os.path.join(getattr(self, 'current_dir', ''), 'classes.txt') if hasattr(self, 'current_dir') else 'classes.txt'
                            try:
                                with open(class_path, 'w', encoding='utf-8') as f:
                                    f.write("\n".join(self.classes))
                            except Exception as e:
                                QMessageBox.warning(self, "Class Save Failed", f"無法寫入 classes.txt：{str(e)}")
                        new_bbox = BoundingBox(x1, y1, w, h, label)
                        self.bboxes.append(new_bbox)
                        self.selected_bbox = new_bbox
                        self.updateColorLegend()
                self.drawing = False
                self.bbox_start = None
                self.bbox_end = None
                self.updateDisplay()
            if self.resize_handle or self.dragging:
                self.push_undo()  # 移動/調整前先記錄
                self.resize_handle = None
                self.dragging = False
                self.original_bbox = None
                self.updateDisplay()
            if self.autosave and self.current_index >= 0:
                self.saveAnnotations()

    def loadClassesFromFile(self):
        """讓使用者自選classes.txt並載入，並複製到目前資料夾下"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Label File", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.classes = [line.strip() for line in f if line.strip()]
                self.class_combo.clear()
                self.class_combo.addItems(self.classes)
                # 複製到目前資料夾下
                class_path = os.path.join(getattr(self, 'current_dir', ''), 'classes.txt') if hasattr(self, 'current_dir') else 'classes.txt'
                with open(class_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.classes))
                QMessageBox.information(self, "Load Success", f"已載入 {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Load Failed", str(e))

    def syncClassComboToClasses(self):
        """將 class_combo 新增的類別自動同步到 self.classes，並寫入目前資料夾的 classes.txt"""
        text = self.class_combo.currentText().strip()
        if text and text not in self.classes:
            self.classes.append(text)
            self.class_combo.addItem(text)
            # 同步寫入目前資料夾下的 classes.txt
            class_path = os.path.join(getattr(self, 'current_dir', ''), 'classes.txt') if hasattr(self, 'current_dir') else 'classes.txt'
            try:
                with open(class_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.classes))
            except Exception as e:
                QMessageBox.warning(self, "Class Save Failed", f"無法寫入 classes.txt：{str(e)}")

    def updateColorLegend(self):
        """Update the color legend to show all labels and their colors with current count"""
        self.legend_list.clear()
        # 計算當前圖片中每個標籤的數量
        current_counts = {}
        for bbox in self.bboxes:
            current_counts[bbox.label] = current_counts.get(bbox.label, 0) + 1
        # 更新所有標籤的顯示
        for label in self.classes:
            color = self.get_label_color(label)
            count = current_counts.get(label, 0)
            item = QListWidgetItem(f"■ {label} ({count})")
            item.setForeground(QColor(color[0], color[1], color[2]))
            item.setFont(QFont("Arial", 10, QFont.Bold))
            self.legend_list.addItem(item)

    def push_undo(self):
        idx = self.current_index
        if idx < 0:
            return
        if idx not in self.undo_stack:
            self.undo_stack[idx] = []
        # 深拷貝 bboxes
        import copy
        self.undo_stack[idx].append(copy.deepcopy(self.bboxes))
        # 限制堆疊長度
        if len(self.undo_stack[idx]) > 30:
            self.undo_stack[idx] = self.undo_stack[idx][-30:]
        # 清空 redo stack
        self.redo_stack[idx] = []

    def undo(self):
        idx = self.current_index
        if idx < 0 or idx not in self.undo_stack or not self.undo_stack[idx]:
            return
        import copy
        if idx not in self.redo_stack:
            self.redo_stack[idx] = []
        self.redo_stack[idx].append(copy.deepcopy(self.bboxes))
        self.bboxes = self.undo_stack[idx].pop()
        self.selected_bboxes = set()
        self.selected_bbox = None
        self.updateDisplay()
        self.updateColorLegend()

    def redo(self):
        idx = self.current_index
        if idx < 0 or idx not in self.redo_stack or not self.redo_stack[idx]:
            return
        import copy
        if idx not in self.undo_stack:
            self.undo_stack[idx] = []
        self.undo_stack[idx].append(copy.deepcopy(self.bboxes))
        self.bboxes = self.redo_stack[idx].pop()
        self.selected_bboxes = set()
        self.selected_bbox = None
        self.updateDisplay()
        self.updateColorLegend()

    def show_data_augmentation_dialog(self):
        if not self.current_image is None:
            dialog = DataAugmentationDialog(self)
            dialog.exec_()
        else:
            QMessageBox.warning(self, tr("warning"), tr("no_image"))
    
    def update_image(self, image):
        """更新顯示的圖像"""
        if image is None:
            return
            
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # 更新標籤框
        self.update_boxes()

class DataAugmentationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(tr("data_aug_title"))
        self.setModal(True)
        
        # 預設輸出資料夾為目前載入的資料夾
        if hasattr(parent, 'current_dir') and parent.current_dir:
            self.output_folder = parent.current_dir
        else:
            self.output_folder = ""
        
        # 其餘初始化...
        self.aug_per_image = 1
        self.preserve_original = True
        self.rotation_range = (-30, 30)
        self.brightness_range = (-0.2, 0.2)
        self.contrast_range = (0.8, 1.2)
        self.saturation_range = (0.8, 1.2)
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # 基本設定組
        basic_group = QGroupBox(tr("aug_settings"))
        basic_layout = QVBoxLayout()
        
        # 每張圖片的增強數量
        aug_count_layout = QHBoxLayout()
        aug_count_label = QLabel(tr("aug_per_image"))
        self.aug_count_spin = QSpinBox()
        self.aug_count_spin.setRange(1, 99999)
        self.aug_count_spin.setValue(1)
        aug_count_layout.addWidget(aug_count_label)
        aug_count_layout.addWidget(self.aug_count_spin)
        basic_layout.addLayout(aug_count_layout)
        
        # 保留原始檔案選項
        self.preserve_checkbox = QCheckBox(tr("preserve_original"))
        self.preserve_checkbox.setChecked(True)
        basic_layout.addWidget(self.preserve_checkbox)
        
        # 輸出資料夾選擇
        folder_layout = QHBoxLayout()
        folder_label = QLabel(tr("output_folder"))
        self.folder_path = QLineEdit()
        self.folder_path.setReadOnly(True)
        self.folder_path.setText(self.output_folder)  # 預設顯示
        folder_btn = QPushButton(tr("select_folder"))
        folder_btn.clicked.connect(self.select_output_folder)
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_path)
        folder_layout.addWidget(folder_btn)
        basic_layout.addLayout(folder_layout)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # 隨機參數設定組
        random_group = QGroupBox(tr("aug_settings"))
        random_layout = QVBoxLayout()
        
        # 旋轉範圍
        rotation_layout = QHBoxLayout()
        rotation_label = QLabel(tr("random_rotation"))
        self.rotation_min = QSpinBox()
        self.rotation_min.setRange(-180, 180)
        self.rotation_min.setValue(-30)
        self.rotation_max = QSpinBox()
        self.rotation_max.setRange(-180, 180)
        self.rotation_max.setValue(30)
        rotation_layout.addWidget(rotation_label)
        rotation_layout.addWidget(self.rotation_min)
        rotation_layout.addWidget(QLabel("to"))
        rotation_layout.addWidget(self.rotation_max)
        random_layout.addLayout(rotation_layout)
        
        # 亮度範圍
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel(tr("random_brightness"))
        self.brightness_min = QDoubleSpinBox()
        self.brightness_min.setRange(-1.0, 1.0)
        self.brightness_min.setValue(-0.2)
        self.brightness_min.setSingleStep(0.1)
        self.brightness_max = QDoubleSpinBox()
        self.brightness_max.setRange(-1.0, 1.0)
        self.brightness_max.setValue(0.2)
        self.brightness_max.setSingleStep(0.1)
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_min)
        brightness_layout.addWidget(QLabel("to"))
        brightness_layout.addWidget(self.brightness_max)
        random_layout.addLayout(brightness_layout)
        
        # 對比度範圍
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel(tr("random_contrast"))
        self.contrast_min = QDoubleSpinBox()
        self.contrast_min.setRange(0.1, 2.0)
        self.contrast_min.setValue(0.8)
        self.contrast_min.setSingleStep(0.1)
        self.contrast_max = QDoubleSpinBox()
        self.contrast_max.setRange(0.1, 2.0)
        self.contrast_max.setValue(1.2)
        self.contrast_max.setSingleStep(0.1)
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_min)
        contrast_layout.addWidget(QLabel("to"))
        contrast_layout.addWidget(self.contrast_max)
        random_layout.addLayout(contrast_layout)
        
        # 飽和度範圍
        saturation_layout = QHBoxLayout()
        saturation_label = QLabel(tr("random_saturation"))
        self.saturation_min = QDoubleSpinBox()
        self.saturation_min.setRange(0.1, 2.0)
        self.saturation_min.setValue(0.8)
        self.saturation_min.setSingleStep(0.1)
        self.saturation_max = QDoubleSpinBox()
        self.saturation_max.setRange(0.1, 2.0)
        self.saturation_max.setValue(1.2)
        self.saturation_max.setSingleStep(0.1)
        saturation_layout.addWidget(saturation_label)
        saturation_layout.addWidget(self.saturation_min)
        saturation_layout.addWidget(QLabel("to"))
        saturation_layout.addWidget(self.saturation_max)
        random_layout.addLayout(saturation_layout)
        
        random_group.setLayout(random_layout)
        layout.addWidget(random_group)
        
        # 翻轉選項
        flip_group = QGroupBox(tr("flip_options"))
        flip_layout = QHBoxLayout()
        self.flip_ud_checkbox = QCheckBox(tr("flip_ud"))
        self.flip_lr_checkbox = QCheckBox(tr("flip_lr"))
        flip_layout.addWidget(self.flip_ud_checkbox)
        flip_layout.addWidget(self.flip_lr_checkbox)
        flip_group.setLayout(flip_layout)
        layout.addWidget(flip_group)
        
        # 按鈕
        button_layout = QHBoxLayout()
        apply_btn = QPushButton(tr("apply"))
        cancel_btn = QPushButton(tr("cancel"))
        button_layout.addWidget(apply_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # 連接信號
        apply_btn.clicked.connect(self.process_images)
        cancel_btn.clicked.connect(self.reject)
        
        self.setLayout(layout)
        
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr("select_folder"))
        if folder:
            self.output_folder = folder
            self.folder_path.setText(folder)
            
    def get_random_params(self):
        """生成隨機增強參數"""
        import random
        return {
            'rotation': random.uniform(self.rotation_min.value(), self.rotation_max.value()),
            'brightness': random.uniform(self.brightness_min.value(), self.brightness_max.value()),
            'contrast': random.uniform(self.contrast_min.value(), self.contrast_max.value()),
            'saturation': random.uniform(self.saturation_min.value(), self.saturation_max.value()),
            'flip_ud': random.choice([True, False]) if self.flip_ud_checkbox.isChecked() else False,
            'flip_lr': random.choice([True, False]) if self.flip_lr_checkbox.isChecked() else False
        }
        
    def apply_augmentation(self, image, bboxes, params):
        """應用資料增強並更新標註框座標"""
        height, width = image.shape[:2]
        
        # 轉換為 HSV 色彩空間進行飽和度調整
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], params['saturation'])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 調整對比度和亮度
        image = cv2.convertScaleAbs(image, alpha=params['contrast'], beta=params['brightness'] * 255)
        
        # 旋轉
        if params['rotation'] != 0:
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, params['rotation'], 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            # 更新標註框座標
            for bbox in bboxes:
                # 將框的座標轉換為角點形式
                corners = np.array([
                    [bbox.x, bbox.y],
                    [bbox.x + bbox.w, bbox.y],
                    [bbox.x + bbox.w, bbox.y + bbox.h],
                    [bbox.x, bbox.y + bbox.h]
                ])
                
                # 應用旋轉矩陣
                corners = cv2.transform(corners.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)
                
                # 更新框的座標
                x_min = np.min(corners[:, 0])
                y_min = np.min(corners[:, 1])
                x_max = np.max(corners[:, 0])
                y_max = np.max(corners[:, 1])
                
                bbox.x = max(0, int(x_min))
                bbox.y = max(0, int(y_min))
                bbox.w = min(width - bbox.x, int(x_max - x_min))
                bbox.h = min(height - bbox.y, int(y_max - y_min))
        
        # 翻轉
        if params['flip_ud']:
            image = cv2.flip(image, 0)
            for bbox in bboxes:
                bbox.y = height - (bbox.y + bbox.h)
                
        if params['flip_lr']:
            image = cv2.flip(image, 1)
            for bbox in bboxes:
                bbox.x = width - (bbox.x + bbox.w)
                
        return image, bboxes
        
    def process_images(self):
        if not self.output_folder:
            QMessageBox.warning(self, "Error", tr("select_folder"))
            return
            
        if not self.parent or not self.parent.image_list:
            return
            
        try:
            # 創建進度對話框
            progress = QProgressDialog(tr("processing"), tr("cancel"), 0, len(self.parent.image_list), self)
            progress.setWindowModality(Qt.WindowModal)
            
            total_augmented = 0
            for i, image_path in enumerate(self.parent.image_list):
                if progress.wasCanceled():
                    break
                    
                progress.setValue(i)
                progress.setLabelText(tr("aug_progress").format(i + 1, len(self.parent.image_list)))
                
                # 讀取圖片和標註
                image = cv2.imread(image_path)
                if image is None:
                    continue
                    
                # 獲取標註檔案路徑
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                label_path = os.path.splitext(image_path)[0] + '.txt'
                
                # 讀取標註
                bboxes = []
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_idx = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                
                                # 轉換為像素座標
                                height, width = image.shape[:2]
                                x = int((x_center - w/2) * width)
                                y = int((y_center - h/2) * height)
                                w_pixels = int(w * width)
                                h_pixels = int(h * height)
                                
                                label = self.parent.classes[class_idx] if class_idx < len(self.parent.classes) else "unknown"
                                bboxes.append(BoundingBox(x, y, w_pixels, h_pixels, label))
                
                # 為每張圖片生成多個增強版本
                for aug_idx in range(self.aug_count_spin.value()):
                    # 生成隨機參數
                    params = self.get_random_params()
                    
                    # 應用增強
                    aug_image, aug_bboxes = self.apply_augmentation(image.copy(), [BoundingBox(b.x, b.y, b.w, b.h, b.label) for b in bboxes], params)
                    
                    # 生成輸出檔名（使用 _aug1, _aug2...）
                    aug_suffix = f"_aug{aug_idx + 1}"
                    aug_image_path = os.path.join(self.output_folder, base_name + aug_suffix + os.path.splitext(image_path)[1])
                    aug_label_path = os.path.join(self.output_folder, base_name + aug_suffix + '.txt')
                    
                    # 儲存增強後的圖片
                    cv2.imwrite(aug_image_path, aug_image)
                    
                    # 儲存標註
                    if aug_bboxes:
                        height, width = aug_image.shape[:2]
                        with open(aug_label_path, 'w') as f:
                            for bbox in aug_bboxes:
                                # 轉換回 YOLO 格式
                                x_center = (bbox.x + bbox.w/2) / width
                                y_center = (bbox.y + bbox.h/2) / height
                                w_norm = bbox.w / width
                                h_norm = bbox.h / height
                                try:
                                    class_idx = self.parent.classes.index(bbox.label)
                                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                                except ValueError:
                                    continue
                    
                    total_augmented += 1
                # 如果需要保留原始檔案，且來源與目標不同才複製
                if self.preserve_checkbox.isChecked():
                    import shutil
                    dst_img = os.path.join(self.output_folder, os.path.basename(image_path))
                    if os.path.abspath(image_path) != os.path.abspath(dst_img):
                        shutil.copy2(image_path, dst_img)
                    if os.path.exists(label_path):
                        dst_label = os.path.join(self.output_folder, os.path.basename(label_path))
                        if os.path.abspath(label_path) != os.path.abspath(dst_label):
                            shutil.copy2(label_path, dst_label)
            
            progress.setValue(len(self.parent.image_list))
            QMessageBox.information(self, tr("aug_complete"), tr("aug_success").format(total_augmented))
            # 如果輸出資料夾就是目前資料夾，自動 reload file list
            if hasattr(self.parent, 'current_dir') and os.path.abspath(self.output_folder) == os.path.abspath(self.parent.current_dir):
                self.parent.openDirectory()
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("aug_error"), str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LabelingTool()
    ex.show()
    sys.exit(app.exec_())
