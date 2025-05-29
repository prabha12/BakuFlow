# 语言切换指南 / Language Switching Guide

本项目支持多种语言界面。以下是更改语言的几种方法：

This project supports multiple language interfaces. Here are several ways to change the language:

## 支持的语言 / Supported Languages

- **English (en)** - 英文 ✅
- **Traditional Chinese (zh-tw)** - 繁体中文 ✅
- **Simplified Chinese (zh-cn)** - 简体中文 ✅
- **Japanese (ja)** - 日文 ✅
- **Italian (it)** - 意大利文 ✅
- **German (de)** - 德文 ✅
- **Norwegian (no)** - 挪威文 ✅
- **Spanish (es)** - 西班牙文 ✅
- **French (fr)** - 法文 ✅

## 方法1：使用语言切换脚本 (推荐)

### 查看当前语言设置
```bash
python change_language.py
```

### 切换到简体中文
```bash
python change_language.py zh-cn
```

### 切换到英文
```bash
python change_language.py en
```

### 切换到其他语言
```bash
python change_language.py zh-tw  # 繁体中文
python change_language.py ja     # 日文
python change_language.py fr     # 法文
python change_language.py de     # 德文
python change_language.py es     # 西班牙文
python change_language.py it     # 意大利文
python change_language.py no     # 挪威文
```

## 方法2：直接编辑配置文件

编辑项目根目录下的 `language_config.py` 文件：

```python
# 将 LANGUAGE 变量设置为你想要的语言代码
LANGUAGE = 'zh-cn'  # 简体中文
# LANGUAGE = 'en'     # 英文
# LANGUAGE = 'zh-tw'  # 繁体中文
# LANGUAGE = 'ja'     # 日文
```

## 方法3：修改本地化文件

编辑 `labelimg/core/localization.py` 文件，找到以下行：

```python
# 手动语言设置 - 如果你想强制使用特定语言，请取消注释并设置以下变量
# 支持的语言代码: 'en', 'zh-tw', 'zh-cn', 'ja', 'it', 'de', 'no', 'es', 'fr'
# MANUAL_LANG = 'zh-cn'  # 取消注释此行并设置你想要的语言代码
```

取消注释最后一行并设置你想要的语言代码：

```python
MANUAL_LANG = 'zh-cn'  # 设置为简体中文
```

## 重要提示 / Important Notes

⚠️ **更改语言设置后，需要重启应用程序才能生效！**

⚠️ **After changing language settings, you need to restart the application for changes to take effect!**

## 语言优先级 / Language Priority

系统按以下优先级选择语言：

1. **手动设置** (`MANUAL_LANG` in `localization.py`)
2. **配置文件** (`language_config.py`)
3. **系统自动检测** (基于操作系统语言设置)
4. **默认语言** (英文)

## 故障排除 / Troubleshooting

### 问题：语言没有切换成功
**解决方案：**
1. 确认语言代码正确（参考上面的支持语言列表）
2. 重启应用程序
3. 检查配置文件是否正确保存

### 问题：出现乱码
**解决方案：**
1. 确保系统支持相应的字符编码
2. 检查字体是否支持相应语言的字符
3. 尝试切换到英文，然后再切换回目标语言

### 问题：找不到配置文件
**解决方案：**
1. 运行 `python change_language.py` 会自动创建配置文件
2. 或者手动创建 `language_config.py` 文件

## 示例 / Examples

### 快速切换到简体中文
```bash
# 方法1：使用脚本
python change_language.py zh-cn

# 方法2：直接编辑配置文件
echo "LANGUAGE = 'zh-cn'" > language_config.py
```

### 查看所有可用语言
```bash
python change_language.py --help
```

---

如果你遇到任何问题，请检查控制台输出的错误信息，或者提交issue。

If you encounter any issues, please check the console output for error messages or submit an issue. 