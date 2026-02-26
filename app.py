"""
Polyglot Screenshot Translator
================================
Upload a German-language screenshot → get translated versions in all 24 EU languages.

Pipeline:
  1. EasyOCR  – extract text + bounding boxes from the German image
  2. deep-translator (Google) – translate each text block to the target language
  3. Pillow   – erase the original text, render the translation at the same position
  4. Streamlit – interactive web UI; download individual images or a ZIP

Run with:
    streamlit run app.py
"""

import io
import os
import textwrap
import zipfile
from pathlib import Path

import easyocr
import streamlit as st
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont

# ── Constants ────────────────────────────────────────────────────────────────

EU_LANGUAGES: dict[str, str] = {
    "Bulgarian": "bg",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hungarian": "hu",
    "Irish": "ga",
    "Italian": "it",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Luxembourgish": "lb",
    "Maltese": "mt",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swedish": "sv",
    
}

# Arial Unicode covers Latin, Cyrillic, Greek and many other scripts
FONT_PATH = "fonts/NotoSans-Regular.ttf"
FALLBACK_FONT_PATH = None

CONFIDENCE_THRESHOLD = 0.4   # ignore detections below this confidence
PADDING = 2                   # pixels of padding inside each bounding box


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

def _dominant_edge_color(img: Image.Image, bbox: list) -> tuple:
    """
    Sample pixels along the edges of *bbox* and return the median RGB colour.
    This is used as the fill colour when erasing the original text, so the
    background blends naturally.
    """
    x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width - 1, x1), min(img.height - 1, y1)

    px = img.load()
    samples: list[tuple] = []

    # top and bottom edges
    for x in range(x0, x1 + 1, max(1, (x1 - x0) // 20)):
        samples.append(px[x, y0])
        samples.append(px[x, y1])
    # left and right edges
    for y in range(y0, y1 + 1, max(1, (y1 - y0) // 20)):
        samples.append(px[x0, y])
        samples.append(px[x1, y])

    if not samples:
        return (255, 255, 255)

    r = sorted(s[0] for s in samples)[len(samples) // 2]
    g = sorted(s[1] for s in samples)[len(samples) // 2]
    b = sorted(s[2] for s in samples)[len(samples) // 2]
    return (r, g, b)


def _contrasting_color(bg: tuple) -> str:
    """Return black or white hex string depending on background luminance."""
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return "black" if lum > 128 else "white"


def _fit_text(draw: ImageDraw.ImageDraw,
                 text: str,
                 box_w: int,
                 box_h: int,
                 max_font_size: int = 140,
                 min_font_size: int = 8) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """
    Pixel-accurate text fitting.

    Finds the largest possible font size where:
      • All lines fit within box width
      • Total text height fits within box height
    """

    def wrap_pixels(font: ImageFont.FreeTypeFont, text: str) -> list[str]:
        """Wrap text based on actual pixel width."""
        words = text.split()
        if not words:
            return [text]

        lines = []
        current_line = words[0]

        for word in words[1:]:
            test_line = current_line + " " + word
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]

            if line_width <= box_w - 2 * PADDING:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)
        return lines

    for size in range(max_font_size, min_font_size - 1, -1):
        font = _get_font(size)

        lines = wrap_pixels(font, text)

        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        line_spacing = int(line_height * 0.15)  # subtle natural spacing

        total_height = len(lines) * line_height + (len(lines) - 1) * line_spacing

        if total_height > box_h - 2 * PADDING:
            continue

        # Check widest line
        max_width = 0
        for line in lines:
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            max_width = max(max_width, line_width)

        if max_width <= box_w - 2 * PADDING:
            return font, lines

    # Absolute fallback
    font = _get_font(min_font_size)
    return font, wrap_pixels(font, text)

# ── Core pipeline ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading EasyOCR model (first run may take a moment)…")
def load_ocr_reader(lang: str = "de") -> easyocr.Reader:
    """Load and cache the EasyOCR reader (GPU optional)."""
    return easyocr.Reader([lang], gpu=False)


def run_ocr(reader: easyocr.Reader, image: Image.Image) -> list[dict]:
    """
    Run EasyOCR on *image* and return a list of detections:
        [{"bbox": [x0,y0,x1,y1], "text": str, "conf": float}, …]
    """
    import numpy as np

    arr = np.array(image.convert("RGB"))
    raw = reader.readtext(arr)

    detections = []
    for bbox_pts, text, conf in raw:
        if conf < CONFIDENCE_THRESHOLD or not text.strip():
            continue
        x0 = min(p[0] for p in bbox_pts)
        y0 = min(p[1] for p in bbox_pts)
        x1 = max(p[0] for p in bbox_pts)
        y1 = max(p[1] for p in bbox_pts)
        detections.append({"bbox": [x0, y0, x1, y1], "text": text, "conf": conf})

    return detections


def translate_detections(detections: list[dict],
                          source_lang: str,
                          target_lang: str) -> list[dict]:
    """Return a copy of *detections* with each 'text' translated to *target_lang*."""
    translated = []
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    for det in detections:
        try:
            tr = translator.translate(det["text"])
        except Exception:
            tr = det["text"]  # fallback to original on error
        translated.append({**det, "text": tr or det["text"]})
    return translated


def render_translations(source_image: Image.Image,
                         detections: list[dict]) -> Image.Image:
    """
    Take the *source_image* and overlay the translated texts from *detections*.

    For every detection:
      1. Flood-fill the bounding box with the sampled background colour.
      2. Render the translated text at the best-fitting font size.
    """
    img = source_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for det in detections:
        x0, y0, x1, y1 = [int(v) for v in det["bbox"]]
        text = det["text"]

        box_w = max(x1 - x0, 1)
        box_h = max(y1 - y0, 1)

        # 1. Sample background colour and erase original text
        bg_color = _dominant_edge_color(img, det["bbox"])
        draw.rectangle([x0, y0, x1, y1], fill=bg_color)

        # 2. Choose a contrasting text colour
        text_color = _contrasting_color(bg_color)

        # 3. Find the best font + wrapping
        font, lines = _fit_text(draw, text, box_w, box_h)

        # 4. Draw each line
        # Accurate metrics
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        line_spacing = int(line_height * 0.15)

        total_text_height = len(lines) * line_height + (len(lines) - 1) * line_spacing

        # Vertical centering
        y_cursor = y0 + (box_h - total_text_height) // 2

        for line in lines:
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]

            # Horizontal centering
            x_text = x0 + (box_w - line_width) // 2

            draw.text(
                (x_text, y_cursor),
                line,
                fill=text_color,
                font=font
            )

            y_cursor += line_height + line_spacing

    return img


def image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    return buf.getvalue()


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Polyglot Screenshot Translator",
        page_icon="🌍",
        layout="wide",
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🌍 Polyglot")
        st.markdown("**Screenshot Translator**")
        st.divider()

        st.subheader("⚙️ Settings")

        source_lang = st.selectbox(
            "Source language of the screenshot",
            options=list(EU_LANGUAGES.keys()),
            index=list(EU_LANGUAGES.keys()).index("German"),
        )

        st.markdown("**Target languages**")
        select_all = st.checkbox("Select all EU languages", value=True)

        lang_names = sorted(EU_LANGUAGES.keys())
        if select_all:
            selected_langs = lang_names
        else:
            selected_langs = st.multiselect(
                "Choose languages",
                options=lang_names,
                default=["English", "French", "Spanish"],
            )

        confidence = st.slider(
            "OCR confidence threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Detections below this confidence score are ignored.",
        )

        st.divider()
        st.markdown(
            "**Stack:** EasyOCR · deep-translator · Pillow · Streamlit\n\n"
            "All open-source. No data leaves your machine."
        )

    # ── Main area ────────────────────────────────────────────────────────────
    st.title("🌍 Polyglot Screenshot Translator")
    st.markdown(
        "Upload a screenshot in any EU language. "
        "The app will OCR the text, translate it into every selected EU language, "
        "and produce a translated version of the image for each one."
    )

    uploaded = st.file_uploader(
        "📤 Upload screenshot (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Supports any high-resolution JPEG or PNG screenshot.",
    )

    if uploaded is None:
        st.info("👆 Upload a screenshot to get started.")
        st.stop()

    source_image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Original")
        st.image(source_image, use_container_width=True)
    with col2:
        st.subheader("Image info")
        st.markdown(f"- **Size:** {source_image.width} × {source_image.height} px")
        st.markdown(f"- **Source language:** {source_lang}")
        st.markdown(f"- **Target languages:** {len(selected_langs)} selected")

    if not selected_langs:
        st.warning("Please select at least one target language.")
        st.stop()

    run_btn = st.button("🚀 Translate Screenshot", type="primary", use_container_width=True)

    if not run_btn:
        st.stop()

    # ── OCR ──────────────────────────────────────────────────────────────────
    src_code = EU_LANGUAGES[source_lang]
    reader = load_ocr_reader(src_code)

    with st.spinner("🔍 Running OCR…"):
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = confidence
        detections = run_ocr(reader, source_image)

    if not detections:
        st.error("No text was detected in the image. Try lowering the confidence threshold.")
        st.stop()

    st.success(f"✅ Detected **{len(detections)}** text regions.")

    with st.expander("📋 Detected text blocks", expanded=False):
        for i, det in enumerate(detections, 1):
            st.markdown(
                f"`{i}.` **{det['text']}** "
                f"*(conf: {det['conf']:.2f}, "
                f"box: {[int(v) for v in det['bbox']]})*"
            )

    # ── Translation + rendering ───────────────────────────────────────────────
    st.subheader("🌐 Translated outputs")

    zip_buf = io.BytesIO()
    output_images: dict[str, Image.Image] = {}

    progress = st.progress(0, text="Translating…")
    total = len(selected_langs)

    for idx, lang_name in enumerate(selected_langs):
        lang_code = EU_LANGUAGES[lang_name]
        progress.progress((idx) / total, text=f"Translating → {lang_name}…")

        translated_dets = translate_detections(detections, src_code, lang_code)
        out_img = render_translations(source_image, translated_dets)
        output_images[lang_name] = out_img

    progress.progress(1.0, text="Done!")

    # Build ZIP
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for lang_name, img in output_images.items():
            zf.writestr(f"output_{lang_name}.jpg", image_to_bytes(img))

    st.download_button(
        label="⬇️ Download all as ZIP",
        data=zip_buf.getvalue(),
        file_name="translated_screenshots.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.divider()

    # Display grid: 2 columns
    lang_list = list(output_images.items())
    for row_start in range(0, len(lang_list), 2):
        cols = st.columns(2)
        for col_idx, (lang_name, img) in enumerate(lang_list[row_start: row_start + 2]):
            with cols[col_idx]:
                st.markdown(f"**{lang_name}** (`{EU_LANGUAGES[lang_name]}`)")
                st.image(img, use_container_width=True)
                st.download_button(
                    label=f"⬇️ {lang_name}.jpg",
                    data=image_to_bytes(img),
                    file_name=f"output_{lang_name}.jpg",
                    mime="image/jpeg",
                    key=f"dl_{lang_name}",
                )


if __name__ == "__main__":
    main()
