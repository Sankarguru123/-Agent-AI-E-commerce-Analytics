from PIL import Image, ImageDraw, ImageFont

# Input & Output paths
input_path = "IMG_20250827_225757.jpg"   # உங்கள் image file பெயர்
output_path = "scrolling_text.gif"       # output GIF

# Load image
img = Image.open(input_path).convert("RGBA")
w, h = img.size

# Tamil text to scroll
msg = "சாஸ்தா சந்தா குரூப்ஸ்"

# Font setup
try:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Linux default
    font = ImageFont.truetype(font_path, size=int(h * 0.08))
except Exception:
    font = ImageFont.load_default()

# Measure text width & height (using textbbox instead of textsize)
dummy_img = Image.new("RGBA", (w, h))
draw = ImageDraw.Draw(dummy_img)
bbox = draw.textbbox((0, 0), msg, font=font)
text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

# Create frames for scrolling (right-to-left)
frames = []
duration_per_frame = 40  # milliseconds
scroll_width = text_width + w

for offset in range(scroll_width):
    base = img.copy()
    draw = ImageDraw.Draw(base)
    x = w - offset  # move text from right to left
    y = int(h * 0.05)  # top margin
    draw.text((x, y), msg, font=font, fill=(255, 255, 255, 255))
    frames.append(base.convert("P", palette=Image.ADAPTIVE))

# Save animated GIF
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=duration_per_frame,
    loop=0
)

print("GIF saved:", output_path)
