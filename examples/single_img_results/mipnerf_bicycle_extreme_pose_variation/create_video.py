import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np


def filename_to_caption(filename):
    """
    Example:
    output_timestep_199.png
    ->
    noise-level: 199
    """
    stem = Path(filename).stem

    if stem.startswith("output_timestep_"):
        value = stem.replace("output_timestep_", "")
        return f"noise-level: {value}"

    return stem


def add_caption(image_path, caption, caption_height, font_size):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # Create canvas with extra space for caption
    canvas = Image.new(
        "RGB",
        (width, height + caption_height),
        color="white"
    )
    
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), caption, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = (width - text_width) // 2
    text_y = (caption_height - text_height) // 2

    draw.text(
        (text_x, text_y),
        caption,
        fill="black",
        font=font
    )
    
    
    canvas.paste(img, (0, caption_height))

    return np.array(canvas)


def create_video(input_folder, fps, caption_height, font_size):
    input_folder = Path(input_folder)

    if not input_folder.exists():
        print(f"Folder does not exist: {input_folder}")
        return

    output_video = f"{input_folder.name}.mp4"

    def extract_suffix_number(path):
        """
        Example:
        output_timestep_49.png -> 49

        If parsing fails, return a large number so fallback sorting still works.
        """
        try:
            stem = path.stem  # output_timestep_49
            return int(stem.split("_")[-1])
        except:
            return float("inf")

    image_paths = [
        p for p in input_folder.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    # First try sorting by numeric suffix, fallback naturally for failures
    image_paths = sorted(
        image_paths,
        key=lambda p: (extract_suffix_number(p), p.name)
    )

    if not image_paths:
        print("No images found.")
        return

    frames = []

    for img_path in image_paths:
        caption = filename_to_caption(img_path.name)

        print(f"Processing: {img_path.name} -> {caption}")
        frame = add_caption(
            img_path,
            caption,
            caption_height,
            font_size
        )
        frames.append(frame)

    print(f"Saving video to: {output_video}")

    writer = imageio.get_writer(output_video, fps=fps)

    for frame in frames:
        writer.append_data(frame)

    writer.close()

    print("Done!")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing images"
    )

    parser.add_argument(
        "-f", "--fps",
        type=int,
        default=0.5,
        help="Frames per second for output video (default: 0.5)"
    )

    parser.add_argument(
        "-c", "--caption_height",
        type=int,
        default=80,
        help="Height of caption area below image (default: 80)"
    )

    parser.add_argument(
        "-s", "--font_size",
        type=int,
        default=72,
        help="Font size for caption text (default: 32)"
    )

    args = parser.parse_args()

    create_video(
        input_folder=args.input_folder,
        fps=args.fps,
        caption_height=args.caption_height,
        font_size=args.font_size
    )


if __name__ == "__main__":
    main()
