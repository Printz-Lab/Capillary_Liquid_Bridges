import numpy as np
from pathlib import Path
from tkinter import filedialog, Tk
import sys 

# Add parent directory to the module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import lines_file, image_dir

def shift_line_y(line, dy):
    """Shift a substrate line in y-direction by dy pixels"""
    if line is None:
        return None
    return (
        (line[0][0], line[0][1] + dy),
        (line[1][0], line[1][1] + dy)
    )

def shift_lines(top_lines, bottom_lines):
    new_top = [shift_line_y(line, +3) if line is not None else None for line in top_lines]
    new_bottom = [shift_line_y(line, -3) if line is not None else None for line in bottom_lines]
    return new_top, new_bottom


# === Main ===
if __name__ == "__main__":
    # File selection
    npz_path = lines_file

    if not npz_path:
        print("❌ No file selected, exiting.")
        exit()

    data = np.load(npz_path, allow_pickle=True)
    top_lines = list(data["top_lines"])
    bottom_lines = list(data["bottom_lines"])

    new_top, new_bottom = shift_lines(top_lines, bottom_lines)

    save_path = Path(npz_path).with_name("shifted_" + Path(npz_path).name)
    np.savez(
        save_path,
        top_lines=np.array(new_top, dtype=object),
        bottom_lines=np.array(new_bottom, dtype=object)
    )

    print(f"✅ Shifted lines saved to: {save_path}")
