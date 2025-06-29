from PIL import Image, ImageDraw, ImageFont
from typing import Sequence
import pathlib, re, numpy as np
from PIL import Image, ImageDraw, ImageFont
import pathlib


ROWS, COLS      = 4, 16                # 4 rows × 16 columns = 64 buckets
CELL_W, CELL_H  = 110, 90             
PAD             = 24                   # outer margin
IMG_W           = PAD*2 + COLS*CELL_W
EXTRA_H = 56
IMG_H           = PAD*2 + ROWS*CELL_H + EXTRA_H   

# Colours (RGB)
BG      = (250, 250, 250)              
FG      = ( 20,  20,  20)              
ACCENT  = (  0, 120, 220)              
GRID    = (220, 220, 220)              

# Typography
FONT_SIZE = 16
FONT_PATH = pathlib.Path(__file__).parent / "fonts" / "DejaVuSans.ttf"
FONT      = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
def _get_font(size:int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        # Fallback: grab any installed mono .ttf
        monos = [p for p in pathlib.Path("/usr/share/fonts").rglob("*.ttf")
                 if re.search("mono", p.name, re.I)]
        if monos:
            return ImageFont.truetype(str(monos[0]), size)
        return ImageFont.load_default()
FONT = _get_font()

FPS      = 6      
TWEEN    = 4      


def _draw_frame(tokens: Sequence[str],
                prev:   Sequence[str] | None,
                k: int, K: int,
                t: float) -> Image.Image:
    """
    Render one tween frame.
      tokens : target tokens of this step
      prev   : tokens from previous step (None for step 0)
      k      : index of *this* diffusion step   (0 … K-1)
      K      : total number of steps
      t      : tween progress toward `tokens` (0.0 … 1.0)
    """
    img = Image.new("RGB", (IMG_W, IMG_H), BG)
    dr  = ImageDraw.Draw(img)


    for r in range(ROWS + 1):
        y = PAD + r*CELL_H
        dr.line([(PAD, y), (IMG_W - PAD, y)], fill=GRID)
    for c in range(COLS + 1):
        x = PAD + c*CELL_W
        dr.line([(x, PAD), (x, PAD + ROWS*CELL_H)], fill=GRID)

    
    EXTRA_H  = 56                                  
    bar_top  = IMG_H - EXTRA_H + 28               
    bar_bot  = bar_top + 12                     
    bar_w    = IMG_W - 2*PAD
    pct      = (k + t) / (K - 1)                  
    fill_w   = int(bar_w * pct)
    fill_w   = min(bar_w, fill_w)              

    # outline
    dr.rectangle([(PAD, bar_top), (PAD + bar_w, bar_bot)],
                 outline=GRID, width=1)
    # fill
    dr.rectangle([(PAD, bar_top), (PAD + fill_w, bar_bot)],
                 fill=ACCENT)

    label_y = bar_top - FONT_SIZE - 4              # plenty of room
    dr.text((PAD, label_y), f"step {k}/{K-1}",
            font=FONT, fill=FG)

-
    for idx, tok in enumerate(tokens):
        r, c = divmod(idx, COLS)
        cx = PAD + c*CELL_W + CELL_W/2
        cy = PAD + r*CELL_H + CELL_H/2
        changed = prev and tok != prev[idx]

        # simple fade-in accent when a token changes
        if changed:
            if t < 0.5:
                blend = int(255 * (t/0.5))          
                col = tuple(FG[i] + (ACCENT[i]-FG[i])*blend//255 for i in range(3))
            else:
                col = ACCENT
        else:
            col = FG

        dr.text((cx, cy), tok, font=FONT, fill=col, anchor="mm")

    return img


def make_video(trace,
               fps: int = 6,
               path: str = "diffusion.mp4",
               tween: int = 2):
    """
    Memory-safe: streams frames to disk, no huge list in RAM.
    """
    K = len(trace)
    if K == 0:
        raise ValueError("trace is empty")
    for i, step in enumerate(trace):
        if len(step) != 64:

            print(f"step {i} has {len(step)} tokens (need 64)")
            step = step[:64] + [""]*(64-len(step))
            trace[i] = step
            print(f"truncated to 64 tokens")

    # • open the encoder once
    import imageio   # ← v2 API
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:

        prev = None
        for k, tokens in enumerate(trace):
            for f in range(tween):
                t = f / tween
                frame = _draw_frame(tokens, prev, k, K, t)
                writer.append_data(np.asarray(frame))
            prev = tokens

    print(f"saved {K*tween} frames to {path}")