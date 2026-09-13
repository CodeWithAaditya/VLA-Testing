"""Generate procedural PNG textures for the SO-101 tabletop scene (no PIL needed)."""
import struct, zlib, numpy as np, os

OUT = "Simulation/SO101/assets/textures"

def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    open(path, "wb").write(png)
    print("wrote", path, os.path.getsize(path) // 1024, "KB")

def fbm(shape, octaves=6, seed=0, tileable=True):
    """Tileable fractal value noise in [0,1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        res = 2 ** (o + 2)
        g = rng.random((res, res))
        # tile by wrapping: sample with periodic bilinear interpolation
        yy = np.linspace(0, res, shape[0], endpoint=False)
        xx = np.linspace(0, res, shape[1], endpoint=False)
        y0 = np.floor(yy).astype(int) % res; x0 = np.floor(xx).astype(int) % res
        y1 = (y0 + 1) % res; x1 = (x0 + 1) % res
        fy = (yy - np.floor(yy))[:, None]; fx = (xx - np.floor(xx))[None, :]
        fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
        top = g[np.ix_(y0, x0)] * (1 - fx) + g[np.ix_(y0, x1)] * fx
        bot = g[np.ix_(y1, x0)] * (1 - fx) + g[np.ix_(y1, x1)] * fx
        out += amp * (top * (1 - fy) + bot * fy)
        total += amp
        amp *= 0.5
    return out / total

N = 512

# ---- Oak table top: fine, mostly-straight grain lines + pore speckle ----
y, x = np.mgrid[0:N, 0:N] / N
warp = fbm((N, N), octaves=5, seed=7)
# Grain runs along +x. Keep the waviness small so it reads as sawn timber,
# not as dunes, and make the dark lines thin.
rings = np.sin((y * 58.0 + warp * 1.1 + np.sin(x * 1.7) * 0.22) * np.pi)
rings = np.abs(rings) ** 0.32
pores = fbm((N, N), octaves=8, seed=21)
grain = 0.62 * rings + 0.38 * (0.35 + 0.65 * pores)

light = np.array([0.74, 0.55, 0.36])   # sapwood
dark = np.array([0.45, 0.31, 0.19])    # grain lines
wood = dark[None, None, :] + (light - dark)[None, None, :] * np.clip(grain, 0, 1)[..., None]
# a few plank seams across the board
seam = (np.abs(((y * 3.0) % 1.0) - 0.5) > 0.492)
wood[seam] *= 0.72
# subtle per-pixel speckle for a matte satin finish
wood += (np.random.default_rng(3).random((N, N, 1)) - 0.5) * 0.022
write_png(f"{OUT}/wood_table.png", np.clip(wood * 255, 0, 255).astype(np.uint8))

# ---- Floor: matte grey studio concrete ----
c = fbm((N, N), octaves=8, seed=42)
c = 0.38 + 0.12 * c
floor = np.repeat(c[..., None], 3, axis=2) * np.array([1.0, 0.99, 0.97])[None, None, :]
floor += (np.random.default_rng(11).random((N, N, 1)) - 0.5) * 0.02
write_png(f"{OUT}/floor_concrete.png", np.clip(floor * 255, 0, 255).astype(np.uint8))
