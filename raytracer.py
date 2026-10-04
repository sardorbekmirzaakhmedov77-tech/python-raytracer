"""
A ray tracer written from scratch in pure Python + NumPy.

No graphics engine, no OpenGL, no game library: every pixel is computed by
shooting rays of light into a 3D scene and doing the geometry by hand.

Features
  * Spheres and an infinite checkerboard floor
  * Blinn-Phong lighting (diffuse + specular highlights)
  * Hard shadows from two coloured lights
  * Recursive mirror reflections with Fresnel (Schlick) falloff
  * Glossy / metallic / matte materials
  * Gradient sky with a sun glow
  * Supersampling anti-aliasing, ACES tone mapping, gamma correction
  * Fully vectorised: every ray in the image is traced at once as NumPy arrays

Usage
  python raytracer.py                 # renders render.png  (1280x720)
  python raytracer.py --animate       # also renders orbit.mp4 + orbit.gif
"""

import argparse
import time
import numpy as np
from PIL import Image

EPS = 1e-4
FAR = 1e9


# --------------------------------------------------------------------------- #
#  Vector helpers
# --------------------------------------------------------------------------- #
def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def dot(a, b):
    return np.sum(a * b, axis=-1)


# --------------------------------------------------------------------------- #
#  Scene objects
# --------------------------------------------------------------------------- #
class Material:
    def __init__(self, color, diffuse=0.9, specular=0.5, shininess=60,
                 reflect=0.0, metal=False):
        self.color = np.array(color, dtype=np.float64)
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflect = reflect        # base reflectivity (0 = matte, 1 = mirror)
        self.metal = metal            # metals tint their reflections


class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=np.float64)
        self.radius = radius
        self.material = material

    def intersect(self, O, D):
        """Distance along each ray to the sphere (FAR if missed)."""
        L = O - self.center
        b = dot(D, L)
        c = dot(L, L) - self.radius ** 2
        disc = b * b - c
        hit = disc > 0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        t0, t1 = -b - sq, -b + sq
        t = np.where(t0 > EPS, t0, np.where(t1 > EPS, t1, FAR))
        return np.where(hit, t, FAR)

    def normal(self, P):
        return normalize(P - self.center)

    def color_at(self, P):
        return np.broadcast_to(self.material.color, P.shape)


class Plane:
    def __init__(self, y, material, alt_color, tile=1.0):
        self.y = y
        self.material = material
        self.alt = np.array(alt_color, dtype=np.float64)
        self.tile = tile

    def intersect(self, O, D):
        dy = D[:, 1]
        safe = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
        t = (self.y - O[:, 1]) / safe
        return np.where((t > EPS) & (np.abs(dy) > 1e-9), t, FAR)

    def normal(self, P):
        n = np.zeros_like(P)
        n[:, 1] = 1.0
        return n

    def color_at(self, P):
        check = (np.floor(P[:, 0] / self.tile) + np.floor(P[:, 2] / self.tile)) % 2
        return np.where(check[:, None] == 0, self.material.color, self.alt)


class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position, dtype=np.float64)
        self.color = np.array(color, dtype=np.float64) * intensity


# --------------------------------------------------------------------------- #
#  Sky
# --------------------------------------------------------------------------- #
SUN_DIR = normalize(np.array([-0.55, 0.10, -1.0]))
HORIZON = np.array([0.95, 0.42, 0.20])
ZENITH = np.array([0.02, 0.035, 0.12])


def sky(D):
    t = np.clip(D[:, 1], 0, 1)[:, None]
    col = ZENITH + (HORIZON - ZENITH) * np.exp(-t * 9.0)
    sun = np.clip(dot(D, SUN_DIR), 0, 1)[:, None]
    col = col + np.array([1.0, 0.65, 0.35]) * (sun ** 3000 * 30 + sun ** 120 * 0.5 + sun ** 10 * 0.12)
    return col


# --------------------------------------------------------------------------- #
#  Tracing
# --------------------------------------------------------------------------- #
def trace(scene, lights, O, D, max_depth=4):
    n = O.shape[0]
    color = np.zeros((n, 3))
    throughput = np.ones((n, 3))
    alive = np.arange(n)

    for depth in range(max_depth + 1):
        if alive.size == 0:
            break
        o, d = O[alive], D[alive]

        # nearest hit across all objects
        dists = np.stack([obj.intersect(o, d) for obj in scene])
        nearest = np.argmin(dists, axis=0)
        t = dists[nearest, np.arange(alive.size)]

        missed = t >= FAR
        color[alive[missed]] += throughput[alive[missed]] * sky(d[missed])

        new_O, new_D, keep = [], [], []
        for k, obj in enumerate(scene):
            sel = (nearest == k) & ~missed
            if not sel.any():
                continue
            idx = alive[sel]
            P = o[sel] + d[sel] * t[sel, None]
            N = obj.normal(P)
            V = -d[sel]
            m = obj.material
            base = obj.color_at(P)
            P_off = P + N * EPS * 10

            # ambient / sky bounce approximation
            shade = base * (0.06 + 0.10 * np.clip(N[:, 1:2], 0, 1))

            for light in lights:
                toL = light.position - P_off
                distL = np.linalg.norm(toL, axis=1)
                Ld = toL / distL[:, None]

                # shadow ray
                shadow_t = np.min(np.stack([s.intersect(P_off, Ld) for s in scene]), axis=0)
                lit = (shadow_t > distL).astype(np.float64)[:, None]

                diff = np.clip(dot(N, Ld), 0, 1)[:, None] * m.diffuse
                H = normalize(Ld + V)
                spec = np.clip(dot(N, H), 0, 1)[:, None] ** m.shininess * m.specular
                spec_col = base if m.metal else 1.0
                shade = shade + lit * light.color * (base * diff + spec_col * spec)

            # Fresnel (Schlick): surfaces get more mirror-like at grazing angles
            cos_i = np.clip(dot(N, V), 0, 1)[:, None]
            fres = m.reflect + (1 - m.reflect) * (1 - cos_i) ** 5
            fres = fres * (1.0 if m.reflect > 0 else 0.25)

            # atmospheric fog: distant surfaces melt into the horizon glow
            fog = np.exp(-t[sel] * 0.018)[:, None]
            color[idx] += throughput[idx] * (shade * (1 - fres) * fog + sky(d[sel]) * (1 - fog))

            if depth < max_depth:
                R = d[sel] - 2 * dot(d[sel], N)[:, None] * N
                tint = base if m.metal else 1.0
                throughput[idx] *= fres * tint * fog
                strong = np.max(throughput[idx], axis=1) > 0.01
                new_O.append(P_off[strong])
                new_D.append(normalize(R[strong]))
                keep.append(idx[strong])

        if not keep:
            break
        alive = np.concatenate(keep)
        O = O.copy()
        D = D.copy()
        O[alive] = np.concatenate(new_O)
        D[alive] = np.concatenate(new_D)

    return color


# --------------------------------------------------------------------------- #
#  Camera
# --------------------------------------------------------------------------- #
def camera_rays(eye, target, width, height, fov=50, ss=2):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = normalize(target - eye)
    right = normalize(np.cross(fwd, [0, 1, 0]))
    up = np.cross(right, fwd)
    aspect = width / height
    scale = np.tan(np.radians(fov) / 2)

    W, H = width * ss, height * ss
    xs = (np.arange(W) + 0.5) / W * 2 - 1
    ys = 1 - (np.arange(H) + 0.5) / H * 2
    px, py = np.meshgrid(xs * aspect * scale, ys * scale)
    D = fwd + px[..., None] * right + py[..., None] * up
    D = normalize(D.reshape(-1, 3))
    O = np.broadcast_to(eye, D.shape).copy()
    return O, D


def tonemap(c):
    # ACES filmic approximation + gamma
    a, b, cc, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    c = np.clip((c * (a * c + b)) / (c * (cc * c + d) + e), 0, 1)
    return (c ** (1 / 2.2) * 255).astype(np.uint8)


def render(scene, lights, eye, target, width, height, ss=2, depth=4):
    O, D = camera_rays(eye, target, width, height, ss=ss)
    col = trace(scene, lights, O, D, depth)
    col = col.reshape(height * ss, width * ss, 3)
    col = col.reshape(height, ss, width, ss, 3).mean(axis=(1, 3))  # anti-alias

    # subtle vignette
    yy, xx = np.mgrid[0:height, 0:width]
    r = np.hypot((xx - width / 2) / width, (yy - height / 2) / height)
    col *= (1 - 0.55 * r ** 2)[..., None]
    return tonemap(col)


# --------------------------------------------------------------------------- #
#  The scene
# --------------------------------------------------------------------------- #
def build_scene(phase=0.0):
    chrome = Material([0.95, 0.95, 0.97], diffuse=0.1, specular=1.0, shininess=200, reflect=0.9, metal=True)
    gold = Material([1.0, 0.76, 0.33], diffuse=0.3, specular=1.0, shininess=120, reflect=0.75, metal=True)
    ruby = Material([0.85, 0.08, 0.12], diffuse=0.9, specular=0.9, shininess=150, reflect=0.08)
    jade = Material([0.10, 0.65, 0.45], diffuse=0.9, specular=0.7, shininess=90, reflect=0.06)
    cobalt = Material([0.12, 0.30, 0.95], diffuse=0.9, specular=0.9, shininess=150, reflect=0.10)
    ivory = Material([0.95, 0.92, 0.85], diffuse=0.95, specular=0.3, shininess=30, reflect=0.03)
    floor = Material([0.78, 0.78, 0.80], diffuse=0.9, specular=0.25, shininess=40, reflect=0.22)

    scene = [
        Plane(0.0, floor, alt_color=[0.08, 0.08, 0.10], tile=1.0),
        Sphere([0.0, 1.2, 0.0], 1.2, chrome),
        Sphere([-2.5, 0.8, 0.8], 0.8, gold),
    ]

    # a ring of small coloured spheres that bob up and down with `phase`
    ring = [ruby, jade, cobalt, ivory, gold, ruby, cobalt, jade]
    for i, mat in enumerate(ring):
        a = i / len(ring) * 2 * np.pi
        r = 0.42 if i % 2 else 0.5
        y = r + 0.25 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi * 2 + a * 2))
        scene.append(Sphere([3.0 * np.cos(a) + 0.5, y, 3.0 * np.sin(a) - 0.8], r, mat))

    scene.append(Sphere([2.2, 0.65, 2.0], 0.65, ruby))

    lights = [
        Light([-6, 9, 6], [1.0, 0.92, 0.80], 1.15),   # warm key light
        Light([7, 5, -3], [0.45, 0.60, 1.00], 0.55),   # cool rim light
    ]
    return scene, lights


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--frames", type=int, default=90)
    args = ap.parse_args()

    target = [0.3, 0.7, -0.2]

    t0 = time.time()
    scene, lights = build_scene(0.0)
    img = render(scene, lights, eye=[4.2, 2.1, 6.8], target=target,
                 width=args.width, height=args.height, ss=args.ss, depth=5)
    Image.fromarray(img).save("render.png")
    print(f"render.png  {args.width}x{args.height}  in {time.time() - t0:.1f}s")

    if args.animate:
        import imageio.v2 as imageio
        frames = []
        t0 = time.time()
        for f in range(args.frames):
            phase = f / args.frames
            ang = phase * 2 * np.pi + 1.0
            eye = [10 * np.cos(ang), 3.0 + 0.8 * np.sin(ang * 2), 10 * np.sin(ang)]
            scene, lights = build_scene(phase)
            frames.append(render(scene, lights, eye, target, 640, 360, ss=2, depth=4))
            print(f"\rframe {f + 1}/{args.frames}", end="", flush=True)
        print(f"\nanimation in {time.time() - t0:.1f}s")
        imageio.mimsave("orbit.mp4", frames, fps=30, quality=9, macro_block_size=8)
        small = [np.asarray(Image.fromarray(fr).resize((480, 270), Image.LANCZOS)) for fr in frames[::2]]
        imageio.mimsave("orbit.gif", small, duration=1 / 15, loop=0)


if __name__ == "__main__":
    main()
