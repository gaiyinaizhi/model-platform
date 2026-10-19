"""Private bounded Pillow renderer. No model, runtime, or database access."""
import hashlib
import io
import os
import signal
import sys
import threading
import time
import warnings

MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_EDGE = 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024


class BoundedOutput(io.BytesIO):
    def write(self, value):
        if self.tell() + len(value) > MAX_OUTPUT_BYTES:
            raise ValueError('Output budget')
        return super().write(value)


def render(raw, width, height, pixel_hash):
    from PIL import Image
    if not 0 < len(raw) <= MAX_SOURCE_BYTES or not 0 < width * height <= MAX_PIXELS:
        raise ValueError('Source budget')
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as original:
            if (original.format not in ('PNG', 'JPEG') or original.size != (width, height)
                    or getattr(original, 'n_frames', 1) != 1):
                raise ValueError('Image identity')
            original.verify()
        with Image.open(io.BytesIO(raw)) as original:
            original.load()
            # Deliberately no EXIF transpose or ICC transform: evaluation used
            # these exact raw-orientation RGB pixels in original coordinates.
            rgb = original.convert('RGB')
            pixels = rgb.tobytes()
            if hashlib.sha256(f'{width}x{height}:'.encode('ascii') + pixels).hexdigest() != pixel_hash:
                raise ValueError('Pixel identity')
            fresh = Image.frombytes('RGB', (width, height), pixels)
            del pixels
            scale = min(1.0, MAX_EDGE / max(width, height))
            target = (max(1, int(width * scale)), max(1, int(height * scale)))
            if target != fresh.size:
                fresh = fresh.resize(target, Image.Resampling.LANCZOS)
            output = BoundedOutput()
            fresh.save(output, format='PNG')
            return output.getvalue()


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.alarm(5)
    parent = int(sys.argv[4])
    def watch_parent():
        while True:
            if os.getppid() != parent:
                os._exit(5)
            time.sleep(.1)
    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        raw = sys.stdin.buffer.read(MAX_SOURCE_BYTES + 1)
        result = render(raw, int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
    except ImportError:
        return 3
    except Exception:
        return 4
    sys.stdout.buffer.write(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
