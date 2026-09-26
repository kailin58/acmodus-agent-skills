# Vyton安全壳 v1(P1):输入白名单+完整性校验+乱码注入+输出白名单
import hashlib, json, os

IMG_MAGICS = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"RIFF", "webp"),
    (b"GIF8", "gif"),
]
MAX_IMG_BYTES = 20 * 1024 * 1024

def validate_input(data):
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0 or len(data) > MAX_IMG_BYTES:
        return None
    for magic, name in IMG_MAGICS:
        if data[:len(magic)] == magic:
            if name == "webp" and data[8:12] != b"WEBP":
                return None
            return name
    return None

def garbage_frame(seed_material, w=256, h=256):
    hsh = hashlib.sha256(b"vyton-salt" + seed_material).digest()
    out = bytearray()
    counter = 0
    while len(out) < w * h * 3:
        out += hashlib.sha256(hsh + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:w*h*3])

def build_manifest(model_dir):
    manifest = {}
    for f in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, f)
        if os.path.isfile(p):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            manifest[f] = {"sha256": h.hexdigest(), "size": os.path.getsize(p)}
    return manifest

def verify_manifest(model_dir, manifest):
    for f, meta in manifest.items():
        p = os.path.join(model_dir, f)
        if not os.path.exists(p) or os.path.getsize(p) != meta["size"]:
            return False
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != meta["sha256"]:
            return False
    return True

def corrupt_weights(path):
    with open(path, "rb") as f:
        data = bytearray(f.read())
    for i in range(0, len(data), 17):
        data[i] ^= 0xA5
    with open(path, "wb") as f:
        f.write(bytes(data))

ALLOWED_SCHEMA = {
    "cutout": {"image": bytes, "foreground_ratio": float},
    "tryon":  {"image": bytes, "quality_score": float},
    "me":     {"mesh": bytes, "face_score": float},
    "glow":   {"image": bytes, "quality_score": float},
    "style":  {"colors": list, "categories": list, "quality_score": float},
    "measure":{"measurements": dict, "confidence": dict},
    "pet":    {"breed": str, "keypoints": list, "personality": dict},
}

def validate_output(module, payload):
    if module not in ALLOWED_SCHEMA:
        return False
    spec = ALLOWED_SCHEMA[module]
    for k, t in spec.items():
        if k not in payload or not isinstance(payload[k], t):
            return False
    if set(payload.keys()) - set(spec.keys()):
        return False
    return True

if __name__ == "__main__":
    ok = 0; total = 0
    def check(name, cond):
        global ok, total
        total += 1; ok += bool(cond)
        print(("PASS" if cond else "FAIL"), name)
    check("jpeg pass", validate_input(b"\xff\xd8\xff\xe0" + b"\x00"*100) == "jpeg")
    check("png pass", validate_input(b"\x89PNG\r\n\x1a\n" + b"\x00"*100) == "png")
    check("webp pass", validate_input(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00"*100) == "webp")
    check("text reject", validate_input(b"hello world") is None)
    check("tensor reject", validate_input(b"\x00\x01\x02\x03"*8) is None)
    check("20MB+ reject", validate_input(b"\xff\xd8\xff" + b"\x00"*(21*1024*1024)) is None)
    check("empty reject", validate_input(b"") is None)
    g1 = garbage_frame(b"test"); g2 = garbage_frame(b"test"); g3 = garbage_frame(b"other")
    check("garbage deterministic", g1 == g2 and g1 != g3 and len(g1) == 256*256*3)
    import tempfile
    d = tempfile.mkdtemp()
    open(os.path.join(d, "m.bin"), "wb").write(b"model-bytes")
    man = build_manifest(d)
    check("manifest pass", verify_manifest(d, man))
    open(os.path.join(d, "m.bin"), "wb").write(b"model-tampered")
    check("tamper detected", not verify_manifest(d, man))
    check("valid output", validate_output("glow", {"image": b"\x00", "quality_score": 0.9}))
    check("extra field reject", not validate_output("glow", {"image": b"\x00", "quality_score": 0.9, "hack": 1}))
    check("unknown module reject", not validate_output("evil", {"image": b"\x00"}))
    print("SECURITY_SHELL " + str(ok) + "/" + str(total))
