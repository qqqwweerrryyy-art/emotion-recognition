import argparse
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from extract_faces import load_detector
from train_audio import MAX_LEN, SR, Net
from train_face import CLASSES, build_model

RU = {"neutral": "нейтральная", "happy": "радость", "sad": "грусть", "angry": "злость",
      "fearful": "страх", "disgust": "отвращение", "surprised": "удивление"}
EN = {k: k for k in RU}
COLORS = {"neutral": (140, 140, 140), "happy": (60, 170, 40), "sad": (200, 110, 30),
          "angry": (40, 40, 220), "fearful": (170, 50, 170), "disgust": (30, 130, 110),
          "surprised": (0, 140, 240)}
FONT_PATHS = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
LABEL_H = 30
SILENCE_RMS = 0.003


def load_font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return None


def draw_labels(frame, labels, font):
    """labels: список (текст, (x, y) левый верхний угол, цвет BGR)."""
    if font is None:
        for text, (x, y), color in labels:
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y), (x + tw + 8, y + th + base + 6), color, -1)
            cv2.putText(frame, text, (x + 4, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for text, (x, y), color in labels:
        l, t, r, b = d.textbbox((x, y), text, font=font)
        d.rectangle([l - 4, t - 3, r + 4, b + 3], fill=(color[2], color[1], color[0]))
        d.text((x, y), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def detect_face(det, frame, max_w=640):
    h, w = frame.shape[:2]
    s = min(1.0, max_w / w)
    small = cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return np.array([x, y, fw, fh], dtype=np.float32) / s


def crop_box(frame, box, margin=0.3, size=224):
    x, y, w, h = box
    cx, cy = int(x + w / 2), int(y + h / 2)
    half = int(max(w, h) * (1 + 2 * margin) / 2)
    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    x1, y1 = min(cx + half, frame.shape[1]), min(cy + half, frame.shape[0])
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


FACE_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def face_prob(model, crop_bgr, device):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = FACE_TF(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(model(x).float(), 1)[0].cpu().numpy()


def voice_windows(video, device, mode):
    """Список (время центра окна, вероятности) или None, если звука нет."""
    if shutil.which("ffmpeg") is None:
        print("ffmpeg не найден: голос не используется.")
        return None
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                            "-vn", "-ac", "1", "-ar", str(SR), str(wav)])
        if r.returncode != 0 or not wav.exists():
            print("В видео нет звуковой дорожки: голос не используется.")
            return None
        raw, _ = librosa.load(str(wav), sr=SR)
    if len(raw) < SR:
        print("Звук слишком короткий: голос не используется.")
        return None
    rms = float(np.sqrt(np.mean(raw ** 2)))
    print(f"Уровень звука (RMS): {rms:.4f}")
    if rms > 0.2:
        print("Внимание: звук очень громкий (возможно, музыка или сильное сжатие),"
              " оценка голоса может быть неверной.")
    if rms < SILENCE_RMS:
        print("Звук почти отсутствует: голос не используется.")
        return None

    segs, centers = [], []
    for s in range(0, max(1, len(raw) - SR), 2 * SR):
        seg = raw[s:s + MAX_LEN]
        if len(seg) < SR or np.sqrt(np.mean(seg ** 2)) < SILENCE_RMS:
            continue
        centers.append(s / SR + len(seg) / (2 * SR))
        seg = (seg - seg.mean()) / (seg.std() + 1e-7)
        if len(seg) < MAX_LEN:
            seg = np.pad(seg, (0, MAX_LEN - len(seg)), mode="wrap")
        segs.append(seg.astype(np.float32))
    if not segs:
        print("В звуке нет речи: голос не используется.")
        return None

    model = Net(len(CLASSES), 4 if mode == "partial" else 0).to(device)
    model.load_state_dict(torch.load(f"outputs/audio_w2v_{mode}.pth", map_location=device))
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(segs), 4):
            x = torch.from_numpy(np.stack(segs[i:i + 4])).to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                out = model(x)
            probs.append(torch.softmax(out.float(), 1).cpu().numpy())
    probs = np.concatenate(probs)
    del model
    return list(zip(centers, probs))


def voice_at(windows, t, max_dist=2.0):
    if not windows:
        return None
    c, p = min(windows, key=lambda w: abs(w[0] - t))
    return p if abs(c - t) <= max_dist else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--face_mode", default="partial")
    ap.add_argument("--audio_mode", default="partial")
    ap.add_argument("--w_face", type=float, default=0.5)
    ap.add_argument("--step", type=int, default=3, help="классифицировать лицо каждые N кадров")
    ap.add_argument("--detect_every", type=int, default=2, help="искать лицо каждые N кадров")
    ap.add_argument("--hold", type=int, default=8, help="сколько проверок держать рамку без лица")
    ap.add_argument("--alpha", type=float, default=0.35, help="сглаживание вероятностей (0-1)")
    ap.add_argument("--no_voice", action="store_true")
    ap.add_argument("--out", default="outputs/annotated")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Файл не найден: {video}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(22)
    names = RU if font is not None else EN
    if font is None:
        print("Шрифт с кириллицей не найден: подписи будут на английском.")

    windows = None if args.no_voice else voice_windows(video, device, args.audio_mode)

    det = load_detector()
    fmodel = build_model(args.face_mode, len(CLASSES)).to(device)
    fmodel.load_state_dict(torch.load(f"outputs/face_{args.face_mode}.pth", map_location=device))
    fmodel.eval()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit("Не удалось открыть видео")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tmp_video = out_dir / f"{video.stem}_noaudio.mp4"
    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise SystemExit("Не удалось создать выходной файл")

    box, lost, ema, idx = None, 0, None, 0
    shown = Counter()
    voice_shown = Counter()
    face_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps

        if idx % args.detect_every == 0:
            new = detect_face(det, frame)
            if new is not None:
                if box is None or np.abs(new[:2] - box[:2]).max() > max(box[2], box[3]):
                    box, ema = new, None  # появилось новое лицо
                else:
                    box = 0.6 * box + 0.4 * new
                lost = 0
            else:
                lost += 1
                if lost > args.hold:
                    box, ema = None, None

        if box is not None and idx % args.step == 0:
            crop = crop_box(frame, box)
            if crop is not None:
                p = face_prob(fmodel, crop, device)
                ema = p if ema is None else args.alpha * p + (1 - args.alpha) * ema

        vp = voice_at(windows, t)
        fp = ema if box is not None else None
        if fp is not None and vp is not None:
            final = args.w_face * fp + (1 - args.w_face) * vp
        else:
            final = fp if fp is not None else vp

        labels = []
        if box is not None and fp is not None:
            k = int(final.argmax())
            color = COLORS[CLASSES[k]]
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            labels.append((f"{names[CLASSES[k]]} {final[k] * 100:.0f}%",
                           (x, max(2, y - LABEL_H)), color))
        if box is not None and fp is not None:
            face_frames += 1
            shown[CLASSES[int(final.argmax())]] += 1
        if vp is not None:
            voice_shown[CLASSES[int(vp.argmax())]] += 1
        if final is not None:
            lines = []
            if fp is not None:
                lines.append(f"Лицо: {names[CLASSES[int(fp.argmax())]]}" if font is not None
                             else f"Face: {names[CLASSES[int(fp.argmax())]]}")
            if vp is not None:
                lines.append(f"Голос: {names[CLASSES[int(vp.argmax())]]}" if font is not None
                             else f"Voice: {names[CLASSES[int(vp.argmax())]]}")
            for i, line in enumerate(lines):
                labels.append((line, (10, 10 + i * (LABEL_H + 4)), (60, 60, 60)))
        if labels:
            frame = draw_labels(frame, labels, font)

        writer.write(frame)
        idx += 1
        if idx % 100 == 0:
            print(f"кадров: {idx}/{total}")

    cap.release()
    writer.release()

    final_path = out_dir / f"{video.stem}_annotated.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_video),
                        "-i", str(video), "-map", "0:v:0", "-map", "1:a:0?",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(final_path)])
    if r.returncode == 0 and final_path.exists():
        tmp_video.unlink()
        print(f"\nГотово: {final_path}")
    else:
        print(f"\nffmpeg не справился, видео без звука: {tmp_video}")

    if idx == 0:
        print("\nВидео не содержит кадров.")
        return
    print(f"\nЛицо найдено в {face_frames / idx * 100:.0f}% кадров ({face_frames} из {idx}).")
    if face_frames / idx < 0.5:
        print("Лицо находится меньше чем в половине кадров: детектор часто его не видит,"
              " поэтому итог по лицу ненадёжен.")

    n = sum(shown.values())
    if n:
        print("\nИтог (лицо + голос), только кадры с найденным лицом:")
        for c, v in shown.most_common():
            print(f"  {names[c]:<12} {v / n * 100:5.1f}%")
    nv = sum(voice_shown.values())
    if nv:
        print("\nГолос отдельно, по всему видео:")
        for c, v in voice_shown.most_common():
            print(f"  {names[c]:<12} {v / nv * 100:5.1f}%")
    if not n and not nv:
        print("\nЛицо и речь не найдены ни в одном кадре.")


if __name__ == "__main__":
    main()