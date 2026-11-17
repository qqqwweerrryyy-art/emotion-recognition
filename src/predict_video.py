import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from torchvision import transforms

from extract_faces import crop_face, load_detector
from train_audio import MAX_LEN, SR, Net, load_wave
from train_face import CLASSES, build_model

RU = {"neutral": "нейтральная", "happy": "радость", "sad": "грусть", "angry": "злость",
      "fearful": "страх", "disgust": "отвращение", "surprised": "удивление"}


def face_probs(video, mode, device, n_frames):
    det = load_detector()
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return None, 0
    faces = []
    for i in np.linspace(0.1 * total, 0.9 * total, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        f = crop_face(det, frame)
        if f is not None:
            faces.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not faces:
        return None, 0
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = torch.stack([tf(f) for f in faces]).to(device)
    model = build_model(mode, len(CLASSES)).to(device)
    model.load_state_dict(torch.load(f"outputs/face_{mode}.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x).float(), 1).mean(0).cpu().numpy()
    return p, len(faces)


def audio_probs(video, mode, device):
    if shutil.which("ffmpeg") is None:
        return None, 0
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                            "-vn", "-ac", "1", "-ar", str(SR), str(wav)])
        if r.returncode != 0 or not wav.exists():
            return None, 0
        raw, _ = librosa.load(str(wav), sr=SR)
        rms = float(np.sqrt(np.mean(raw ** 2))) if len(raw) else 0.0
        print(f"Уровень звука (RMS): {rms:.4f}")
        if rms < 0.003:
            return None, 0
        y = load_wave(wav)
    if len(y) <= MAX_LEN:
        chunks = [y]
    else:
        chunks = [y[s:s + MAX_LEN] for s in range(0, len(y), MAX_LEN) if len(y[s:s + MAX_LEN]) >= SR]
    chunks = [np.pad(c, (0, MAX_LEN - len(c)), mode="wrap") if len(c) < MAX_LEN else c
              for c in chunks[:8]]
    model = Net(len(CLASSES), 4 if mode == "partial" else 0).to(device)
    model.load_state_dict(torch.load(f"outputs/audio_w2v_{mode}.pth", map_location=device))
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(chunks), 4):
            x = torch.from_numpy(np.stack(chunks[i:i + 4])).to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                out = model(x)
            probs.append(torch.softmax(out.float(), 1).cpu().numpy())
    return np.concatenate(probs).mean(0), len(chunks)


def show(title, p):
    print(f"\n{title}")
    for k in np.argsort(-p):
        print(f"  {RU[CLASSES[k]]:<12} {p[k]*100:5.1f}%  " + "#" * int(p[k] * 30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--face_mode", default="partial")
    ap.add_argument("--audio_mode", default="partial")
    ap.add_argument("--w_face", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pf, nf = face_probs(args.video, args.face_mode, device, args.frames)
    pa, na = audio_probs(args.video, args.audio_mode, device)

    if pf is None:
        print("Лицо не найдено в видео (нужно лицо анфас, крупно). Используется только голос.")
    else:
        show(f"ЛИЦО (кадров с лицом: {nf})", pf)
    if pa is None:
        print("Звук не найден или не прочитан. Используется только лицо.")
    else:
        show(f"ГОЛОС (фрагментов: {na})", pa)
    if pf is None and pa is None:
        raise SystemExit("Нечего анализировать.")

    if pf is None:
        final = pa
    elif pa is None:
        final = pf
    else:
        final = args.w_face * pf + (1 - args.w_face) * pa
    show("ИТОГ", final)
    print(f"\nРезультат: {RU[CLASSES[int(final.argmax())]].upper()}")


if __name__ == "__main__":
    main()

