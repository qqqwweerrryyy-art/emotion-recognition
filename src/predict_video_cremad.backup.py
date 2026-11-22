
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from extract_faces import load_detector, crop_face
from train_face_cremad import CLASSES, build_model
from train_audio_cremad import Net, SR, MAX_LEN, load_wave


def get_face_probs(video_path, model, device, n_frames=10):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("В видео не удалось определить количество кадров")

    indices = np.linspace(
        0.2 * total, 0.8 * total, n_frames
    ).astype(int)

    detector = load_detector()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    probabilities = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        face = crop_face(detector, frame)
        if face is None:
            continue

        image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        x = tf(image).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.autocast(
                device_type=device, enabled=(device == "cuda")
            ):
                logits = model(x)

            probs = torch.softmax(logits.float(), dim=1)
            probabilities.append(probs.cpu().numpy()[0])

    cap.release()

    if not probabilities:
        return None

    return np.mean(probabilities, axis=0)


def get_audio_path(video_path, temp_dir):
    # Для оригинальных CREMA-D-видео используем соответствующий WAV.
    dataset_wav = Path("data/cremad/AudioWAV") / (
        video_path.stem + ".wav"
    )
    if dataset_wav.exists():
        return dataset_wav

    # Для других видео извлекаем звук через ffmpeg.
    output_wav = Path(temp_dir) / "extracted_audio.wav"
    command = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(SR),
        str(output_wav)
    ]
    result = subprocess.run(
        command, capture_output=True, text=True
    )
    if result.returncode != 0 or not output_wav.exists():
        raise RuntimeError(
            "Не удалось извлечь аудио через ffmpeg:\n"
            + result.stderr[-2000:]
        )

    return output_wav


def get_audio_probs(audio_path, model, device):
    y = load_wave(audio_path)

    if len(y) > MAX_LEN:
        start = (len(y) - MAX_LEN) // 2
        y = y[start:start + MAX_LEN]
    elif len(y) < MAX_LEN:
        repeats = int(np.ceil(MAX_LEN / len(y)))
        y = np.tile(y, repeats)[:MAX_LEN]

    x = torch.from_numpy(y.copy()).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(
            device_type=device, enabled=(device == "cuda")
        ):
            logits = model(x)

        probs = torch.softmax(logits.float(), dim=1)

    return probs.cpu().numpy()[0]


def print_probs(title, probs):
    print(f"\n{title}")
    if probs is None:
        print("Лицо не обнаружено.")
        return

    order = np.argsort(probs)[::-1]
    for i in order:
        print(f"  {CLASSES[i]:10s}: {probs[i]:.3f}")

    print(f"  Предсказание: {CLASSES[int(np.argmax(probs))]}")


def main():
    parser = argparse.ArgumentParser(
        description="CREMA-D video emotion prediction"
    )
    parser.add_argument("video", help="Путь к видео")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--w-face", type=float, default=0.5)
    parser.add_argument(
        "--face-weights",
        default="outputs/face_cremad_frozen.pth"
    )
    parser.add_argument(
        "--audio-weights",
        default="outputs/audio_cremad_w2v_frozen.pth"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Видео не найдено: {video_path}")

    for weight_path in [args.face_weights, args.audio_weights]:
        if not Path(weight_path).exists():
            raise SystemExit(f"Не найдены веса модели: {weight_path}")

    if not 0 <= args.w_face <= 1:
        raise SystemExit("--w-face должен быть от 0 до 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Устройство:", device)
    print("Видео:", video_path)

    # Загружаем CREMA-D face model (6 классов).
    face_model = build_model("frozen", len(CLASSES)).to(device)
    face_state = torch.load(
        args.face_weights, map_location=device, weights_only=True
    )
    face_model.load_state_dict(face_state)
    face_model.eval()

    # Загружаем CREMA-D audio model (6 классов).
    audio_model = Net(len(CLASSES), 0).to(device)
    audio_state = torch.load(
        args.audio_weights, map_location=device, weights_only=True
    )
    audio_model.load_state_dict(audio_state)
    audio_model.eval()

    face_probs = get_face_probs(
        video_path, face_model, device, args.frames
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = get_audio_path(video_path, temp_dir)
        audio_probs = get_audio_probs(
            audio_path, audio_model, device
        )

    print_probs("=== FACE CREMA-D ===", face_probs)
    print_probs("=== AUDIO CREMA-D ===", audio_probs)

    if face_probs is not None:
        fused = (
            args.w_face * face_probs
            + (1 - args.w_face) * audio_probs
        )
        print_probs("=== FUSED CREMA-D ===", fused)
    else:
        print("\nЛицо не найдено: итоговое предсказание только по голосу.")
        print_probs("=== AUDIO-ONLY RESULT ===", audio_probs)

    # Метка из имени файла — это ожидаемый ответ для исходной записи.
    parts = video_path.stem.split("_")
    code_to_emotion = {
        "NEU": "neutral",
        "HAP": "happy",
        "SAD": "sad",
        "ANG": "angry",
        "FEA": "fearful",
        "DIS": "disgust",
    }
    expected = code_to_emotion.get(parts[2]) if len(parts) >= 3 else None

    if expected:
        print("\nОжидаемая эмоция по имени файла:", expected)
        print("(Это метка записи, а не результат модели.)")


if __name__ == "__main__":
    main()
