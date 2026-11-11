import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from train_face import CLASSES, CODE2CLS, FaceDS, build_model, split_actors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/faces/labels.csv")
    ap.add_argument("--mode", choices=["frozen", "partial"], default="partial")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.labels)
    _, va_a, te_a = split_actors(df["actor"].unique().tolist(), args.seed)
    print(f"Актёры: val={va_a} test={te_a}")

    model = build_model(args.mode, len(CLASSES)).to(device)
    model.load_state_dict(torch.load(f"outputs/face_{args.mode}.pth", map_location=device))
    model.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    pcols = [f"p_{c}" for c in CLASSES]
    parts = []
    for split_name, actors in [("val", va_a), ("test", te_a)]:
        sub = df[df["actor"].isin(actors)].reset_index(drop=True)
        dl = DataLoader(FaceDS(sub, tf), batch_size=args.batch, shuffle=False, num_workers=2)
        probs = []
        with torch.no_grad():
            for x, _ in dl:
                with torch.autocast(device_type=device, enabled=(device == "cuda")):
                    out = model(x.to(device))
                probs.append(torch.softmax(out.float(), 1).cpu().numpy())
        probs = np.concatenate(probs)
        for k, c in enumerate(pcols):
            sub[c] = probs[:, k]
        sub["split"] = split_name
        parts.append(sub)

    frames = pd.concat(parts, ignore_index=True)
    frames["label"] = frames["emotion_code"].map(CODE2CLS)
    video = frames.groupby(["video", "actor", "split"], as_index=False).agg(
        {"label": "first", **{c: "mean" for c in pcols}})
    video.to_csv(f"outputs/face_{args.mode}_probs.csv", index=False)

    t_fr = frames[frames["split"] == "test"]
    t_vi = video[video["split"] == "test"]
    for name, d in [("кадры", t_fr), ("видео (среднее по кадрам)", t_vi)]:
        pred = d[pcols].values.argmax(1)
        print(f"TEST, {name}: accuracy={accuracy_score(d['label'], pred):.3f}  "
              f"macro-F1={f1_score(d['label'], pred, average='macro'):.3f}  (n={len(d)})")


if __name__ == "__main__":
    main()
