import os
import sys
import numpy as np
import imgproc as ip

FILTERS = [("без фильтра", "none", 3, 1.0),
           ("усредн. 7x7", "box", 7, 1.0),
           ("гаусс 7x7 s=1.5", "gauss", 7, 1.5)]


def run(folder="images"):
    if not os.path.isdir(folder):
        return f"Папка не найдена: {folder}"
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(ip.IMAGE_EXT))
    if not files:
        return f"В папке {folder} нет изображений."
    head = (f"{'изображение':28s}{'фильтр':18s}{'T Оцу':>7s}{'T итер':>8s}"
            f"{'расхожд.%':>11s}{'белых Оцу%':>12s}{'белых итер%':>13s}")
    lines = [head, "-" * len(head)]
    for f in files:
        img = ip.load_gray(os.path.join(folder, f))
        for title, m, k, s in FILTERS:
            im = ip.smooth(img, m, k, s)
            to = ip.otsu_threshold(im)
            ti, _ = ip.iterative_threshold(im)
            bo, bi = ip.binarize(im, to), ip.binarize(im, ti)
            lines.append(f"{f[:27]:28s}{title:18s}{to:7.0f}{ti:8.1f}"
                         f"{np.mean(bo != bi) * 100:11.2f}{np.mean(bo > 0) * 100:12.1f}"
                         f"{np.mean(bi > 0) * 100:13.1f}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run(sys.argv[1] if len(sys.argv) > 1 else "images"))
