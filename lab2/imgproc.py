import numpy as np
from PIL import Image

def load_gray(path):
    im = Image.open(path)
    if im.mode.startswith("I") or im.mode == "F":      # 16/32-битные серые
        a = np.asarray(im, dtype=np.float64)
        if a.max() > 255:
            a = a / 256.0 if a.max() > 255 and a.max() <= 65535 else a * 255.0 / a.max()
        return np.clip(np.round(a), 0, 255).astype(np.uint8)
    img = np.asarray(im.convert("RGB"), dtype=np.float64)
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def histogram(img):
    return np.bincount(img.ravel(), minlength=256).astype(np.float64)


# ------------------------------------------------- низкочастотные фильтры
def convolve2d(img, kernel):
    """Свёртка с зеркальным дополнением границ. Ядра симметричны, поэтому
    корреляция == свёртка."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    src = np.pad(img.astype(np.float64), ((ph, ph), (pw, pw)), mode="reflect")
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * src[i:i + h, j:j + w]
    return out


def box_kernel(k):
    """Усредняющий фильтр k x k: все веса равны 1/k^2."""
    return np.full((k, k), 1.0 / (k * k))


def gaussian_kernel(k, sigma):
    """Гауссово ядро k x k, нормированное на сумму 1."""
    ax = np.arange(k) - k // 2
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    kern = np.outer(g, g)
    return kern / kern.sum()


def weighted_kernel():
    """Классическое взвешенное среднее 3x3 (приближение Гаусса)."""
    return np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0


def convolve1d(img, k1d, axis):
    """Одномерная свёртка вдоль оси (0 — по строкам, 1 — по столбцам), зеркальные границы."""
    r = len(k1d) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    src = np.pad(img.astype(np.float64), pad, mode="reflect")
    n = img.shape[axis]
    out = np.zeros(img.shape, dtype=np.float64)
    for i, w in enumerate(k1d):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, i + n)
        out += w * src[tuple(sl)]
    return out


def kernel_1d(method, k=3, sigma=1.0):
    """Одномерное ядро; все три фильтра раздельны: K2D = outer(k1d, k1d)."""
    if method == "box":
        return np.full(k, 1.0 / k)
    if method == "gauss":
        ax = np.arange(k) - k // 2
        g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
        return g / g.sum()
    if method == "weighted":
        return np.array([1.0, 2.0, 1.0]) / 4.0
    raise ValueError(method)


def smooth(img, method, k=3, sigma=1.0):
    """Низкочастотная фильтрация. Раздельная свёртка даёт тот же результат,
    что и 2D-ядро, но за O(k) операций на пиксель вместо O(k^2)."""
    if method == "none":
        return img.copy()
    k1 = kernel_1d(method, k, sigma)
    out = convolve1d(convolve1d(img, k1, 0), k1, 1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ------------------------------------------------ шум и оценка качества
def add_gaussian_noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(np.round(img + rng.normal(0, sigma, img.shape)), 0, 255).astype(np.uint8)


def add_salt_pepper(img, p, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.random(img.shape)
    out = img.copy()
    out[r < p / 2] = 0
    out[r > 1 - p / 2] = 255
    return out


def psnr(a, b):
    """Пиковое отношение сигнал/шум в дБ (чем больше, тем ближе a к b)."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


# ------------------------------------------ глобальная пороговая обработка
def otsu_threshold(img):
    """Метод Оцу: порог максимизирует межклассовую дисперсию."""
    p = histogram(img)
    p /= p.sum()
    levels = np.arange(256)
    omega = np.cumsum(p)              # вес класса фона
    mu = np.cumsum(p * levels)        # накопленное среднее
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[denom == 0] = 0
    best = np.flatnonzero(sigma_b >= sigma_b.max() - 1e-12)
    return float(best.mean())         # при плато максимума берём его середину


def iterative_threshold(img, eps=0.5, max_iter=1000, init="mean"):
    """Итерационный метод (Ridler-Calvard / isodata).
    T0 = среднее яркости (init="mean") либо (min+max)/2 (init="midrange");
    далее T = (m1 + m2) / 2, где m1, m2 — средние яркости классов <=T и >T;
    остановка, когда |T_new - T| < eps."""
    p = histogram(img)
    levels = np.arange(256)
    if init == "midrange":
        T = (float(img.min()) + float(img.max())) / 2.0
    else:
        T = float((p * levels).sum() / p.sum())
    it = 0
    for it in range(1, max_iter + 1):
        low = levels <= T
        high = ~low
        if p[low].sum() == 0 or p[high].sum() == 0:
            break
        m1 = (p[low] * levels[low]).sum() / p[low].sum()
        m2 = (p[high] * levels[high]).sum() / p[high].sum()
        new_T = 0.5 * (m1 + m2)
        if abs(new_T - T) < eps:
            T = new_T
            break
        T = new_T
    return float(T), it


def binarize(img, T):
    return ((img > T) * 255).astype(np.uint8)



# ------------------------------------------------- вспомогательная статистика
def image_stats(img):
    """Краткая характеристика изображения (полезна для пояснений о контрасте)."""
    return {"min": int(img.min()), "max": int(img.max()),
            "mean": float(img.mean()), "std": float(img.std())}


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

