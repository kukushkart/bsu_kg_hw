import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import imgproc as ip
import benchmark

FILTERS = {"Без фильтра": "none", "Усредняющий": "box",
           "Гауссов": "gauss", "Взвешенный 3×3": "weighted"}
NOISES = ["Нет", "Гауссов", "Соль и перец"]
INITS = {"Среднее яркости": "mean", "(min+max)/2": "midrange"}
HERE = os.path.dirname(os.path.abspath(__file__))


def fmt_psnr(v):
    return "—" if np.isinf(v) else f"{v:.1f} дБ"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР2 (вариант 7): НЧ-фильтры и глобальная пороговая обработка")
        self.geometry("1150x780")
        self.minsize(950, 650)
        self.original = None   # исходное серое изображение (эталон для PSNR)
        self.gray = None       # вход фильтра: оригинал или оригинал + шум
        self.path = None
        self.folder = os.path.join(HERE, "images")
        self.result = None
        self.eps = 0.5
        self._build_ui()
        self._fill_folder_list()

    # ------------------------------------------------------------ интерфейс
    def _build_ui(self):
        style = ttk.Style(self)
        style.configure("TLabelframe.Label", font=("Segoe UI", 9, "bold"))

        top = ttk.Frame(self, padding=(10, 8, 10, 0))
        top.pack(side=tk.TOP, fill=tk.X)

        # --- блок «Изображение»
        f_img = ttk.LabelFrame(top, text="Изображение", padding=6)
        f_img.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        ttk.Button(f_img, text="Открыть файл…", command=self.open_file).grid(row=0, column=0, sticky="we")
        ttk.Button(f_img, text="Выбрать папку…", command=self.open_folder).grid(row=0, column=1, sticky="we", padx=(4, 0))
        self.list_var = tk.StringVar()
        self.list_box = ttk.Combobox(f_img, textvariable=self.list_var, state="readonly", width=30)
        self.list_box.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self.list_box.bind("<<ComboboxSelected>>", self.on_list_selected)

        # --- блок «Шум»
        f_noise = ttk.LabelFrame(top, text="Шум (для проверки фильтров)", padding=6)
        f_noise.grid(row=0, column=1, sticky="ns", padx=(0, 8))
        self.noise_var = tk.StringVar(value=NOISES[0])
        self._combo(f_noise, self.noise_var, NOISES, 14).grid(row=0, column=0, columnspan=2, sticky="we")
        ttk.Label(f_noise, text="Уровень:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.noise_amt = tk.DoubleVar(value=20.0)
        self._spin(f_noise, self.noise_amt, 0, 100, 1, 7).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(f_noise, text="σ для гауссова, 0…1 для соль/перец",
                  foreground="gray", font=("Segoe UI", 8)).grid(row=2, column=0, columnspan=2, sticky="w")

        # --- блок «НЧ-фильтр»
        f_flt = ttk.LabelFrame(top, text="НЧ-фильтр", padding=6)
        f_flt.grid(row=0, column=2, sticky="ns", padx=(0, 8))
        self.filter_var = tk.StringVar(value="Гауссов")
        self._combo(f_flt, self.filter_var, list(FILTERS), 16).grid(row=0, column=0, columnspan=2, sticky="we")
        ttk.Label(f_flt, text="Окно k:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.k_var = tk.IntVar(value=7)
        self.k_spin = self._spin(f_flt, self.k_var, 3, 51, 2, 6)
        self.k_spin.grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(f_flt, text="σ:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.sigma_var = tk.DoubleVar(value=1.5)
        self.sigma_spin = self._spin(f_flt, self.sigma_var, 0.3, 20, 0.1, 6)
        self.sigma_spin.grid(row=2, column=1, sticky="w", pady=(4, 0))

        # --- блок «Итерационный метод»
        f_it = ttk.LabelFrame(top, text="Итерационный метод", padding=6)
        f_it.grid(row=0, column=3, sticky="ns")
        ttk.Label(f_it, text="Начальный порог T0:").grid(row=0, column=0, sticky="w")
        self.init_var = tk.StringVar(value="Среднее яркости")
        self._combo(f_it, self.init_var, list(INITS), 16).grid(row=1, column=0, columnspan=2, sticky="we")
        ttk.Label(f_it, text="ε (остановка):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.eps_var = tk.DoubleVar(value=0.5)
        self._spin(f_it, self.eps_var, 0.001, 5, 0.1, 6).grid(row=2, column=1, sticky="w", pady=(6, 0))

        # --- кнопки действий
        bar = ttk.Frame(self, padding=(10, 6, 10, 0))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Сравнить все фильтры", command=self.compare_filters).pack(side=tk.LEFT)
        ttk.Button(bar, text="Таблица по папке", command=self.show_benchmark).pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="Сохранить результаты", command=self.save_results).pack(side=tk.LEFT)

        # --- строка результатов (внизу)
        self.status = ttk.Label(self, text="", font=("Consolas", 10), justify="left",
                                padding=(12, 6), relief="groove", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        # --- графики
        mid = ttk.Frame(self, padding=(10, 4))
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid)
        NavigationToolbar2Tk(self.canvas, mid)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _combo(self, parent, var, values, width):
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
        cb.bind("<<ComboboxSelected>>", lambda e: self.process())
        return cb

    def _spin(self, parent, var, lo, hi, step, width):
        sp = ttk.Spinbox(parent, from_=lo, to=hi, increment=step, textvariable=var,
                         width=width, command=self.process)
        sp.bind("<Return>", lambda e: self.process())
        sp.bind("<FocusOut>", lambda e: self.process())
        return sp

    # ------------------------------------------------------------ загрузка
    def _fill_folder_list(self):
        files = []
        if os.path.isdir(self.folder):
            files = sorted(f for f in os.listdir(self.folder) if f.lower().endswith(ip.IMAGE_EXT))
        self.list_box["values"] = files
        if files:
            self.list_var.set(files[0])
            self.load(os.path.join(self.folder, files[0]))
        else:
            self.list_var.set("")
            self.status.config(text="Положите фотографии в папку images/ или нажмите «Открыть файл…»")

    def open_file(self):
        p = filedialog.askopenfilename(
            filetypes=[("Изображения", " ".join("*" + e for e in ip.IMAGE_EXT))])
        if p:
            self.load(p)

    def open_folder(self):
        d = filedialog.askdirectory(title="Папка с изображениями")
        if d:
            self.folder = d
            self._fill_folder_list()

    def on_list_selected(self, _):
        self.load(os.path.join(self.folder, self.list_var.get()))

    def load(self, path):
        try:
            self.original = ip.load_gray(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            return
        self.path = path
        self.process()

    def _params(self):
        try:
            k = int(self.k_var.get())
            sigma = max(float(self.sigma_var.get()), 0.1)
            amt = float(self.noise_amt.get())
            self.eps = max(float(self.eps_var.get()), 1e-6)
        except (tk.TclError, ValueError):
            return None
        if k % 2 == 0:
            k += 1
            self.k_var.set(k)
        return k, sigma, amt

    def _noisy_input(self, amt):
        mode = self.noise_var.get()
        if mode == NOISES[1]:
            return ip.add_gaussian_noise(self.original, amt)
        if mode == NOISES[2]:
            return ip.add_salt_pepper(self.original, min(max(amt, 0.0), 1.0))
        return self.original

    def _update_states(self):
        m = FILTERS[self.filter_var.get()]
        self.k_spin.state(["!disabled"] if m in ("box", "gauss") else ["disabled"])
        self.sigma_spin.state(["!disabled"] if m == "gauss" else ["disabled"])

    def process(self):
        if self.original is None:
            return
        self._update_states()
        p = self._params()
        if p is None:
            return
        k, sigma, amt = p
        self.gray = self._noisy_input(amt)
        sm = ip.smooth(self.gray, FILTERS[self.filter_var.get()], k, sigma)
        t_otsu = ip.otsu_threshold(sm)
        t_iter, n_it = ip.iterative_threshold(sm, self.eps, init=INITS[self.init_var.get()])
        b_otsu, b_iter = ip.binarize(sm, t_otsu), ip.binarize(sm, t_iter)
        self.result = (sm, b_otsu, b_iter)
        self.draw(sm, t_otsu, t_iter, b_otsu, b_iter)
        self.write_status(sm, t_otsu, t_iter, n_it, b_otsu, b_iter)

    def draw(self, sm, t_otsu, t_iter, b_otsu, b_iter):
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 3, hspace=0.30, wspace=0.08,
                                   left=0.02, right=0.98, top=0.94, bottom=0.05)
        noisy = self.gray is not self.original
        panels = [
            (gs[0, 0], self.gray, "Вход (с шумом)" if noisy else "Исходное"),
            (gs[0, 1], sm, "После НЧ-фильтра"),
            (gs[1, 0], b_otsu, f"Оцу, T = {t_otsu:.0f}"),
            (gs[1, 1], b_iter, f"Итерационный, T = {t_iter:.1f}"),
            (gs[1, 2], b_otsu != b_iter, "Расхождение методов (белое)"),
        ]
        for spec, img, title in panels:
            a = self.fig.add_subplot(spec)
            a.imshow(img, cmap="gray", vmin=0, vmax=255 if img.dtype != bool else 1)
            a.set_title(title, fontsize=10)
            a.axis("off")
        h = self.fig.add_subplot(gs[0, 2])
        h.bar(np.arange(256), ip.histogram(sm), width=1.0, color="0.55")
        h.axvline(t_otsu, color="red", lw=1.5, label="Оцу")
        h.axvline(t_iter, color="blue", lw=1.5, ls="--", label="Итерационный")
        h.set_title("Гистограмма после фильтра", fontsize=10)
        h.set_yticks([])
        h.set_xlim(0, 255)
        h.legend(fontsize=8, frameon=False)
        for side in ("top", "right", "left"):
            h.spines[side].set_visible(False)
        self.canvas.draw()

    def write_status(self, sm, t_otsu, t_iter, n_it, b_otsu, b_iter):
        st = ip.image_stats(self.original)
        l1 = (f"{os.path.basename(self.path)}  {self.original.shape[1]}×{self.original.shape[0]}   "
              f"яркость {st['min']}…{st['max']}, σ={st['std']:.1f}   "
              f"PSNR: вход {fmt_psnr(ip.psnr(self.original, self.gray))}, "
              f"после фильтра {fmt_psnr(ip.psnr(self.original, sm))}")
        l2 = (f"Оцу T={t_otsu:.1f} (белых {np.mean(b_otsu > 0) * 100:.1f}%)   "
              f"Итерационный T={t_iter:.1f} ({n_it} ит., белых {np.mean(b_iter > 0) * 100:.1f}%)   "
              f"расхождение {np.mean(b_otsu != b_iter) * 100:.1f}% пикс.")
        self.status.config(text=l1 + "\n" + l2)

    # ------------------------------------------------------------ сервис
    def compare_filters(self):
        """Окно: все НЧ-фильтры при текущих k, σ и шуме + результат Оцу для каждого."""
        if self.original is None:
            return
        p = self._params()
        if p is None:
            return
        k, sigma, amt = p
        inp = self._noisy_input(amt)
        win = tk.Toplevel(self)
        win.title("Сравнение низкочастотных фильтров")
        win.geometry("1150x650")
        fig = Figure(figsize=(11, 6), dpi=100)
        cv = FigureCanvasTkAgg(fig, master=win)
        NavigationToolbar2Tk(cv, win)
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ax = fig.subplots(2, 4)
        for col, (name, m) in enumerate(FILTERS.items()):
            sm = ip.smooth(inp, m, k, sigma)
            t = ip.otsu_threshold(sm)
            ax[0, col].imshow(sm, cmap="gray", vmin=0, vmax=255)
            ax[0, col].set_title(f"{name}\nPSNR {fmt_psnr(ip.psnr(self.original, sm))}", fontsize=9)
            ax[1, col].imshow(ip.binarize(sm, t), cmap="gray", vmin=0, vmax=255)
            ax[1, col].set_title(f"Оцу, T = {t:.0f}", fontsize=9)
            ax[0, col].axis("off")
            ax[1, col].axis("off")
        noise = self.noise_var.get() + (f" ({amt:g})" if self.noise_var.get() != NOISES[0] else "")
        fig.suptitle(f"k = {k}, σ = {sigma:g}, шум: {noise}", fontsize=10)
        fig.tight_layout()
        cv.draw()

    def save_results(self):
        if self.result is None:
            return
        d = filedialog.askdirectory(title="Папка для сохранения")
        if not d:
            return
        base = os.path.splitext(os.path.basename(self.path))[0]
        for tag, arr in zip(("smoothed", "otsu", "iterative"), self.result):
            Image.fromarray(arr).save(os.path.join(d, f"{base}_{tag}.png"))
        messagebox.showinfo("Готово", f"Сохранено в {d}")

    def show_benchmark(self):
        win = tk.Toplevel(self)
        win.title("Пороги по всей папке")
        win.geometry("1000x600")
        box = scrolledtext.ScrolledText(win, font=("Consolas", 10))
        box.pack(fill=tk.BOTH, expand=True)
        box.insert(tk.END, benchmark.run(self.folder))


if __name__ == "__main__":
    App().mainloop()
