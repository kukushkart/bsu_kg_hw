import tkinter as tk
from tkinter import ttk, colorchooser


def rgb_to_cmyk(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    k = 1.0 - max(r, g, b)
    if k >= 1.0:                      # чёрный: C, M, Y не определены -> 0
        return 0.0, 0.0, 0.0, 1.0
    c = (1.0 - r - k) / (1.0 - k)
    m = (1.0 - g - k) / (1.0 - k)
    y = (1.0 - b - k) / (1.0 - k)
    return c, m, y, k


def cmyk_to_rgb(c, m, y, k):
    return (255.0 * (1.0 - c) * (1.0 - k),
            255.0 * (1.0 - m) * (1.0 - k),
            255.0 * (1.0 - y) * (1.0 - k))


def rgb_to_hls(r, g, b, h_default=0.0):
    """Возвращает (H, L, S). Для серых цветов H не определён -> h_default."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    l = (mx + mn) / 2.0
    if d < 1e-9:
        return h_default, l, 0.0
    s = d / (1.0 - abs(2.0 * l - 1.0))
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    return h % 360.0, l, min(max(s, 0.0), 1.0)


def hls_to_rgb(h, l, s):
    c = (1.0 - abs(2.0 * l - 1.0)) * s          # цветность (chroma)
    hp = (h % 360.0) / 60.0
    x = c * (1.0 - abs(hp % 2.0 - 1.0))
    if hp < 1:   r, g, b = c, x, 0
    elif hp < 2: r, g, b = x, c, 0
    elif hp < 3: r, g, b = 0, c, x
    elif hp < 4: r, g, b = 0, x, c
    elif hp < 5: r, g, b = x, 0, c
    else:        r, g, b = c, 0, x
    m = l - c / 2.0
    return 255.0 * (r + m), 255.0 * (g + m), 255.0 * (b + m)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def to_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(clamp(v, 0, 255))) for v in rgb)

MODELS = {
    "RGB":  [("R", 255, "красный"), ("G", 255, "зелёный"), ("B", 255, "синий")],
    "CMYK": [("C", 100, "голубой, %"), ("M", 100, "пурпурный, %"),
             ("Y", 100, "жёлтый, %"), ("K", 100, "чёрный, %")],
    "HLS":  [("H", 360, "оттенок, °"), ("L", 100, "светлота, %"),
             ("S", 100, "насыщенность, %")],
}

PAL_W, PAL_H, PAL_ZOOM = 180, 100, 2     # палитра рисуется 180x100 и увеличивается в 2 раза


def model_to_rgb(model, v):
    if model == "RGB":
        return tuple(v)
    if model == "CMYK":
        return cmyk_to_rgb(*(x / 100.0 for x in v))
    return hls_to_rgb(v[0], v[1] / 100.0, v[2] / 100.0)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Цветовые модели: CMYK – RGB – HLS")
        self.resizable(False, False)

        self.lock = False            # защита от рекурсии при программном обновлении
        self.rgb = (255.0, 128.0, 0.0)
        self.h_mem = 0.0             # запоминаем H для серых цветов
        self.pal_sat = None
        self.pal_img = None

        self.scales, self.scale_vars, self.entry_vars = {}, {}, {}
        self._build_ui()
        self.refresh("RGB", list(self.rgb))


    def _build_ui(self):
        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="n")
        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="n")

        self.swatch = tk.Label(left, width=34, height=5, relief="sunken", bd=2)
        self.swatch.pack(pady=(0, 8))

        ttk.Label(left, text="Палитра: H – по горизонтали, L – по вертикали\n"
                             "(насыщенность S берётся из ползунка)").pack()
        self.canvas = tk.Canvas(left, width=PAL_W * PAL_ZOOM, height=PAL_H * PAL_ZOOM,
                                cursor="crosshair", highlightthickness=1,
                                highlightbackground="#888")
        self.canvas.pack(pady=4)
        self.canvas_img_id = self.canvas.create_image(0, 0, anchor="nw")
        self.marker = self.canvas.create_oval(0, 0, 0, 0, outline="black", width=2)
        self.marker2 = self.canvas.create_oval(0, 0, 0, 0, outline="white", width=1)
        self.canvas.bind("<Button-1>", self._on_palette)
        self.canvas.bind("<B1-Motion>", self._on_palette)

        row = ttk.Frame(left)
        row.pack(pady=6, fill="x")
        ttk.Label(row, text="HEX:").pack(side="left")
        self.hex_var = tk.StringVar()
        hex_entry = ttk.Entry(row, textvariable=self.hex_var, width=10)
        hex_entry.pack(side="left", padx=6)
        hex_entry.bind("<Return>", self._on_hex)
        hex_entry.bind("<FocusOut>", self._on_hex)
        ttk.Button(row, text="Системная палитра…",
                   command=self._on_system_picker).pack(side="left")

        for model, comps in MODELS.items():
            box = ttk.LabelFrame(right, text=model, padding=8)
            box.pack(fill="x", pady=4)
            for i, (name, mx, hint) in enumerate(comps):
                ttk.Label(box, text=name, width=2, font=("TkDefaultFont", 10, "bold")
                          ).grid(row=i, column=0)
                sv = tk.DoubleVar(value=0)
                sc = ttk.Scale(box, from_=0, to=mx, variable=sv, length=260,
                               command=lambda _v, m=model: self._on_slider(m))
                sc.grid(row=i, column=1, padx=6, pady=2)
                ev = tk.StringVar()
                en = ttk.Entry(box, textvariable=ev, width=7, justify="right")
                en.grid(row=i, column=2)
                en.bind("<KeyRelease>", lambda _e, m=model: self._on_entry(m))
                en.bind("<Return>", lambda _e, m=model: self._on_entry(m))
                ttk.Label(box, text=hint, foreground="#666").grid(row=i, column=3, padx=6, sticky="w")
                self.scale_vars[(model, i)] = sv
                self.entry_vars[(model, i)] = ev
                self.scales[(model, i)] = sc

    def _read_sliders(self, model):
        return [self.scale_vars[(model, i)].get() for i in range(len(MODELS[model]))]

    def _read_entries(self, model):
        vals = []
        for i, (_n, mx, _h) in enumerate(MODELS[model]):
            txt = self.entry_vars[(model, i)].get().strip().replace(",", ".")
            try:
                vals.append(clamp(float(txt), 0, mx))
            except ValueError:
                return None            # пока введено некорректное значение — ничего не делаем
        return vals

    def _on_slider(self, model):
        if self.lock:
            return
        self.refresh(model, self._read_sliders(model))

    def _on_entry(self, model):
        if self.lock:
            return
        vals = self._read_entries(model)
        if vals is not None:
            self.refresh(model, vals, keep_entries=True)

    def _on_palette(self, event):
        x = clamp(event.x, 0, PAL_W * PAL_ZOOM - 1)
        y = clamp(event.y, 0, PAL_H * PAL_ZOOM - 1)
        h = x / (PAL_W * PAL_ZOOM - 1) * 360.0
        l = (1.0 - y / (PAL_H * PAL_ZOOM - 1)) * 100.0
        s = self.scale_vars[("HLS", 2)].get()
        self.refresh("HLS", [h, l, s])

    def _on_hex(self, _event=None):
        if self.lock:
            return
        t = self.hex_var.get().strip().lstrip("#")
        if len(t) == 6:
            try:
                self.refresh("RGB", [int(t[i:i + 2], 16) for i in (0, 2, 4)])
            except ValueError:
                pass

    def _on_system_picker(self):
        res = colorchooser.askcolor(color=to_hex(self.rgb), parent=self)
        if res and res[0]:
            self.refresh("RGB", [float(c) for c in res[0]])

    def refresh(self, source, src_vals, keep_entries=False):
        """source — модель, в которой пользователь менял цвет; src_vals — её значения.
        Остальные модели считаются через RGB. Значения источника не пересчитываются
        обратно, чтобы не терять информацию (например, H при L=0 или CMY при K=100)."""
        self.lock = True
        try:
            self.rgb = tuple(clamp(c, 0, 255) for c in model_to_rgb(source, src_vals))
            r, g, b = self.rgb
            if source == "HLS":
                self.h_mem = src_vals[0]

            values = {
                "RGB": [r, g, b],
                "CMYK": [v * 100 for v in rgb_to_cmyk(r, g, b)],
            }
            if source == "HLS":
                values["HLS"] = list(src_vals)
            else:
                h, l, s = rgb_to_hls(r, g, b, self.h_mem)
                self.h_mem = h
                values["HLS"] = [h, l * 100, s * 100]
            values[source] = list(src_vals)          # источник остаётся как ввёл пользователь

            for model, vals in values.items():
                for i, v in enumerate(vals):
                    self.scale_vars[(model, i)].set(v)
                    if keep_entries and model == source:
                        continue                      # не мешаем набирать текст
                    fmt = "%d" % round(v) if model == "RGB" else ("%.1f" % v).rstrip("0").rstrip(".")
                    self.entry_vars[(model, i)].set(fmt or "0")

            hx = to_hex(self.rgb)
            self.hex_var.set(hx)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            self.swatch.config(bg=hx, text=hx, fg="black" if lum > 140 else "white",
                               font=("TkDefaultFont", 14, "bold"))
            self._update_palette(values["HLS"])
        finally:
            self.lock = False

    def _update_palette(self, hls):
        sat = int(round(hls[2]))
        if sat != self.pal_sat:                       # перерисовка только при смене S
            self.pal_sat = sat
            img = tk.PhotoImage(width=PAL_W, height=PAL_H)
            s = sat / 100.0
            for y in range(PAL_H):
                l = 1.0 - y / (PAL_H - 1)
                row = [to_hex(hls_to_rgb(x / (PAL_W - 1) * 360.0, l, s)) for x in range(PAL_W)]
                img.put("{" + " ".join(row) + "}", to=(0, y))
            self.pal_img = img.zoom(PAL_ZOOM, PAL_ZOOM)
            self.canvas.itemconfig(self.canvas_img_id, image=self.pal_img)
        cx = hls[0] / 360.0 * (PAL_W * PAL_ZOOM - 1)
        cy = (1.0 - hls[1] / 100.0) * (PAL_H * PAL_ZOOM - 1)
        self.canvas.coords(self.marker, cx - 6, cy - 6, cx + 6, cy + 6)
        self.canvas.coords(self.marker2, cx - 4, cy - 4, cx + 4, cy + 4)


if __name__ == "__main__":
    App().mainloop()
