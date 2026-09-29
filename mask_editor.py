import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import os

# The three allowed colors (RGB)
COLOR_RED   = (255, 0, 0)      # High-priority zone
COLOR_GREEN = (0, 255, 0)      # Medium-priority zone
COLOR_BLACK = (0, 0, 0)        # Dead zone / ignored

COLOR_MAP = {
    "red":   COLOR_RED,
    "green": COLOR_GREEN,
    "black": COLOR_BLACK,
}
HEX_MAP = {
    "red":   "#ff0000",
    "green": "#00ff00",
    "black": "#000000",
}

TOOL_BRUSH     = "brush"
TOOL_FILL      = "fill"
TOOL_CIRCLE    = "circle"
TOOL_RECTANGLE = "rectangle"


class MaskEditor:
    """A Paint-like Toplevel window for editing the zone mask.

    Only three colours are allowed (red, green, black).
    Tools: brush, flood-fill, circle, rectangle.
    Saves back to mask_path on 'Save', or discards on 'Cancel'.
    Calls on_save_callback() after saving so the main app can reload.
    """

    def __init__(self, parent, mask_path="Assets/mask.png", on_save_callback=None):
        self.mask_path = mask_path
        self.on_save_callback = on_save_callback

        # Load existing mask (or create blank green one)
        if os.path.exists(mask_path):
            self.pil_image = Image.open(mask_path).convert("RGB")
        else:
            self.pil_image = Image.new("RGB", (640, 480), COLOR_GREEN)

        self.img_w, self.img_h = self.pil_image.size
        self.draw = ImageDraw.Draw(self.pil_image)

        # State
        self.current_color = "green"
        self.current_tool  = TOOL_BRUSH
        self.brush_size    = 12
        self._drag_start   = None    # for shape tools

        # ─── Window ───────────────────────────────────────────
        self.win = tk.Toplevel(parent)
        self.win.title("Mask Editor — Zone Painter")
        self.win.configure(bg='#c0c0c0')
        self.win.transient(parent)
        try:
            self.win.grab_set()      # modal
        except tk.TclError:
            pass

        font = ('MS Sans Serif', 8)
        font_bold = ('MS Sans Serif', 9, 'bold')

        # Top banner
        tk.Label(self.win, text="Zone Mask Editor", bg='#000080', fg='white',
                 font=font_bold, anchor='w', padx=8).pack(fill=tk.X)

        body = tk.Frame(self.win, bg='#c0c0c0')
        body.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ─── Left: Toolbar ────────────────────────────────────
        toolbar = tk.LabelFrame(body, text="Tools", bg='#c0c0c0',
                                font=font, relief=tk.GROOVE, bd=2)
        toolbar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        # Color selector
        color_frame = tk.LabelFrame(toolbar, text="Color", bg='#c0c0c0',
                                    font=font, relief=tk.GROOVE, bd=2)
        color_frame.pack(fill=tk.X, padx=5, pady=5)

        self.color_var = tk.StringVar(value="green")
        for name, hex_c in HEX_MAP.items():
            label = name.capitalize()
            if name == "red":
                label += " (High Pri.)"
            elif name == "green":
                label += " (Med. Pri.)"
            elif name == "black":
                label += " (Dead Zone)"
            rb = tk.Radiobutton(color_frame, text=label, variable=self.color_var,
                                value=name, bg='#c0c0c0', font=font,
                                activebackground='#c0c0c0',
                                command=self._on_color_change)
            rb.pack(anchor='w')

        self.color_preview = tk.Label(color_frame, bg=HEX_MAP[self.current_color],
                                       width=10, height=1, relief=tk.SUNKEN, bd=2)
        self.color_preview.pack(pady=5)

        # Tool selector
        tool_frame = tk.LabelFrame(toolbar, text="Draw Tool", bg='#c0c0c0',
                                   font=font, relief=tk.GROOVE, bd=2)
        tool_frame.pack(fill=tk.X, padx=5, pady=5)

        self.tool_var = tk.StringVar(value=TOOL_BRUSH)
        for tool_name, label in [(TOOL_BRUSH, "Brush"),
                                  (TOOL_FILL, "Fill (Bucket)"),
                                  (TOOL_CIRCLE, "Circle"),
                                  (TOOL_RECTANGLE, "Rectangle")]:
            rb = tk.Radiobutton(tool_frame, text=label, variable=self.tool_var,
                                value=tool_name, bg='#c0c0c0', font=font,
                                activebackground='#c0c0c0',
                                command=self._on_tool_change)
            rb.pack(anchor='w')

        # Brush size slider
        self.brush_frame = tk.LabelFrame(toolbar, text="Brush Size", bg='#c0c0c0',
                                         font=font, relief=tk.GROOVE, bd=2)
        self.brush_frame.pack(fill=tk.X, padx=5, pady=5)

        self.scale_brush = tk.Scale(self.brush_frame, from_=2, to=60, resolution=1,
                                    orient=tk.HORIZONTAL, bg='#c0c0c0', font=font,
                                    length=100, command=self._on_brush_size)
        self.scale_brush.set(self.brush_size)
        self.scale_brush.pack()

        # Save / Cancel buttons
        btn_frame = tk.Frame(toolbar, bg='#c0c0c0')
        btn_frame.pack(fill=tk.X, padx=5, pady=10)

        tk.Button(btn_frame, text="Save", command=self._save,
                  relief=tk.RAISED, bd=2, bg='#c0c0c0', font=font,
                  width=10).pack(pady=2, fill=tk.X)
        tk.Button(btn_frame, text="Cancel", command=self._cancel,
                  relief=tk.RAISED, bd=2, bg='#c0c0c0', font=font,
                  width=10).pack(pady=2, fill=tk.X)

        # Legend
        legend = tk.LabelFrame(toolbar, text="Legend", bg='#c0c0c0',
                               font=font, relief=tk.GROOVE, bd=2)
        legend.pack(fill=tk.X, padx=5, pady=5)
        for color_hex, desc in [("#00ff00", "Medium Priority"),
                                ("#ff0000", "High Priority"),
                                ("#000000", "Ignored / Dead")]:
            row = tk.Frame(legend, bg='#c0c0c0')
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, bg=color_hex, width=3, height=1,
                     relief=tk.SUNKEN, bd=1).pack(side=tk.LEFT, padx=2)
            tk.Label(row, text=desc, bg='#c0c0c0', font=font).pack(side=tk.LEFT)

        # ─── Right: Canvas ────────────────────────────────────
        canvas_frame = tk.Frame(body, bg='#c0c0c0', relief=tk.SUNKEN, bd=2)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Scale image to fit in the editor (max 800x600) while preserving aspect
        self.scale_factor = min(800 / self.img_w, 600 / self.img_h, 1.0)
        self.disp_w = int(self.img_w * self.scale_factor)
        self.disp_h = int(self.img_h * self.scale_factor)

        self.canvas = tk.Canvas(canvas_frame, width=self.disp_w, height=self.disp_h,
                                bg='grey', highlightthickness=0)
        self.canvas.pack(padx=2, pady=2)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Shape preview rectangle/oval ID
        self._preview_id = None

        self._refresh_canvas()

    # ─── Coordinate mapping ───────────────────────────────────

    def _canvas_to_image(self, cx, cy):
        """Convert canvas coordinates to image coordinates."""
        ix = int(cx / self.scale_factor)
        iy = int(cy / self.scale_factor)
        return max(0, min(ix, self.img_w - 1)), max(0, min(iy, self.img_h - 1))

    # ─── Canvas refresh ───────────────────────────────────────

    def _refresh_canvas(self):
        disp = self.pil_image.resize((self.disp_w, self.disp_h), Image.NEAREST)
        self._tk_photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("img")
        self.canvas.create_image(0, 0, image=self._tk_photo, anchor=tk.NW, tags="img")

    # ─── UI callbacks ─────────────────────────────────────────

    def _on_color_change(self):
        self.current_color = self.color_var.get()
        self.color_preview.config(bg=HEX_MAP[self.current_color])

    def _on_tool_change(self):
        self.current_tool = self.tool_var.get()

    def _on_brush_size(self, val):
        self.brush_size = int(float(val))

    # ─── Mouse handlers ───────────────────────────────────────

    def _on_press(self, event):
        ix, iy = self._canvas_to_image(event.x, event.y)
        rgb = COLOR_MAP[self.current_color]

        if self.current_tool == TOOL_BRUSH:
            r = self.brush_size // 2
            self.draw.ellipse([ix - r, iy - r, ix + r, iy + r], fill=rgb)
            self._refresh_canvas()

        elif self.current_tool == TOOL_FILL:
            self._flood_fill(ix, iy, rgb)
            self._refresh_canvas()

        elif self.current_tool in (TOOL_CIRCLE, TOOL_RECTANGLE):
            self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self.current_tool == TOOL_BRUSH:
            ix, iy = self._canvas_to_image(event.x, event.y)
            rgb = COLOR_MAP[self.current_color]
            r = self.brush_size // 2
            self.draw.ellipse([ix - r, iy - r, ix + r, iy + r], fill=rgb)
            self._refresh_canvas()

        elif self.current_tool in (TOOL_CIRCLE, TOOL_RECTANGLE) and self._drag_start:
            # Draw live preview on canvas
            if self._preview_id:
                self.canvas.delete(self._preview_id)
            x0, y0 = self._drag_start
            x1, y1 = event.x, event.y
            hex_c = HEX_MAP[self.current_color]
            if self.current_tool == TOOL_RECTANGLE:
                self._preview_id = self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline=hex_c, width=2, dash=(4, 4))
            else:
                self._preview_id = self.canvas.create_oval(
                    x0, y0, x1, y1, outline=hex_c, width=2, dash=(4, 4))

    def _on_release(self, event):
        if self.current_tool in (TOOL_CIRCLE, TOOL_RECTANGLE) and self._drag_start:
            if self._preview_id:
                self.canvas.delete(self._preview_id)
                self._preview_id = None

            sx, sy = self._drag_start
            ex, ey = event.x, event.y
            ix0, iy0 = self._canvas_to_image(sx, sy)
            ix1, iy1 = self._canvas_to_image(ex, ey)
            # Normalise
            x0, x1 = min(ix0, ix1), max(ix0, ix1)
            y0, y1 = min(iy0, iy1), max(iy0, iy1)

            rgb = COLOR_MAP[self.current_color]
            if self.current_tool == TOOL_RECTANGLE:
                self.draw.rectangle([x0, y0, x1, y1], fill=rgb)
            else:
                self.draw.ellipse([x0, y0, x1, y1], fill=rgb)

            self._drag_start = None
            self._refresh_canvas()

    # ─── Flood fill ───────────────────────────────────────────

    def _flood_fill(self, x, y, fill_color):
        """Simple scanline flood fill on the PIL image."""
        pixels = self.pil_image.load()
        target_color = pixels[x, y]
        if target_color == fill_color:
            return

        stack = [(x, y)]
        visited = set()
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            if cx < 0 or cx >= self.img_w or cy < 0 or cy >= self.img_h:
                continue
            # Snap target: only fill pixels that match the clicked color
            pix = pixels[cx, cy]
            if pix != target_color:
                continue
            visited.add((cx, cy))
            pixels[cx, cy] = fill_color
            stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])

    # ─── Save / Cancel ────────────────────────────────────────

    def _save(self):
        # Save as PNG (RGB). OpenCV reads as BGR, but the color comparisons
        # in main.py use BGR constants that correspond to these RGB values:
        #   RGB (255,0,0) → BGR (0,0,255) = gb.high_priority_color  ✓
        #   RGB (0,255,0) → BGR (0,255,0) = gb.medium_priority_region ✓
        #   RGB (0,0,0)   → BGR (0,0,0)   = gb.dead_zone_color ✓
        self.pil_image.save(self.mask_path)
        if self.on_save_callback:
            self.on_save_callback()
        self.win.destroy()

    def _cancel(self):
        self.win.destroy()
