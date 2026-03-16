import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import threading

import torch
import torchaudio
import torchaudio.functional as AF
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import PianoNet
from inference import predict_note_multi_window, index_to_note_name

AUDIO_EXTS = (".wav", ".ogg", ".flac", ".mp3", ".aiff")
MODEL_PATH = "piano_net.pth"
TARGET_SR  = 22050

# Note names for y-axis ticks (show C notes + A4)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_TICK_LABELS = {i: index_to_note_name(i)
                for i in range(88)
                if _NOTE_NAMES[(i + 21) % 12] == "C" or i == 57}  # C notes + A4


def _play_file_thread(filepath):
    """Load audio and play via sounddevice (non-blocking)."""
    try:
        wf, sr = torchaudio.load(filepath)
        if wf.shape[0] > 1:
            wf = wf.mean(0, keepdim=True)
        if sr != TARGET_SR:
            wf = AF.resample(wf, sr, TARGET_SR)
        sd.stop()
        sd.play(wf.squeeze(0).numpy(), samplerate=TARGET_SR)
    except Exception:
        pass


class ToneRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Piano Tone AI")
        self.root.geometry("960x720")
        self.root.minsize(800, 580)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None
        self._current_folder = ""
        self._result     = None
        self._sel_window = 0

        self._build_ui()
        self._load_model()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self.root, padding=(10, 8, 10, 4))
        top.pack(fill=tk.X)
        ttk.Label(top, text="Piano Tone AI", font=("Helvetica", 15, "bold")).pack(side=tk.LEFT)
        ttk.Button(top, text="Reload Model", command=self._load_model).pack(side=tk.RIGHT)

        # Horizontal split: file list | results
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # ── Left: folder + file list ─────────────────────────────────────────
        left = ttk.Frame(pane, padding=4)
        pane.add(left, weight=1)

        folder_row = ttk.Frame(left)
        folder_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(folder_row, text="Select Folder", command=self._select_folder).pack(side=tk.LEFT)
        self.folder_label = ttk.Label(folder_row, text="  No folder selected",
                                      font=("Helvetica", 9, "italic"))
        self.folder_label.pack(side=tk.LEFT, padx=6)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.file_list = tk.Listbox(list_frame, yscrollcommand=scroll.set,
                                    font=("Consolas", 10), activestyle="dotbox",
                                    selectmode=tk.SINGLE, exportselection=False)
        scroll.config(command=self.file_list.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list.pack(fill=tk.BOTH, expand=True)
        self.file_list.bind("<<ListboxSelect>>", self._on_select)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_row, text="▶  Play again", command=self._play_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="⟳  Refresh",    command=self._refresh_list).pack(side=tk.LEFT, padx=2)

        # ── Right: results + charts ──────────────────────────────────────────
        right = ttk.Frame(pane, padding=(12, 4, 4, 4))
        pane.add(right, weight=2)

        # Summary row: big note + confidence
        summary = ttk.Frame(right)
        summary.pack(fill=tk.X)

        self.note_label = ttk.Label(summary, text="--",
                                    font=("Helvetica", 52, "bold"), foreground="#1a6ebd",
                                    width=5, anchor="center")
        self.note_label.pack(side=tk.LEFT, padx=(0, 16))

        detail = ttk.Frame(summary)
        detail.pack(side=tk.LEFT, anchor=tk.W)
        self.conf_label = ttk.Label(detail, text="", font=("Helvetica", 13))
        self.conf_label.pack(anchor=tk.W)
        self.windows_label = ttk.Label(detail, text="", font=("Helvetica", 9), foreground="gray")
        self.windows_label.pack(anchor=tk.W)
        self.top3_label = ttk.Label(detail, text="", font=("Helvetica", 10), justify=tk.LEFT)
        self.top3_label.pack(anchor=tk.W, pady=(6, 0))

        self.progress = ttk.Progressbar(right, mode="indeterminate", length=180)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 6))

        # Matplotlib: scatter (top) + mel spectrogram (bottom)
        self.fig, (self.ax_scatter, self.ax_mel) = plt.subplots(
            2, 1, figsize=(5, 4.2),
            gridspec_kw={"height_ratios": [1, 1.5]},
        )
        self.fig.patch.set_facecolor("#f8f8f8")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_chart_click)
        self._draw_empty_chart()

        # Status bar
        self.status_var = tk.StringVar(value="Loading model…")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W,
                  font=("Helvetica", 9)).pack(fill=tk.X, side=tk.BOTTOM)

    # ── Chart ────────────────────────────────────────────────────────────────

    def _draw_empty_chart(self):
        for ax, msg in [
            (self.ax_scatter, "Select a file to see window predictions"),
            (self.ax_mel,     "Mel spectrogram appears here"),
        ]:
            ax.clear()
            ax.set_facecolor("#f0f0f0")
            ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")
            ax.tick_params(labelsize=7)
        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()

    def _draw_scatter(self, result, sel_win):
        labels    = result["window_labels"]
        confs     = result["window_confs"]
        final     = result["note_name"]
        final_idx = next(i for i in range(88) if index_to_note_name(i) == final)

        ax = self.ax_scatter
        ax.clear()
        ax.set_facecolor("#f9f9f9")

        n  = len(labels)
        xs = list(range(1, n + 1))

        # Alternating octave bands
        for oct_start in range(0, 88, 12):
            color = "#e8e8e8" if (oct_start // 12) % 2 == 0 else "#f5f5f5"
            ax.axhspan(oct_start - 0.5, min(oct_start + 11.5, 87.5), color=color, zorder=0)

        # Final averaged prediction: dashed line
        ax.axhline(final_idx, color="#1a6ebd", linewidth=1.2,
                   linestyle="--", alpha=0.7, zorder=1)

        # Highlight selected window
        ax.axvline(sel_win + 1, color="#cc0000", alpha=0.30, linewidth=8, zorder=1)

        # Per-window dots
        for x, lbl, conf in zip(xs, labels, confs):
            color = "#2ca02c" if lbl == final_idx else "#ff7f0e"
            ax.scatter(x, lbl, s=80 + 120 * conf, color=color,
                       zorder=3, alpha=0.85, edgecolors="white", linewidths=0.5)

        # Y-axis: C notes + A4
        tick_pos = sorted(_TICK_LABELS.keys())
        ax.set_yticks(tick_pos)
        ax.set_yticklabels([_TICK_LABELS[t] for t in tick_pos], fontsize=7)
        ax.set_ylim(-1, 88)

        ax.set_xlim(0.3, n + 0.7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("Window # — click to view mel spec", fontsize=7)
        ax.set_ylabel("Predicted note", fontsize=8)

        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
                   markersize=7, label='agrees'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
                   markersize=7, label='differs'),
            Line2D([0], [0], color='#1a6ebd', linestyle='--',
                   linewidth=1.2, label=f'avg → {final}'),
        ], fontsize=7, loc="upper right", framealpha=0.8)

    def _draw_mel(self, result, win_idx):
        ax   = self.ax_mel
        spec = result["window_specs"][win_idx]   # numpy [N_MELS, T]
        note = index_to_note_name(result["window_labels"][win_idx])
        conf = result["window_confs"][win_idx]

        ax.clear()
        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma",
                  interpolation="nearest")
        ax.set_title(f"Window {win_idx + 1}  —  {note}  ({conf:.0%})", fontsize=8, pad=3)
        ax.set_xlabel("Time frame", fontsize=7)
        ax.set_ylabel("Mel bin", fontsize=7)
        ax.tick_params(labelsize=6)

    def _draw_window_chart(self, result):
        self._draw_scatter(result, self._sel_window)
        self._draw_mel(result, self._sel_window)
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

    def _on_chart_click(self, event):
        if event.inaxes != self.ax_scatter or self._result is None:
            return
        x = event.xdata
        if x is None:
            return
        n   = len(self._result["window_labels"])
        idx = max(0, min(n - 1, round(x) - 1))
        if idx == self._sel_window:
            return
        self._sel_window = idx
        self._draw_scatter(self._result, idx)
        self._draw_mel(self._result, idx)
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

    # ── Model ────────────────────────────────────────────────────────────────

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showwarning("Model Not Found",
                                   f"Could not find '{MODEL_PATH}'.\nPlease train the model first.")
            self._set_status("Model not found.")
            return
        try:
            m = PianoNet(num_classes=88)
            m.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
            m.to(self.device).eval()
            self.model = m
            self._set_status(f"Model ready  |  {MODEL_PATH}  |  device: {self.device}")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load model:\n{e}")
            self._set_status("Model load failed.")

    # ── Folder / file list ───────────────────────────────────────────────────

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select folder with audio files")
        if not folder:
            return
        self._current_folder = folder
        self.folder_label.config(text=f"  {os.path.basename(folder)}")
        self._refresh_list()

    def _refresh_list(self):
        if not self._current_folder:
            return
        files = sorted(f for f in os.listdir(self._current_folder)
                       if os.path.splitext(f)[1].lower() in AUDIO_EXTS)
        self.file_list.delete(0, tk.END)
        for f in files:
            self.file_list.insert(tk.END, f)
        count = len(files)
        self._set_status(f"{count} audio file{'s' if count != 1 else ''} found  |  "
                         f"Model: {MODEL_PATH}  |  device: {self.device}")

    def _selected_path(self):
        sel = self.file_list.curselection()
        if not sel or not self._current_folder:
            return None
        return os.path.join(self._current_folder, self.file_list.get(sel[0]))

    # ── Selection → play + infer ─────────────────────────────────────────────

    def _on_select(self, _event=None):
        path = self._selected_path()
        if not path:
            return
        threading.Thread(target=_play_file_thread, args=(path,), daemon=True).start()
        self._start_inference(path)

    def _play_selected(self):
        path = self._selected_path()
        if path:
            threading.Thread(target=_play_file_thread, args=(path,), daemon=True).start()

    # ── Inference ────────────────────────────────────────────────────────────

    def _start_inference(self, path):
        if self.model is None:
            return
        self._infer_version = getattr(self, "_infer_version", 0) + 1
        version = self._infer_version

        self.note_label.config(text="…")
        self.conf_label.config(text="")
        self.windows_label.config(text="")
        self.top3_label.config(text="")
        self.progress.pack(pady=4)
        self.progress.start()
        self._set_status(f"Running inference on {os.path.basename(path)} …")

        def _run():
            try:
                result = predict_note_multi_window(path, self.model, self.device)
                self.root.after(0, self._show_result, result, version)
            except Exception as e:
                self.root.after(0, self._show_error, str(e), version)

        threading.Thread(target=_run, daemon=True).start()

    def _show_result(self, result, version):
        if version != self._infer_version:
            return
        self.progress.stop()
        self.progress.pack_forget()

        self.note_label.config(text=result["note_name"])
        self.conf_label.config(text=f"Confidence: {result['confidence']:.1%}")
        n = result["n_windows"]
        self.windows_label.config(text=f"{n} window{'s' if n != 1 else ''}")

        lines = ["Runner-ups:"]
        for name, prob in zip(result["top3_names"][1:], result["top3_probs"][1:]):
            lines.append(f"   {name:<5}  {prob:.1%}")
        self.top3_label.config(text="\n".join(lines))

        # Auto-select most confident window
        self._result     = result
        self._sel_window = result["window_confs"].index(max(result["window_confs"]))
        self._draw_window_chart(result)

        self._set_status(f"Done  |  {result['note_name']}  ({result['confidence']:.1%})"
                         f"  |  {n} window{'s' if n != 1 else ''}")

    def _show_error(self, msg, version):
        if version != self._infer_version:
            return
        self.progress.stop()
        self.progress.pack_forget()
        self.note_label.config(text="Error")
        self._set_status(f"Error: {msg}")

    def _set_status(self, text):
        self.status_var.set(text)


if __name__ == "__main__":
    root = tk.Tk()
    app = ToneRecognitionApp(root)
    root.mainloop()
