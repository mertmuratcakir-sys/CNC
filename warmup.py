"""Build sırasında çalışır: matplotlib yazı tipi önbelleğini hazırlar (ilk açılışı hızlandırır)."""
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache"))
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    f = Figure()
    FigureCanvasAgg(f)
    f.add_subplot(111).set_title("warmup")
    f.canvas.draw()
    print("matplotlib yazı tipi önbelleği hazır")
except ImportError:
    pass
