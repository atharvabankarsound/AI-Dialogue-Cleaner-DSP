"""Assemble docs/index.html from the template, the JS DSP port and the exported model."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
html = (root / "web/template.html").read_text()
html = html.replace("/*__DSP__*/", (root / "web/dsp.js").read_text()).replace("__MODEL__", (root / "web_model.json").read_text())
(root / "docs/index.html").write_text(html)
print("docs/index.html", len(html) // 1024, "KiB")
