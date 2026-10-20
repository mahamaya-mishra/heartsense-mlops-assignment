"""Run a shell command and render its transcript as a terminal-style PNG for the report.

python tools/render_terminal.py --out screenshots/tests_01_pytest.png --title "pytest" -- pytest -q
"""

from __future__ import annotations

import argparse
import os
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def font(size: int):
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def render(
    title: str,
    prompt_lines: list[str],
    output: str,
    out: Path,
    width_chars: int = 118,
    max_lines: int = 60,
):
    lines = [f"$ {p}" for p in prompt_lines]
    for raw in output.rstrip().splitlines():
        lines.extend(
            textwrap.wrap(raw, width_chars, replace_whitespace=False, drop_whitespace=False) or [""]
        )
    if len(lines) > max_lines:
        lines = [*lines[: max_lines // 2], "   ... (truncated) ...", *lines[-(max_lines // 2) :]]
    fnt = font(15)
    lh = 21
    pad = 18
    w = int(width_chars * 9.1) + 2 * pad
    h = 40 + pad * 2 + lh * len(lines)
    img = Image.new("RGB", (w, h), (30, 32, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 40], fill=(50, 53, 62))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + i * 22, 13, 28 + i * 22, 27], fill=c)
    d.text((w // 2, 20), title, fill=(200, 200, 210), font=font(14), anchor="mm")
    y = 40 + pad
    for ln in lines:
        color = (120, 220, 140) if ln.startswith("$ ") else (225, 225, 230)
        if "PASSED" in ln or "passed" in ln or ln.strip() == "ok":
            color = (120, 220, 140)
        if "FAIL" in ln or "error" in ln.lower():
            color = (255, 120, 110)
        d.text((pad, y), ln, fill=color, font=fnt)
        y += lh
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--title", default="Terminal")
    p.add_argument("--cwd", default=".")
    p.add_argument("--max-lines", type=int, default=60)
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="command after --")
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    shell_cmd = " ".join(cmd)
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
    res = subprocess.run(shell_cmd, shell=True, cwd=a.cwd, capture_output=True, text=True, env=env)
    out = res.stdout + (("\n" + res.stderr) if res.stderr.strip() else "")
    render(a.title, [shell_cmd], out, a.out, max_lines=a.max_lines)
    print(f"wrote {a.out} (exit {res.returncode})")


if __name__ == "__main__":
    main()
