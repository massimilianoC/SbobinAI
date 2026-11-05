"""Optional real-WebView smoke check on generated audio and explicitly mock inference.

Run separately from headless tests: python scripts/check-desktop-ui.py
Opens a temporary desktop window, exercises the guided steps (including the
automatic waveform, the MP4 chip and the optional review step with a real media
element), then closes it. Does not use private recordings or download models.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
SHOTS = ROOT / ".local" / "desktop-review"


def luminance(rgb):
    channels = [c / 255 for c in rgb]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def parse_rgb(text):
    return tuple(int(v) for v in text[text.index("(") + 1 : text.index(")")].split(",")[:3])


def capture_window(window, name):
    """Best-effort PNG of the native window via PrintWindow; failure only prints a note."""
    try:
        hwnd = int(window.native.Handle.ToInt64())
        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        width, height = rect.right - rect.left, rect.bottom - rect.top
        hdc = user32.GetWindowDC(hwnd)
        mem = gdi32.CreateCompatibleDC(hdc)
        bitmap = gdi32.CreateCompatibleBitmap(hdc, width, height)
        gdi32.SelectObject(mem, bitmap)
        user32.PrintWindow(hwnd, mem, 2)  # PW_RENDERFULLCONTENT: includes the WebView2 surface

        class Header(ctypes.Structure):
            _fields_ = [
                ("size", ctypes.c_uint32), ("width", ctypes.c_int32), ("height", ctypes.c_int32),
                ("planes", ctypes.c_uint16), ("bits", ctypes.c_uint16), ("compression", ctypes.c_uint32),
                ("image", ctypes.c_uint32), ("xppm", ctypes.c_int32), ("yppm", ctypes.c_int32),
                ("used", ctypes.c_uint32), ("important", ctypes.c_uint32),
            ]  # fmt: skip

        header = Header(ctypes.sizeof(Header), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
        buffer = ctypes.create_string_buffer(width * height * 4)
        gdi32.GetDIBits(mem, bitmap, 0, height, buffer, ctypes.byref(header), 0)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(hwnd, hdc)
        raw = buffer.raw
        rows = bytearray()
        for y in range(height):
            line = raw[y * width * 4 : (y + 1) * width * 4]
            rgb = bytearray(width * 3)
            rgb[0::3], rgb[1::3], rgb[2::3] = line[2::4], line[1::4], line[0::4]
            rows += b"\x00" + rgb

        def chunk(kind, data):
            body = kind + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

        png = b"\x89PNG\r\n\x1a\n" + chunk(
            b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        )
        png += chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b"")
        SHOTS.mkdir(parents=True, exist_ok=True)
        (SHOTS / f"{name}.png").write_bytes(png)
        print(f"screenshot: .local/desktop-review/{name}.png", flush=True)
    except Exception as exc:  # screenshots are evidence, not a gate
        print(f"screenshot {name} skipped: {exc}", flush=True)


def main() -> int:
    import webview

    from audio_transcript.desktop.bridge import DesktopBridge

    with tempfile.TemporaryDirectory(prefix="sbobinai-ui-check-") as directory:
        root = Path(directory)
        queue = root / "input"
        queue.mkdir()
        source = queue / "Generated test tone — 日本語.wav"
        with wave.open(str(source), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            samples = (int(8000 * math.sin(i * 2 * math.pi * 220 / 16000)) for i in range(64000))
            stream.writeframes(struct.pack("<64000h", *samples))
        config = root / "config.toml"
        config.write_text(
            '[audio_transcript]\nbackend="mock"\nvad="none"\nlanguage="it"\nmonitor=true\n'
            + "\n".join(
                f"{key}={json.dumps(str(root / key.removesuffix('_dir')))}"
                for key in ("input_dir", "process_dir", "output_dir", "processed_dir")
            ),
            encoding="utf-8",
        )
        bridge = DesktopBridge(config)
        window = webview.create_window(
            "SbobinAI — integration check",
            str(ROOT / "src/audio_transcript/desktop/assets/index.html"),
            js_api=bridge,
            width=1100,
            height=830,
            min_size=(820, 700),
        )
        bridge._window = window
        failures = []

        def wait(expression, timeout=20):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    if window.evaluate_js(expression):
                        return
                except Exception:
                    pass
                time.sleep(0.15)
            try:
                alert = window.evaluate_js("document.getElementById('alert').textContent")
            except Exception:
                alert = "?"
            raise AssertionError(f"UI condition timed out: {expression} (alert: {alert!r})")

        def fits(label):
            sizes = window.evaluate_js(
                "[document.documentElement.scrollHeight, window.innerHeight].join('/')"
            )
            scroll, inner = (int(v) for v in sizes.split("/"))
            assert scroll <= inner + 2, (
                f"{label} should fit the 1100x830 window without page scroll ({scroll} > {inner})"
            )

        def stacked(first, second):
            return window.evaluate_js(
                f"(()=>{{const a=document.querySelector('{first}').getBoundingClientRect(),"
                f"b=document.querySelector('{second}').getBoundingClientRect();return b.top>=a.bottom-1;}})()"
            )

        def check_results(*, reused):
            wait(
                "document.getElementById('files-summary').textContent.startsWith('All ') && "
                "document.getElementById('files-summary').textContent.endsWith(' files checked')",
                40,
            )
            assert window.evaluate_js(
                "(()=>{const m=[...document.querySelectorAll('#files-table .mark')];"
                "return m.length>0 && m.every(x=>x.classList.contains('verified')||x.classList.contains('complete'));})()"
            ), "Every produced file should be marked verified or complete"
            queued = window.evaluate_js(
                "[...document.querySelectorAll('#result-summary div')].find(r=>r.firstChild.textContent==='Queued file').lastChild.textContent"
            )
            assert queued.startswith("Kept in input"), queued  # mock output never archives
            # Both accordions are closed on success, yet their summaries carry the essentials.
            assert window.evaluate_js(
                "!document.getElementById('outcome-panel').open && !document.getElementById('times-panel').open"
            )
            brief = window.evaluate_js("document.getElementById('outcome-brief').textContent")
            assert brief.startswith("Completed") and "Kept in input" in brief, brief
            total = window.evaluate_js("document.getElementById('times-brief').textContent")
            assert total.startswith("Total "), total
            assert window.evaluate_js(
                "document.getElementById('outcome-brief').offsetParent!==null && document.getElementById('result-summary').checkVisibility({contentVisibilityAuto:true})===false"
            ), "Summaries are visible while the bodies stay hidden"
            times = window.evaluate_js("document.getElementById('result-times').textContent")
            assert "Total" in times and "Time per section" not in times
            if reused:
                assert "Preview video" in times
            assert window.evaluate_js("document.getElementById('result-original').hidden") != reused
            if reused:
                assert "reused" in window.evaluate_js(
                    "document.getElementById('result-description').textContent"
                )
                assert "Original transcription" in window.evaluate_js(
                    "document.getElementById('result-original').textContent"
                )
            # The review action is the dominant one; opening the folder is secondary.
            style = window.evaluate_js(
                "(()=>{const r=getComputedStyle(document.getElementById('review-open')),"
                "o=getComputedStyle(document.getElementById('open-result'));"
                "return JSON.stringify({rb:r.backgroundColor,rc:r.color,rs:parseFloat(r.fontSize),ob:o.backgroundColor,os:parseFloat(o.fontSize),"
                "oq:document.getElementById('open-result').classList.contains('quiet')});})()"
            )
            style = json.loads(style)
            assert style["rb"] == "rgb(36, 108, 85)" and style["rs"] > style["os"], style
            assert style["oq"] and style["ob"] != style["rb"], style
            assert contrast(parse_rgb(style["rc"]), parse_rgb(style["rb"])) >= 4.5, style
            folder = json.loads(
                window.evaluate_js(
                    "(()=>{const o=document.getElementById('open-result'),c=getComputedStyle(o),q=getComputedStyle(document.getElementById('again'));"
                    "return JSON.stringify({w:parseFloat(c.borderTopWidth),c:c.borderTopColor,qw:parseFloat(q.borderTopWidth),svg:!!o.querySelector('svg')});})()"
                )
            )
            assert folder["svg"] and folder["w"] >= 2 and folder["w"] > folder["qw"], folder
            assert folder["c"] == "rgb(36, 108, 85)", folder
            fits("The results screen")
            window.resize(820, 700)
            time.sleep(0.6)
            assert stacked("#outcome-panel", "#times-panel")
            window.resize(1100, 830)
            time.sleep(0.6)

        def check_failed_results():
            """The failed-run view is not reachable with a mock worker: feed the page a failure."""
            window.evaluate_js(
                "finish({result:{ok:false,jobs:[{error:'Synthetic failure for the layout check'}]},summary:{status:'failed'},can_open:false,can_review:false,stages:{},progress:{}})"
            )
            wait("!document.querySelector('[data-step=\"5\"]').hidden")
            assert window.evaluate_js("document.getElementById('outcome-panel').open")
            assert not window.evaluate_js("document.getElementById('times-panel').open")
            assert "Synthetic failure" in window.evaluate_js(
                "document.getElementById('result-description').textContent"
            )
            assert window.evaluate_js(
                "document.getElementById('open-result').disabled && document.getElementById('review-open').hidden"
            )
            fits("The failed results screen")

        def check_layout():
            """Step 3 fits the default window; at the minimum size the footer stays reachable."""
            assert window.evaluate_js(
                "document.documentElement.scrollHeight <= window.innerHeight + 2"
            ), "Step 3 should fit the 1100x830 window"
            assert window.evaluate_js(
                "document.querySelectorAll('.out-group').length===5 && "
                "[...document.querySelectorAll('.out-group>span')].map(e=>e.textContent).join()"
                "==='Text,Subtitles,Audio,Video,Data & reports'"
            )
            assert window.evaluate_js(
                "(()=>{const a=document.querySelector('.retention-col').getBoundingClientRect(),"
                "b=document.querySelector('.output-options').getBoundingClientRect();"
                "return a.right<=b.left+1 && a.width<b.width && Math.abs(a.top-b.top)<30;})()"
            ), "Retention should sit left of the wider output groups"
            window.resize(820, 700)
            time.sleep(0.6)
            assert window.evaluate_js(
                "(()=>{const a=document.querySelector('.retention-col').getBoundingClientRect(),"
                "b=document.querySelector('.output-options').getBoundingClientRect();"
                "return b.top>=a.bottom-1;})()"
            ), "Narrow widths should use one column"
            assert window.evaluate_js(
                "(()=>{const f=document.querySelector('footer').getBoundingClientRect();"
                "return f.bottom<=window.innerHeight+1 && f.top>=0;})()"
            ), "Footer buttons must stay reachable at the minimum size"
            window.resize(1100, 830)
            time.sleep(0.6)

        def layout_rects(label):
            """No two blocks of step 6 may intersect and the footer content sits below the grid."""
            report = json.loads(
                window.evaluate_js(
                    "(()=>{const r=q=>{const e=document.querySelector(q);return e&&!e.hidden&&e.offsetParent!==null?e.getBoundingClientRect():null};"
                    "const items={wave:r('#review-wave'),player:r('#player'),subtitle:r('#subtitle'),cues:r('#cues'),hint:r('.review-hint'),actions:r('.review-actions')};"
                    "const names=Object.keys(items).filter(k=>items[k]),bad=[];"
                    "for(let i=0;i<names.length;i++)for(let j=i+1;j<names.length;j++){const a=items[names[i]],b=items[names[j]];"
                    "if(a.left<b.right-1&&b.left<a.right-1&&a.top<b.bottom-1&&b.top<a.bottom-1)bad.push(names[i]+'/'+names[j]);}"
                    "const g=document.querySelector('.review-grid').getBoundingClientRect();"
                    "return JSON.stringify({bad,ordered:g.bottom<=items.hint.top+1&&items.hint.top<=items.actions.top+1,names});})()"
                )
            )
            assert not report["bad"], f"{label}: overlapping blocks {report['bad']}"
            assert report["ordered"], f"{label}: the hint and buttons must sit below the grid"

        def subtitle_is_fixed():
            """The subtitle box keeps its size and never moves its neighbours, whatever the cue."""
            window.evaluate_js("document.getElementById('player').pause()")
            long_text = " ".join(f"word{i}" for i in range(400))
            seen = []
            for text in ("", "A short cue.", long_text):
                seen.append(
                    json.loads(
                        window.evaluate_js(
                            f"(()=>{{const s=document.getElementById('subtitle');s.textContent={json.dumps(text)};"
                            "const t=q=>{const e=document.querySelector(q);return e&&e.offsetParent!==null?Math.round(e.getBoundingClientRect().top*10)/10:null};"
                            "return JSON.stringify({h:s.getBoundingClientRect().height,over:s.scrollHeight>s.clientHeight+1,"
                            "tops:[t('#review-wave'),t('#player'),t('.review-grid'),t('.review-hint'),t('.review-actions'),t('#cues')]});})()"
                        )
                    )
                )
            assert seen[0]["h"] == seen[1]["h"] == seen[2]["h"], seen
            assert seen[0]["tops"] == seen[1]["tops"] == seen[2]["tops"], seen
            assert seen[2]["over"] and not seen[0]["over"], seen
            layout_rects("long cue")
            window.evaluate_js(
                "document.getElementById('subtitle').textContent='';cueIndex=-1;document.querySelectorAll('#cues li.active').forEach(e=>e.classList.remove('active'))"
            )

        def check_review(expect_video):
            assert window.evaluate_js("!document.getElementById('review-open').hidden")
            window.evaluate_js("document.getElementById('review-open').click()")
            wait("!document.querySelector('[data-step=\"6\"]').hidden")
            wait("document.querySelectorAll('#cues li').length>0")
            wait("document.getElementById('player').readyState>=1", 30)
            window.evaluate_js("document.getElementById('player').muted=true")
            fits("Step 6")
            layout_rects("after loadedmetadata")
            # The overlap bug appeared only while real media played.
            window.evaluate_js("document.getElementById('player').play()")
            wait(
                "!document.getElementById('player').paused && document.getElementById('player').currentTime>0.3",
                15,
            )
            layout_rects("while playing")
            fits("Step 6 while playing")
            if expect_video:
                capture_window(window, "step6")
            subtitle_is_fixed()
            if expect_video:
                window.resize(820, 700)
                time.sleep(0.6)
                assert stacked(".review-left", ".cues-col"), (
                    "Step 6 should use one column when narrow"
                )
                layout_rects("narrow")
                window.resize(1100, 830)
                time.sleep(0.6)
            try:
                wait(
                    f"document.getElementById('player').classList.contains('audio-only')==={str(not expect_video).lower()}",
                    10,
                )
            except AssertionError:
                info = window.evaluate_js(
                    "(()=>{const v=document.getElementById('player');return JSON.stringify({src:v.currentSrc.slice(-12),q:v.getVideoPlaybackQuality&&v.getVideoPlaybackQuality().totalVideoFrames,paused:v.paused,t:v.currentTime,w:v.videoWidth,h:v.videoHeight,rs:v.readyState,dur:v.duration,h264:v.canPlayType('video/mp4; codecs=\"avc1.64001F, mp4a.40.2\"'),cls:v.className,err:v.error&&v.error.code,label:document.getElementById('review-source').textContent});})()"
                )
                raise AssertionError(
                    f"picture area should follow the media (video={expect_video}): {info} served={bridge._playback_server._path.name}"
                ) from None
            last = window.evaluate_js("document.querySelectorAll('#cues li').length-1")
            for index in sorted({0, last}):
                window.evaluate_js(f"document.querySelectorAll('#cues li')[{index}].click()")
                wait(
                    f"document.getElementById('subtitle').textContent===document.querySelectorAll('#cues li span')[{index}].textContent && "
                    f"document.querySelectorAll('#cues li')[{index}].classList.contains('active')"
                )
            assert window.evaluate_js(
                "document.getElementById('review-source').textContent.startsWith('Playing')"
            )
            window.evaluate_js("document.getElementById('review-back').click()")
            wait("!document.querySelector('[data-step=\"5\"]').hidden")
            assert window.evaluate_js(
                "!document.getElementById('player').hasAttribute('src') && document.getElementById('player').networkState===0"
            ), "Back must release the media element"
            assert bridge._playback_server._path is None, "Back must release the served file"

        def check():
            try:
                wait(
                    "document.querySelectorAll('.file-card').length===1 && !document.getElementById('next').disabled"
                )
                assert window.evaluate_js("document.querySelector('input[name=file]').checked")
                window.evaluate_js("document.getElementById('next').click()")
                wait("!document.querySelector('[data-step=\"2\"]').hidden")
                window.evaluate_js("document.getElementById('next').click()")
                wait("!document.querySelector('[data-step=\"3\"]').hidden")
                # A recording that was already in input defaults to keep (GUI-08).
                assert window.evaluate_js("document.getElementById('retention').value==='keep-all'")
                assert window.evaluate_js("document.querySelector('[data-output=mp3]').checked")
                # Audio-only: the MP4 chip is disabled with a reason and unchecked.
                assert window.evaluate_js(
                    "document.getElementById('mp4').disabled && !document.getElementById('mp4').checked"
                )
                assert window.evaluate_js(
                    "document.getElementById('mp4-note').textContent.includes('Audio-only')"
                )
                # The waveform starts by itself; no button exists any more.
                assert window.evaluate_js("document.getElementById('preview')===null")
                wait("document.getElementById('wave-panel').dataset.state==='ready'")
                assert window.evaluate_js("document.getElementById('wave-failed').hidden")
                check_layout()
                capture_window(window, "step3")
                window.evaluate_js(
                    "document.querySelectorAll('[data-output]').forEach(el=>{el.checked=el.dataset.output==='srt';el.dispatchEvent(new Event('change'));})"
                )
                window.evaluate_js(
                    "document.getElementById('start-range').value=1; document.getElementById('start-range').dispatchEvent(new Event('input'))"
                )
                assert window.evaluate_js("document.getElementById('retention').disabled")
                assert window.evaluate_js("document.getElementById('start-time').value==='1'")
                window.evaluate_js("document.getElementById('next').click()")
                wait("!document.querySelector('[data-step=\"4\"]').hidden")
                assert window.evaluate_js(
                    "document.getElementById('recap').textContent.includes('00:00:01')"
                )
                window.evaluate_js("document.getElementById('start').click()")
                wait(
                    "!document.getElementById('recap-panel').open && !document.getElementById('run-progress').hidden"
                )
                assert window.evaluate_js(
                    "document.documentElement.scrollHeight <= window.innerHeight + 2"
                ), "Collapsed processing view should fit the desktop window"
                wait("!document.querySelector('[data-step=\"5\"]').hidden", 40)
                assert window.evaluate_js("!document.getElementById('recap-panel').open")
                assert window.evaluate_js(
                    "document.getElementById('result-title').textContent.includes('ready')"
                )
                assert window.evaluate_js("!document.getElementById('open-result').disabled")
                assert bridge._result_folder is not None
                assert (bridge._result_folder / "transcript.srt").is_file()
                assert not (bridge._result_folder / "video.mp4").exists()
                assert not (bridge._result_folder / "transcript.txt").exists()
                assert (bridge._result_folder / "intermediate.zip").is_file()
                assert not (bridge._result_folder / "intermediate").exists()
                assert source.exists()
                check_results(reused=False)
                check_review(expect_video=False)
                # A generated H.264 video, run twice: first without MP4, then with MP4 added.
                video = queue / "Second generated recording.mp4"
                subprocess.run(
                    [
                        "ffmpeg", "-v", "error", "-y",
                        "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=10:duration=120",
                        "-f", "lavfi", "-i", "sine=frequency=330:duration=120",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(video),
                    ],
                    check=True,
                )  # fmt: skip
                window.evaluate_js("document.getElementById('again').click()")
                wait(
                    "document.querySelectorAll('.file-card').length===2 && !document.getElementById('refresh').disabled"
                )
                assert window.evaluate_js("document.getElementById('next').disabled")
                assert window.evaluate_js(
                    "document.querySelectorAll('input[name=file]:checked').length===0"
                )
                window.evaluate_js("document.querySelectorAll('input[name=file]')[1].click()")
                wait("!document.getElementById('next').disabled")
                wait("document.getElementById('wave-panel').dataset.state==='ready'")

                def run_video(*, with_mp4, timeout):
                    for _ in range(2):
                        window.evaluate_js("document.getElementById('next').click()")
                    wait("!document.querySelector('[data-step=\"3\"]').hidden")
                    assert window.evaluate_js("!document.getElementById('mp4').disabled")
                    assert window.evaluate_js("!document.getElementById('mp4').checked")
                    if with_mp4:
                        # The default deliveries (every text, subtitle, data and report format
                        # plus MP3) and MP4: ten result files, the case a real user sees.
                        window.evaluate_js(
                            "document.querySelectorAll('[data-output]').forEach(el=>{el.checked=el.dataset.output!=='flac';el.dispatchEvent(new Event('change'));})"
                        )
                        assert window.evaluate_js(
                            "document.getElementById('mp4-note').textContent.includes('CPU')"
                        )
                        fits("Step 3 with MP4 checked and its note visible")
                    window.evaluate_js("document.getElementById('next').click()")
                    wait("!document.querySelector('[data-step=\"4\"]').hidden")
                    # Record what the user would see between polls.
                    window.evaluate_js(
                        "window.__seen=[];const note=()=>window.__seen.push([document.getElementById('phase').textContent,"
                        "document.getElementById('progress-detail').textContent,document.getElementById('progress').hasAttribute('value'),"
                        "[...document.getElementById('stage-rail').children].map(e=>e.className).join('|'),!document.getElementById('reuse-note').hidden,"
                        "document.querySelector('[data-res=gpu] .res-sub').textContent,document.querySelector('[data-res=cpu] .res-num').textContent,"
                        "document.querySelector('[data-res=ram] .res-num').textContent,document.querySelector('[data-res=vram] .res-num').textContent,"
                        "document.querySelector('[data-res=gpu] .res-num').textContent]);"
                        "new MutationObserver(note).observe(document.querySelector('[data-step=\"4\"]'),{subtree:true,childList:true,characterData:true,attributes:true});"
                    )
                    window.evaluate_js("document.getElementById('start').click()")
                    if with_mp4:  # look at the processing view with its cards during the render
                        rendering = False
                        try:
                            wait(
                                "document.getElementById('phase').textContent.startsWith('Rendering') && "
                                "document.getElementById('percent').textContent!=='…'",
                                30,
                            )
                            rendering = True
                        except AssertionError:
                            pass  # a very fast render may already be over; the observer covers it
                        if rendering:
                            fits("The processing view with resource cards")
                            try:  # let the first resource samples arrive so the picture shows values
                                wait(
                                    "document.querySelector('[data-res=cpu] .res-num').textContent!=='—'",
                                    6,
                                )
                            except AssertionError:
                                pass
                            time.sleep(0.7)  # let the level-bar transition finish
                            if window.evaluate_js(
                                "document.querySelector('[data-res=ram] .res-num').textContent!=='—'"
                            ):
                                ram_bar = window.evaluate_js(
                                    "parseFloat(document.querySelector('[data-res=ram] .res-bar i').style.width)"
                                )
                                assert ram_bar > 3, f"the RAM level bar is empty ({ram_bar})"
                            capture_window(window, "step4")
                    wait("!document.querySelector('[data-step=\"5\"]').hidden", timeout)

                run_video(with_mp4=False, timeout=60)
                first_phases = json.loads(window.evaluate_js("JSON.stringify(window.__seen)"))
                check_results(reused=False)
                assert not (bridge._result_folder / "video.mp4").exists()
                first_job = bridge._result_job[0]
                window.evaluate_js("document.getElementById('again').click()")
                wait(
                    "document.querySelectorAll('.file-card').length===2 && !document.getElementById('refresh').disabled"
                )
                window.evaluate_js("document.querySelectorAll('input[name=file]')[1].click()")
                wait("!document.getElementById('next').disabled")
                wait("document.getElementById('wave-panel').dataset.state==='ready'")
                run_video(with_mp4=True, timeout=180)
                assert bridge._result_job[0] == first_job, (
                    "identical settings must reuse the version"
                )
                assert (bridge._result_folder / "video.mp4").is_file(), "the MP4 was not rendered"
                seen = window.evaluate_js("JSON.stringify(window.__seen)")
                phases = json.loads(seen)
                assert any(item[4] for item in phases), "reuse was never announced"
                assert not any(item[4] for item in first_phases), "a fresh run must not claim reuse"
                assert any(
                    "Rendering preview video" in d and "%" in d and "remaining" in d and has
                    for _, d, has, *_ in phases
                ), "no determinate render progress was shown"
                assert any("reused" in item[3] for item in phases), (
                    "the rail never marked the model steps as reused"
                )
                # Which encoder ran on this machine, and live resource cards with real values.
                nvenc = bridge._processor.nvenc_usable()
                label = "NVENC" if nvenc else "CPU"
                assert any(
                    "Rendering preview video" in item[1] and label in item[1] for item in phases
                ), f"the render line should name the encoder ({label})"
                assert any(item[6] not in ("—", "") for item in phases), (
                    "CPU card never showed a value"
                )
                assert any(item[7] not in ("—", "") for item in phases), (
                    "RAM card never showed a value"
                )
                if shutil.which("nvidia-smi"):
                    assert any(item[9] not in ("—", "") for item in phases), (
                        "GPU card never showed a value"
                    )
                    assert any(item[8] not in ("—", "") for item in phases), (
                        "VRAM card never showed a value"
                    )
                    assert any(
                        "Rendering preview video" in item[0] and "Encoder" in item[5]
                        for item in phases
                    ), "the GPU card should show encoder use while rendering"
                notes = window.evaluate_js(
                    "[...document.querySelectorAll('#files-table tr')].filter(r=>r.cells[0].textContent==='video.mp4').map(r=>r.cells[4].textContent).join()"
                )
                assert ("NVENC" if nvenc else "CPU (x264)") in notes, notes
                assert "not checked" not in notes, notes  # the data end of the MP4 was verified
                check_results(reused=True)
                # The default selection produces ten files; the page must not scroll with them.
                rows = window.evaluate_js("document.querySelectorAll('#files-table tr').length")
                assert rows >= 10, f"expected the ten default files in the table, got {rows}"
                fits("Step 5 with ten files")
                assert window.evaluate_js(
                    "(()=>{const a=document.querySelector('aside').getBoundingClientRect();"
                    "return a.bottom<=window.innerHeight+2;})()"
                ), "The side panel must not be cut off"
                capture_window(window, "step5")
                assert window.evaluate_js(
                    "[...document.querySelectorAll('#files-table td:first-child')].some(t=>t.textContent==='video.mp4') && "
                    "[...document.querySelectorAll('#files-table .note')].some(t=>t.textContent.startsWith('720p'))"
                )
                check_review(expect_video=True)
                check_failed_results()
                print(
                    "PASS: real WebView, automatic selection and waveform, keep default, "
                    "MP4 chip and render, reuse feedback with determinate render progress, "
                    "results sections and file checks, step 3/5/6 layout, mock worker, "
                    "review playback with live subtitles for audio and H.264 video.",
                    flush=True,
                )
            except Exception as exc:
                failures.append(str(exc))
                print(f"FAIL: {exc}", file=sys.stderr, flush=True)
            finally:
                # Never close a still-running worker; it owns recoverable state.
                deadline = time.monotonic() + 45
                while not bridge.can_close() and time.monotonic() < deadline:
                    time.sleep(0.2)
                window.destroy()

        webview.start(check)
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
