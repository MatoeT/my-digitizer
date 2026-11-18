"""
NE 111 Term Project - Plot Digitizer
------------------------------------
A single-page Streamlit app that pulls x-y data out of a picture of a graph.

How it works, in two phases:
  1. CALIBRATE: the user clicks three spots on the image (the origin, a point
     on the x-axis, a point on the y-axis) and types the real values there.
  2. PICK: the user clicks data points. Each click is stored in image pixels
     and converted to real x-y values with the calibration.

Run locally with:   uv run streamlit run app.py
Extra package:      uv add streamlit-image-coordinates
"""

import csv
import html
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

COLORS = ["#ff1aa8", "#00b8d4", "#64dd17", "#ff9100", "#3d5afe",
          "#00bfa5", "#c51162", "#795548", "#aeea00", "#000000"]
MAX_DISPLAY_W = 850
MAX_DISPLAY_H = 650
THUMB_W = 260
PNG_LIMIT = 350_000
CAL_NAMES = ["origin", "point on the x-axis", "point on the y-axis"]
CAL_TAGS = ["O", "X", "Y"]
MODE_CAL = "1 · Set up axes"
MODE_PICK = "2 · Pick data points"
DEFAULT_SETTINGS = {"x_label": "x", "y_label": "y", "x_scale": "Linear", "y_scale": "Linear",
                    "x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
SAMPLE_SETTINGS = {"x_label": "Wavelength (nm)", "y_label": "Absorbance (a.u.)",
                   "x_scale": "Linear", "y_scale": "Linear",
                   "x0": 400.0, "y0": 0.0, "x1": 700.0, "y1": 1.0}


def load_image(file) -> Image.Image:
    img = Image.open(file)
    img.seek(0)
    img = ImageOps.exif_transpose(img)

    if img.mode.startswith("I") or img.mode == "F":
        arr = np.asarray(img, dtype=float)
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8))

    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(white, img)

    return img.convert("RGB")


def choose_format(img) -> str:
    preview = img.copy()
    preview.thumbnail((MAX_DISPLAY_W, MAX_DISPLAY_H))
    buf = io.BytesIO()
    preview.save(buf, "PNG", compress_level=6)
    return "PNG" if buf.tell() < PNG_LIMIT else "JPEG"


def get_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_sample_image() -> Image.Image:
    W, H = 900, 600
    left, right, top, bottom = 110, 850, 60, 500
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    font, small = get_font(18), get_font(14)

    def to_px(x, y):
        return (left + (x - 400) / 300 * (right - left),
                bottom - y / 1.0 * (bottom - top))

    d.line([left, top, left, bottom, right, bottom], fill="black", width=2)
    for x in range(400, 701, 50):
        px, py = to_px(x, 0)
        d.line([px, py, px, py + 7], fill="black", width=2)
        d.text((px, py + 12), str(x), fill="black", font=small, anchor="ma")
    for i in range(0, 11, 2):
        px, py = to_px(400, i / 10)
        d.line([px - 7, py, px, py], fill="black", width=2)
        d.text((px - 12, py), f"{i / 10:.1f}", fill="black", font=small, anchor="rm")
    d.text(((left + right) / 2, bottom + 45), "Wavelength (nm)",
           fill="black", font=font, anchor="ma")
    d.text((left - 10, top - 35), "Absorbance (a.u.)", fill="black", font=font)
    d.text((right, top - 35), "Gold nanoparticles (sample graph)",
           fill="gray", font=small, anchor="ra")

    for peak, width, height, colour, shape in [(520, 45, 0.85, "#1f4e9c", "circle"),
                                               (570, 70, 0.60, "#c0392b", "square")]:
        for x in range(410, 700, 20):
            y = 0.08 + height * math.exp(-((x - peak) / width) ** 2)
            px, py = to_px(x, y)
            box = [px - 6, py - 6, px + 6, py + 6]
            if shape == "circle":
                d.ellipse(box, fill=colour)
            else:
                d.rectangle(box, fill=colour)

    return img.rotate(2, expand=True, fillcolor="white", resample=Image.BICUBIC)


def make_transform(cal_pts, x0, x1, y0, y1, x_log, y_log):
    if len(cal_pts) < 3:
        raise ValueError("Click all three calibration points first.")
    if x0 == x1:
        raise ValueError("The two x values must be different.")
    if y0 == y1:
        raise ValueError("The two y values must be different.")
    if x_log and (x0 <= 0 or x1 <= 0):
        raise ValueError("A log x-axis needs x values greater than zero.")
    if y_log and (y0 <= 0 or y1 <= 0):
        raise ValueError("A log y-axis needs y values greater than zero.")

    origin = np.array(cal_pts[0], dtype=float)
    u = np.array(cal_pts[1], dtype=float) - origin
    v = np.array(cal_pts[2], dtype=float) - origin
    M = np.column_stack([u, v])
    if abs(np.linalg.det(M)) < 1e-6 * (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12):
        raise ValueError("The three calibration points are in a straight line. "
                         "Reset the clicks and place them again.")
    M_inv = np.linalg.inv(M)

    fx = math.log10 if x_log else float
    fy = math.log10 if y_log else float
    gx = (lambda t: 10 ** t) if x_log else float
    gy = (lambda t: 10 ** t) if y_log else float
    X0, X1, Y0, Y1 = fx(x0), fx(x1), fy(y0), fy(y1)

    def to_data(px, py):
        a, b = M_inv @ (np.array([px, py], dtype=float) - origin)
        return gx(X0 + a * (X1 - X0)), gy(Y0 + b * (Y1 - Y0))

    def to_pixel(x, y):
        a = (fx(x) - X0) / (X1 - X0)
        b = (fy(y) - Y0) / (Y1 - Y0)
        p = origin + a * u + b * v
        return float(p[0]), float(p[1])

    return to_data, to_pixel


def current_transform():
    ss = st.session_state
    s = ss.settings
    return make_transform(ss.cal_pts, s["x0"], s["x1"], s["y0"], s["y1"],
                          s["x_scale"] == "Log", s["y_scale"] == "Log")


def rotation_angle(cal_pts) -> float:
    (ox, oy), (xx, xy) = cal_pts[0], cal_pts[1]
    return math.degrees(math.atan2(-(xy - oy), xx - ox))


def view_box(img, zoom, pan_x, pan_y):
    W, H = img.size
    crop_w, crop_h = W / zoom, H / zoom
    left = round(pan_x * (W - crop_w))
    top = round(pan_y * (H - crop_h))
    return left, top, max(left + 1, round(left + crop_w)), max(top + 1, round(top + crop_h))


def base_view(img, zoom, pan_x, pan_y):
    ss = st.session_state
    key = (ss.image_id, zoom, pan_x, pan_y)
    if ss.get("base_key") != key:
        box = view_box(img, zoom, pan_x, pan_y)
        crop_w, crop_h = box[2] - box[0], box[3] - box[1]
        disp_w = int(max(200, min(MAX_DISPLAY_W, MAX_DISPLAY_H * crop_w / crop_h)))
        disp_h = max(1, round(crop_h * disp_w / crop_w))
        ss.base_img = img.crop(box).resize((disp_w, disp_h), Image.LANCZOS)
        ss.base_box = box
        ss.base_key = key
    return ss.base_img, ss.base_box


def draw_marks(base, box, cal_pts, datasets, active_id):
    view = base.copy()
    d = ImageDraw.Draw(view)
    sx = view.width / (box[2] - box[0])
    sy = view.height / (box[3] - box[1])
    font, small = get_font(15), get_font(12)

    def on_screen(p):
        return (p[0] - box[0]) * sx, (p[1] - box[1]) * sy

    for tag, p in zip(CAL_TAGS, cal_pts):
        x, y = on_screen(p)
        for w, colour in [(5, "white"), (2, "#7b1fa2")]:
            d.line([x - 11, y, x + 11, y], fill=colour, width=w)
            d.line([x, y - 11, x, y + 11], fill=colour, width=w)
        d.text((x + 9, y - 22), tag, fill="#7b1fa2", font=font,
               stroke_width=2, stroke_fill="white")

    for ds in sorted(datasets, key=lambda ds: ds["id"] == active_id):
        active = ds["id"] == active_id
        r = 6 if active else 4
        for i, p in enumerate(ds["points"], start=1):
            x, y = on_screen(p)
            d.ellipse([x - r, y - r, x + r, y + r], fill=ds["color"],
                      outline="white", width=2)
            if active:
                d.text((x + 7, y - 17), str(i), fill=ds["color"], font=small,
                       stroke_width=2, stroke_fill="white")
    return view


def overview(img, zoom, pan_x, pan_y):
    ss = st.session_state
    if ss.get("thumb_id") != ss.image_id:
        thumb = img.copy()
        thumb.thumbnail((THUMB_W, THUMB_W * 2))
        ss.thumb, ss.thumb_id = thumb, ss.image_id
    thumb = ss.thumb.copy()
    s = thumb.width / img.width
    l, t, r, b = view_box(img, zoom, pan_x, pan_y)
    ImageDraw.Draw(thumb).rectangle([l * s, t * s, r * s, b * s], outline="#e6194b", width=2)
    return thumb


def build_csv(datasets, to_data, x_label, y_label) -> str:
    columns = []
    for ds in datasets:
        if ds["points"]:
            columns.append((ds["name"], [to_data(*p) for p in ds["points"]]))

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([cell for name, _ in columns for cell in (name, "")])
    writer.writerow([cell for _ in columns for cell in (x_label, y_label)])
    longest = max((len(vals) for _, vals in columns), default=0)
    for i in range(longest):
        row = []
        for _, vals in columns:
            row += [f"{vals[i][0]:.6g}", f"{vals[i][1]:.6g}"] if i < len(vals) else ["", ""]
        writer.writerow(row)
    return out.getvalue()


def new_dataset():
    ss = st.session_state
    ds = {"id": ss.next_id, "name": f"Dataset {ss.next_id}",
          "color": COLORS[(ss.next_id - 1) % len(COLORS)], "points": []}
    ss.next_id += 1
    ss.datasets.append(ds)
    ss.active_id = ds["id"]
    ss.version += 1


def reset_work():
    ss = st.session_state
    ss.cal_pts = []
    ss.datasets = []
    ss.next_id = 1
    ss.version = ss.get("version", 0) + 1
    new_dataset()
    ss.mode = MODE_CAL


def apply_settings(values):
    st.session_state.settings = dict(values)


def init_state():
    ss = st.session_state
    if "datasets" not in ss:
        ss.image_id = None
        ss.img_format = "PNG"
        ss.settings = dict(DEFAULT_SETTINGS)
        reset_work()


def cb_setting(name, then=None, *args):
    ss = st.session_state
    ss.settings[name] = ss["w_" + name]
    if then:
        then(*args)


def setting(widget, label, name, then=None, args=(), **kwargs):
    ss = st.session_state
    key = "w_" + name
    ss[key] = ss.settings[name]
    widget(label, key=key, on_change=cb_setting, args=(name, then, *args), **kwargs)
    return ss.settings[name]


def cb_rename(ds_id):
    ss = st.session_state
    for ds in ss.datasets:
        if ds["id"] == ds_id:
            ds["name"] = ss[f"name_{ds_id}"] or ds["name"]


def active_dataset():
    ss = st.session_state
    for ds in ss.datasets:
        if ds["id"] == ss.active_id:
            return ds
    return ss.datasets[0]


def cb_click():
    ss = st.session_state
    click, geom = ss.get("clicker"), ss.get("view_geom")
    if not click or not geom:
        return
    (left, top, right, bottom), disp_w, disp_h = geom
    shown_w = click.get("width") or disp_w
    shown_h = click.get("height") or disp_h
    px = left + click["x"] * (right - left) / shown_w
    py = top + click["y"] * (bottom - top) / shown_h

    if ss.mode == MODE_CAL:
        if len(ss.cal_pts) < 3:
            ss.cal_pts.append((px, py))
        else:
            ss.flash = "All three axis points are set. Press Start picking points."
    elif len(ss.cal_pts) < 3:
        ss.flash = "Set up the axes first (step 1)."
    else:
        active_dataset()["points"].append((px, py))
        ss.version += 1


def cb_nav():
    ss = st.session_state
    click, img = ss.get("nav"), ss.get("image")
    if not click or img is None:
        return
    W, H = img.size
    zoom = ss.get("zoom", 1.0)
    cx = click["x"] / (click.get("width") or ss.thumb.width) * W
    cy = click["y"] / (click.get("height") or ss.thumb.height) * H
    for key, c, full in (("pan_x", cx, W), ("pan_y", cy, H)):
        crop = full / zoom
        ss[key] = round(min(1.0, max(0.0, (c - crop / 2) / (full - crop))), 2)


def cb_scale(axis):
    s = st.session_state.settings
    if s[f"{axis}_scale"] == "Log":
        lo, hi = f"{axis}0", f"{axis}1"
        if s[lo] <= 0:
            s[lo] = 1.0
        if s[hi] <= 0 or s[hi] == s[lo]:
            s[hi] = s[lo] * 10


def cb_select():
    st.session_state.active_id = st.session_state.active_select


def cb_start():
    st.session_state.mode = MODE_PICK


def cb_undo_cal():
    if st.session_state.cal_pts:
        st.session_state.cal_pts.pop()


def cb_reset_cal():
    st.session_state.cal_pts = []


def cb_undo():
    ds = active_dataset()
    if ds["points"]:
        ds["points"].pop()
        st.session_state.version += 1


def cb_sort():
    try:
        to_data, _ = current_transform()
    except ValueError:
        return
    ds = active_dataset()
    ds["points"].sort(key=lambda p: to_data(*p)[0])
    st.session_state.version += 1


def cb_clear():
    active_dataset()["points"] = []
    st.session_state.version += 1


def cb_delete():
    ss = st.session_state
    ss.pop(f"name_{ss.active_id}", None)
    ss.datasets = [ds for ds in ss.datasets if ds["id"] != ss.active_id]
    if not ss.datasets:
        new_dataset()
    ss.active_id = ss.datasets[-1]["id"]
    ss.version += 1


def axes_panel(calibrated, cal_error, is_sample):
    ss = st.session_state
    st.subheader("Set up the axes")
    c1, c2 = st.columns(2)
    x_label = setting(c1.text_input, "x-axis name", "x_label") or "x"
    y_label = setting(c2.text_input, "y-axis name", "y_label") or "y"
    setting(c1.selectbox, "x scale", "x_scale", cb_scale, ("x",), options=["Linear", "Log"])
    setting(c2.selectbox, "y scale", "y_scale", cb_scale, ("y",), options=["Linear", "Log"])

    n = len(ss.cal_pts)
    fields = [[("x0", x_label), ("y0", y_label)], [("x1", x_label)], [("y1", y_label)]]
    for i, tag in enumerate(CAL_TAGS):
        state = "✓ clicked" if i < n else ("**← click it now**" if i == n else "not clicked yet")
        st.markdown(f"**{tag}** · {CAL_NAMES[i]} &nbsp; {state}")
        cols = st.columns(2)
        for col, (name, axis_name) in zip(cols, fields[i]):
            setting(col.number_input, f"{axis_name} at {tag}", name, format="%g")

    if is_sample:
        st.caption("Sample graph: the values are filled in. Click the corner where the "
                   "axes meet (O), the 700 tick (X) and the 1.0 tick (Y).")
    b1, b2 = st.columns(2)
    b1.button("Undo last click", on_click=cb_undo_cal, disabled=n == 0, width="stretch")
    b2.button("Reset clicks", on_click=cb_reset_cal, disabled=n == 0, width="stretch")
    if cal_error and calibrated:
        st.error(cal_error)
    st.button("Start picking points →", type="primary", on_click=cb_start,
              disabled=not calibrated or bool(cal_error), width="stretch")


def datasets_panel(to_data, to_pixel):
    ss = st.session_state
    s = ss.settings
    x_label, y_label = s["x_label"] or "x", s["y_label"] or "y"
    x_log, y_log = s["x_scale"] == "Log", s["y_scale"] == "Log"

    st.subheader("Datasets")
    names = {ds["id"]: ds["name"] for ds in ss.datasets}
    ss.active_select = ss.active_id
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.selectbox("Active dataset (clicks go here)", list(names), on_change=cb_select,
                 format_func=lambda i: names[i], key="active_select")
    c2.button("＋ New dataset", on_click=new_dataset, width="stretch")
    active = active_dataset()
    ss[f"name_{active['id']}"] = active["name"]
    st.text_input("Dataset name", key=f"name_{active['id']}",
                  on_change=cb_rename, args=(active["id"],))

    n = len(active["points"])
    b1, b2 = st.columns(2)
    b1.button("Undo last point", on_click=cb_undo, disabled=n == 0, width="stretch")
    b2.button("Sort by x", on_click=cb_sort, disabled=n < 2, width="stretch")
    with b1.popover("Clear points", disabled=n == 0, width="stretch"):
        st.write(f"Remove all {n} points from **{active['name']}**?")
        st.button("Yes, clear them", on_click=cb_clear, type="primary")
    with b2.popover("Delete dataset", width="stretch"):
        st.write(f"Delete **{active['name']}** and its points?")
        st.button("Yes, delete it", on_click=cb_delete, type="primary")

    values = [to_data(*p) for p in active["points"]]
    df = pd.DataFrame({"#": pd.Series(range(1, n + 1), dtype="Int64"),
                       "x": pd.Series([v[0] for v in values], dtype=float),
                       "y": pd.Series([v[1] for v in values], dtype=float)})
    edited = st.data_editor(
        df, num_rows="dynamic", width="stretch", hide_index=True,
        key=f"editor_{active['id']}_{ss.version}",
        column_config={"#": st.column_config.NumberColumn("#", disabled=True, width=40),
                       "x": st.column_config.NumberColumn(x_label, format="%.5g", width=135),
                       "y": st.column_config.NumberColumn(y_label, format="%.5g", width=135)})
    st.caption("Edit a number, or add and delete rows, and the marks on the graph move to match.")

    rows = edited[["x", "y"]].dropna(how="all")
    if rows.isna().any().any():
        st.caption("Fill in both x and y to place the new point.")
    else:
        new = rows.to_numpy(dtype=float).reshape(-1, 2)
        old = df[["x", "y"]].to_numpy(dtype=float).reshape(-1, 2)
        if new.shape != old.shape or not np.allclose(new, old, rtol=1e-9, atol=0):
            if (x_log and (new[:, 0] <= 0).any()) or (y_log and (new[:, 1] <= 0).any()):
                st.error("Values on a log axis must be greater than zero.")
            else:
                active["points"] = [to_pixel(x, y) for x, y in new]
                ss.version += 1
                st.rerun()

    st.subheader("Download")
    csv_text = build_csv(ss.datasets, to_data, x_label, y_label)
    filled = sum(1 for ds in ss.datasets if ds["points"])
    st.download_button(f"Download CSV ({filled} dataset{'s' * (filled != 1)})", csv_text,
                       file_name="digitized_data.csv", mime="text/csv", type="primary",
                       on_click="ignore", disabled=filled == 0, width="stretch")
    with st.expander("Preview the CSV file"):
        st.code(csv_text or "(no points yet)", language=None)


def main():
    st.set_page_config(page_title="Plot Digitizer", page_icon="📈", layout="wide")
    init_state()
    ss = st.session_state

    st.title("Plot Digitizer")
    st.caption("Turn a picture of a graph back into numbers: set up the axes, "
               "click the points, download the CSV.")
    with st.expander("How to use this app"):
        st.markdown(
            "1. **Upload** a graph in the sidebar (png, jpg, gif or tiff), or tick "
            "*Use the sample graph*.\n"
            "2. **Set up axes:** name each axis and choose Linear or Log. Click the origin "
            "**O**, a point on the x-axis **X** and a point on the y-axis **Y**, and type the "
            "real value at each. A tilted scan is fine: the three clicks correct for it.\n"
            "3. **Pick data points:** click each point on the graph. Use **＋ New dataset** "
            "for each extra line or marker type. The numbers on the graph match the table rows.\n"
            "4. **Fix mistakes** with *Undo last point*, or type a new value in the table.\n"
            "5. **Zoom** in the sidebar for precise clicks, then **Download CSV**.")

    with st.sidebar:
        st.header("Image")
        file = st.file_uploader("Upload a graph image",
                                type=["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp"])
        use_sample = st.checkbox("Use the sample graph instead", value=False,
                                 disabled=file is not None)

    if file is not None:
        image_id = f"{file.name}-{file.size}"
    elif use_sample:
        image_id = "sample"
    else:
        image_id = None

    if image_id != ss.image_id:
        ss.image_id = image_id
        ss.image = None
        if image_id is not None:
            try:
                ss.image = make_sample_image() if image_id == "sample" else load_image(file)
                ss.img_format = choose_format(ss.image)
            except Exception as err:
                st.error(f"That file could not be read as an image ({err}).")
        reset_work()
        apply_settings(SAMPLE_SETTINGS if image_id == "sample" else DEFAULT_SETTINGS)
        for key in ("zoom", "pan_x", "pan_y"):
            ss.pop(key, None)

    img = ss.get("image")
    if img is None:
        st.info("Upload an image in the sidebar, or tick **Use the sample graph** to try the app.")
        return

    with st.sidebar:
        st.header("Zoom")
        zoom = st.slider("Zoom", 1.0, 8.0, 1.0, 0.5, key="zoom",
                         help="Zoom in to place clicks more precisely.")
        pan_x = pan_y = 0.5
        if zoom > 1:
            for key in ("pan_x", "pan_y"):
                if key not in ss:
                    ss[key] = 0.5
            streamlit_image_coordinates(overview(img, zoom, ss.pan_x, ss.pan_y), key="nav",
                                        on_click=cb_nav, width="stretch", cursor="pointer")
            st.caption("Click this overview to move the zoomed view there.")
            pan_x = st.slider("Move left ↔ right", 0.0, 1.0, step=0.01, key="pan_x")
            pan_y = st.slider("Move up ↕ down", 0.0, 1.0, step=0.01, key="pan_y")

    if ss.active_id not in [ds["id"] for ds in ss.datasets]:
        ss.active_id = ss.datasets[0]["id"]
    calibrated = len(ss.cal_pts) == 3
    to_data = to_pixel = None
    cal_error = None
    if calibrated:
        try:
            to_data, to_pixel = current_transform()
        except ValueError as err:
            cal_error = str(err)

    mode = st.radio("Step", [MODE_CAL, MODE_PICK], key="mode", horizontal=True,
                    label_visibility="collapsed")
    if ss.get("flash"):
        st.toast(ss.pop("flash"))

    active = active_dataset()
    if mode == MODE_CAL:
        if not calibrated:
            n = len(ss.cal_pts)
            st.info(f"**Axis click {n + 1} of 3:** click the **{CAL_NAMES[n]}** "
                    f"({CAL_TAGS[n]}) on the graph, then type its value on the right.")
        elif cal_error:
            st.warning("Fix the axis values on the right to continue.")
        else:
            st.success("All three axis points are set. Check the values on the right, "
                       "then press **Start picking points**.")
    elif not calibrated or cal_error:
        st.warning("Set up the axes first (step 1).")
    else:
        st.info(f"Click the graph to add points to **{active['name']}**.")

    col_img, col_side = st.columns([3, 2], gap="large")

    with col_img:
        base, box = base_view(img, zoom, pan_x, pan_y)
        view = draw_marks(base, box, ss.cal_pts, ss.datasets, active["id"])
        ss.view_geom = (box, view.width, view.height)
        streamlit_image_coordinates(view, key="clicker", on_click=cb_click, width="stretch",
                                    cursor="crosshair", image_format=ss.img_format,
                                    png_compression_level=6, jpeg_quality=90)

        legend = " &nbsp;&nbsp; ".join(
            f"<span style='color:{ds['color']};font-size:1.2em'>●</span> "
            f"<span style='font-weight:{700 if ds['id'] == active['id'] else 400}'>"
            f"{html.escape(ds['name'])}</span> ({len(ds['points'])})"
            for ds in ss.datasets)
        st.markdown(legend, unsafe_allow_html=True)
        if calibrated and not cal_error:
            st.caption(f"Detected tilt of the scan: {rotation_angle(ss.cal_pts):+.2f}°. "
                       "The calibration corrects for it.")

    with col_side:
        if mode == MODE_CAL:
            axes_panel(calibrated, cal_error, ss.image_id == "sample")
        elif to_data is None:
            st.subheader("Datasets")
            st.write("The datasets and table appear once the axes are set up.")
        else:
            datasets_panel(to_data, to_pixel)


if __name__ == "__main__":
    main()