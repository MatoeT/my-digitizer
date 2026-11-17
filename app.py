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
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

COLORS = ["#e6194b", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
          "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#000000"]
MAX_DISPLAY_W = 850
MAX_DISPLAY_H = 650
CAL_NAMES = ["origin", "x-axis point", "y-axis point"]
CAL_TAGS = ["O", "X", "Y"]
MODE_CAL = "1 · Calibrate axes"
MODE_PICK = "2 · Pick data points"


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
            d.ellipse(box, fill=colour) if shape == "circle" else d.rectangle(box, fill=colour)

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
                         "Reset the calibration and click them again.")
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


def rotation_angle(cal_pts) -> float:
    (ox, oy), (xx, xy) = cal_pts[0], cal_pts[1]
    return math.degrees(math.atan2(-(xy - oy), xx - ox))


def render_view(img, cal_pts, datasets, active_id, zoom, pan_x, pan_y):
    W, H = img.size
    crop_w, crop_h = W / zoom, H / zoom
    left = pan_x * (W - crop_w)
    top = pan_y * (H - crop_h)

    disp_w = int(max(200, min(MAX_DISPLAY_W, MAX_DISPLAY_H * crop_w / crop_h)))
    scale = disp_w / crop_w
    disp_h = max(1, round(crop_h * scale))

    view = img.crop((round(left), round(top), round(left + crop_w), round(top + crop_h)))
    view = view.resize((disp_w, disp_h), Image.LANCZOS)
    d = ImageDraw.Draw(view)
    font = get_font(15)

    def on_screen(p):
        return (p[0] - left) * scale, (p[1] - top) * scale

    for tag, p in zip(CAL_TAGS, cal_pts):
        x, y = on_screen(p)
        for w, colour in [(5, "white"), (2, "#7b1fa2")]:
            d.line([x - 11, y, x + 11, y], fill=colour, width=w)
            d.line([x, y - 11, x, y + 11], fill=colour, width=w)
        d.text((x + 9, y - 22), tag, fill="#7b1fa2", font=font,
               stroke_width=2, stroke_fill="white")

    for ds in datasets:
        r = 6 if ds["id"] == active_id else 4
        for p in ds["points"]:
            x, y = on_screen(p)
            d.ellipse([x - r, y - r, x + r, y + r], fill=ds["color"],
                      outline="white", width=2)

    return view, left, top, scale


def build_csv(datasets, to_data, x_label, y_label) -> str:
    columns = []
    for ds in datasets:
        values = [to_data(*p) for p in ds["points"]]
        columns.append((ds["name"], values))

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
    ss.pending_mode = MODE_CAL


def init_state():
    ss = st.session_state
    if "datasets" not in ss:
        ss.last_click = None
        ss.image_id = None
        reset_work()


def active_dataset():
    ss = st.session_state
    for ds in ss.datasets:
        if ds["id"] == ss.active_id:
            return ds
    return ss.datasets[0]


def cb_undo():
    ds = active_dataset()
    if ds["points"]:
        ds["points"].pop()
        st.session_state.version += 1


def cb_clear():
    active_dataset()["points"] = []
    st.session_state.version += 1


def cb_delete():
    ss = st.session_state
    ss.datasets = [ds for ds in ss.datasets if ds["id"] != ss.active_id]
    if not ss.datasets:
        new_dataset()
    ss.active_id = ss.datasets[-1]["id"]
    ss.version += 1


def cb_reset_cal():
    st.session_state.cal_pts = []
    st.session_state.mode = MODE_CAL


def cb_undo_cal():
    if st.session_state.cal_pts:
        st.session_state.cal_pts.pop()


def main():
    st.set_page_config(page_title="Plot Digitizer", page_icon="📈", layout="wide")
    init_state()
    ss = st.session_state

    if "pending_mode" in ss:
        ss.mode = ss.pop("pending_mode")

    st.title("Plot Digitizer")
    st.caption("Turn a picture of a graph back into numbers: calibrate the axes, "
               "click the points, download the CSV.")

    with st.sidebar:
        st.header("1 · Image")
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
            except Exception as err:
                st.error(f"That file could not be read as an image ({err}).")
        reset_work()
        ss.mode = ss.pop("pending_mode")

    img = ss.get("image")
    if img is None:
        st.info("Upload an image in the sidebar, or tick **Use the sample graph** to try the app.")
        return

    with st.sidebar:
        st.header("2 · Axes")
        c1, c2 = st.columns(2)
        x_label = c1.text_input("x-axis name", "x")
        y_label = c2.text_input("y-axis name", "y")
        x_log = c1.selectbox("x scale", ["Linear", "Log"]) == "Log"
        y_log = c2.selectbox("y scale", ["Linear", "Log"]) == "Log"
        st.caption("Real values at your three calibration clicks:")
        x0 = c1.number_input("x at origin (O)", value=0.0, format="%g")
        y0 = c2.number_input("y at origin (O)", value=0.0, format="%g")
        x1 = c1.number_input("x at x-axis point (X)", value=1.0, format="%g")
        y1 = c2.number_input("y at y-axis point (Y)", value=1.0, format="%g")
        if image_id == "sample":
            st.caption("Sample graph: use the axis corner as O with x = 400, y = 0, "
                       "the 700 tick as X, and the 1.0 tick as Y.")
        b1, b2 = st.columns(2)
        b1.button("Undo calibration click", on_click=cb_undo_cal,
                  disabled=not ss.cal_pts, width="stretch")
        b2.button("Reset calibration", on_click=cb_reset_cal,
                  disabled=not ss.cal_pts, width="stretch")

        st.header("3 · Datasets")
        for ds in ss.datasets:
            ds["name"] = ss.get(f"name_{ds['id']}", ds["name"])
        ids = [ds["id"] for ds in ss.datasets]
        names = {ds["id"]: ds["name"] for ds in ss.datasets}
        if ss.get("active_id") not in ids:
            ss.active_id = ids[0]
        st.selectbox("Active dataset (clicks go here)", ids,
                     format_func=lambda i: names[i], key="active_id")
        active = active_dataset()
        st.text_input("Dataset name", value=active["name"], key=f"name_{active['id']}")
        st.markdown(f"Colour: <span style='color:{active['color']};font-size:1.3em'>●</span> "
                    f"&nbsp; {len(active['points'])} point(s)", unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        d1.button("Undo last point", on_click=cb_undo,
                  disabled=not active["points"], width="stretch")
        d2.button("Clear points", on_click=cb_clear,
                  disabled=not active["points"], width="stretch")
        d1.button("Add dataset", on_click=new_dataset, width="stretch")
        d2.button("Delete dataset", on_click=cb_delete, width="stretch")

        st.header("4 · Zoom")
        zoom = st.slider("Zoom", 1.0, 8.0, 1.0, 0.5)
        pan_x = pan_y = 0.5
        if zoom > 1:
            pan_x = st.slider("Move left ↔ right", 0.0, 1.0, 0.5, 0.01)
            pan_y = st.slider("Move up ↕ down", 0.0, 1.0, 0.5, 0.01)

    calibrated = len(ss.cal_pts) == 3
    to_data = to_pixel = None
    cal_error = None
    if calibrated:
        try:
            to_data, to_pixel = make_transform(ss.cal_pts, x0, x1, y0, y1, x_log, y_log)
        except ValueError as err:
            cal_error = str(err)

    mode = st.radio("Step", [MODE_CAL, MODE_PICK], key="mode", horizontal=True,
                    label_visibility="collapsed")

    if mode == MODE_CAL:
        if not calibrated:
            n = len(ss.cal_pts)
            hints = ["where the two axes meet (or any spot whose x and y you know)",
                     "a spot far along the x-axis whose x value you know",
                     "a spot far up the y-axis whose y value you know"]
            st.info(f"**Calibration click {n + 1} of 3:** click the **{CAL_NAMES[n]}** "
                    f"({CAL_TAGS[n]}), {hints[n]}.")
        else:
            st.success("All three calibration points are set. Check the values in the "
                       "sidebar, then switch to **Pick data points**.")
    else:
        if not calibrated:
            st.warning("Calibrate the axes first (step 1).")
        else:
            st.info(f"Click on the graph to add points to **{active['name']}**.")
    if cal_error:
        st.error(cal_error)
    elif calibrated:
        st.caption(f"Detected rotation of the scan: {rotation_angle(ss.cal_pts):+.2f}°. "
                   "The calibration corrects for it.")

    col_img, col_table = st.columns([3, 2], gap="large")

    with col_img:
        view, left, top, scale = render_view(img, ss.cal_pts, ss.datasets,
                                             active["id"], zoom, pan_x, pan_y)
        click = streamlit_image_coordinates(view, key="clicker", cursor="crosshair")

        if click is not None and click != ss.last_click:
            ss.last_click = click
            shown_w = click.get("width") or view.width
            px = left + click["x"] * (view.width / shown_w) / scale
            py = top + click["y"] * (view.width / shown_w) / scale
            if mode == MODE_CAL and not calibrated:
                ss.cal_pts.append((px, py))
                if len(ss.cal_pts) == 3:
                    ss.pending_mode = MODE_PICK
                st.rerun()
            elif mode == MODE_PICK and calibrated:
                active["points"].append((px, py))
                ss.version += 1
                st.rerun()

    with col_table:
        st.subheader(active["name"])
        if to_data is None:
            st.write("The table appears once the axes are calibrated.")
        else:
            values = [to_data(*p) for p in active["points"]]
            df = pd.DataFrame(values, columns=["x", "y"], dtype=float)
            edited = st.data_editor(
                df, num_rows="dynamic", width="stretch", hide_index=True,
                key=f"editor_{active['id']}_{ss.version}",
                column_config={"x": st.column_config.NumberColumn(x_label or "x", format="%.5g"),
                               "y": st.column_config.NumberColumn(y_label or "y", format="%.5g")})
            st.caption("Edit a number, or add and delete rows, and the marks on the "
                       "graph move to match.")

            rows = edited.dropna(how="all")
            if rows.isna().any().any():
                st.caption("Fill in both x and y to place the new point.")
            else:
                new = rows.to_numpy(dtype=float).reshape(-1, 2)
                old = df.to_numpy(dtype=float).reshape(-1, 2)
                changed = new.shape != old.shape or not np.allclose(new, old, rtol=1e-9, atol=0)
                if changed:
                    if (x_log and (new[:, 0] <= 0).any()) or (y_log and (new[:, 1] <= 0).any()):
                        st.error("Values on a log axis must be greater than zero.")
                    else:
                        active["points"] = [to_pixel(x, y) for x, y in new]
                        ss.version += 1
                        st.rerun()

    st.divider()
    st.subheader("Download")
    if to_data is None:
        st.write("Calibrate the axes to enable the download.")
    else:
        csv_text = build_csv(ss.datasets, to_data, x_label, y_label)
        total = sum(len(ds["points"]) for ds in ss.datasets)
        st.download_button("Download CSV", csv_text, file_name="digitized_data.csv",
                           mime="text/csv", disabled=total == 0)
        with st.expander("Preview the CSV file"):
            st.code(csv_text, language=None)


if __name__ == "__main__":
    main()