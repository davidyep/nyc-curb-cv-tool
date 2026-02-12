from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="NYC Curb Utilization Dashboard", layout="wide")
st.title("NYC Curb Utilization Dashboard")

# ── Session state defaults ──────────────────────────────────────────────────
if "zones" not in st.session_state:
    st.session_state.zones = []
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

tab1, tab2, tab3 = st.tabs([
    "1. Upload & Define Zones",
    "2. Analysis Results",
    "3. Historical Analytics",
])


# ── Helpers ─────────────────────────────────────────────────────────────────

def _parse_canvas_shapes(
    objects: list[dict], zone_type: str, img_width: int, img_height: int,
) -> list[dict]:
    """Convert streamlit-drawable-canvas JSON objects to ZoneDefinition dicts."""
    zones: list[dict] = []
    for obj in objects:
        polygon: list[tuple[float, float]] = []

        if obj.get("type") == "path":
            # Polygon drawn with the polygon tool — path commands
            for cmd in obj.get("path", []):
                if len(cmd) >= 3 and cmd[0] in ("M", "L"):
                    polygon.append((float(cmd[1]), float(cmd[2])))
        elif obj.get("type") == "rect":
            # Rectangle
            left = obj["left"]
            top = obj["top"]
            w = obj["width"] * obj.get("scaleX", 1)
            h = obj["height"] * obj.get("scaleY", 1)
            polygon = [
                (left, top),
                (left + w, top),
                (left + w, top + h),
                (left, top + h),
            ]

        if len(polygon) >= 3:
            zones.append({
                "zone_id": f"zone_{uuid.uuid4().hex[:6]}",
                "zone_type": zone_type,
                "polygon": polygon,
                "label": f"{zone_type.replace('_', ' ').title()}",
            })
    return zones


def _run_analysis(
    uploaded_file,
    zones: list[dict],
    borough: str,
    segment_id: str,
    camera_id: str,
) -> None:
    """POST to /analyze/image and store result in session state."""
    request_payload = {
        "frame": {
            "frame_id": f"frame_{uuid.uuid4().hex[:8]}",
            "camera_id": camera_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "borough": borough,
            "segment_id": segment_id,
        },
        "zones": zones,
    }

    uploaded_file.seek(0)
    try:
        resp = requests.post(
            f"{API_BASE}/analyze/image",
            files={"image": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)},
            data={"request_json": json.dumps(request_payload)},
            timeout=120,
        )
        resp.raise_for_status()
        st.session_state.analysis_result = resp.json()
        st.success("Analysis complete! Switch to the **Analysis Results** tab.")
    except requests.ConnectionError:
        st.error(
            "Cannot reach the API server. Start it with: "
            "`uvicorn app.main:app --reload`"
        )
    except requests.HTTPError as exc:
        st.error(f"API error: {exc.response.status_code} — {exc.response.text[:300]}")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 1: Upload & Define Zones
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Upload Street Image & Define Infrastructure Zones")
    st.caption(
        "Upload a photo of an NYC street, draw polygons to define zones "
        "(parking, bus lane, etc.), then run the CV analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload a street image", type=["jpg", "jpeg", "png"], key="img_upload",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_w, img_h = image.size

        # Scale canvas to fit in dashboard (max 900px wide)
        scale = min(900 / img_w, 1.0)
        canvas_w = int(img_w * scale)
        canvas_h = int(img_h * scale)

        # ── Sidebar controls ────────────────────────────────────────────
        st.sidebar.header("Zone Drawing")
        zone_type = st.sidebar.selectbox("Zone Type", [
            "parking", "no_parking", "bus_lane", "bike_lane",
            "loading_zone", "fire_hydrant", "double_parking", "travel_lane",
        ])
        drawing_mode = st.sidebar.radio("Draw Mode", ["polygon", "rect"])
        stroke_color = st.sidebar.color_picker("Stroke color", "#FF0000")

        st.sidebar.header("Frame Context")
        borough = st.sidebar.selectbox("Borough", [
            "manhattan", "brooklyn", "queens", "bronx", "staten_island",
        ])
        segment_id = st.sidebar.text_input("Segment ID", "seg_0001")
        camera_id = st.sidebar.text_input("Camera ID", "cam_01")

        # ── Canvas ──────────────────────────────────────────────────────
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color=stroke_color,
            background_image=image,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="polygon" if drawing_mode == "polygon" else "rect",
            key="canvas",
        )

        # ── Parse drawn shapes ──────────────────────────────────────────
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                new_zones = _parse_canvas_shapes(objects, zone_type, img_w, img_h)
                # Rescale polygon coords back to original image size
                if scale != 1.0:
                    for z in new_zones:
                        z["polygon"] = [
                            (x / scale, y / scale) for x, y in z["polygon"]
                        ]
                st.session_state.zones = new_zones

        # ── Zone summary ────────────────────────────────────────────────
        zones = st.session_state.zones
        if zones:
            st.subheader(f"{len(zones)} zone(s) defined")
            for z in zones:
                st.caption(
                    f"**{z['zone_id']}** — {z['zone_type'].replace('_', ' ').title()} "
                    f"({len(z['polygon'])} vertices)"
                )
        else:
            st.info("Draw zones on the image above, then click **Run Analysis**.")

        # ── Run button ──────────────────────────────────────────────────
        if st.button("Run Analysis", type="primary", use_container_width=True):
            if not zones:
                st.warning("Please draw at least one zone before running analysis.")
            else:
                with st.spinner("Running CV analysis..."):
                    _run_analysis(uploaded_file, zones, borough, segment_id, camera_id)
    else:
        st.info("Upload a street image to get started.")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2: Analysis Results
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Analysis Results")

    result = st.session_state.analysis_result
    if result is None:
        st.info("Upload an image and run analysis in **Tab 1** first.")
    else:
        # ── Annotated image ─────────────────────────────────────────────
        if result.get("annotated_image_b64"):
            st.subheader("Annotated Image")
            ann_bytes = base64.b64decode(result["annotated_image_b64"])
            st.image(ann_bytes, use_container_width=True)

        # ── Summary metrics ─────────────────────────────────────────────
        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vehicles Detected", len(result.get("detections", [])))
        c2.metric("Occupancy Rate", f"{result.get('occupancy_rate', 0):.1%}")
        c3.metric("Likely Illegal", result.get("summary", {}).get("likely_illegal", 0))
        c4.metric("Legal", result.get("summary", {}).get("legal", 0))

        # ── Detection details table ─────────────────────────────────────
        st.subheader("Detection Details")
        det_rows = []
        for za in result.get("zone_assignments", []):
            det = za.get("detection", {})
            zone = za.get("zone")
            det_rows.append({
                "ID": det.get("detection_id", ""),
                "Type": za.get("vehicle_type", ""),
                "Zone": zone["zone_type"].replace("_", " ").title() if zone else "None",
                "Confidence": f"{det.get('confidence', 0):.2f}",
            })
        if det_rows:
            st.dataframe(pd.DataFrame(det_rows), use_container_width=True)
        else:
            st.caption("No vehicles detected.")

        # ── Legality decisions table ────────────────────────────────────
        st.subheader("Legality Decisions")
        dec_rows = []
        for d in result.get("decisions", []):
            dec_rows.append({
                "Track ID": d["track_id"],
                "Status": d["status"],
                "Reason Codes": ", ".join(d.get("reason_codes", [])),
                "Confidence": f"{d.get('confidence', 0):.2f}",
            })
        if dec_rows:
            st.dataframe(pd.DataFrame(dec_rows), use_container_width=True)

        # ── Violation breakdown chart ───────────────────────────────────
        reason_counts: dict[str, int] = {}
        for d in result.get("decisions", []):
            for rc in d.get("reason_codes", []):
                reason_counts[rc] = reason_counts.get(rc, 0) + 1
        if reason_counts:
            st.subheader("Violation Breakdown")
            fig = px.bar(
                x=list(reason_counts.keys()),
                y=list(reason_counts.values()),
                labels={"x": "Violation Type", "y": "Count"},
                color=list(reason_counts.keys()),
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── Vehicle type distribution ───────────────────────────────────
        type_counts: dict[str, int] = {}
        for za in result.get("zone_assignments", []):
            vt = za.get("vehicle_type", "other")
            type_counts[vt] = type_counts.get(vt, 0) + 1
        if type_counts:
            st.subheader("Vehicle Type Distribution")
            fig2 = px.pie(
                names=list(type_counts.keys()),
                values=list(type_counts.values()),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3: Historical Analytics
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Historical Analytics")
    st.caption("Trend data from previous analysis sessions.")

    try:
        df = pd.read_csv("data/sample_results.csv")

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Occupancy", f"{df['occupancy_rate'].mean():.1%}")
        c2.metric("Likely Illegal Events", int(df["likely_illegal"].sum()))
        c3.metric("Legal Events", int(df["legal"].sum()))

        st.subheader("Segment Performance")
        st.dataframe(df, use_container_width=True)

        st.subheader("Occupancy Trend")
        fig3 = px.line(df, x="timestamp", y="occupancy_rate", title="Occupancy Over Time")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Legality Breakdown")
        fig4 = px.bar(
            df, x="timestamp", y=["legal", "likely_illegal", "uncertain"],
            barmode="group", title="Legality Status Over Time",
        )
        st.plotly_chart(fig4, use_container_width=True)
    except FileNotFoundError:
        st.warning("No historical data found. Run analyses to build history.")
