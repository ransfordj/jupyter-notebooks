"""Data Request Review Dashboard — extracted logic for Voila landing-page pattern.

All blocking I/O and widget-building functions live here so the notebook
cells stay thin (imports, skeleton, launch).  The ``orchestrate()`` coroutine
is the async pipeline that populates placeholder widgets progressively.
"""

import asyncio
import base64
import html as html_module
import io
import logging
import os
import ssl
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import pandas as pd
import requests
import xnat
from ipywidgets import (
    HTML,
    Button,
    Dropdown,
    HBox,
    Layout,
    Text,
    VBox,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

warnings.filterwarnings("ignore")
logging.getLogger("xnat").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# CSS shared by all table widgets
# ---------------------------------------------------------------------------
BUTTON_STYLE_CSS = """
<style>
.header-sort-button button,
.header-sort-button .widget-button,
.header-sort-button .jupyter-button,
.header-sort-button button .bp3-button-text,
.header-sort-button button span,
.header-sort-button button p {
    border-radius: 0 !important;
    font-weight: bold !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-bottom: 2px solid #555 !important;
    background-color: #333 !important;
    color: #fff !important;
}
.header-sort-button button:hover {
    background-color: #444 !important;
}
</style>
"""

# SOP classes excluded by site anonymization script
EXCLUDED_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.7": "Secondary Capture Image",
    "1.2.840.10008.5.1.4.1.1.7.1": "Multi-frame Single Bit SC Image",
    "1.2.840.10008.5.1.4.1.1.7.2": "Multi-frame Grayscale Byte SC Image",
    "1.2.840.10008.5.1.4.1.1.7.3": "Multi-frame Grayscale Word SC Image",
    "1.2.840.10008.5.1.4.1.1.7.4": "Multi-frame True Color SC Image",
    "1.2.840.10008.5.1.4.1.1.104.1": "Encapsulated PDF",
    "1.2.840.10008.5.1.4.1.1.104.2": "Encapsulated CDA",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_msg(text, color="#856404", icon=""):
    """Return an HTML snippet for the status banner."""
    icon_html = f"{icon} " if icon else ""
    return (
        f'<span style="color: {color};">{icon_html}'
        f"<strong>{text}</strong></span>"
    )


def _apply_tls_patch():
    """Monkey-patch ``requests.Session`` for legacy SSL servers."""

    class _TLSAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            ctx = create_urllib3_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
            try:
                ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
            except AttributeError:
                pass
            kwargs["ssl_context"] = ctx
            return super().init_poolmanager(*args, **kwargs)

    _orig_init = requests.Session.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.mount("https://", _TLSAdapter())
        self.verify = False

    requests.Session.__init__ = _patched_init


def _get_jsession(xnat_host, xnat_user, xnat_pass):
    """Acquire a JSESSION token from XNAT."""
    resp = requests.post(
        f"{xnat_host}/data/JSESSION",
        auth=(xnat_user, xnat_pass),
        verify=False,
    )
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def connect_xnat(xnat_host, xnat_user, xnat_pass):
    """Return ``(session, jsession_id)`` with TLS-fallback."""
    try:
        jsession_id = _get_jsession(xnat_host, xnat_user, xnat_pass)
    except Exception as e:
        if "ssl" in type(e).__name__.lower() or "ssl" in str(e).lower():
            _apply_tls_patch()
            jsession_id = _get_jsession(xnat_host, xnat_user, xnat_pass)
        else:
            raise

    session = xnat.connect(
        server=xnat_host,
        user=xnat_user,
        jsession=jsession_id,
        detect_redirect=False,
        verify=False,
    )
    return session, jsession_id


def disconnect_xnat(session, xnat_host, jsession_id):
    """Disconnect the pyxnat session and invalidate the JSESSION cookie."""
    try:
        session.disconnect()
    except Exception:
        pass
    try:
        requests.delete(
            f"{xnat_host}/data/JSESSION",
            cookies={"JSESSIONID": jsession_id},
            verify=False,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_api(session, project_id, endpoint_suffix, page_size=1000, progress_callback=None):
    """Paginate through a Discovery cohort API endpoint (keyset cursor)."""
    all_records = []
    cursor = 0
    while True:
        resp = session.get(
            f"/xapi/discovery/cohorts/{project_id}/{endpoint_suffix}",
            query={"pageSize": page_size, "cursor": cursor},
        )
        records = resp.json()
        all_records.extend(records)
        if progress_callback:
            progress_callback(len(all_records))
        if len(records) != page_size:
            break
        cursor = records[-1]["id"]
    return all_records


def dedupe_studies(df):
    """Deduplicate studies across data requests."""
    if df.empty:
        return df

    df = df.copy()
    df["status"] = df["status"].str.upper()
    df = df.sort_values("_execution_time", ascending=False)

    if "studyInstanceUid" in df.columns:
        df = df.drop_duplicates(subset="studyInstanceUid", keep="first")

    available = df[df["status"] == "AVAILABLE"].copy()
    other = df[df["status"] != "AVAILABLE"].copy()

    if not available.empty and "experimentLabel" in available.columns:
        has_label = available["experimentLabel"].notna()
        available_with = available[has_label].drop_duplicates(subset="experimentLabel", keep="first")
        available_without = available[~has_label]
        available = pd.concat([available_with, available_without], ignore_index=True)

    return pd.concat([available, other], ignore_index=True)


def load_studies(session, project_id):
    """Load + dedupe studies from the Discovery API.  Returns ``(study_df, series_df)``."""
    study_records = load_from_api(session, project_id, "studies")
    study_df = pd.DataFrame(study_records)

    if not study_df.empty:
        study_df.rename(
            columns={
                "lastExecutionTimestamp": "_execution_time",
                "dataRequestId": "_data_request_id",
            },
            inplace=True,
        )
        study_df = dedupe_studies(study_df)

    series_df = pd.DataFrame()
    return study_df, series_df


# ---------------------------------------------------------------------------
# Metadata & stats
# ---------------------------------------------------------------------------

def clean_xsi_type(xsi_type, suffix="SessionData"):
    """Strip ``xnat:`` prefix and *suffix*, uppercase result."""
    if not xsi_type or xsi_type == "Unknown":
        return "Unknown"
    result = str(xsi_type)
    if result.startswith("xnat:"):
        result = result[5:]
    if result.endswith(suffix):
        return result[: -len(suffix)].upper()
    return xsi_type


def load_xnat_metadata(session, project_id):
    """Fetch experiments and subjects from XNAT.

    Returns ``(xnat_experiments, xnat_subjects, exp_map, exp_xsi_map, xnat_exp_labels)``.
    """
    exp_response = session.get(
        f"/data/projects/{project_id}/experiments",
        query={"format": "json", "columns": "ID,label,xsiType,subject_label"},
    )
    xnat_experiments = exp_response.json().get("ResultSet", {}).get("Result", [])
    xnat_exp_labels = {e["label"] for e in xnat_experiments}
    exp_map = {e["label"]: e["ID"] for e in xnat_experiments}
    exp_xsi_map = {
        e["label"]: e.get("xsiType", "Unknown") or "Unknown"
        for e in xnat_experiments
    }

    subj_response = session.get(
        f"/data/projects/{project_id}/subjects",
        query={"format": "json", "columns": "ID,label"},
    )
    xnat_subjects = subj_response.json().get("ResultSet", {}).get("Result", [])

    return xnat_experiments, xnat_subjects, exp_map, exp_xsi_map, xnat_exp_labels


def compute_stats(study_df, xnat_experiments, xnat_subjects, exp_map, exp_xsi_map, xnat_exp_labels):
    """Derive all dashboard statistics from study_df + XNAT metadata.

    Returns a dict with every metric needed by the widget builders.
    """
    project_experiment_count = len(xnat_experiments)
    project_subject_count = len(xnat_subjects)

    if study_df.empty:
        return {
            "project_experiment_count": project_experiment_count,
            "project_subject_count": project_subject_count,
            "empty": True,
        }

    study_df = study_df.copy()
    study_df["status"] = study_df["status"].str.upper()

    # Enrich with session type
    study_df["sessionType"] = study_df["experimentLabel"].apply(
        lambda x: clean_xsi_type(exp_xsi_map.get(x, ""), "SessionData")
        if pd.notna(x) and x in exp_xsi_map
        else ""
    )

    total = len(study_df)
    available_count = (study_df["status"] == "AVAILABLE").sum()
    errors = (study_df["status"] == "ERROR").sum()
    rejected = (study_df["status"] == "REJECTED").sum()
    unique_accessions = (
        study_df["accessionNumber"].nunique()
        if "accessionNumber" in study_df.columns
        else 0
    )
    extra_studies = total - unique_accessions
    unique_patients = (
        study_df["patientId"]
        .dropna()
        .loc[lambda s: (s != "") & (s != "Unknown")]
        .nunique()
        if "patientId" in study_df.columns
        else 0
    )

    # --- Missing patients ---
    missing_patients_df = pd.DataFrame(columns=["patientId", "subjectLabel", "missing", "expected"])
    missing_patient_count = 0
    completely_missing_count = 0
    unknown_patient_study_count = 0

    if "patientId" in study_df.columns:
        valid_pid_mask = (
            study_df["patientId"].notna()
            & (study_df["patientId"] != "")
            & (study_df["patientId"] != "Unknown")
        )
        valid_studies = study_df[valid_pid_mask].copy()

        expected_per_patient = valid_studies.groupby("patientId").size().reset_index(name="expected")

        in_project_mask = (valid_studies["status"] == "AVAILABLE") & valid_studies[
            "experimentLabel"
        ].apply(lambda x: pd.notna(x) and str(x).strip() in xnat_exp_labels)
        in_project_counts = (
            valid_studies[in_project_mask]
            .groupby("patientId")
            .size()
            .reset_index(name="in_project")
        )

        patient_summary = expected_per_patient.merge(in_project_counts, on="patientId", how="left")
        patient_summary["in_project"] = patient_summary["in_project"].fillna(0).astype(int)
        patient_summary["missing"] = patient_summary["expected"] - patient_summary["in_project"]

        subject_labels = valid_studies.drop_duplicates("patientId")[["patientId", "subjectLabel"]]
        patient_summary = patient_summary.merge(subject_labels, on="patientId", how="left")

        missing_patients_df = (
            patient_summary[patient_summary["missing"] > 0]
            .sort_values("patientId")
            .reset_index(drop=True)
        )
        missing_patient_count = len(missing_patients_df)
        completely_missing_count = len(patient_summary[patient_summary["in_project"] == 0])

        unknown_patient_study_count = len(
            study_df[
                study_df["patientId"].isna()
                | (study_df["patientId"] == "")
                | (study_df["patientId"] == "Unknown")
            ]
        )

    # --- Missing from project ---
    available_df = study_df[study_df["status"] == "AVAILABLE"].copy()
    missing_from_project_count = 0
    if "experimentLabel" in available_df.columns:

        def _is_missing(row):
            exp_label = row.get("experimentLabel")
            if pd.isna(exp_label) or exp_label == "" or exp_label is None:
                return True
            return str(exp_label).strip() not in xnat_exp_labels

        missing_from_project_count = available_df.apply(_is_missing, axis=1).sum()

    # --- Multi-accession studies ---
    multi_study_df = pd.DataFrame()
    multi_study_accessions_count = 0
    if "accessionNumber" in study_df.columns:
        accession_counts = study_df.groupby("accessionNumber").size()
        multi_accessions = accession_counts[accession_counts > 1].index.tolist()
        multi_study_accessions_count = len(multi_accessions)
        if multi_accessions:
            multi_study_df = study_df[study_df["accessionNumber"].isin(multi_accessions)].copy()
            multi_study_df = multi_study_df.sort_values(["accessionNumber", "studyInstanceUid"])

    return {
        "study_df": study_df,
        "project_experiment_count": project_experiment_count,
        "project_subject_count": project_subject_count,
        "total": total,
        "available_count": available_count,
        "errors": errors,
        "rejected": rejected,
        "unique_accessions": unique_accessions,
        "extra_studies": extra_studies,
        "unique_patients": unique_patients,
        "missing_patients_df": missing_patients_df,
        "missing_patient_count": missing_patient_count,
        "completely_missing_count": completely_missing_count,
        "unknown_patient_study_count": unknown_patient_study_count,
        "missing_from_project_count": missing_from_project_count,
        "multi_study_df": multi_study_df,
        "multi_study_accessions_count": multi_study_accessions_count,
        "available_df": available_df,
        "exp_map": exp_map,
        "exp_xsi_map": exp_xsi_map,
        "xnat_exp_labels": xnat_exp_labels,
        "empty": False,
    }


# ---------------------------------------------------------------------------
# Widget builders
# ---------------------------------------------------------------------------

def stat_card(label, value, color, tooltip=""):
    """Create a uniformly-sized status card widget with optional tooltip."""
    title_attr = f'title="{tooltip}"' if tooltip else ""
    return HTML(
        f"""
        <div {title_attr} style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 15px 20px;
            margin: 5px;
            min-width: 170px;
            width: 170px;
            height: 70px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            justify-content: center;
            cursor: help;
        ">
            <div style="font-size: 28px; font-weight: bold; color: {color};">{value:,}</div>
            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; white-space: nowrap;">{label}</div>
        </div>
    """
    )


def build_patient_cards(stats):
    """Return a list of stat-card widgets for the patient row."""
    return [
        stat_card(
            "Patients In Project",
            stats["project_subject_count"],
            "#1abc9c",
            "Unique subjects in XNAT project",
        ),
        stat_card(
            "Patients Requested",
            stats["unique_patients"],
            "#8e44ad",
            "Unique patient IDs across all requested studies",
        ),
        stat_card(
            "Missing Patients",
            stats["completely_missing_count"],
            "#e74c3c",
            f'{stats["completely_missing_count"]} patients completely missing from project',
        ),
    ]


def build_study_cards(stats):
    """Return a list of stat-card widgets for the study row."""
    return [
        stat_card(
            "Studies In Project",
            stats["project_experiment_count"],
            "#1abc9c",
            "Actual studies in the project",
        ),
        stat_card(
            "Studies Requested",
            stats["total"],
            "#8e44ad",
            "Unique Study Instance UIDs",
        ),
        stat_card(
            "Accessions Requested",
            stats["unique_accessions"],
            "#9b59b6",
            "Unique Accession Numbers",
        ),
        stat_card(
            "Additional Studies",
            stats["extra_studies"],
            "#3498db",
            f'{stats["extra_studies"]} additional studies found in PACS',
        ),
        stat_card(
            "Missing Studies",
            stats["errors"] + stats["missing_from_project_count"] + stats["rejected"],
            "#e74c3c",
            "Error, missing from project, or excluded studies",
        ),
    ]


def build_session_type_chart(stats):
    """Return an HTML widget containing the session-types donut chart (or empty)."""
    available_df = stats["available_df"]
    exp_map = stats["exp_map"]
    exp_xsi_map = stats["exp_xsi_map"]

    available_in_xnat = available_df[available_df["experimentLabel"].isin(exp_map.keys())]
    session_types = []
    for _, row in available_in_xnat.iterrows():
        exp_label = row.get("experimentLabel", "")
        if pd.isna(exp_label) or exp_label not in exp_map:
            continue
        xsi_type = exp_xsi_map.get(exp_label, "Unknown")
        session_types.append(clean_xsi_type(xsi_type, "SessionData"))

    widget = HTML()
    if not session_types:
        return widget, ""

    fig = Figure(figsize=(6, 4.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    counts = pd.Series(session_types).value_counts()
    total = len(session_types)
    colors = plt.cm.tab20(range(len(counts)))

    wedges, _ = ax.pie(
        counts.values,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.5, edgecolor="white"),
    )
    ax.text(
        0, 0, f"{total:,}\nSessions",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#333",
    )
    legend_labels = [
        f"{stype} - {c:,} ({c / total * 100:.1f}%)"
        for stype, c in zip(counts.index, counts.values)
    ]
    ax.legend(
        wedges, legend_labels, title="Type",
        loc="center left", bbox_to_anchor=(1.05, 0.5),
        fontsize=9, title_fontsize=10,
    )
    ax.set_title("Session Types", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    session_b64 = base64.b64encode(buf.read()).decode("ascii")

    widget.value = (
        '<div style="display:flex;gap:20px;align-items:flex-start;">'
        f'<img src="data:image/png;base64,{session_b64}">'
        "</div>"
    )
    return widget, session_b64


# ---------------------------------------------------------------------------
# Series banner  (button-triggered async series loading)
# ---------------------------------------------------------------------------

def _sop_with_tooltip(sop_uid):
    sop_uid = str(sop_uid).strip() if sop_uid else ""
    if not sop_uid or sop_uid == "nan":
        return ""
    if sop_uid in EXCLUDED_SOP_CLASSES:
        name = EXCLUDED_SOP_CLASSES[sop_uid]
        return f'<span title="{html_module.escape(sop_uid)}" style="cursor: help;">{html_module.escape(name)}</span>'
    return sop_uid


def _truncate_with_tooltip(text, max_len=50):
    text = str(text)
    if len(text) <= max_len:
        return html_module.escape(text)
    truncated = text[:max_len].rsplit(" ", 1)[0] + "..."
    return (
        f'<span title="{html_module.escape(text)}" style="cursor: help;">'
        f"{html_module.escape(truncated)}</span>"
    )


def build_series_banner(stats, charts_html, session_b64, xnat_host, xnat_user, xnat_pass, project_id):
    """Build the "Load Series Data" banner and wire up the async handler.

    Returns ``(series_banner_widget, missing_series_output_widget)``.
    """
    study_df = stats["study_df"]
    exp_map = stats["exp_map"]
    xnat_exp_labels = stats["xnat_exp_labels"]

    scans_by_experiment = {}

    load_btn = Button(
        description="Load Series Data",
        button_style="warning",
        tooltip="Fetch series data from XNAT to view scan type breakdown and missing series analysis",
        icon="download",
        layout=Layout(margin="0 15px 0 0"),
    )

    progress_html = HTML(
        '<span style="color: #856404;">Click to load series data for scan type breakdown '
        "and missing series analysis</span>"
    )

    banner_content = HBox([load_btn, progress_html], layout=Layout(align_items="center"))
    series_note = HTML(
        '<span style="color: #856404; font-size: 12px; opacity: 0.8;">'
        "Note: Loading series data can take several minutes.</span>"
    )
    series_banner = VBox(
        [banner_content, series_note],
        layout=Layout(
            background="#fff8e6",
            border="1px solid #f39c12",
            border_radius="10px",
            padding="15px 20px",
            margin="10px 5px",
            width="1500px",
        ),
    )

    missing_series_output = VBox()

    # --- async handler ---
    async def _load_series_async():
        loop = asyncio.get_event_loop()
        total_experiments = len(exp_map)

        progress_html.value = '<span style="color: #856404;">Connecting to XNAT...</span>'

        try:
            jsession_resp = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{xnat_host}/data/JSESSION",
                    auth=(xnat_user, xnat_pass),
                    verify=False,
                ),
            )
            jsession_resp.raise_for_status()
            series_jsession_id = jsession_resp.text

            new_session = await loop.run_in_executor(
                None,
                lambda: xnat.connect(
                    server=xnat_host,
                    user=xnat_user,
                    jsession=series_jsession_id,
                    detect_redirect=False,
                    verify=False,
                ),
            )
        except Exception as e:
            progress_html.value = f'<span style="color: #c00;">Failed to connect: {e}</span>'
            return

        # Fetch scan data in background thread
        def _fetch_scans():
            exp_items = list(exp_map.items())
            for i, (exp_label, exp_id) in enumerate(exp_items):
                progress_html.value = (
                    f'<span style="color: #856404;">Loading series data from experiment '
                    f"{i + 1:,} / {total_experiments:,}...</span>"
                )
                try:
                    scans_response = new_session.get(
                        f"/data/experiments/{exp_id}/scans",
                        query={"format": "json", "columns": "ID,UID,series_description,xsiType"},
                    )
                    scans_data = scans_response.json().get("ResultSet", {}).get("Result", [])
                    scans_by_experiment[exp_label] = [
                        {
                            "id": scan.get("ID", ""),
                            "uid": scan.get("UID", ""),
                            "series_description": scan.get("series_description", ""),
                            "xsiType": scan.get("xsiType", "Unknown") or "Unknown",
                        }
                        for scan in scans_data
                    ]
                except Exception:
                    scans_by_experiment[exp_label] = []

        await loop.run_in_executor(None, _fetch_scans)

        # Build scan type data
        all_project_scans = []
        for exp_label, scans in scans_by_experiment.items():
            for scan in scans:
                all_project_scans.append({"type": clean_xsi_type(scan["xsiType"], "ScanData")})

        exps_with_scans = sum(1 for scans in scans_by_experiment.values() if scans)
        total_scans = len(all_project_scans)

        load_btn.disabled = True
        load_btn.description = "Series Loaded"
        load_btn.button_style = ""
        load_btn.icon = "check"
        load_btn.style.button_color = "#6c757d"

        # Scan types chart
        if all_project_scans:
            try:
                fig = Figure(figsize=(6, 4.5))
                FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)

                scan_type_counts = pd.DataFrame(all_project_scans)["type"].value_counts()
                total_series = len(all_project_scans)
                colors = plt.cm.tab20(range(len(scan_type_counts)))

                wedges, _ = ax.pie(
                    scan_type_counts.values,
                    startangle=90,
                    colors=colors,
                    wedgeprops=dict(width=0.5, edgecolor="white"),
                )
                ax.text(
                    0, 0, f"{total_series:,}\nScans",
                    ha="center", va="center", fontsize=14, fontweight="bold", color="#333",
                )
                legend_labels = [
                    f"{stype} - {c:,} ({c / total_series * 100:.1f}%)"
                    for stype, c in zip(scan_type_counts.index, scan_type_counts.values)
                ]
                ax.legend(
                    wedges, legend_labels, title="Type",
                    loc="center left", bbox_to_anchor=(1.05, 0.5),
                    fontsize=9, title_fontsize=10,
                )
                ax.set_title("Scan Types", fontsize=14, fontweight="bold", pad=15)
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                scan_b64 = base64.b64encode(buf.read()).decode("ascii")

                imgs = '<div style="display:flex;gap:20px;align-items:flex-start;">'
                if session_b64:
                    imgs += f'<img src="data:image/png;base64,{session_b64}">'
                imgs += f'<img src="data:image/png;base64,{scan_b64}"></div>'
                charts_html.value = imgs
            except Exception as _chart_err:
                imgs = '<div style="display:flex;gap:20px;align-items:flex-start;">'
                if session_b64:
                    imgs += f'<img src="data:image/png;base64,{session_b64}">'
                imgs += f'<div style="color:red;padding:10px;">Chart error: {type(_chart_err).__name__}: {_chart_err}</div></div>'
                charts_html.value = imgs

        # --- Missing series analysis ---
        studies_with_missing = []
        available = study_df[study_df["status"] == "AVAILABLE"]

        progress_html.value = '<span style="color: #856404;">Loading series data from Discovery API...</span>'

        def _fetch_series():
            try:
                series_records = load_from_api(
                    new_session,
                    project_id,
                    "series",
                    progress_callback=lambda n: setattr(
                        progress_html,
                        "value",
                        f'<span style="color: #856404;">Loading series data from Discovery API... ({n:,} records)</span>',
                    ),
                )
                return pd.DataFrame(series_records)
            except Exception:
                return pd.DataFrame()

        api_series_df = await loop.run_in_executor(None, _fetch_series)
        series_by_exp = (
            api_series_df.groupby("experimentLabel")
            if not api_series_df.empty and "experimentLabel" in api_series_df.columns
            else None
        )

        progress_html.value = '<span style="color: #856404;">Analyzing missing series...</span>'

        try:
            for _, row in available.iterrows():
                exp_label = row.get("experimentLabel", "")
                if pd.isna(exp_label) or exp_label not in exp_map:
                    continue

                xnat_scans = scans_by_experiment.get(exp_label, [])
                actual_count = len(xnat_scans)
                xnat_series_uids = {
                    scan.get("uid", "").strip() for scan in xnat_scans if scan.get("uid")
                }

                if series_by_exp is not None and exp_label in series_by_exp.groups:
                    expected = series_by_exp.get_group(exp_label)
                    expected_count = len(expected)

                    def _is_missing(r):
                        uid = str(r.get("seriesInstanceUid", "")).strip()
                        if not uid or uid == "nan":
                            return True
                        return uid not in xnat_series_uids

                    all_missing_series = expected[expected.apply(_is_missing, axis=1)].copy()

                    if len(all_missing_series) > 0:
                        excluded_series = all_missing_series[
                            all_missing_series["sopClassUid"].isin(EXCLUDED_SOP_CLASSES.keys())
                        ].copy()
                        actually_missing = all_missing_series[
                            ~all_missing_series["sopClassUid"].isin(EXCLUDED_SOP_CLASSES.keys())
                        ].copy()

                        excluded_count = len(excluded_series)
                        unexplained_count = len(actually_missing)

                        if unexplained_count > 0:
                            study_uid = row.get("studyInstanceUid", "")
                            if pd.isna(study_uid):
                                study_uid = ""
                            studies_with_missing.append(
                                {
                                    "experimentLabel": exp_label,
                                    "accessionNumber": row.get("accessionNumber", ""),
                                    "studyInstanceUid": study_uid,
                                    "expected": expected_count,
                                    "actual": actual_count,
                                    "missing": excluded_count + unexplained_count,
                                    "excluded_sop": excluded_count,
                                    "unexplained": unexplained_count,
                                    "excluded_series": excluded_series,
                                    "actually_missing": actually_missing,
                                }
                            )
        except Exception as e:
            progress_html.value = (
                f'<span style="color: #c00;">Error analyzing series: '
                f"{type(e).__name__}: {e}</span>"
            )
            return

        # Cleanup series session
        new_session.disconnect()
        await loop.run_in_executor(
            None,
            lambda: requests.delete(
                f"{xnat_host}/data/JSESSION",
                cookies={"JSESSIONID": series_jsession_id},
                verify=False,
            ),
        )

        series_count = len(api_series_df) if not api_series_df.empty else 0
        progress_html.value = (
            f'<span style="color: #155724;">&#10003; <strong>Loaded {total_scans:,} scans '
            f"from {exps_with_scans:,} experiments, {series_count:,} series from Discovery API."
            "</strong> Scroll down to view Missing Series Analysis.</span>"
        )
        series_banner.layout.background = "#d4edda"
        series_banner.layout.border = "1px solid #28a745"
        series_note.value = ""

        # Build missing series widget
        if studies_with_missing:
            _build_missing_series_widget(
                studies_with_missing, missing_series_output, xnat_host, project_id
            )
        else:
            missing_series_output.children = [
                HTML(
                    '<div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); '
                    "color: white; padding: 20px; border-radius: 10px; text-align: center; "
                    'width: 1500px; margin: 10px 5px;">'
                    "<h3 style=\"margin: 0;\">All Series Complete</h3>"
                    '<p style="margin: 10px 0 0 0; opacity: 0.9;">No missing series detected.</p>'
                    "</div>"
                )
            ]

    def _on_click(b):
        b.disabled = True
        asyncio.ensure_future(_load_series_async())

    load_btn.on_click(_on_click)

    return series_banner, missing_series_output


def _build_missing_series_widget(studies_with_missing, container, xnat_host, project_id):
    """Populate *container* with the missing-series dropdown UI."""
    total_missing_studies = len(studies_with_missing)
    total_excluded = sum(s["excluded_sop"] for s in studies_with_missing)
    total_unexplained = sum(s["unexplained"] for s in studies_with_missing)

    ms_header = HTML(
        f"""
        <div style="
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white; padding: 20px; border-radius: 10px 10px 0 0;
        ">
            <h3 style="margin: 0; font-size: 18px;">Missing Series</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 13px;">{total_missing_studies} studies with missing series</p>
        </div>
    """
    )

    ms_summary = HTML(
        f"""
        <div style="display: flex; gap: 15px; padding: 15px 20px; background: #f8f9fa;">
            <div style="text-align: center; padding: 10px 20px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 24px; font-weight: bold; color: #e74c3c;">{total_missing_studies}</div>
                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Studies Affected</div>
            </div>
            <div style="text-align: center; padding: 10px 20px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 24px; font-weight: bold; color: #f39c12;">{total_excluded}</div>
                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Excluded</div>
            </div>
            <div style="text-align: center; padding: 10px 20px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 24px; font-weight: bold; color: #c0392b;">{total_unexplained}</div>
                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Missing</div>
            </div>
        </div>
    """
    )

    options = []
    for s in studies_with_missing:
        label = (
            f"{s['experimentLabel']} \u2014 {s['missing']} missing "
            f"({s['excluded_sop']} excluded, {s['unexplained']} missing)"
        )
        options.append((label, s))

    ms_dropdown = Dropdown(options=options, value=studies_with_missing[0], layout=Layout(width="100%"))
    ms_detail_output = VBox()

    def _on_select(change):
        study = change["new"]
        if study is None:
            ms_detail_output.children = []
            return

        xnat_url = f"{xnat_host}/data/archive/projects/{project_id}/experiments/{study['experimentLabel']}"

        header_w = HTML(
            f"""
            <div style="background: white; padding: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 16px; font-weight: bold;">
                            <a href="{xnat_url}" target="_blank" style="color: #e74c3c; text-decoration: none;">{study['experimentLabel']}</a>
                        </div>
                        <div style="font-size: 12px; color: #888; margin-top: 4px;">Accession: {study['accessionNumber']}</div>
                        <div style="font-size: 11px; color: #aaa; margin-top: 2px;">Study Instance UID: {study['studyInstanceUid']}</div>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <div style="text-align: center; padding: 8px 15px; background: #e8f4fd; border-radius: 6px;">
                            <div style="font-size: 18px; font-weight: bold; color: #3498db;">{study['expected']}</div>
                            <div style="font-size: 10px; color: #666;">Expected</div>
                        </div>
                        <div style="text-align: center; padding: 8px 15px; background: #e8f8e8; border-radius: 6px;">
                            <div style="font-size: 18px; font-weight: bold; color: #27ae60;">{study['actual']}</div>
                            <div style="font-size: 10px; color: #666;">In XNAT</div>
                        </div>
                        <div style="text-align: center; padding: 8px 15px; background: #fff8e6; border-radius: 6px;">
                            <div style="font-size: 18px; font-weight: bold; color: #f39c12;">{study['excluded_sop']}</div>
                            <div style="font-size: 10px; color: #856404;">Excluded</div>
                        </div>
                        <div style="text-align: center; padding: 8px 15px; background: #fdeaea; border-radius: 6px;">
                            <div style="font-size: 18px; font-weight: bold; color: #e74c3c;">{study['unexplained']}</div>
                            <div style="font-size: 10px; color: #721c21;">Missing</div>
                        </div>
                    </div>
                </div>
            </div>
        """
        )

        all_missing = []
        is_excluded_flags = []

        for _, r in study["excluded_series"].iterrows():
            all_missing.append(
                {
                    "Series UID": r.get("seriesInstanceUid", ""),
                    "Status": "Excluded",
                    "Series Description": r.get("seriesDescription", ""),
                    "Modality": r.get("modality", ""),
                    "SOP Class": _sop_with_tooltip(r.get("sopClassUid", "")),
                }
            )
            is_excluded_flags.append(True)

        for _, r in study["actually_missing"].iterrows():
            all_missing.append(
                {
                    "Series UID": r.get("seriesInstanceUid", ""),
                    "Status": "Missing",
                    "Series Description": r.get("seriesDescription", ""),
                    "Modality": r.get("modality", ""),
                    "SOP Class": _sop_with_tooltip(r.get("sopClassUid", "")),
                }
            )
            is_excluded_flags.append(False)

        table_widget = HTML()
        if all_missing:
            display_df = pd.DataFrame(all_missing)

            def _style_row(row_idx):
                if is_excluded_flags[row_idx]:
                    return ["background-color: #fff8e6; color: #856404;"] * len(display_df.columns)
                return ["background-color: #fdeaea; color: #721c24;"] * len(display_df.columns)

            styled = (
                display_df.style.apply(lambda x: _style_row(x.name), axis=1)
                .set_properties(**{"text-align": "left", "font-size": "12px", "padding": "10px"})
                .set_table_styles(
                    [
                        {"selector": "", "props": [("width", "100%"), ("border-collapse", "collapse")]},
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", "#f8f9fa"),
                                ("font-size", "11px"),
                                ("text-transform", "uppercase"),
                                ("color", "#666"),
                                ("padding", "10px"),
                                ("border-bottom", "2px solid #ddd"),
                            ],
                        },
                        {"selector": "td", "props": [("border-bottom", "1px solid #eee")]},
                    ]
                )
                .hide(axis="index")
            )

            table_html = styled.to_html()
            table_html = (
                table_html.replace("&lt;span title=", "<span title=")
                .replace('style=&quot;cursor: help;&quot;&gt;', 'style="cursor: help;">')
                .replace("&lt;/span&gt;", "</span>")
                .replace("&quot;", '"')
            )
            table_widget = HTML(f'<div style="background: white; padding: 15px 20px;">{table_html}</div>')

        ms_detail_output.children = [header_w, table_widget] if all_missing else [header_w]

    ms_dropdown.observe(_on_select, names="value")

    ms_dropdown_label = HTML(
        '<div style="font-size: 12px; font-weight: bold; color: #666; margin-bottom: 5px;">Select Study</div>'
    )

    ms_widget = VBox(
        [
            ms_header,
            ms_summary,
            VBox(
                [ms_dropdown_label, ms_dropdown],
                layout=Layout(padding="15px 20px", width="100%", overflow="hidden"),
            ),
            ms_detail_output,
        ],
        layout=Layout(
            width="1500px",
            border_radius="10px",
            overflow="hidden",
            background="#f8f9fa",
            margin="10px 5px",
        ),
    )

    _on_select({"new": studies_with_missing[0]})
    container.children = [ms_widget]


# ---------------------------------------------------------------------------
# Table builders (missing patients, additional studies, errors/missing)
# ---------------------------------------------------------------------------

def build_missing_patients_table(stats):
    """Return a widget for the Missing Patients table, or ``None``."""
    study_df = stats["study_df"]
    missing_patients_df = stats["missing_patients_df"]
    missing_patient_count = stats["missing_patient_count"]
    unknown_patient_study_count = stats["unknown_patient_study_count"]

    if study_df.empty or missing_patient_count == 0 or missing_patients_df.empty:
        return None

    mp_details = []
    for _, row in missing_patients_df.iterrows():
        patient_id = str(row.get("patientId", "")).strip()
        if not patient_id or patient_id in ("nan", "None"):
            patient_id = "Unknown"
        subject_label = str(row.get("subjectLabel", "")).strip()
        if not subject_label or subject_label in ("nan", "None"):
            subject_label = "Unknown"
        mp_details.append(
            {
                "Patient ID": patient_id,
                "Subject Label": subject_label,
                "Missing Studies": int(row.get("missing", 0)),
                "Expected Studies": int(row.get("expected", 0)),
            }
        )

    mp_table_df = pd.DataFrame(mp_details)
    MP_PAGE_SIZE = 15
    mp_state = {"page": 0, "sort_col": "Missing Studies", "sort_asc": False, "filter": ""}

    mp_columns = [
        ("Patient ID", "30%"),
        ("Subject Label", "30%"),
        ("Missing Studies", "20%"),
        ("Expected Studies", "20%"),
    ]

    mp_filter_input = Text(placeholder="Search", layout=Layout(width="300px"))
    mp_prev_btn = Button(description="Previous", button_style="info", layout=Layout(width="100px"))
    mp_next_btn = Button(description="Next", button_style="info", layout=Layout(width="100px"))
    mp_page_label = HTML()
    mp_table_body = HTML()
    mp_header_buttons = []

    def _update_headers():
        for i, (col_name, _) in enumerate(mp_columns):
            arrow = ""
            if mp_state["sort_col"] == col_name:
                arrow = " \u2191" if mp_state["sort_asc"] else " \u2193"
            mp_header_buttons[i].description = f"{col_name}{arrow}"

    def _make_sort(col_name):
        def handler(b):
            if mp_state["sort_col"] == col_name:
                mp_state["sort_asc"] = not mp_state["sort_asc"]
            else:
                mp_state["sort_col"] = col_name
                mp_state["sort_asc"] = True
            mp_state["page"] = 0
            _update_headers()
            _render()

        return handler

    def _render():
        filtered = mp_table_df.copy()
        if mp_state["filter"]:
            pat = mp_state["filter"].replace("*", ".*")
            mask = (
                filtered["Patient ID"].astype(str).str.contains(pat, case=False, regex=True, na=False)
                | filtered["Subject Label"].astype(str).str.contains(pat, case=False, regex=True, na=False)
            )
            filtered = filtered[mask]

        sort_col = mp_state["sort_col"]
        if sort_col in ("Missing Studies", "Expected Studies"):
            sorted_df = filtered.sort_values(by=sort_col, ascending=mp_state["sort_asc"]).reset_index(drop=True)
        else:
            sorted_df = filtered.sort_values(
                by=sort_col, ascending=mp_state["sort_asc"],
                key=lambda x: x.str.lower() if x.dtype == "object" else x,
            ).reset_index(drop=True)

        total_pages = max(1, (len(sorted_df) + MP_PAGE_SIZE - 1) // MP_PAGE_SIZE)
        mp_state["page"] = max(0, min(mp_state["page"], total_pages - 1))
        start = mp_state["page"] * MP_PAGE_SIZE
        page_df = sorted_df.iloc[start : start + MP_PAGE_SIZE]

        filter_text = f" (filtered: {len(filtered)})" if mp_state["filter"] else ""
        mp_page_label.value = (
            f'<span style="font-size: 13px; color: #666;">'
            f'Page {mp_state["page"] + 1} of {total_pages} ({len(sorted_df)} patients{filter_text})</span>'
        )
        mp_prev_btn.disabled = mp_state["page"] == 0
        mp_next_btn.disabled = mp_state["page"] >= total_pages - 1

        rows = []
        for _, r in page_df.iterrows():
            pid = html_module.escape(str(r["Patient ID"]))
            slabel = html_module.escape(str(r["Subject Label"]))
            rows.append(
                f'<tr style="background-color: #fdeaea; color: #721c24;">'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 30%;">{pid}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 30%;">{slabel}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 20%; font-weight: bold;">{int(r["Missing Studies"])}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 20%;">{int(r["Expected Studies"])}</td>'
                f"</tr>"
            )

        mp_table_body.value = (
            '<table style="width: 100%; border-collapse: collapse; font-size: 12px; '
            'table-layout: fixed; background: white; margin: 0; padding: 0;">'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )

    for col_name, width in mp_columns:
        arrow = ""
        if mp_state["sort_col"] == col_name:
            arrow = " \u2191" if mp_state["sort_asc"] else " \u2193"
        btn = Button(description=f"{col_name}{arrow}")
        btn.layout = Layout(width=width, height="36px", margin="0", padding="0")
        btn.style.button_color = "#333"
        btn.add_class("header-sort-button")
        btn.on_click(_make_sort(col_name))
        mp_header_buttons.append(btn)

    mp_filter_input.observe(lambda c: (mp_state.update(filter=c["new"], page=0), _render()), names="value")
    mp_prev_btn.on_click(lambda b: (mp_state.update(page=mp_state["page"] - 1), _render()))
    mp_next_btn.on_click(lambda b: (mp_state.update(page=mp_state["page"] + 1), _render()))

    _render()

    mp_desc = f"{missing_patient_count} patients with missing studies"
    completely_missing = len(mp_table_df[mp_table_df["Missing Studies"] == mp_table_df["Expected Studies"]])
    if completely_missing > 0:
        mp_desc += f". {completely_missing} patients completely missing"
    if unknown_patient_study_count > 0:
        mp_desc += (
            f'<br><span style="opacity: 0.85;">Note: The Patient ID is unknown for '
            f"{unknown_patient_study_count} studies. Please check the missing studies table.</span>"
        )

    mp_title = HTML(
        f"""
        <div style="
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white; padding: 20px; border-radius: 10px 10px 0 0;
        ">
            <h3 style="margin: 0; font-size: 18px;">Missing Patients</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 13px;">{mp_desc}</p>
        </div>
    """
    )

    mp_controls = HBox(
        [
            HBox([mp_filter_input], layout=Layout(align_items="center")),
            HBox([mp_prev_btn, mp_page_label, mp_next_btn], layout=Layout(align_items="center", gap="10px")),
        ],
        layout=Layout(
            padding="15px 20px",
            justify_content="space-between",
            align_items="center",
            background="#f5f5f5",
            width="100%",
        ),
    )

    mp_header_row = HBox(mp_header_buttons)
    mp_header_row.layout = Layout(width="100%", margin="0", padding="0", background="#333", overflow="hidden")
    mp_header_row.add_class("no-gap")

    mp_table_wrapper = VBox([mp_table_body])
    mp_table_wrapper.layout = Layout(background="white", padding="0", margin="0", width="100%", overflow="hidden")

    mp_table_container = VBox([mp_header_row, mp_table_wrapper])
    mp_table_container.layout = Layout(background="#333", padding="0", margin="0", width="100%", overflow="hidden")
    mp_table_container.add_class("no-gap")

    widget = VBox([mp_title, mp_controls, mp_table_container])
    widget.layout = Layout(width="1500px", border_radius="10px", overflow="hidden", background="#f5f5f5", margin="10px 5px")
    widget.add_class("no-gap")
    return widget


def build_additional_studies_table(stats, xnat_host, project_id):
    """Return a widget for the Additional Studies table, or ``None``."""
    study_df = stats["study_df"]
    extra_studies = stats["extra_studies"]
    multi_study_df = stats["multi_study_df"]
    multi_study_accessions_count = stats["multi_study_accessions_count"]
    xnat_exp_labels = stats["xnat_exp_labels"]

    if study_df.empty or extra_studies <= 0 or multi_study_df.empty:
        return None

    # Inject button CSS
    css_widget = HTML(BUTTON_STYLE_CSS)

    additional_details = []
    for _, row in multi_study_df.iterrows():
        exp_label = str(row.get("experimentLabel", "")).strip()
        if not exp_label or exp_label in ("nan", "None"):
            exp_label = "Unknown"
        study_uid = str(row.get("studyInstanceUid", "")).strip()
        if not study_uid or study_uid in ("nan", "None"):
            study_uid = "Unknown"

        status = row.get("status", "")
        if status == "ERROR":
            reason = str(row.get("statusMessage", "")).strip() or "Unknown"
        elif status == "REJECTED":
            reason = str(row.get("statusMessage", "")).strip() or "Rejected by site"
        else:
            reason = ""

        session_type = row.get("sessionType", "")
        if pd.isna(session_type):
            session_type = ""

        patient_id = str(row.get("patientId", "")).strip()
        if not patient_id or patient_id in ("nan", "None"):
            patient_id = "Unknown"
        subject_label = str(row.get("subjectLabel", "")).strip()
        if not subject_label or subject_label in ("nan", "None"):
            subject_label = "Unknown"

        additional_details.append(
            {
                "Experiment Label": str(exp_label),
                "Patient ID": str(patient_id),
                "Subject Label": str(subject_label),
                "Session Type": str(session_type),
                "Accession Number": str(row.get("accessionNumber", "")),
                "Study UID": study_uid,
                "Status": status,
                "Reason": reason,
            }
        )

    add_table_df = pd.DataFrame(additional_details)
    PAGE_SIZE = 15
    add_state = {"page": 0, "sort_col": "Accession Number", "sort_asc": True, "filter": ""}

    add_columns = [
        ("Experiment Label", "14%"),
        ("Patient ID", "9%"),
        ("Subject Label", "9%"),
        ("Session Type", "10%"),
        ("Accession Number", "10%"),
        ("Study UID", "26%"),
        ("Status", "6%"),
        ("Reason", "16%"),
    ]

    add_filter = Text(placeholder="Search", layout=Layout(width="300px"))
    add_prev = Button(description="Previous", button_style="info", layout=Layout(width="100px"))
    add_next = Button(description="Next", button_style="info", layout=Layout(width="100px"))
    add_page_label = HTML()
    add_table_body = HTML()
    add_header_buttons = []

    def _update_headers():
        for i, (col_name, _) in enumerate(add_columns):
            arrow = ""
            if add_state["sort_col"] == col_name:
                arrow = " \u2191" if add_state["sort_asc"] else " \u2193"
            add_header_buttons[i].description = f"{col_name}{arrow}"

    def _make_sort(col_name):
        def handler(b):
            if add_state["sort_col"] == col_name:
                add_state["sort_asc"] = not add_state["sort_asc"]
            else:
                add_state["sort_col"] = col_name
                add_state["sort_asc"] = True
            add_state["page"] = 0
            _update_headers()
            _render()

        return handler

    def _render():
        filtered = add_table_df.copy()
        if add_state["filter"]:
            pat = add_state["filter"].replace("*", ".*")
            mask = pd.Series(False, index=filtered.index)
            for col in add_table_df.columns:
                mask = mask | filtered[col].str.contains(pat, case=False, regex=True, na=False)
            filtered = filtered[mask]

        sorted_df = filtered.sort_values(
            by=add_state["sort_col"],
            ascending=add_state["sort_asc"],
            key=lambda x: x.str.lower() if x.dtype == "object" else x,
        ).reset_index(drop=True)

        total_pages = max(1, (len(sorted_df) + PAGE_SIZE - 1) // PAGE_SIZE)
        add_state["page"] = max(0, min(add_state["page"], total_pages - 1))
        start = add_state["page"] * PAGE_SIZE
        page_df = sorted_df.iloc[start : start + PAGE_SIZE]

        filter_text = f" (filtered: {len(filtered)})" if add_state["filter"] else ""
        add_page_label.value = (
            f'<span style="font-size: 13px; color: #666;">'
            f'Page {add_state["page"] + 1} of {total_pages} ({len(sorted_df)} studies{filter_text})</span>'
        )
        add_prev.disabled = add_state["page"] == 0
        add_next.disabled = add_state["page"] >= total_pages - 1

        rows = []
        for _, r in page_df.iterrows():
            exp_label = str(r["Experiment Label"])
            pid = html_module.escape(str(r["Patient ID"]))
            slabel = html_module.escape(str(r["Subject Label"]))
            sess_type = html_module.escape(str(r["Session Type"]))
            acc = html_module.escape(str(r["Accession Number"]))
            uid = html_module.escape(str(r["Study UID"]))
            stat = html_module.escape(str(r["Status"]))
            reason = _truncate_with_tooltip(str(r["Reason"]), 50)

            status_upper = str(r["Status"]).upper()
            if status_upper == "ERROR":
                bg, tc = "#fdeaea", "#721c24"
            elif status_upper == "REJECTED":
                bg, tc = "#fff8e6", "#856404"
            else:
                bg, tc = "#f0f7ff", "#2c5282"

            if status_upper == "AVAILABLE" and exp_label and exp_label != "Unknown":
                xnat_url = f"{xnat_host}/data/archive/projects/{project_id}/experiments/{exp_label}"
                exp_display = f'<a href="{xnat_url}" target="_blank" style="color: {tc}; text-decoration: underline;">{html_module.escape(exp_label)}</a>'
            else:
                exp_display = html_module.escape(exp_label)

            rows.append(
                f'<tr style="background-color: {bg}; color: {tc};">'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 14%;">{exp_display}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 9%;">{pid}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 9%;">{slabel}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 10%;">{sess_type}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 10%;">{acc}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 26%; word-break: break-all;">{uid}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 6%;">{stat}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 16%;">{reason}</td>'
                f"</tr>"
            )

        add_table_body.value = (
            '<table style="width: 100%; border-collapse: collapse; font-size: 12px; '
            'table-layout: fixed; background: white; margin: 0; padding: 0;">'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )

    for col_name, width in add_columns:
        arrow = ""
        if add_state["sort_col"] == col_name:
            arrow = " \u2191" if add_state["sort_asc"] else " \u2193"
        btn = Button(description=f"{col_name}{arrow}")
        btn.layout = Layout(width=width, height="36px", margin="0", padding="0")
        btn.style.button_color = "#333"
        btn.add_class("header-sort-button")
        btn.on_click(_make_sort(col_name))
        add_header_buttons.append(btn)

    add_filter.observe(lambda c: (add_state.update(filter=c["new"], page=0), _render()), names="value")
    add_prev.on_click(lambda b: (add_state.update(page=add_state["page"] - 1), _render()))
    add_next.on_click(lambda b: (add_state.update(page=add_state["page"] + 1), _render()))

    _render()

    add_title = HTML(
        f"""
        <div style="
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white; padding: 20px; border-radius: 10px 10px 0 0;
        ">
            <h3 style="margin: 0; font-size: 18px;">Additional Studies</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 13px;">{extra_studies} additional studies found. {multi_study_accessions_count} accession numbers have multiple studies in the PACS</p>
        </div>
    """
    )

    add_controls = HBox(
        [
            HBox([add_filter], layout=Layout(align_items="center")),
            HBox([add_prev, add_page_label, add_next], layout=Layout(align_items="center", gap="10px")),
        ],
        layout=Layout(
            padding="15px 20px",
            justify_content="space-between",
            align_items="center",
            background="#f5f5f5",
            width="100%",
        ),
    )

    add_header_row = HBox(add_header_buttons)
    add_header_row.layout = Layout(width="100%", margin="0", padding="0", background="#333", overflow="hidden")
    add_header_row.add_class("no-gap")

    add_table_wrapper = VBox([add_table_body])
    add_table_wrapper.layout = Layout(background="white", padding="0", margin="0", width="100%", overflow="hidden")

    add_table_container = VBox([add_header_row, add_table_wrapper])
    add_table_container.layout = Layout(background="#333", padding="0", margin="0", width="100%", overflow="hidden")
    add_table_container.add_class("no-gap")

    widget = VBox([css_widget, add_title, add_controls, add_table_container])
    widget.layout = Layout(width="1500px", border_radius="10px", overflow="hidden", background="#f5f5f5", margin="10px 5px")
    widget.add_class("no-gap")
    return widget


def build_errors_missing_table(stats, xnat_host, project_id):
    """Return a widget for the Study Errors / Missing table, or ``None``."""
    study_df = stats["study_df"]
    xnat_exp_labels = stats["xnat_exp_labels"]
    errors = stats["errors"]
    rejected = stats["rejected"]

    if study_df.empty:
        return None

    # Inject button CSS
    css_widget = HTML(BUTTON_STYLE_CSS)

    available_df = study_df[study_df["status"] == "AVAILABLE"].copy()
    if "experimentLabel" in available_df.columns:

        def _is_missing(row):
            exp_label = row.get("experimentLabel")
            if pd.isna(exp_label) or exp_label == "" or exp_label is None:
                return True
            return str(exp_label).strip() not in xnat_exp_labels

        missing_from_project = available_df[available_df.apply(_is_missing, axis=1)]
    else:
        missing_from_project = pd.DataFrame()

    missing_from_project_count = len(missing_from_project)
    failed_count = errors + rejected + missing_from_project_count

    if failed_count == 0:
        return HTML(
            """
            <div style="
                background: #d4edda; border: 1px solid #28a745; border-radius: 6px;
                padding: 15px 20px; margin: 10px 5px; font-size: 13px; color: #155724;
                width: 1500px; box-sizing: border-box;
            ">
                <strong>All studies processed successfully.</strong> No errors, rejections, or missing studies.
            </div>
        """
        )

    failed_df = study_df[study_df["status"].isin(["ERROR", "REJECTED"])].copy()
    error_details = []

    for _, row in failed_df.iterrows():
        status = row.get("status", "")
        if status == "ERROR":
            status_display = "Error"
            reason = str(row.get("statusMessage", "")).strip() or "Unknown"
        else:
            status_display = "Rejected"
            reason = str(row.get("statusMessage", "")).strip() or "Rejected by site"

        study_uid = str(row.get("studyInstanceUid", "")).strip()
        if not study_uid or study_uid in ("nan", "None"):
            study_uid = "Unknown"
        exp_label = str(row.get("experimentLabel", "")).strip()
        if not exp_label or exp_label in ("nan", "None"):
            exp_label = "Unknown"
        patient_id = str(row.get("patientId", "")).strip()
        if not patient_id or patient_id in ("nan", "None"):
            patient_id = "Unknown"
        subject_label = str(row.get("subjectLabel", "")).strip()
        if not subject_label or subject_label in ("nan", "None"):
            subject_label = "Unknown"

        error_details.append(
            {
                "Experiment Label": str(exp_label),
                "Patient ID": str(patient_id),
                "Subject Label": str(subject_label),
                "Accession Number": str(row.get("accessionNumber", "")),
                "Study UID": study_uid,
                "Status": status_display,
                "Reason": reason,
            }
        )

    for _, row in missing_from_project.iterrows():
        study_uid = str(row.get("studyInstanceUid", "")).strip()
        if not study_uid or study_uid in ("nan", "None"):
            study_uid = "Unknown"
        exp_label = str(row.get("experimentLabel", "")).strip()
        if not exp_label or exp_label in ("nan", "None"):
            exp_label = "Unknown"
        patient_id = str(row.get("patientId", "")).strip()
        if not patient_id or patient_id in ("nan", "None"):
            patient_id = "Unknown"
        subject_label = str(row.get("subjectLabel", "")).strip()
        if not subject_label or subject_label in ("nan", "None"):
            subject_label = "Unknown"

        error_details.append(
            {
                "Experiment Label": str(exp_label),
                "Patient ID": str(patient_id),
                "Subject Label": str(subject_label),
                "Accession Number": str(row.get("accessionNumber", "")),
                "Study UID": study_uid,
                "Status": "Missing",
                "Reason": "Available study missing from Project",
            }
        )

    error_table_df = pd.DataFrame(error_details)

    # ---- Error distribution sub-table ----
    dist_widget = None
    if len(error_table_df) > 0:
        error_distribution = (
            error_table_df.groupby("Reason").size().reset_index(name="Count")
        )
        error_distribution = error_distribution.sort_values("Count", ascending=False).reset_index(drop=True)

        DIST_PAGE_SIZE = 5
        dist_state = {"page": 0}
        dist_table_body = HTML()
        dist_prev = Button(description="Previous", button_style="info", layout=Layout(width="100px"))
        dist_next = Button(description="Next", button_style="info", layout=Layout(width="100px"))
        dist_page_label = HTML()

        def _render_dist():
            tp = max(1, (len(error_distribution) + DIST_PAGE_SIZE - 1) // DIST_PAGE_SIZE)
            dist_state["page"] = max(0, min(dist_state["page"], tp - 1))
            start = dist_state["page"] * DIST_PAGE_SIZE
            page_data = error_distribution.iloc[start : start + DIST_PAGE_SIZE]

            dist_page_label.value = f'<span style="font-size: 12px; color: #666;">Page {dist_state["page"] + 1} of {tp}</span>'
            dist_prev.disabled = dist_state["page"] == 0
            dist_next.disabled = dist_state["page"] >= tp - 1

            rows = []
            for _, r in page_data.iterrows():
                reason_text = html_module.escape(str(r["Reason"]))
                rows.append(
                    f"<tr>"
                    f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; color: #333; text-align: left !important;">{reason_text}</td>'
                    f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; text-align: center; font-weight: bold; width: 80px; color: #333;">{r["Count"]}</td>'
                    f"</tr>"
                )

            dist_table_body.value = (
                '<table style="width: 100%; border-collapse: collapse; font-size: 12px; background: white;">'
                '<thead><tr style="background: #f8f9fa;">'
                '<th style="padding: 10px 12px; text-align: left !important; font-size: 11px; text-transform: uppercase; color: #666; border-bottom: 2px solid #ddd;">Reason</th>'
                '<th style="padding: 10px 12px; text-align: center; font-size: 11px; text-transform: uppercase; color: #666; border-bottom: 2px solid #ddd; width: 80px;">Count</th>'
                '</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table>'
            )

        dist_prev.on_click(lambda b: (dist_state.update(page=dist_state["page"] - 1), _render_dist()))
        dist_next.on_click(lambda b: (dist_state.update(page=dist_state["page"] + 1), _render_dist()))
        _render_dist()

        dist_header = HTML(
            f"""
            <div style="
                background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
                color: white; padding: 20px; border-radius: 10px 10px 0 0;
            ">
                <h3 style="margin: 0; font-size: 18px;">Study Error Distribution</h3>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 13px;">{len(error_distribution)} unique error reasons</p>
            </div>
        """
        )

        dist_pagination = HBox(
            [dist_prev, dist_page_label, dist_next],
            layout=Layout(
                padding="8px 12px",
                justify_content="flex-end",
                align_items="center",
                gap="10px",
                background="#f8f9fa",
                border_top="1px solid #eee",
                border_radius="0 0 10px 10px",
            ),
        )

        dist_widget = VBox(
            [
                dist_header,
                dist_pagination,
                VBox([dist_table_body], layout=Layout(height="230px", overflow="hidden", background="white")),
            ],
            layout=Layout(
                width="1500px",
                border_radius="10px",
                overflow="hidden",
                background="#f5f5f5",
                margin="10px 5px",
            ),
        )

    # ---- Main errors table ----
    PAGE_SIZE = 15
    state = {"page": 0, "sort_col": "Status", "sort_asc": True, "filter": ""}

    columns = [
        ("Experiment Label", "14%"),
        ("Patient ID", "9%"),
        ("Subject Label", "9%"),
        ("Accession Number", "10%"),
        ("Study UID", "30%"),
        ("Status", "6%"),
        ("Reason", "22%"),
    ]

    filter_input = Text(placeholder="Search", layout=Layout(width="300px"))
    prev_btn = Button(description="Previous", button_style="info", layout=Layout(width="100px"))
    next_btn = Button(description="Next", button_style="info", layout=Layout(width="100px"))
    page_label = HTML()
    table_body = HTML()
    header_buttons = []

    def _update_headers():
        for i, (col_name, _) in enumerate(columns):
            arrow = ""
            if state["sort_col"] == col_name:
                arrow = " \u2191" if state["sort_asc"] else " \u2193"
            header_buttons[i].description = f"{col_name}{arrow}"

    def _make_sort(col_name):
        def handler(b):
            if state["sort_col"] == col_name:
                state["sort_asc"] = not state["sort_asc"]
            else:
                state["sort_col"] = col_name
                state["sort_asc"] = True
            state["page"] = 0
            _update_headers()
            _render()

        return handler

    def _render():
        filtered = error_table_df.copy()
        if state["filter"]:
            pat = state["filter"].replace("*", ".*")
            mask = pd.Series(False, index=filtered.index)
            for col in error_table_df.columns:
                mask = mask | filtered[col].str.contains(pat, case=False, regex=True, na=False)
            filtered = filtered[mask]

        sorted_df = filtered.sort_values(
            by=state["sort_col"],
            ascending=state["sort_asc"],
            key=lambda x: x.str.lower() if x.dtype == "object" else x,
        ).reset_index(drop=True)

        tp = max(1, (len(sorted_df) + PAGE_SIZE - 1) // PAGE_SIZE)
        state["page"] = max(0, min(state["page"], tp - 1))
        start = state["page"] * PAGE_SIZE
        page_df = sorted_df.iloc[start : start + PAGE_SIZE]

        filter_text = f" (filtered: {len(filtered)})" if state["filter"] else ""
        page_label.value = (
            f'<span style="font-size: 13px; color: #666;">'
            f'Page {state["page"] + 1} of {tp} ({len(sorted_df)} issues{filter_text})</span>'
        )
        prev_btn.disabled = state["page"] == 0
        next_btn.disabled = state["page"] >= tp - 1

        if len(page_df) == 0:
            table_body.value = (
                '<div style="padding: 20px; text-align: center; color: #666; background: white;">'
                "No matching studies found.</div>"
            )
            return

        rows = []
        for _, r in page_df.iterrows():
            status = str(r["Status"])
            if status == "Error":
                bg, tc = "#fdeaea", "#721c24"
            elif status == "Rejected":
                bg, tc = "#fff8e6", "#856404"
            else:
                bg, tc = "#e8daef", "#4a235a"

            exp_label = str(r["Experiment Label"])
            pid = html_module.escape(str(r["Patient ID"]))
            slabel = html_module.escape(str(r["Subject Label"]))
            acc = html_module.escape(str(r["Accession Number"]))
            uid = html_module.escape(str(r["Study UID"]))
            stat = html_module.escape(str(r["Status"]))
            reason_trunc = _truncate_with_tooltip(str(r["Reason"]), 60)

            if exp_label and exp_label != "Unknown" and exp_label.strip() in xnat_exp_labels:
                xnat_url = f"{xnat_host}/data/archive/projects/{project_id}/experiments/{exp_label.strip()}"
                exp_display = f'<a href="{xnat_url}" target="_blank" style="color: {tc}; text-decoration: underline;">{html_module.escape(exp_label)}</a>'
            else:
                exp_display = html_module.escape(exp_label)

            rows.append(
                f'<tr style="background-color: {bg}; color: {tc};">'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 14%;">{exp_display}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 9%;">{pid}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 9%;">{slabel}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 10%;">{acc}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 30%; word-break: break-all;">{uid}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 6%;">{stat}</td>'
                f'<td style="padding: 8px 12px; border-bottom: 1px solid #eee; width: 22%;">{reason_trunc}</td>'
                f"</tr>"
            )

        thtml = (
            '<table style="width: 100%; border-collapse: collapse; font-size: 12px; '
            'table-layout: fixed; background: white; margin: 0; padding: 0;">'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )
        thtml = (
            thtml.replace("&lt;span title=", "<span title=")
            .replace('style=&quot;cursor: help;&quot;&gt;', 'style="cursor: help;">')
            .replace("&lt;/span&gt;", "</span>")
            .replace("&quot;", '"')
        )

        table_body.value = thtml

    for col_name, width in columns:
        arrow = ""
        if state["sort_col"] == col_name:
            arrow = " \u2191" if state["sort_asc"] else " \u2193"
        btn = Button(description=f"{col_name}{arrow}")
        btn.layout = Layout(width=width, height="36px", margin="0", padding="0")
        btn.style.button_color = "#333"
        btn.add_class("header-sort-button")
        btn.on_click(_make_sort(col_name))
        header_buttons.append(btn)

    filter_input.observe(lambda c: (state.update(filter=c["new"], page=0), _render()), names="value")
    prev_btn.on_click(lambda b: (state.update(page=state["page"] - 1), _render()))
    next_btn.on_click(lambda b: (state.update(page=state["page"] + 1), _render()))

    _render()

    title_widget = HTML(
        f"""
        <div style="
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white; padding: 20px; border-radius: 10px 10px 0 0;
        ">
            <h3 style="margin: 0; font-size: 18px;">Missing Studies</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 13px;">{errors} errors, {rejected} rejected, {missing_from_project_count} missing from project</p>
        </div>
    """
    )

    controls = HBox(
        [
            HBox([filter_input], layout=Layout(align_items="center")),
            HBox([prev_btn, page_label, next_btn], layout=Layout(align_items="center", gap="10px")),
        ],
        layout=Layout(
            padding="15px 20px",
            justify_content="space-between",
            align_items="center",
            background="#f5f5f5",
            width="100%",
        ),
    )

    header_row = HBox(header_buttons)
    header_row.layout = Layout(width="100%", margin="0", padding="0", background="#333", overflow="hidden")
    header_row.add_class("no-gap")

    legend = HTML(
        """
        <div style="
            padding: 10px 20px; font-size: 11px; color: #666;
            background: #f8f9fa; border-top: 1px solid #eee; border-radius: 0 0 10px 10px;
        ">
            <span style="background: #fdeaea; padding: 2px 8px; border-radius: 3px; margin-right: 15px; color: #721c24;">Error</span> Failed to process
            <span style="margin-left: 20px; background: #fff8e6; padding: 2px 8px; border-radius: 3px; margin-right: 15px; color: #856404;">Rejected</span> Rejected by site
            <span style="margin-left: 20px; background: #e8daef; padding: 2px 8px; border-radius: 3px; margin-right: 15px; color: #4a235a;">Missing</span> Available study missing from project
        </div>
    """
    )

    table_wrapper = VBox([table_body])
    table_wrapper.layout = Layout(background="white", padding="0", margin="0", width="100%", overflow="hidden")

    table_container = VBox([header_row, table_wrapper])
    table_container.layout = Layout(background="#333", padding="0", margin="0", width="100%", overflow="hidden")
    table_container.add_class("no-gap")

    errors_widget = VBox([css_widget, title_widget, controls, table_container, legend])
    errors_widget.layout = Layout(
        width="1500px", border_radius="10px", overflow="hidden", background="#f5f5f5", margin="10px 5px"
    )
    errors_widget.add_class("no-gap")

    if dist_widget is not None:
        return VBox([dist_widget, errors_widget])
    return errors_widget


# ---------------------------------------------------------------------------
# Orchestrate — the async pipeline
# ---------------------------------------------------------------------------

async def orchestrate(
    status_html,
    patient_cards_box,
    study_cards_box,
    charts_box,
    series_banner_box,
    missing_patients_box,
    additional_studies_box,
    errors_missing_box,
    missing_series_box,
):
    """Async pipeline: connect, load, compute, render progressively."""
    xnat_host = os.environ.get("XNAT_HOST")
    xnat_user = os.environ.get("XNAT_USER")
    xnat_pass = os.environ.get("XNAT_PASS")
    project_id = os.environ.get("XNAT_PROJECT", "YOUR_PROJECT_ID")

    loop = asyncio.get_event_loop()
    session = None
    jsession_id = None

    try:
        # Phase 1: Connect
        status_html.value = _status_msg("Connecting to XNAT...")

        session, jsession_id = await loop.run_in_executor(
            None, lambda: connect_xnat(xnat_host, xnat_user, xnat_pass)
        )

        # Phase 2: Load studies + metadata in parallel
        status_html.value = _status_msg("Loading studies and metadata...")

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_studies = loop.run_in_executor(pool, lambda: load_studies(session, project_id))
            future_meta = loop.run_in_executor(pool, lambda: load_xnat_metadata(session, project_id))
            study_df, series_df = await future_studies
            xnat_experiments, xnat_subjects, exp_map, exp_xsi_map, xnat_exp_labels = await future_meta

        # Phase 3: Compute stats
        status_html.value = _status_msg("Computing statistics...")

        stats = await loop.run_in_executor(
            None,
            lambda: compute_stats(
                study_df, xnat_experiments, xnat_subjects, exp_map, exp_xsi_map, xnat_exp_labels
            ),
        )

        if stats.get("empty"):
            status_html.value = _status_msg("No study data available.", color="#856404", icon="\u26a0")
            return

        # Phase 4: Stat cards
        patient_cards_box.children = build_patient_cards(stats)
        study_cards_box.children = build_study_cards(stats)
        await asyncio.sleep(0)  # flush widget updates to frontend

        # Phase 5: Charts
        status_html.value = _status_msg("Building charts...")
        charts_widget, session_b64 = build_session_type_chart(stats)
        charts_box.children = [charts_widget]
        await asyncio.sleep(0)

        # Phase 6: Series banner
        series_banner, missing_series_output = build_series_banner(
            stats, charts_widget, session_b64, xnat_host, xnat_user, xnat_pass, project_id
        )
        series_banner_box.children = [series_banner]
        await asyncio.sleep(0)

        # Phase 7: Detail tables
        status_html.value = _status_msg("Building tables...")
        await asyncio.sleep(0)

        mp_widget = build_missing_patients_table(stats)
        if mp_widget is not None:
            missing_patients_box.children = [mp_widget]
            await asyncio.sleep(0)

        add_widget = build_additional_studies_table(stats, xnat_host, project_id)
        if add_widget is not None:
            additional_studies_box.children = [add_widget]
            await asyncio.sleep(0)

        err_widget = build_errors_missing_table(stats, xnat_host, project_id)
        if err_widget is not None:
            errors_missing_box.children = [err_widget]
            await asyncio.sleep(0)

        # Missing series placeholder (populated after button click)
        missing_series_box.children = [missing_series_output]
        await asyncio.sleep(0)

        # Phase 8: Done
        total = stats["total"]
        subj = stats["project_subject_count"]
        status_html.value = _status_msg(
            f"Loaded {total:,} studies, {subj:,} subjects.",
            color="#155724",
            icon="&#10003;",
        )

    except Exception as exc:
        status_html.value = (
            f'<div style="background: #fdeaea; border: 1px solid #e74c3c; border-radius: 6px; '
            f'padding: 15px; margin: 10px 0; color: #721c24;">'
            f"<strong>Error:</strong> <code>{type(exc).__name__}: {exc}</code></div>"
        )

    finally:
        if session is not None and jsession_id is not None:
            try:
                await loop.run_in_executor(
                    None, lambda: disconnect_xnat(session, xnat_host, jsession_id)
                )
            except Exception:
                pass
