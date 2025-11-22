import streamlit as st
import pandas as pd
import json
import random
import os
from pathlib import Path
from datetime import datetime
import math
import html
import base64
import uuid

# -----------------------------
# Load Gemini key securely
# -----------------------------
# Prefer Streamlit secrets (deployed) then environment variable (local dev)
GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Add it to Streamlit Secrets (recommended) or set environment variable locally.")
    st.stop()

# Fixed model name (your working model)
GEMINI_MODEL_NAME = "models/gemini-2.5-flash-lite-preview-09-2025"
# -----------------------------
# IMPORT GENERATIVE CLIENT
# -----------------------------
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel, configure
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# -----------------------------
# STREAMLIT PAGE SETUP
# -----------------------------
st.set_page_config(page_title="VenomVerse – Antivenom Research Copilot",
                   page_icon="🐍",
                   layout="wide")
st.title("🐍 VenomVerse – Antivenom Research Copilot")
st.subheader("AI-assisted venom research hypotheses (prototype)")

st.markdown("""
<div style='padding:12px; background:#ADD8E6; border-radius:8px; border:1px solid #cc0000'>
<b>⚠️ PROTOTYPE ONLY · NOT MEDICAL OR CLINICAL ADVICE · AI-GENERATED HYPOTHESES</b>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SANITY CHECK KEY
# -----------------------------
if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("<PUT_"):
    st.sidebar.error("❌ Please edit app.py and insert your GEMINI_API_KEY.")
    st.stop()

if GENAI_AVAILABLE:
    try:
        configure(api_key=GEMINI_API_KEY)
    except Exception:
        # Non-fatal; actual calls will reveal issues
        pass

# -----------------------------
# ACTIVITY / TRANSITION LOG HELPERS
# -----------------------------
# Moved above authentication so authentication events can be logged.
if "activity_log" not in st.session_state:
    # store as list of dicts: [{"time": "...", "user":"...", "event":"..."}]
    st.session_state.activity_log = []

def current_time_str():
    # local server time string (YYYY-MM-DD HH:MM:SS)
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_activity(event: str, user: str = None):
    """Append an activity entry and keep max 200 entries to avoid growing state unlimited."""
    entry = {"time": current_time_str(), "user": user or st.session_state.get("username", "Unknown"), "event": event}
    st.session_state.activity_log.insert(0, entry)  # newest first
    # cap log length
    if len(st.session_state.activity_log) > 200:
        st.session_state.activity_log = st.session_state.activity_log[:200]


# -----------------------------
# SIMPLE NAME AUTHENTICATION
# -----------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 VenomVerse Access Portal")

    username = st.text_input("Enter your name to access the portal:")

    if st.button("Enter Portal"):
        if username.strip() != "":
            st.session_state.authenticated = True
            st.session_state.username = username
            # log authentication
            log_activity("User entered portal (authentication)", user=username)
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Please enter your name to continue.")

    st.stop()  # Prevent the main app from loading


# -----------------------------
# LOAD CSV DATA
# -----------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

try:
    venom_df = load_csv(DATA_DIR / "venoms.csv")
    antivenom_df = load_csv(DATA_DIR / "antivenoms.csv")
except FileNotFoundError as e:
    st.error(f"Missing CSV: {e}")
    st.stop()

# -----------------------------
# MODEL CALL + PARSING HELPERS
# -----------------------------
def _extract_json_from_text(text: str) -> str:
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError("No JSON found in model output.")
    return text[first:last+1]

def _parse_gemini_response(resp):
    if hasattr(resp, "text"):
        return resp.text
    if isinstance(resp, dict):
        for key in ("text", "content", "output", "result"):
            if key in resp:
                return str(resp[key])
        return json.dumps(resp)
    return str(resp)

def analyze_species(species_common: str) -> dict:
    """
    Call the fixed Gemini model and return parsed JSON dict according to the schema.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-generativeai is not installed. Run: pip install google-generativeai")

    vrow = venom_df[venom_df["species_common"] == species_common]
    if vrow.empty:
        raise ValueError("Species not found.")
    venom_info = vrow.iloc[0].to_dict()
    arow = antivenom_df[antivenom_df["species_scientific"] == venom_info["species_scientific"]]
    antivenom_info = arow.iloc[0].to_dict() if not arow.empty else {}

    context = {"query_species": species_common, "venom": venom_info, "antivenom": antivenom_info}

    prompt = f"""
You are a scientific hypothesis generator for venom research.

RULES:
- Provide ONLY hypothetical, qualitative, non-clinical analysis.
- DO NOT include statistics, clinical claims, PDB IDs, or real economic figures.
- Return ONLY a JSON object in EXACTLY this schema (no extra commentary):

{{
  "venom_analysis": {{
    "key_mechanisms": ["..."],
    "related_pathways": ["..."],
    "notes": "..."
  }},
  "top_prediction": {{
    "research_direction": "string",
    "confidence": 0.0
  }},
  "synthetic_protein": {{
    "hypothetical_peptide_name": "string",
    "sequence": "string (<=20 aa)",
    "notes": "string"
  }},
  "market_analysis": {{
    "priority": "low/medium/high",
    "qualitative_summary": "string"
  }},
  "research_abstract": "string (AI-generated, hypothesis only)."
}}

Context:
{json.dumps(context, indent=2)}
"""

    model = GenerativeModel(GEMINI_MODEL_NAME)
    try:
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 800, "temperature": 0.2})
    except TypeError:
        response = model.generate(prompt, max_output_tokens=800, temperature=0.2)
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")

    raw_text = _parse_gemini_response(response)

    try:
        json_text = _extract_json_from_text(raw_text)
        parsed = json.loads(json_text)
    except Exception as e:
        preview = raw_text[:800].replace("\n", " ")
        raise RuntimeError(f"JSON parse error. Model preview: {preview}") from e

    # minimal validation
    required_keys = {"venom_analysis", "top_prediction", "synthetic_protein", "market_analysis", "research_abstract"}
    if not required_keys.issubset(set(parsed.keys())):
        raise RuntimeError("Model output missing required keys.")

    return parsed

# -----------------------------
# 3D MOCK PDB GENERATOR
# -----------------------------
def make_mock_pdb(sequence: str) -> str:
    """
    Generate a simple conceptual PDB string with CA atoms along a helix/spiral
    based on the sequence length. This is a purely conceptual placeholder
    for visualization only.
    """
    lines = []
    seq = sequence.strip().upper()
    # limit length for visualization
    max_len = min(len(seq), 60)
    a = 1.5  # rise per residue (Å)
    r = 2.0  # radius of helix
    # Use simple helix angles
    for i in range(max_len):
        aa = seq[i]
        # helix parameters
        theta = i * (100.0 * math.pi / 180.0)  # 100 degrees per residue approx
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = i * a
        atom_serial = i + 1
        res_seq = i + 1
        # Use CA atom line
        # Format: "ATOM  {serial:5d}  CA  {resname:>3s} A{resSeq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        resname = aa if len(aa) == 1 else aa[0]
        line = f"ATOM  {atom_serial:5d}  CA  {resname:>3s} A{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        lines.append(line)
    # end marker
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines)

# -----------------------------
# 3DMOL HTML EMBED HELPER
# -----------------------------
def render_3dmol_from_pdb(pdb_string: str, width: int = 700, height: int = 500, style: str = "cartoon"):
    """
    Return HTML that embeds 3Dmol.js viewer with the given PDB string.
    style: 'cartoon', 'stick', 'sphere' etc.
    Uses base64 encoding to safely pass the PDB into the iframe script and
    generates a unique element id to avoid collisions. Adds JS error handling.
    """
    # safe base64 encode the PDB text
    b64 = base64.b64encode(pdb_string.encode("utf-8")).decode("ascii")
    element_id = f"viewer-{uuid.uuid4().hex}"
    html_template = f"""
    <div id="{element_id}" style="width: {width}px; height: {height}px; position: relative; background:#ffffff;"></div>
    <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <script>
    (function() {{
        const element = document.getElementById("{element_id}");
        element.innerHTML = "";
        try {{
            const pdb = atob("{b64}");
            const config = {{ backgroundColor: "0xeeeeee" }};
            const viewer = $3Dmol.createViewer(element, config);
            viewer.addModel(pdb, "pdb");
            if ("{style}" === "stick") {{
                viewer.setStyle({{}}, {{stick:{{radius:0.2}}}});
            }} else if ("{style}" === "sphere") {{
                viewer.setStyle({{}}, {{sphere:{{scale:0.5}}}});
            }} else {{
                viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            }}
            viewer.zoomTo();
            viewer.render();
        }} catch (err) {{
            console.error("3Dmol render error:", err);
            element.innerHTML = "<div style='color:#900;padding:10px;'>3Dmol render error: " + (err && err.message ? err.message : String(err)) + "</div>";
        }}
    }})();
    </script>
    """
    return html_template

# -----------------------------
# EXPORT HELPERS
# -----------------------------
def export_json(data: dict):
    return json.dumps(data, indent=2)

def export_txt(species, d):
    return f"""VenomVerse Prototype Output
Species: {species}

Top Research Direction:
{d['top_prediction'].get('research_direction', '')}

Research Priority:
{d['market_analysis'].get('priority', '')}

Abstract (snippet):
{d.get('research_abstract','')[:250]}...

Generated {datetime.utcnow().isoformat()}Z
PROTOTYPE ONLY · NOT MEDICAL ADVICE
"""

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Select Species")
species_selected = st.sidebar.selectbox("Species", venom_df["species_common"].tolist())

# Log species selection change
if st.session_state.get("last_species_selected") != species_selected:
    st.session_state["last_species_selected"] = species_selected
    log_activity(f"Selected species: {species_selected}")

st.sidebar.info(f"Using Gemini model: **{GEMINI_MODEL_NAME}**")
run_button = st.sidebar.button("Analyze Venom", type="primary")

# Log analyze button press (this records immediate press; actual generation handled later)
if run_button:
    log_activity("Pressed 'Analyze Venom' button")


# -----------------------------
# SIDEBAR ACTIVITY LOG DISPLAY
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔔 Activity Log")
if st.session_state.activity_log:
    # show latest 10 entries
    for e in st.session_state.activity_log[:10]:
        user_str = f" ({e['user']})" if e.get("user") else ""
        st.sidebar.markdown(f"- **{e['time']}**: {e['event']}{user_str}")
else:
    st.sidebar.info("No activity yet.")
st.sidebar.markdown("---")


# -----------------------------
# MAIN APP FLOW
# -----------------------------
if run_button:
    with st.spinner("Generating hypothesis…"):
        try:
            result = analyze_species(species_selected)
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.stop()

    # Safe extraction
    top_area = result["top_prediction"].get("research_direction", "—")
    confidence = result["top_prediction"].get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    confidence_pct = f"{confidence*100:.1f}%"
    try:
        antivenom_avail = antivenom_df[
            antivenom_df["species_scientific"] == venom_df[venom_df["species_common"] == species_selected].iloc[0]["species_scientific"]
        ].iloc[0]["antivenom_available"]
    except Exception:
        antivenom_avail = "Unknown"

        # Metrics — improved Top Research Area wrapping
    st.markdown("### 📊 Summary Metrics")
    c1, c2, c3, c4 = st.columns([2,1,1,1])  # make the first column wider for long text

    # Top Research Area: show a wrapped, multi-line block (not metric) so long text doesn't get cut
    with c1:
        st.markdown("**Top Research Area**")
        st.markdown(
            f"<div style='white-space:normal; word-wrap:break-word; line-height:1.2;'>{top_area}</div>",
            unsafe_allow_html=True
        )
        # optionally show a small subtitle about confidence
        st.caption(f"Confidence (qualitative): {confidence_pct}")

    # other metrics kept compact
    c2.metric("Confidence", confidence_pct)
    c3.metric("Antivenom Available", antivenom_avail)
    c4.metric("Prototype Status", "Conceptual Only")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Venom & Disease Hypotheses",
        "Hypothetical Peptide",
        "Simulated 3D Structure",
        "Research Priority",
        "Abstract + Export"
    ])

    with tab1:
        st.subheader("Hypothetical Venom Mechanisms")
        mech = result["venom_analysis"].get("key_mechanisms", [])
        if mech:
            df = pd.DataFrame({"Mechanism": mech, "Relevance Score": [random.randint(1,5) for _ in mech]})
            st.bar_chart(df.set_index("Mechanism"))
        st.write("**Notes:**")
        st.write(result["venom_analysis"].get("notes", ""))

    with tab2:
        st.subheader("Hypothetical Peptide Concept (AI-generated, not validated)")
        syn = result.get("synthetic_protein", {})
        sequence = syn.get("sequence", "").strip()
        # If model returned empty sequence, create a conceptual random peptide
        if not sequence:
            aa_choices = list("ACDEFGHIKLMNPQRSTVWY")
            sequence = "".join(random.choice(aa_choices) for _ in range(12))
            syn["sequence"] = sequence
            syn["hypothetical_peptide_name"] = syn.get("hypothetical_peptide_name", "ConceptualPeptide-1")
            syn["notes"] = syn.get("notes", "Conceptual peptide sequence generated for visualization; not validated.")
        st.markdown("**Peptide sequence (conceptual):**")
        st.code(sequence)
        st.table(pd.DataFrame([{
            "Name": syn.get("hypothetical_peptide_name", ""),
            "Sequence": sequence,
            "Notes": syn.get("notes", "")
        }]))

    with tab3:
            st.subheader("Simulated DNA-like Structure (conceptual)")
            st.info("Conceptual double-helix visualization only — not experimental or validated.")

            # ---------- DNA PDB generator ----------
            def make_mock_dna_pdb(n_bp: int = 20, rise: float = 3.4, twist_deg: float = 36.0, radius: float = 10.0):
                """
                Build a conceptual PDB of a DNA-like double helix with n_bp base pairs.
                Geometric only — NOT real DNA coordinates.
                """
                lines = []
                twist = math.radians(twist_deg)
                atom_serial = 1

                bases = ["DA", "DT", "DG", "DC"]

                for i in range(n_bp):
                    theta = i * twist
                    z = i * rise

                    # Strand A
                    xA = radius * math.cos(theta)
                    yA = radius * math.sin(theta)

                    # Strand B (180° opposite)
                    xB = radius * math.cos(theta + math.pi)
                    yB = radius * math.sin(theta + math.pi)

                    res_seq = i + 1
                    baseA = bases[i % 4]
                    baseB = bases[(i + 1) % 4]

                    lineA = f"ATOM  {atom_serial:5d}  P   {baseA} A{res_seq:4d}    {xA:8.3f}{yA:8.3f}{z:8.3f}  1.00 20.00           P"
                    atom_serial += 1
                    lineB = f"ATOM  {atom_serial:5d}  P   {baseB} B{res_seq:4d}    {xB:8.3f}{yB:8.3f}{z:8.3f}  1.00 20.00           P"
                    atom_serial += 1

                    lines.append(lineA)
                    lines.append(lineB)

                lines.append("TER")
                lines.append("END")
                return "\n".join(lines)

            # Build conceptual DNA PDB
            dna_pdb = make_mock_dna_pdb(n_bp=24)

            # ---------- Parse coordinates ----------
            def parse_pdb_coords(pdb_text):
                xs, ys, zs = [], [], []
                for line in pdb_text.splitlines():
                    if line.startswith("ATOM"):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            xs.append(x); ys.append(y); zs.append(z)
                        except:
                            continue
                return xs, ys, zs

            xs, ys, zs = parse_pdb_coords(dna_pdb)

            # ---------- STATIC PNG VISUALIZATION ----------
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                import io

                fig = plt.figure(figsize=(7,5))
                ax = fig.add_subplot(111, projection='3d')

                if xs and ys and zs:
                    # strand A (even indices)
                    ax.plot(xs[0::2], ys[0::2], zs[0::2], linewidth=2, label="Strand A")
                    ax.scatter(xs[0::2], ys[0::2], zs[0::2], s=30)

                    # strand B (odd indices)
                    ax.plot(xs[1::2], ys[1::2], zs[1::2], linewidth=2, label="Strand B")
                    ax.scatter(xs[1::2], ys[1::2], zs[1::2], s=30)

                    # Connect base pairs
                    for i in range(0, len(xs), 2):
                        ax.plot(
                            [xs[i], xs[i+1]],
                            [ys[i], ys[i+1]],
                            [zs[i], zs[i+1]],
                            color="gray",
                            linewidth=1,
                            alpha=0.6
                        )

                    ax.view_init(elev=20, azim=120)
                    ax.set_axis_off()
                else:
                    ax.text(0.5, 0.5, 0.5, "No coordinates parsed", horizontalalignment='center')
                    ax.set_axis_off()

                buf = io.BytesIO()
                plt.tight_layout()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                buf.seek(0)
                png_bytes = buf.read()

                st.markdown("**Static conceptual double-helix rendering:**")
                st.image(png_bytes)

                _download_started = st.download_button(
                    "⬇ Download conceptual DNA PNG",
                    png_bytes,
                    file_name=f"{species_selected}_conceptual_dna.png",
                    mime="image/png"
                )
                if _download_started:
                    log_activity("Downloaded conceptual DNA PNG", user=st.session_state.get("username"))
            except Exception as e:
                st.warning("Static DNA rendering failed: " + str(e))

            # ---------- Show & download PDB ----------
            with st.expander("Show conceptual DNA PDB (click to expand)"):
                st.code(dna_pdb)

            _download_started = st.download_button(
                "⬇ Download conceptual DNA PDB (text)",
                dna_pdb,
                file_name=f"{species_selected}_conceptual_dna.pdb",
                mime="text/plain"
            )
            if _download_started:
                log_activity("Downloaded conceptual DNA PDB", user=st.session_state.get("username"))

    with tab4:
        st.subheader("Research Priority")
        st.write(f"**Priority:** {result['market_analysis'].get('priority', 'N/A')}")
        st.write(result['market_analysis'].get('qualitative_summary', ''))

    with tab5:
        st.subheader("AI-Generated Abstract")
        st.write(result.get("research_abstract", ""))
        st.markdown("---")
        st.subheader("Export")
        _download_started = st.download_button("⬇ Download JSON", export_json(result), file_name=f"{species_selected}_analysis.json", mime="application/json")
        if _download_started:
            log_activity("Downloaded JSON export", user=st.session_state.get("username"))

        _download_started = st.download_button("⬇ Download TXT Summary", export_txt(species_selected, result), file_name=f"{species_selected}_summary.txt", mime="text/plain")
        if _download_started:
            log_activity("Downloaded TXT summary", user=st.session_state.get("username"))

        st.caption("Prototype export — not for scientific/clinical use.")
