# app.py
import os
import streamlit as st
import pandas as pd
import json
import random
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
# Import Gemini client
# -----------------------------
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel, configure
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

if GENAI_AVAILABLE:
    try:
        configure(api_key=GEMINI_API_KEY)
    except Exception:
        # Some client versions may not require configure; ignore non-fatal errors
        pass

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="VenomVerse – Antivenom Research Copilot",
                   page_icon="🐍",
                   layout="wide")
st.title("🐍 VenomVerse – Antivenom Research Copilot")
st.subheader("AI-assisted venom research hypotheses (prototype)")
st.markdown("""
<div style='padding:12px; background:#ffdddd; border-radius:8px; border:1px solid #cc0000'>
<b>⚠️ PROTOTYPE ONLY · NOT MEDICAL OR CLINICAL ADVICE · AI-GENERATED HYPOTHESES</b>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Data loading
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
    st.error(f"Missing CSV: {e}. Place venoms.csv and antivenoms.csv inside the `data/` folder.")
    st.stop()

# -----------------------------
# Helper: parse model output
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

# -----------------------------
# Analysis function
# -----------------------------
def analyze_species(species_common: str) -> dict:
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

    # call Gemini model
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

    required_keys = {"venom_analysis", "top_prediction", "synthetic_protein", "market_analysis", "research_abstract"}
    if not required_keys.issubset(set(parsed.keys())):
        raise RuntimeError("Model output missing required keys.")

    return parsed

# -----------------------------
# 3D / PDB helpers
# -----------------------------
def make_mock_pdb(sequence: str) -> str:
    lines = []
    seq = (sequence or "").strip().upper()
    if not seq:
        seq = "AAAAAAAAAAAA"
    max_len = min(len(seq), 60)
    a = 1.5
    r = 2.0
    for i in range(max_len):
        aa = seq[i]
        theta = i * (100.0 * math.pi / 180.0)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = i * a
        atom_serial = i + 1
        res_seq = i + 1
        resname = aa if len(aa) == 1 else aa[0]
        line = f"ATOM  {atom_serial:5d}  CA  {resname:>3s} A{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        lines.append(line)
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines)

def render_3dmol_from_pdb(pdb_string: str, width: int = 700, height: int = 500, style: str = "cartoon"):
    b64 = base64.b64encode(pdb_string.encode("utf-8")).decode("ascii")
    element_id = f"viewer-{uuid.uuid4().hex}"
    html_template = f"""
    <div id="{element_id}" style="width: {width}px; height: {height}px; position: relative; background:#ffffff; border:1px solid #ddd;"></div>
    <script>
    (function() {{
      function loadScript(url, onload, onerror) {{
        var s = document.createElement('script');
        s.src = url;
        s.async = true;
        s.onload = onload;
        s.onerror = onerror;
        document.head.appendChild(s);
      }}
      const cdns = [
        "https://3dmol.csb.pitt.edu/build/3Dmol-min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/1.8.0/3Dmol-min.js"
      ];
      let attempt = 0;
      function tryLoad() {{
        if (attempt >= cdns.length) {{
          console.warn("3Dmol failed to load from CDNs. Interactive viewer may be blocked.");
          const el = document.getElementById("{element_id}");
          el.innerHTML = "<div style='padding:12px;color:#b00;'>Interactive 3D viewer failed to load. See browser console for details.</div>";
          return;
        }}
        const url = cdns[attempt++];
        loadScript(url, initViewer, tryLoad);
      }}
      function initViewer() {{
        try {{
          setTimeout(function() {{
            const el = document.getElementById("{element_id}");
            el.innerHTML = "";
            const pdb = atob("{b64}");
            const config = {{ backgroundColor: "0xeeeeee" }};
            const viewer = $3Dmol.createViewer(el, config);
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
            window.addEventListener('resize', function() {{ viewer.resize(); viewer.render(); }});
            console.info("3Dmol viewer initialized.");
          }}, 50);
        }} catch (err) {{
          console.error("3Dmol init error:", err);
          tryLoad();
        }}
      }}
      tryLoad();
    }})();
    </script>
    """
    return html_template

# -----------------------------
# Exports
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
# Sidebar & controls
# -----------------------------
st.sidebar.header("Select Species")
species_selected = st.sidebar.selectbox("Species", venom_df["species_common"].tolist())
st.sidebar.info(f"Using Gemini model: **{GEMINI_MODEL_NAME}**")
run_button = st.sidebar.button("Analyze Venom", type="primary")

# -----------------------------
# Main app flow
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

    # Top Research Area display (wrapped)
    with c1:
        st.markdown("**Top Research Area**")
        st.markdown(
            f"<div style='white-space:normal; word-wrap:break-word; line-height:1.2;'>{top_area}</div>",
            unsafe_allow_html=True
        )
        st.caption(f"Confidence (qualitative): {confidence_pct}")

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

    # Tab 1 — safe plotting
    with tab1:
        st.subheader("Hypothetical Venom Mechanisms")
        mech = result["venom_analysis"].get("key_mechanisms", [])
        notes = result["venom_analysis"].get("notes", "")
        if mech is None:
            mech = []
        mech = [str(m).strip() for m in mech if str(m).strip()]
        if len(mech) == 0:
            st.info("No discrete mechanisms returned by the model to plot. See notes below.")
            st.write("**Notes:**")
            st.write(notes or "_No additional notes provided._")
        else:
            # deterministic simple relevance 1-5
            relevance = []
            for i in range(len(mech)):
                val = 3 + ((i * 37) % 3)
                val = max(1, min(5, int(val)))
                relevance.append(val)
            df_plot = pd.DataFrame({"Mechanism": mech, "Relevance Score": relevance})
            try:
                import altair as alt
                chart = (
                    alt.Chart(df_plot)
                    .mark_bar()
                    .encode(
                        x=alt.X("Mechanism:N", sort='-y', title="Mechanism"),
                        y=alt.Y("Relevance Score:Q", scale=alt.Scale(domain=[0, 5.5]), title="Relevance (1–5)")
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                st.bar_chart(df_plot.set_index("Mechanism"))
            st.write("**Notes:**")
            st.write(notes or "_No additional notes provided._")

    # Tab 2 — peptide info
    with tab2:
        st.subheader("Hypothetical Peptide Concept (AI-generated, not validated)")
        syn = result.get("synthetic_protein", {})
        sequence = syn.get("sequence", "").strip()
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

    # Tab 3 — 3D + static fallback + pdb download
    with tab3:
        st.subheader("Simulated 3D Structure (conceptual)")
        st.info("This is a simulated visualization created from a simple geometric backbone. NOT a validated experimental structure.")

        # parse CA coords for static fallback
        def parse_pdb_ca_coords(pdb_text):
            xs, ys, zs = [], [], []
            for line in pdb_text.splitlines():
                if line.startswith("ATOM") and " CA " in line:
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        xs.append(x); ys.append(y); zs.append(z)
                    except Exception:
                        continue
            return xs, ys, zs

        xs, ys, zs = parse_pdb_ca_coords(mock_pdb)

        # static PNG fallback
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            import io
            fig = plt.figure(figsize=(6,4.5))
            ax = fig.add_subplot(111, projection='3d')
            if xs and ys and zs:
                ax.plot(xs, ys, zs, linewidth=2, alpha=0.9)
                ax.scatter(xs, ys, zs, s=50)
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

            st.markdown("**Static conceptual rendering (fallback if interactive viewer blocked):**")
            st.image(png_bytes, use_column_width=False)
            st.download_button("⬇ Download conceptual PNG", png_bytes, file_name=f"{species_selected}_conceptual_peptide.png", mime="image/png")
        except Exception as e:
            st.warning("Static fallback rendering failed: " + str(e))

        # Show and download PDB
        with st.expander("Show conceptual PDB text (click to expand)"):
            st.code(mock_pdb)
        st.download_button("⬇ Download conceptual PDB (text)", mock_pdb, file_name=f"{species_selected}_conceptual_peptide.pdb", mime="text/plain")

    # Tab 4 — research priority
    with tab4:
        st.subheader("Research Priority")
        st.write(f"**Priority:** {result['market_analysis'].get('priority', 'N/A')}")
        st.write(result['market_analysis'].get('qualitative_summary', ''))

    # Tab 5 — abstract + exports
    with tab5:
        st.subheader("AI-Generated Abstract")
        st.write(result.get("research_abstract", ""))
        st.markdown("---")
        st.subheader("Export")
        st.download_button("⬇ Download JSON", export_json(result), file_name=f"{species_selected}_analysis.json", mime="application/json")
        st.download_button("⬇ Download TXT Summary", export_txt(species_selected, result), file_name=f"{species_selected}_summary.txt", mime="text/plain")
        st.caption("Prototype export — not for scientific/clinical use.")

