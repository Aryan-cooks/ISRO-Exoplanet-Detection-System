import streamlit as st
import lightkurve as lk
import numpy as np
from astropy.timeseries import BoxLeastSquares
import batman
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import streamlit.components.v1 as components

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Exoplanet Detection Pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
    background: transparent !important;
    color: #c8d8f0;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: rgba(7, 11, 20, 0.8) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

[data-testid="stSidebar"] {
    position: relative !important;
}

[data-testid="stSidebarUserContent"] {
    padding-bottom: 120px !important;
}

div.element-container:has(#sidebar-footer) {
    position: absolute !important;
    bottom: 2rem !important;
    left: 1.5rem !important;
    width: calc(100% - 3rem) !important;
    margin-top: 0 !important;
    padding-bottom: 0 !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

/* ── Disable Fullscreen Feature ── */
button[title="Fullscreen"],
button[aria-label="Fullscreen"],
button[title="View fullscreen"],
button[aria-label="View fullscreen"],
button[aria-label*="ullscreen"],
button[title*="ullscreen"],
[title="Fullscreen"],
[data-testid="StyledFullScreenButton"],
[data-testid="stFullScreenButton"],
[data-testid="stImageFullScreenButton"],
[data-testid="stImage"] button,
[data-testid="stPyplot"] button,
.st-emotion-cache-12w0qpk,
.st-emotion-cache-1yxdwbs {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

</style>
""", unsafe_allow_html=True)

# ── ANIMATED BACKGROUND (SideRays) ───────────────────────────
components.html("""
<script type="module">
  import { Renderer, Program, Triangle, Mesh } from 'https://unpkg.com/ogl?module';

  if (!window.parent.document.getElementById('siderays-canvas')) {
      const container = window.parent.document.createElement('div');
      container.id = 'siderays-canvas';
      container.style.position = 'fixed';
      container.style.top = '0';
      container.style.left = '0';
      container.style.width = '100vw';
      container.style.height = '100vh';
      container.style.zIndex = '-1';
      container.style.pointerEvents = 'none';
      container.style.overflow = 'hidden';
      container.style.backgroundColor = '#05080e';
      window.parent.document.body.prepend(container);

      // Add a custom floating button to reopen the sidebar (nav bar) if native one is hidden
      if (!window.parent.document.getElementById('custom-nav-open-btn')) {
          const navBtn = window.parent.document.createElement('button');
          navBtn.id = 'custom-nav-open-btn';
          navBtn.innerHTML = '☰ Open Config';
          Object.assign(navBtn.style, {
              position: 'fixed',
              top: '20px',
              left: '20px',
              zIndex: '999999',
              background: 'rgba(24, 24, 27, 0.6)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              borderRadius: '12px',
              color: '#f4f4f5',
              padding: '10px 16px',
              fontSize: '0.9rem',
              fontWeight: '500',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              display: 'none',
              boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              fontFamily: 'Inter, sans-serif'
          });
          navBtn.onmouseover = () => { navBtn.style.background = 'rgba(39, 39, 42, 0.8)'; navBtn.style.borderColor = 'rgba(255, 255, 255, 0.3)'; navBtn.style.transform = 'scale(1.05)'; };
          navBtn.onmouseout = () => { navBtn.style.background = 'rgba(24, 24, 27, 0.6)'; navBtn.style.borderColor = 'rgba(255, 255, 255, 0.15)'; navBtn.style.transform = 'scale(1)'; };
          navBtn.onclick = () => {
              const nativeBtn = window.parent.document.querySelector('[data-testid="collapsedControl"]') || window.parent.document.querySelector('[data-testid="stSidebarCollapsedControl"]');
              if (nativeBtn) {
                  nativeBtn.click();
              } else {
                  const header = window.parent.document.querySelector('header');
                  if (header) {
                      const buttons = header.querySelectorAll('button');
                      if (buttons.length > 0) buttons[0].click();
                  }
              }
          };
          window.parent.document.body.appendChild(navBtn);
          
          setInterval(() => {
              const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
              let isClosed = false;
              if (!sidebar) {
                  isClosed = true;
              } else {
                  const style = window.parent.getComputedStyle(sidebar);
                  const aria = sidebar.getAttribute('aria-expanded');
                  if (aria === 'false' || style.width === '0px' || style.transform.includes('-')) {
                      isClosed = true;
                  }
              }
              
              if (isClosed) {
                  navBtn.style.display = 'flex';
                  navBtn.style.alignItems = 'center';
                  navBtn.style.gap = '8px';
              } else {
                  navBtn.style.display = 'none';
              }
          }, 500);
      }

      // Add particle attract effect to Streamlit buttons
      setInterval(() => {
          const buttons = window.parent.document.querySelectorAll('.stButton button, [data-testid="stDownloadButton"] button');
          buttons.forEach(btn => {
              if (!btn.dataset.attractBound) {
                  btn.dataset.attractBound = "true";
                  
                  // Make sure button can contain absolute elements relative to its center, 
                  // but we want particles to spawn outside so we keep overflow visible
                  btn.style.overflow = "visible";
                  
                  // Container for particles
                  const pContainer = window.parent.document.createElement("div");
                  Object.assign(pContainer.style, {
                      position: "absolute",
                      top: "50%",
                      left: "50%",
                      width: "0px",
                      height: "0px",
                      pointerEvents: "none",
                      zIndex: "0"
                  });
                  btn.insertBefore(pContainer, btn.firstChild);

                  // Ensure text stays above particles
                  const textSpans = btn.querySelectorAll("div, p, span");
                  textSpans.forEach(s => { s.style.position = "relative"; s.style.zIndex = "1"; });

                  const particleCount = 12;
                  const particles = [];
                  for (let i = 0; i < particleCount; i++) {
                      const p = window.parent.document.createElement("div");
                      const initX = Math.random() * 360 - 180;
                      const initY = Math.random() * 360 - 180;
                      Object.assign(p.style, {
                          position: "absolute",
                          width: "6px",
                          height: "6px",
                          backgroundColor: "#c4b5fd",
                          borderRadius: "50%",
                          opacity: "0",
                          transform: `translate(${initX}px, ${initY}px)`,
                          transition: "all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)"
                      });
                      pContainer.appendChild(p);
                      particles.push({ el: p, initX, initY });
                  }

                  btn.addEventListener("mouseenter", () => {
                      particles.forEach(p => {
                          p.el.style.opacity = "1";
                          p.el.style.transform = "translate(-3px, -3px)"; // Move to center
                      });
                  });
                  
                  btn.addEventListener("mouseleave", () => {
                      particles.forEach(p => {
                          p.el.style.opacity = "0";
                          p.el.style.transform = `translate(${p.initX}px, ${p.initY}px)`;
                      });
                  });
              }
          });
      }, 1000);

      const renderer = new Renderer({
          dpr: Math.min(window.devicePixelRatio, 2),
          alpha: true
      });
      const gl = renderer.gl;
      gl.canvas.style.width = '100%';
      gl.canvas.style.height = '100%';
      container.appendChild(gl.canvas);

      const vert = `
      attribute vec2 position;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
      }`;

      const frag = `precision highp float;
      uniform float iTime;
      uniform vec2 iResolution;
      uniform float iSpeed;
      uniform vec3 iRayColor1;
      uniform vec3 iRayColor2;
      uniform float iIntensity;
      uniform float iSpread;
      uniform float iFlipX;
      uniform float iFlipY;
      uniform float iTilt;
      uniform float iSaturation;
      uniform float iBlend;
      uniform float iFalloff;
      uniform float iOpacity;

      float rayStrength(vec2 raySource, vec2 rayRefDirection, vec2 coord, float seedA, float seedB, float speed) {
        vec2 sourceToCoord = coord - raySource;
        float cosAngle = dot(normalize(sourceToCoord), rayRefDirection);
        return clamp(
          (0.45 + 0.15 * sin(cosAngle * seedA + iTime * speed)) +
          (0.3 + 0.2 * cos(-cosAngle * seedB + iTime * speed)),
          0.0, 1.0) *
          clamp((iResolution.x - length(sourceToCoord)) / iResolution.x, 0.5, 1.0);
      }

      void main() {
        vec2 fragCoord = gl_FragCoord.xy;
        if (iFlipX > 0.5) fragCoord.x = iResolution.x - fragCoord.x;
        if (iFlipY > 0.5) fragCoord.y = iResolution.y - fragCoord.y;

        vec2 coord = vec2(fragCoord.x, iResolution.y - fragCoord.y);
        vec2 rayPos = vec2(iResolution.x * 1.1, -0.5 * iResolution.y);

        float tiltRad = iTilt * 3.14159265 / 180.0;
        float cs = cos(tiltRad);
        float sn = sin(tiltRad);
        vec2 rel = coord - rayPos;
        vec2 tiltedCoord = vec2(rel.x * cs - rel.y * sn, rel.x * sn + rel.y * cs) + rayPos;

        float halfSpread = iSpread * 0.275;
        vec2 rayRefDir1 = normalize(vec2(cos(0.785398 + halfSpread), sin(0.785398 + halfSpread)));
        vec2 rayRefDir2 = normalize(vec2(cos(0.785398 - halfSpread), sin(0.785398 - halfSpread)));

        vec4 rays1 = vec4(iRayColor1, 1.0) * rayStrength(rayPos, rayRefDir1, tiltedCoord, 36.2214, 21.11349, iSpeed);
        vec4 rays2 = vec4(iRayColor2, 1.0) * rayStrength(rayPos, rayRefDir2, tiltedCoord, 22.3991, 18.0234, iSpeed * 0.2);

        vec4 color = rays1 * (1.0 - iBlend) * 0.9 + rays2 * iBlend * 0.9;

        float distanceToLight = length(fragCoord.xy - vec2(rayPos.x, iResolution.y - rayPos.y)) / iResolution.y;
        float brightness = iIntensity * 0.4 / pow(max(distanceToLight, 0.001), iFalloff);
        color.rgb *= brightness;

        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        color.rgb = mix(vec3(gray), color.rgb, iSaturation);

        color.a = max(color.r, max(color.g, color.b)) * iOpacity;
        gl_FragColor = color;
      }`;

      const hexToRgb = hex => {
        const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return m ? [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255] : [1, 1, 1];
      };

      const uniforms = {
        iTime: { value: 0 },
        iResolution: { value: [1, 1] },
        iSpeed: { value: 2.5 },
        iRayColor1: { value: hexToRgb('#EAB308') },
        iRayColor2: { value: hexToRgb('#96c8ff') },
        iIntensity: { value: 3.0 },
        iSpread: { value: 2.7 },
        iFlipX: { value: 0 }, 
        iFlipY: { value: 0 },
        iTilt: { value: 0 },
        iSaturation: { value: 1.7 },
        iBlend: { value: 0.89 },
        iFalloff: { value: 1.6 },
        iOpacity: { value: 0.8 } 
      };

      const geometry = new Triangle(gl);
      const program = new Program(gl, { vertex: vert, fragment: frag, uniforms });
      const mesh = new Mesh(gl, { geometry, program });

      const updateSize = () => {
        renderer.dpr = Math.min(window.devicePixelRatio, 2);
        const w = window.parent.innerWidth;
        const h = window.parent.innerHeight;
        renderer.setSize(w, h);
        uniforms.iResolution.value = [w * renderer.dpr, h * renderer.dpr];
      };

      const loop = t => {
        uniforms.iTime.value = t * 0.001;
        renderer.render({ scene: mesh });
        window.parent.requestAnimationFrame(loop);
      };

      window.parent.addEventListener('resize', updateSize);
      updateSize();
      window.parent.requestAnimationFrame(loop);
  }
</script>
""", height=0)

st.markdown("""
<style>

/* ── Hero ── */
.hero {
    padding: 3.5rem 0 2.5rem 0;
    margin-bottom: 2.5rem;
    position: relative;
    text-align: center;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, rgba(59,130,246,0) 0%, rgba(59,130,246,0.6) 50%, rgba(59,130,246,0) 100%);
}
.hero-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.3em;
    color: #60a5fa;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    text-shadow: 0 0 10px rgba(96, 165, 250, 0.5);
}
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #8bb1ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 0.8rem;
    letter-spacing: -0.03em;
}
.hero-sub {
    font-size: 1.05rem;
    color: #7a9acc;
    font-weight: 300;
}

/* ── Sidebar ── */
.sidebar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #60a5fa;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.target-card {
    background: rgba(15, 30, 53, 0.4);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    cursor: pointer;
    transition: all 0.25s ease;
    font-size: 0.82rem;
}
.target-card:hover { 
    background: rgba(20, 40, 70, 0.7);
    border-color: rgba(59, 130, 246, 0.5); 
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
}
.target-name { color: #e8f0ff; font-weight: 600; }
.target-meta { color: #7a9acc; font-size: 0.75rem; margin-top: 0.2rem; }

/* ── Stage tracker ── */
.stage-track {
    display: flex;
    gap: 0;
    margin-bottom: 2.5rem;
    background: rgba(11, 17, 32, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
.stage-item {
    flex: 1;
    padding: 1rem 0.5rem;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    color: #4a5a7a;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    position: relative;
    transition: all 0.3s ease;
}
.stage-item:last-child { border-right: none; }
.stage-item.done { color: #4ade80; background: rgba(10, 31, 18, 0.5); }
.stage-item.active { 
    color: #60a5fa; 
    background: rgba(15, 35, 70, 0.5); 
    box-shadow: inset 0 -2px 0 #3b82f6;
}
.stage-item.error { color: #f87171; background: rgba(30, 10, 10, 0.5); }
.stage-dot {
    display: block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    margin: 0 auto 0.4rem auto;
    box-shadow: 0 0 8px currentColor;
}

/* ── Section & Expanders ── */
[data-testid="stExpander"] {
    background: rgba(11, 17, 32, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15) !important;
    transition: border-color 0.3s ease, transform 0.3s ease !important;
    margin-bottom: 1.5rem !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(59, 130, 246, 0.3) !important;
}
[data-testid="stExpander"] > summary {
    background: rgba(255, 255, 255, 0.02) !important;
    padding: 1rem 1.5rem !important;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    width: 100%;
}
.section-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #60a5fa;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.1);
}
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e8f0ff;
    letter-spacing: 0.02em;
}
.section-badge {
    margin-left: auto;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-right: 1rem;
}
.badge-ok { background: rgba(34, 197, 94, 0.1); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); box-shadow: 0 0 10px rgba(34, 197, 94, 0.1); }
.badge-warn { background: rgba(245, 158, 11, 0.1); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); box-shadow: 0 0 10px rgba(245, 158, 11, 0.1); }
.badge-info { background: rgba(59, 130, 246, 0.1); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }

/* ── Data readouts ── */
.readout-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.readout {
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    transition: transform 0.2s ease, background 0.2s ease;
}
.readout:hover {
    background: rgba(255, 255, 255, 0.03);
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.1);
}
.readout-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #60a5fa;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.readout-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.25rem;
    color: #ffffff;
    font-weight: 700;
}
.readout-unit {
    font-size: 0.7rem;
    color: #7a9acc;
    margin-top: 0.2rem;
}

/* ── Verdict card ── */
.verdict {
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.verdict-transit { background: rgba(22, 101, 52, 0.2); border: 1px solid rgba(74, 222, 128, 0.3); box-shadow: 0 0 30px rgba(74, 222, 128, 0.1); }
.verdict-binary   { background: rgba(146, 64, 14, 0.2); border: 1px solid rgba(251, 191, 36, 0.3); box-shadow: 0 0 30px rgba(251, 191, 36, 0.1); }
.verdict-none     { background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(148, 163, 184, 0.2); }
.verdict-icon { font-size: 3rem; filter: drop-shadow(0 0 10px currentColor); }
.verdict-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.verdict-title { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; }
.verdict-transit .verdict-label, .verdict-transit .verdict-title { color: #4ade80; }
.verdict-binary .verdict-label, .verdict-binary .verdict-title { color: #fbbf24; }
.verdict-none .verdict-label, .verdict-none .verdict-title { color: #94a3b8; }
.verdict-desc { font-size: 0.95rem; color: #cbd5e1; margin-top: 0.5rem; line-height: 1.5; }

/* ── Error / info banners ── */
.banner-error, .banner-warn, .banner-ok {
    border-radius: 8px;
    padding: 1rem 1.4rem;
    font-size: 0.9rem;
    margin: 1.2rem 0;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.banner-error { background: rgba(127, 29, 29, 0.2); border: 1px solid rgba(239, 68, 68, 0.3); border-left: 4px solid #ef4444; color: #fecaca; }
.banner-warn { background: rgba(120, 53, 15, 0.2); border: 1px solid rgba(245, 158, 11, 0.3); border-left: 4px solid #f59e0b; color: #fde68a; }
.banner-ok { background: rgba(20, 83, 45, 0.2); border: 1px solid rgba(34, 197, 94, 0.3); border-left: 4px solid #22c55e; color: #bbf7d0; }

/* ── Matplotlib container ── */
[data-testid="stImage"] img, .stPyplot img {
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(0, 0, 0, 0.2);
    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.4);
    padding: 0.5rem;
}

/* ── Streamlit input overrides ── */
[data-testid="stTextInput"] input {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
    background: rgba(15, 30, 53, 0.6) !important;
}

.stButton button {
    background: #4c1d95 !important;
    color: #c4b5fd !important;
    border: 1px solid #6d28d9 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    position: relative !important;
}
.stButton button:hover { 
    background: #5b21b6 !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #4c1d95 !important;
    color: #c4b5fd !important;
    border: 1px solid #6d28d9 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.2rem !important;
    width: auto !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    transition: all 0.3s ease !important;
    position: relative !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #5b21b6 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #60a5fa !important; }

/* ── Profile-style Popover ── */
[data-testid="stPopover"] > button {
    background: rgba(24, 24, 27, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 12px 16px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    color: #f4f4f5 !important;
    transition: all 0.2s ease !important;
    text-align: left !important;
    line-height: 1.2 !important;
}
[data-testid="stPopover"] > button:hover {
    background: rgba(39, 39, 42, 0.6) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}
[data-testid="stPopover"] > button p {
    font-weight: 500;
    font-size: 0.9rem;
    margin: 0;
    color: #f4f4f5;
}
[data-testid="stPopover"] > button p::after {
    content: "\A Select a target system";
    white-space: pre;
    font-size: 0.75rem;
    font-weight: 400;
    color: #a1a1aa;
    display: block;
    margin-top: 2px;
}
[data-testid="stPopover"] > button::after {
    content: "🪐";
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    min-width: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #a855f7, #ec4899, #f97316);
    margin-left: auto;
    font-size: 16px;
    box-shadow: inset 0 0 0 2px #18181b;
}

[data-testid="stPopoverBody"] {
    background: rgba(24, 24, 27, 0.95) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 8px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
    min-width: 250px !important;
    overflow: hidden !important;
}

[data-testid="stPopoverBody"] > div,
[data-testid="stPopoverBody"] * {
    scrollbar-width: none !important;
}
[data-testid="stPopoverBody"]::-webkit-scrollbar,
[data-testid="stPopoverBody"] *::-webkit-scrollbar {
    display: none !important;
}

[data-testid="stPopoverBody"] [data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    color: #e4e4e7 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
[data-testid="stPopoverBody"] [data-testid="stButton"] button p {
    margin: 0 !important;
}
[data-testid="stPopoverBody"] [data-testid="stButton"] button:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
}

[data-testid="stPopoverBody"] .element-container:nth-child(1) button::after {
    content: "Hot Jupiter"; margin-left: auto; font-size: 0.7rem; color: #c084fc; background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.2); padding: 2px 6px; border-radius: 6px;
}
[data-testid="stPopoverBody"] .element-container:nth-child(2) button::after {
    content: "Rocky planet"; margin-left: auto; font-size: 0.7rem; color: #60a5fa; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); padding: 2px 6px; border-radius: 6px;
}
[data-testid="stPopoverBody"] .element-container:nth-child(3) button::after,
[data-testid="stPopoverBody"] .element-container:nth-child(4) button::after {
    content: "EB"; margin-left: auto; font-size: 0.7rem; color: #fbbf24; background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.2); padding: 2px 6px; border-radius: 6px;
}

/* ── Animated Globe ── */
@keyframes earthRotate {
    0% { background-position: 0 0; }
    100% { background-position: 400px 0; }
}
@keyframes twinkling { 0%,100% { opacity:0.1; } 50% { opacity:1; } }
@keyframes twinkling-slow { 0%,100% { opacity:0.1; } 50% { opacity:1; } }
@keyframes twinkling-long { 0%,100% { opacity:0.1; } 50% { opacity:1; } }
@keyframes twinkling-fast { 0%,100% { opacity:0.1; } 50% { opacity:1; } }

.globe-container {
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    width: 250px;
    margin: 0 auto;
}
.globe-sphere {
    position: relative;
    width: 250px;
    height: 250px;
    border-radius: 50%;
    overflow: hidden;
    box-shadow: 0 0 20px rgba(255,255,255,0.2),
                -5px 0 8px #c3f4ff inset,
                15px 2px 25px #000 inset,
                -24px -2px 34px #c3f4ff99 inset,
                250px 0 44px #00000066 inset,
                150px 0 38px #000000aa inset;
    background-image: url('https://pub-940ccf6255b54fa799a9b01050e6c227.r2.dev/globe.jpeg');
    background-size: cover;
    background-position: left;
    animation: earthRotate 30s linear infinite;
}
.globe-star {
    position: absolute;
    width: 3px;
    height: 3px;
    background-color: white;
    border-radius: 50%;
}
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB STYLE ─────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#080c14",
    "axes.facecolor":    "#0b1120",
    "axes.edgecolor":    "#1a2540",
    "axes.labelcolor":   "#7a9acc",
    "axes.titlecolor":   "#c8d8f0",
    "xtick.color":       "#3a5a8a",
    "ytick.color":       "#3a5a8a",
    "grid.color":        "#1a2540",
    "grid.linewidth":    0.5,
    "text.color":        "#c8d8f0",
    "lines.color":       "#3b82f6",
    "figure.dpi":        120,
    "font.family":       "monospace",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
})

# ── HELPERS ───────────────────────────────────────────────────
def readout(label, value, unit=""):
    unit_str = f"<div class='readout-unit'>{unit}</div>" if unit else ""
    return f'<div class="readout"><div class="readout-label">{label}</div><div class="readout-value">{value}</div>{unit_str}</div>'

def section(num, title, badge_text="", badge_type="info"):
    badge = f'<span class="section-badge badge-{badge_type}">{badge_text}</span>' if badge_text else ""
    return f'<div class="section-header"><span class="section-num">{num:02d}</span><span class="section-title">{title}</span>{badge}</div>'

def banner(msg, kind="ok"):
    return f'<div class="banner-{kind}">{msg}</div>'

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-label">⬡ Pipeline Configuration</div>', unsafe_allow_html=True)

    if "tic_input" not in st.session_state:
        st.session_state["tic_input"] = "100100827"
    if "analyzed" not in st.session_state:
        st.session_state["analyzed"] = False

    if "quick_select" not in st.session_state:
        st.session_state["quick_select"] = "-- Select a target --"

    def reset_analyzed():
        st.session_state["analyzed"] = False
        st.session_state["quick_select"] = "-- Select a target --"

    tic_id = st.text_input("TIC ID", key="tic_input", placeholder="Enter TIC ID…", on_change=reset_analyzed)

    st.markdown('<div style="font-family:\'Space Mono\', monospace; font-size: 0.65rem; color:#60a5fa; text-transform:uppercase; margin-bottom: 0.5rem; margin-top: 1rem;">Quick targets</div>', unsafe_allow_html=True)
    
    def handle_quick_target(tid):
        st.session_state["tic_input"] = tid
        st.session_state["analyzed"] = True

    with st.popover("Quick Targets", use_container_width=True):
        st.button("🪐 WASP-18 b", on_click=handle_quick_target, args=("100100827",), use_container_width=True)
        st.button("🌍 HD 219134 b", on_click=handle_quick_target, args=("283722336",), use_container_width=True)
        st.button("⭐ Eclipsing Binary #1", on_click=handle_quick_target, args=("38699825",), use_container_width=True)
        st.button("⭐ Eclipsing Binary #2", on_click=handle_quick_target, args=("150361911",), use_container_width=True)

    st.markdown("---")
    def handle_analyse():
        st.session_state["analyzed"] = True

    st.button("🔭  Analyse Target", type="primary", on_click=handle_analyse)
    analyse_btn = st.session_state["analyzed"]

    st.markdown("""
    <div id="sidebar-footer" style="font-size:0.72rem; color:#2a3a5a; line-height:1.6;">
    Data: NASA MAST / TESS SPOC<br>
    Detector: BLS + batman<br>
    Classifier: Odd-Even + SNR
    </div>
    """, unsafe_allow_html=True)



# ── HERO ─────────────────────────────────────────────────────
hero_col1, hero_col2 = st.columns([2, 1])
with hero_col1:
    st.markdown("""
    <div class="hero" style="border:none; margin-bottom:0; padding-bottom:0; text-align:left;">
        <div class="hero-label" style="text-align:left;">◈ Bharatiya Antariksh Hackathon 2026</div>
        <div class="hero-title" style="text-align:left;">Exoplanet Detection Pipeline</div>
        <div class="hero-sub" style="text-align:left;">AI-enabled classification of transit signals in TESS light curves</div>
    </div>
    """, unsafe_allow_html=True)
with hero_col2:
    st.markdown("""
    <div class="globe-container">
        <!-- Stars -->
        <div class="globe-star" style="left: 10px; top: 20px; animation: twinkling 3s infinite;"></div>
        <div class="globe-star" style="left: -20px; top: 120px; animation: twinkling-slow 2s infinite;"></div>
        <div class="globe-star" style="left: 270px; top: 90px; animation: twinkling-long 4s infinite;"></div>
        <div class="globe-star" style="left: 200px; top: 280px; animation: twinkling 3s infinite;"></div>
        <div class="globe-star" style="left: 50px; top: 270px; animation: twinkling-fast 1.5s infinite;"></div>
        <div class="globe-star" style="left: 230px; top: -10px; animation: twinkling-long 4s infinite;"></div>
        <div class="globe-sphere"></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid rgba(255,255,255,0.05); margin: 2.5rem 0;' />", unsafe_allow_html=True)

# ── STAGE TRACKER ────────────────────────────────────────────
stages = ["Download", "Detrend", "BLS", "Odd-Even", "Batman", "Verdict"]

stage_ph = st.empty()

def render_stage_ph(done=-1, active=-1, error=-1):
    html = '<div class="stage-track">'
    for i, s in enumerate(stages):
        if i == error: cls = "error"
        elif i <= done: cls = "done"
        elif i == active: cls = "active"
        else: cls = ""
        html += f'<div class="stage-item {cls}"><span class="stage-dot"></span>{s}</div>'
    html += "</div>"
    stage_ph.markdown(html, unsafe_allow_html=True)

render_stage_ph()

# ── CACHED PIPELINE FUNCTIONS ────────────────────────────────
@st.cache_data(show_spinner=False)
def download_data(tic):
    try:
        sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC", exptime=120)
        if len(sr) == 0:
            return None, "No SPOC 120s data found for this TIC ID."
        if "sequence_number" in sr.table.columns:
            sr = sr[np.argsort(sr.table["sequence_number"])]
        lc = sr[0].download(flux_column="pdcsap_flux")
        if lc is None:
            return None, "Download returned empty data."
        lc = lc.remove_nans()
        sector = int(sr.table["sequence_number"][0]) if "sequence_number" in sr.table.columns else "?"
        n_pts = len(lc)
        return {
            "time": lc.time.value,
            "flux": lc.flux.value,
            "flux_err": lc.flux_err.value,
            "sector": sector,
            "n_pts": n_pts,
        }, None
    except Exception as e:
        return None, f"Error: {e}"

@st.cache_data(show_spinner=False)
def run_bls(time, flux, flux_err):
    baseline = np.nanmax(time) - np.nanmin(time)
    max_period = min(20.0, baseline / 2.0)
    bls = BoxLeastSquares(time, flux, flux_err)
    periods = np.linspace(0.5, max_period, 10000)
    durations = np.linspace(0.01, 0.2, 50)
    pg = bls.power(periods, durations)
    idx = np.argmax(pg.power)
    per = float(pg.period[idx])
    t0  = float(pg.transit_time[idx])
    dur = float(pg.duration[idx])
    depth = float(pg.depth[idx])
    in_tr = bls.transit_mask(time, per, dur, t0)
    scatter = float(np.nanstd(flux[~in_tr]))
    N_in = int(np.sum(in_tr))
    snr = depth / scatter * np.sqrt(N_in) if scatter > 0 else 0.0
    return {
        "period": per, "t0": t0, "duration": dur, "depth": depth,
        "snr": snr, "scatter": scatter,
        "periods": periods, "power": np.array(pg.power),
    }

# ── MAIN ANALYSIS ────────────────────────────────────────────
if not analyse_btn:
    st.markdown("""
    <div style="margin-top:4rem; text-align:center; color:#1e2e4a;">
        <div style="font-size:3rem; margin-bottom:1rem;">⬡</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:0.2em;">
            ENTER A TIC ID AND HIT ANALYSE
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ───────────────────────────────────────────────────────────────
# STEP 1 — DOWNLOAD
# ───────────────────────────────────────────────────────────────

render_stage_ph(active=0)

with st.spinner("Querying MAST archive…"):
    raw, err = download_data(tic_id)

if err:
    render_stage_ph(error=0)
    st.markdown(f'<div class="banner-error">⚠ {err}</div>', unsafe_allow_html=True)
    st.stop()

render_stage_ph(done=0, active=1)

with st.expander("01 · Data Acquisition", expanded=True):
    st.markdown(section(1, "Data Acquisition", "COMPLETE", "ok"), unsafe_allow_html=True)
    st.markdown(f"""
    <div class="readout-grid">
        {readout("TIC ID", tic_id)}
        {readout("SECTOR", str(raw["sector"]))}
        {readout("DATA POINTS", f"{raw['n_pts']:,}")}
        {readout("CADENCE", "120 s")}
        {readout("SOURCE", "SPOC")}
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# STEP 2 — DETREND
# ───────────────────────────────────────────────────────────────
with st.spinner("Detrending light curve…"):
    time_arr = raw["time"]
    flux_arr = raw["flux"]
    ferr_arr = raw["flux_err"]

    lc = lk.LightCurve(time=time_arr, flux=flux_arr, flux_err=ferr_arr)
    lc = lc.normalize().remove_outliers(sigma=5)
    n_removed = raw["n_pts"] - len(lc)

    dt = float(np.nanmedian(np.diff(lc.time.value)))
    w12 = max(3, int((12/24)/dt) | 1)
    flat12, trend12 = lc.flatten(window_length=w12, return_trend=True)
    vraw = float(np.nanvar(lc.flux.value))
    vflat = float(np.nanvar(flat12.flux.value))
    vfrac = 1.0 - vflat/vraw if vraw > 0 else 0.0

    if vfrac > 0.5:
        w24 = max(3, int((24/24)/dt) | 1)
        flat_lc, trend = lc.flatten(window_length=w24, return_trend=True)
        win_used = "24 h"
    else:
        flat_lc, trend = flat12, trend12
        win_used = "12 h"

render_stage_ph(done=1, active=2)

with st.expander("02 · Preprocessing & Detrending", expanded=True):
    st.markdown(section(2, "Preprocessing & Detrending",
                        "⚠ WIDE WINDOW" if vfrac > 0.5 else "COMPLETE",
                        "warn" if vfrac > 0.5 else "ok"), unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    fig.patch.set_facecolor("#080c14")
    axes[0].plot(lc.time.value, lc.flux.value, ".", ms=1.2, color="#2a4a7a", alpha=0.7, label="Raw flux")
    axes[0].plot(trend.time.value, trend.flux.value, "-", color="#ef4444", lw=1.5, label=f"S-G trend ({win_used})")
    axes[0].set_ylabel("Norm. flux")
    axes[0].legend(fontsize=7, framealpha=0.2)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(flat_lc.time.value, flat_lc.flux.value, ".", ms=1.2, color="#3b82f6", alpha=0.7, label="Detrended")
    axes[1].axhline(1.0, color="#1a2540", lw=0.8)
    axes[1].set_xlabel("Time (BTJD)")
    axes[1].set_ylabel("Relative flux")
    axes[1].legend(fontsize=7, framealpha=0.2)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(f"""
    <div class="readout-grid">
        {readout("OUTLIERS REMOVED", str(n_removed))}
        {readout("WINDOW", win_used)}
        {readout("VAR REMOVED", f"{vfrac*100:.1f}%")}
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# STEP 3 — BLS
# ───────────────────────────────────────────────────────────────
time_flat = flat_lc.time.value
flux_flat = flat_lc.flux.value
ferr_flat = flat_lc.flux_err.value

with st.spinner("Running Box Least Squares period search…"):
    bls_res = run_bls(time_flat, flux_flat, ferr_flat)

render_stage_ph(done=2, active=3)

with st.expander("03 · Periodic Signal Detection (BLS)", expanded=True):
    st.markdown(section(3, "Periodic Signal Detection — BLS", "COMPLETE", "ok"), unsafe_allow_html=True)

    per  = bls_res["period"]
    t0   = bls_res["t0"]
    dur  = bls_res["duration"]
    dep  = bls_res["depth"]
    snr  = bls_res["snr"]

    phase_all = (time_flat - t0 + per/2) % per - per/2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#080c14")

    ax1.plot(bls_res["periods"], bls_res["power"], "-", color="#2a4a7a", lw=0.8)
    ax1.axvline(per, color="#3b82f6", lw=1.5, ls="--", alpha=0.8, label=f"P = {per:.4f} d")
    ax1.fill_between(bls_res["periods"], bls_res["power"],
                     alpha=0.08, color="#3b82f6")
    ax1.set_xlabel("Period (days)")
    ax1.set_ylabel("BLS Power")
    ax1.set_title("Periodogram")
    ax1.legend(fontsize=7, framealpha=0.2)
    ax1.grid(True, alpha=0.3)

    win = max(dur * 3, 0.08)
    mask_ph = np.abs(phase_all) < win
    ax2.plot(phase_all[~mask_ph], flux_flat[~mask_ph], ".", ms=1, color="#1a3060", alpha=0.4)
    ax2.plot(phase_all[mask_ph], flux_flat[mask_ph], ".", ms=1.8, color="#3b82f6", alpha=0.8)

    bins = np.linspace(-win, win, 60)
    bm, be = np.histogram(phase_all, bins=bins, weights=flux_flat)
    bc, _  = np.histogram(phase_all, bins=bins)
    msk = bc > 0
    bc2 = (be[:-1] + be[1:]) / 2
    ax2.plot(bc2[msk], bm[msk]/bc[msk], "-", color="#60a5fa", lw=2, drawstyle="steps-mid")

    ax2.set_xlim(-win, win)
    ax2.set_xlabel("Phase (days)")
    ax2.set_ylabel("Norm. flux")
    ax2.set_title("Phase-folded (transit zoom)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(f"""
    <div class="readout-grid">
        {readout("PERIOD", f"{per:.4f}", "days")}
        {readout("EPOCH T₀", f"{t0:.4f}", "BTJD")}
        {readout("DURATION", f"{dur*24:.2f}", "hours")}
        {readout("DEPTH", f"{dep:.4f}", "frac. flux")}
        {readout("SNR", f"{snr:.1f}")}
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# STEP 4 — ODD-EVEN
# ───────────────────────────────────────────────────────────────
with st.spinner("Running odd-even eclipse test…"):
    phases_cyc = (time_flat - t0) / per
    cycles     = np.round(phases_cyc)
    in_tr_all  = np.abs(phases_cyc - cycles) < (dur / per / 2)

    odd_mask  = (cycles % 2 != 0) & in_tr_all
    even_mask = (cycles % 2 == 0) & in_tr_all
    out_mask  = ~in_tr_all

    out_mean = float(np.nanmean(flux_flat[out_mask]))
    odd_flux  = flux_flat[odd_mask]
    even_flux = flux_flat[even_mask]

    odd_d  = out_mean - float(np.nanmean(odd_flux))  if len(odd_flux)  > 0 else 0.0
    even_d = out_mean - float(np.nanmean(even_flux)) if len(even_flux) > 0 else 0.0
    sc     = bls_res["scatter"]
    odd_se  = sc / np.sqrt(len(odd_flux))  if len(odd_flux)  > 0 else np.inf
    even_se = sc / np.sqrt(len(even_flux)) if len(even_flux) > 0 else np.inf

    denom = np.sqrt(odd_se**2 + even_se**2)
    sigma_diff = float(abs(odd_d - even_d) / denom) if denom > 0 and not np.isinf(denom) else 0.0

render_stage_ph(done=3, active=4)

with st.expander("04 · Eclipsing Binary Check (Odd-Even)", expanded=True):
    sig_label = f"{sigma_diff:.1f}σ"
    badge_t = "warn" if sigma_diff >= 3 else "ok"
    badge_l = f"⚠ {sig_label} ASYMMETRY" if sigma_diff >= 3 else f"✓ {sig_label}"
    st.markdown(section(4, "Odd-Even Eclipse Test", badge_l, badge_t), unsafe_allow_html=True)

    fig, (ax_o, ax_e) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor("#080c14")
    win_oe = max(dur*3, 0.08)

    for ax, lbl, mask, depth_val, col in [
        (ax_o, "Odd cycles",  cycles % 2 != 0, odd_d,  "#f59e0b"),
        (ax_e, "Even cycles", cycles % 2 == 0, even_d, "#3b82f6"),
    ]:
        ph = (time_flat[mask] - t0 + per/2) % per - per/2
        ax.plot(ph, flux_flat[mask], ".", ms=1.5, color=col, alpha=0.5)
        ax.axhline(out_mean, color="#1a2540", lw=0.8, ls="--")
        ax.axhline(out_mean - depth_val, color=col, lw=1.2, ls=":", alpha=0.8)
        ax.set_xlim(-win_oe, win_oe)
        ax.set_xlabel("Phase (days)")
        ax.set_title(f"{lbl}  depth={depth_val:.4f}")
        ax.grid(True, alpha=0.3)

    ax_o.set_ylabel("Norm. flux")
    fig.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(f"""
    <div class="readout-grid">
        {readout("ODD DEPTH",  f"{odd_d:.4f}")}
        {readout("EVEN DEPTH", f"{even_d:.4f}")}
        {readout("DIFFERENCE", f"{abs(odd_d-even_d):.4f}")}
        {readout("SIGNIFICANCE", f"{sigma_diff:.2f}σ")}
    </div>
    """, unsafe_allow_html=True)

    if sigma_diff >= 3:
        st.markdown(banner(
            f"Significant odd-even depth asymmetry ({sigma_diff:.1f}σ ≥ 3σ threshold) — "
            "consistent with alternating primary/secondary eclipses in a binary star system.", "warn"),
            unsafe_allow_html=True)
    else:
        st.markdown(banner(
            f"No significant odd-even asymmetry ({sigma_diff:.1f}σ < 3σ) — "
            "consistent with a symmetric, repeating transit.", "ok"),
            unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# STEP 5 — BATMAN
# ───────────────────────────────────────────────────────────────
batman_res = None

with st.spinner("Fitting batman transit model…"):
    def batman_model(t, rp, a, inc):
        p = batman.TransitParams()
        p.t0 = 0.; p.per = per; p.rp = rp
        p.a = a; p.inc = inc; p.ecc = 0.; p.w = 90.
        p.u = [0.3, 0.2]; p.limb_dark = "quadratic"
        return batman.TransitModel(p, t).light_curve(p)

    rp_g = float(np.sqrt(abs(dep)))
    a_g  = float((per / np.pi) * (1 + rp_g) / dur) if dur > 0 else 10.0
    p0   = [max(0.01, min(rp_g, 0.9)), max(1.5, min(a_g, 90.0)), 85.0]

    idx_sort = np.argsort(phase_all)
    ps = phase_all[idx_sort]
    fs = flux_flat[idx_sort]
    fes = ferr_flat[idx_sort]

    try:
        popt, pcov = curve_fit(
            batman_model, ps, fs,
            p0=p0,
            bounds=([0.0, 1.0, 60.0], [1.0, 100.0, 90.0]),
            sigma=fes, maxfev=3000,
        )
        perr = np.sqrt(np.diag(pcov))
        rp_fit, a_fit, inc_fit = popt
        rp_err, a_err, inc_err = perr

        val = np.clip(np.sqrt((1 + rp_fit)**2) / a_fit, 0, 1)
        dur_fit = float((per / np.pi) * np.arcsin(val))
        dep_fit = float(rp_fit**2)

        t_mod = np.linspace(-dur*3, dur*3, 600)
        f_mod = batman_model(t_mod, *popt)

        batman_res = {
            "rp": rp_fit, "rp_err": rp_err,
            "depth": dep_fit, "duration": dur_fit,
            "a": a_fit, "inc": inc_fit,
        }
        bat_ok = True
    except Exception:
        bat_ok = False

render_stage_ph(done=4, active=5)

with st.expander("05 · Transit Model Fitting (batman)", expanded=True):
    if bat_ok:
        st.markdown(section(5, "Transit Model Fitting", "CONVERGED", "ok"), unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#080c14")
        win_bat = max(dur * 3, 0.08)
        m_zoom = np.abs(ps) < win_bat
        ax.plot(ps[~m_zoom], fs[~m_zoom], ".", ms=1, color="#1a3060", alpha=0.3)
        ax.plot(ps[m_zoom],  fs[m_zoom],  ".", ms=1.8, color="#3b82f6", alpha=0.7, label="Data")
        ax.plot(t_mod, f_mod, "-", color="#ef4444", lw=2.5, label="batman fit", zorder=5)
        ax.set_xlim(-win_bat, win_bat)
        ax.set_xlabel("Phase (days)")
        ax.set_ylabel("Norm. flux")
        ax.set_title("Phase-folded with batman model")
        ax.legend(fontsize=8, framealpha=0.2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=1.2)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"""
        <div class="readout-grid">
            {readout("Rp/Rs", f"{batman_res['rp']:.4f} ± {batman_res['rp_err']:.4f}")}
            {readout("DEPTH (fit)", f"{batman_res['depth']:.4f}")}
            {readout("DURATION (fit)", f"{batman_res['duration']*24:.2f}", "hours")}
            {readout("a/Rs", f"{batman_res['a']:.2f}")}
            {readout("INCLINATION", f"{batman_res['inc']:.1f}", "deg")}
        </div>
        """, unsafe_allow_html=True)

        if batman_res["rp"] > 0.2:
            st.markdown(banner(
                f"Rp/Rs = {batman_res['rp']:.3f} greatly exceeds the planetary range (<0.2). "
                "This independently confirms a stellar companion rather than a planet.", "warn"),
                unsafe_allow_html=True)
    else:
        st.markdown(section(5, "Transit Model Fitting", "FIT FAILED", "warn"), unsafe_allow_html=True)
        st.markdown(banner("Batman model failed to converge. BLS parameters are used for final verdict.", "warn"),
                    unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# STEP 6 — VERDICT
# ───────────────────────────────────────────────────────────────
render_stage_ph(done=5)

final_per  = per
final_dep  = batman_res["depth"]    if batman_res else dep
final_dur  = batman_res["duration"] if batman_res else dur
final_rp   = batman_res["rp"]       if batman_res else float(np.sqrt(abs(dep)))

is_transit = snr > 7.0 and sigma_diff < 3
is_eb      = snr > 7.0 and sigma_diff >= 3

if is_transit:
    verdict_cls  = "transit"
    verdict_icon = "🪐"
    verdict_title = "Transit Candidate"
    verdict_desc  = f"SNR {snr:.1f} (>7.0) · Odd-even {sigma_diff:.1f}σ (<3σ) · Consistent with a planetary transit."
elif is_eb:
    verdict_cls   = "binary"
    verdict_icon  = "⭐"
    verdict_title = "Eclipsing Binary"
    verdict_desc  = f"SNR {snr:.1f} (>7.0) · Odd-even {sigma_diff:.1f}σ (≥3σ) · Alternating eclipse depths indicate a binary star system."
else:
    verdict_cls   = "none"
    verdict_icon  = "◌"
    verdict_title = "No Confident Detection"
    verdict_desc  = f"SNR {snr:.1f} (≤7.0) · Signal does not exceed detection threshold. Period may exceed search range or signal is too shallow."

st.markdown(f"""
<div class="verdict verdict-{verdict_cls}">
    <div class="verdict-icon">{verdict_icon}</div>
    <div>
        <div class="verdict-label">Classification verdict</div>
        <div class="verdict-title">{verdict_title}</div>
        <div class="verdict-desc">{verdict_desc}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="readout-grid">
    {readout("SNR",             f"{snr:.1f}")}
    {readout("ODD-EVEN σ",     f"{sigma_diff:.2f}")}
    {readout("PERIOD",         f"{final_per:.4f}", "days")}
    {readout("DEPTH",          f"{final_dep:.4f}")}
    {readout("DURATION",       f"{final_dur*24:.2f}", "hours")}
    {readout("Rp/Rs",          f"{final_rp:.4f}")}
</div>
""", unsafe_allow_html=True)

# ── DOWNLOAD CSV ─────────────────────────────────────────────
res_df = pd.DataFrame([{
    "TIC_ID": tic_id, "Sector": raw["sector"], "Data_Points": raw["n_pts"],
    "Detrend_Window": win_used,
    "BLS_Period_d": round(per, 6), "BLS_Depth": round(dep, 6),
    "BLS_Duration_h": round(dur*24, 4), "SNR": round(snr, 2),
    "Odd_Even_Sigma": round(sigma_diff, 2),
    "Batman_Rp_Rs": round(batman_res["rp"], 6) if batman_res else None,
    "Batman_Depth": round(batman_res["depth"], 6) if batman_res else None,
    "Batman_Duration_h": round(batman_res["duration"]*24, 4) if batman_res else None,
    "Classification": verdict_title,
}])

st.markdown("<br>", unsafe_allow_html=True)
st.download_button(
    "⬇  Download Results CSV",
    data=res_df.to_csv(index=False),
    file_name=f"TIC_{tic_id}_results.csv",
    mime="text/csv",
)
