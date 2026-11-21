"""Entry point aplikasi Streamlit: pemetaan tutupan lahan (K-Means Sentinel-2).

Langkah 1: baru judul dan lencana backend aktif. Layout penuh ada di Langkah 6.
"""

from __future__ import annotations

import streamlit as st

import config
from ui.badges import backend_badge

TITLE = "Pemetaan Tutupan Lahan"
SUBTITLE = "K-Means piksel RGB Sentinel-2: CPU, OpenMP, dan CUDA"


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(SUBTITLE)
    st.markdown(backend_badge(config.BACKEND))


main()
