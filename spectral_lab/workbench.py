"""Presentation helpers for the desktop-style spectral workspace."""
from html import escape
from pathlib import Path

import streamlit as st


def apply_style():
    css = Path(__file__).with_name("workbench.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def panel_heading(text):
    st.markdown(f'<div class="panel-heading">{escape(text)}</div>', unsafe_allow_html=True)


def document_strip(name, view, unit):
    st.markdown(
        f'<div class="document-strip"><span class="document-tab">'
        f'<span class="document-mark">∿</span> {escape(name)}</span>'
        f'<span class="document-view">{escape(view)} · {escape(unit)}</span></div>',
        unsafe_allow_html=True)


def chart_style(figure, *, height=540, grid=True, dragmode="zoom", revision="default"):
    figure.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="#242424", plot_bgcolor="#191919",
        font=dict(family="Segoe UI, sans-serif", size=11, color="#bdbdbd"),
        margin=dict(l=56, r=24, t=34, b=52), hovermode="x unified", dragmode=dragmode,
        uirevision=revision,
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#353535", bordercolor="#666666", font_size=12))
    figure.update_xaxes(showgrid=grid, gridcolor="#303030", zeroline=False, showline=True,
                        linecolor="#535353", mirror=True, ticks="outside", tickcolor="#666666",
                        title_font_size=11)
    figure.update_yaxes(showgrid=grid, gridcolor="#303030", zeroline=False, showline=True,
                        linecolor="#535353", mirror=True, ticks="outside", tickcolor="#666666",
                        title_font_size=11)
    return figure


def status_bar(samples, axis_unit, view, has_error):
    st.markdown(
        '<div class="workbench-status"><span><i></i> ローカル処理</span>'
        f'<span>{samples:,} サンプル</span><span>{escape(axis_unit)}</span>'
        f'<span>{"1σ誤差あり" if has_error else "入力誤差なし"}</span>'
        f'<span class="status-view">{escape(view)} · FITS Spectral Lab</span></div>',
        unsafe_allow_html=True)
