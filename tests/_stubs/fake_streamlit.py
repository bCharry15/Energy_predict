"""Doble de pruebas MÍNIMO de Streamlit y Plotly.

Solo se usa en tests/test_ui_smoke.py cuando Streamlit o Plotly NO están instalados, para
poder ejecutar el código de todas las vistas (lógica, cálculos y flujo) sin navegador.
Reproduce las validaciones de Streamlit que más errores causan: tipos mezclados y valores
fuera de rango en number_input/slider, opciones por defecto inexistentes y claves duplicadas.
No sustituye a la prueba real con streamlit.testing (tests/test_app_streamlit.py).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


class RerunException(BaseException):
    pass


class StopException(BaseException):
    pass


class _Ctx:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class SessionState(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.overrides: dict = {}      # key -> valor devuelto por el widget
        self.press = lambda label: False  # decide qué botones se "pulsan"
        self.check_all = False
        self.reset_run()
        self.sidebar = _Ctx(self)
        self.__version__ = "1.40.0"

    # -------------------------------------------------------------- control
    def reset_run(self):
        self.log: list[tuple[str, str]] = []
        self._ids: set = set()
        self.charts = 0

    def _register(self, kind, label, key):
        ident = key if key is not None else (kind, label)
        if ident in self._ids:
            raise RuntimeError(f"DuplicateWidgetID: {ident!r}")
        self._ids.add(ident)

    def _value(self, key, default):
        if key is not None and key in self.overrides:
            return self.overrides[key]
        return default

    def messages(self, kind):
        return [m for k, m in self.log if k == kind]

    # --------------------------------------------------------------- salida
    def _out(kind):
        def f(self, *a, **k):
            self.log.append((kind, str(a[0]) if a else ""))
        return f

    markdown = _out("markdown"); caption = _out("caption"); info = _out("info"); warning = _out("warning")
    error = _out("error"); success = _out("success"); write = _out("write"); exception = _out("exception")
    toast = _out("toast"); set_page_config = _out("config")

    def plotly_chart(self, fig, use_container_width=None, width=None, theme="streamlit", config=None, key=None):
        self._register("plotly", id(fig), key)
        self.charts += 1

    def dataframe(self, data, width=None, height=None, use_container_width=None, hide_index=None):
        import pandas as pd
        assert isinstance(data, pd.DataFrame)
        data.to_json()  # falla si hay tipos no serializables

    # ----------------------------------------------------------- contenedores
    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        assert n >= 1
        return [_Ctx(self) for _ in range(n)]

    def tabs(self, labels):
        return [_Ctx(self) for _ in labels]

    def form(self, key, **kw):
        self._register("form", key, key)
        return _Ctx(self)

    def expander(self, label, **kw):
        return _Ctx(self)

    container = spinner = expander

    # --------------------------------------------------------------- widgets
    @staticmethod
    def _check_numeric(label, lo, hi, value, step):
        nums = [v for v in (lo, hi, value, step) if v is not None]
        tipos = {isinstance(v, int) and not isinstance(v, bool) for v in nums}
        if len(tipos) > 1:
            raise TypeError(f"{label}: mezcla de int y float en min/max/value/step: {nums}")
        if lo is not None and value < lo:
            raise ValueError(f"{label}: value {value} < min {lo}")
        if hi is not None and value > hi:
            raise ValueError(f"{label}: value {value} > max {hi}")

    def number_input(self, label, min_value=None, max_value=None, value="min", step=None, format=None,
                     key=None, help=None, disabled=False, **kw):
        assert not kw, kw
        self._register("number_input", label, key)
        if value == "min":
            value = min_value if min_value is not None else 0.0
        self._check_numeric(label, min_value, max_value, value, step)
        return self._value(key, value)

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, format=None, key=None,
               help=None, **kw):
        assert not kw, kw
        self._register("slider", label, key)
        self._check_numeric(label, min_value, max_value, value, step)
        return self._value(key, value)

    def select_slider(self, label, options=(), value=None, key=None, **kw):
        self._register("select_slider", label, key)
        assert value in options
        return self._value(key, value)

    def selectbox(self, label, options, index=0, format_func=str, key=None, help=None, **kw):
        assert not kw, kw
        self._register("selectbox", label, key)
        options = list(options)
        for o in options:
            format_func(o)
        if not options:
            return None
        return self._value(key, options[index])

    def radio(self, label, options, index=0, key=None, label_visibility=None, **kw):
        self._register("radio", label, key)
        options = list(options)
        if key in self.session_state:
            return self.session_state[key]
        return self._value(key, options[index])

    def multiselect(self, label, options, default=None, format_func=str, key=None, **kw):
        assert not kw, kw
        self._register("multiselect", label, key)
        options = list(options)
        for o in options:
            format_func(o)
        default = list(default or [])
        assert all(d in options for d in default), "default fuera de options"
        return self._value(key, default)

    def text_input(self, label, value="", max_chars=None, key=None, **kw):
        self._register("text_input", label, key)
        return self._value(key, value)

    def checkbox(self, label, value=False, key=None, **kw):
        self._register("checkbox", label, key)
        return self._value(key, True if self.check_all else value)

    def button(self, label, key=None, type="secondary", disabled=False, on_click=None, args=(), **kw):
        assert not kw, kw
        self._register("button", label, key)
        pulsado = bool(self.press(label)) and not disabled
        if pulsado and on_click:
            on_click(*args)
        return pulsado

    def form_submit_button(self, label="Submit", type="secondary", **kw):
        self._register("submit", label, None)
        return bool(self.press(label))

    def download_button(self, label, data, file_name=None, mime=None, disabled=False, key=None, **kw):
        self._register("download", label, key)
        assert isinstance(data, (bytes, str))
        return False

    def file_uploader(self, label, type=None, key=None, **kw):
        self._register("uploader", label, key)
        return self._value(key, None)

    def rerun(self):
        raise RerunException()

    def stop(self):
        raise StopException()


def install() -> FakeStreamlit:
    """Instala los dobles en sys.modules y devuelve el falso Streamlit."""
    st = FakeStreamlit()
    sys.modules["streamlit"] = st
    plotly = MagicMock(name="plotly")
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = plotly.graph_objects
    return st
