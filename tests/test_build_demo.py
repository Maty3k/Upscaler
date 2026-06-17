"""Smoke test: the whole Gradio app wiring constructs without launching.

app.build_demo() is the most wiring-fragile surface (every tab, every .click
input/output list). Constructing it exercises config.load() seeding and the
full component tree, catching mismatched input/output lists before runtime —
without starting a server or downloading any weights.
"""

import pytest


def test_build_demo_constructs():
    gr = pytest.importorskip("gradio")
    import app

    demo = app.build_demo()
    assert isinstance(demo, gr.Blocks)
