"""
Flitter templating
"""

from pathlib import Path

import mako.lookup


TemplateLoader = mako.lookup.TemplateLookup(directories=[Path(__file__).parent])
