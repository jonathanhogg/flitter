"""
Flitter plugins
"""

import importlib.metadata

from loguru import logger


PluginEntryPoints = {}
PluginObjectCache = {}


def get_plugin(group, name, quiet=False):
    if group in PluginObjectCache:
        cache = PluginObjectCache[group]
        if name in cache:
            return cache[name]
    else:
        cache = {}
        PluginObjectCache[group] = cache
    if group in PluginEntryPoints:
        entry_points = PluginEntryPoints[group]
    else:
        entry_points = importlib.metadata.entry_points(group=group)
        if entry_points.names:
            logger.debug("Available {} plugins: {}", group, ', '.join(sorted(entry_points.names)))
        PluginEntryPoints[group] = entry_points
    if name in entry_points.names:
        try:
            plugin = entry_points[name].load()
        except Exception:
            logger.exception("Unable to load {} plugin: {}", group, name)
            plugin = None
    else:
        if not quiet:
            logger.warning("No {} plugin found for: {}", group, name)
        plugin = None
    cache[name] = plugin
    return plugin


def get_plugin_names(group):
    if group in PluginEntryPoints:
        entry_points = PluginEntryPoints[group]
    else:
        entry_points = importlib.metadata.entry_points(group=group)
        logger.debug("Available {} plugins: {}", group, ', '.join(entry_points.names))
        PluginEntryPoints[group] = entry_points
    return entry_points.names
