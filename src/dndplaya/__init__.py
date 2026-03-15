"""DnDPlaya - AI-powered D&D dungeon playtesting tool."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dndplaya")
except PackageNotFoundError:
    __version__ = "0.1.0"
