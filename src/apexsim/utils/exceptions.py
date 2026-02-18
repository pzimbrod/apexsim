"""Custom exceptions for lap time simulation."""


class LapSimError(Exception):
    """Base exception for simulation errors."""


class ConfigurationError(LapSimError):
    """Raised when model or solver configuration is invalid."""


class TrackDataError(LapSimError):
    """Raised when track data cannot be parsed or validated."""
