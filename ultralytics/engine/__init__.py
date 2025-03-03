# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Make NormalizedSGD available through the engine module

try:
    from .NormalizedSGD import NormalizedSGD
except ImportError:
    pass  # Silent failure if file not found
