import os
__all__ = [f.replace('.py', '') for f in os.listdir('./classes') if '__' not in f]
