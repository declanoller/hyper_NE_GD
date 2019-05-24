import os
__all__ = [f.replace('.py', '') for f in os.listdir('./agent_classes') if '__' not in f]
