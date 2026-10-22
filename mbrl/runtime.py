"""Resolve separately installed robot/project assets without machine-specific paths."""
import os
import sys
from pathlib import Path


def configure_runtime():
    raw=os.environ.get('MBRL_PROJECT_ROOT')
    if not raw:return
    root=Path(raw).expanduser().resolve()
    if not (root/'src/assets/robots').is_dir():
        raise ValueError('MBRL_PROJECT_ROOT must contain src/assets/robots from the configured G1 project')
    sys.path.insert(0,str(root))
    if (root/'vendor').is_dir():sys.path.insert(0,str(root/'vendor'))
