import sys, importlib.machinery, importlib.util, enum

class InterpolationMode(enum.Enum):
    NEAREST = 'nearest'
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'
    BOX = 'box'
    HAMMING = 'hamming'
    LANCZOS = 'lanczos'
    NEAREST_EXACT = 'nearest-exact'

def _mod(name):
    s = importlib.machinery.ModuleSpec(name, None, is_package=True)
    m = importlib.util.module_from_spec(s)
    return m

_transforms = _mod('torchvision.transforms')
_transforms.InterpolationMode = InterpolationMode
_transforms_v2 = _mod('torchvision.transforms.v2')
_transforms_v2.functional = _mod('torchvision.transforms.v2.functional')
_io = _mod('torchvision.io')
_tv = _mod('torchvision')
_tv.transforms = _transforms
_tv.io = _io

sys.modules['torchvision'] = _tv
sys.modules['torchvision.transforms'] = _transforms
sys.modules['torchvision.transforms.v2'] = _transforms_v2
sys.modules['torchvision.transforms.v2.functional'] = _transforms_v2.functional
sys.modules['torchvision.io'] = _io
for _s in ['_meta_registrations', 'datasets', 'models', 'ops', 'utils', 'extension']:
    sys.modules['torchvision.' + _s] = _mod('torchvision.' + _s)

import runpy
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
