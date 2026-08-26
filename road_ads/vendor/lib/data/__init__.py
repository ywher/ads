from .acdc import ACDCDataset, ACDCDepDataset
from .cityscapes_cv2 import CityscapesDataset, CityscapesDepDataset
from .gta import GTADataset
from .mapillary import MapillaryDataset, MapillaryDepDataset
from .muses import MusesDataset, MusesDepDataset
from .synthia import CityscapesSynCityDataset, SYNTHIADataset
from .wilddash import WildDashDataset, WildDashDepDataset


__all__ = [
    'ACDCDataset', 'ACDCDepDataset',
    'CityscapesDataset', 'CityscapesDepDataset',
    'GTADataset',
    'SYNTHIADataset', 'CityscapesSynCityDataset',
    'MapillaryDataset', 'MapillaryDepDataset',
    'MusesDataset', 'MusesDepDataset',
    'WildDashDataset', 'WildDashDepDataset',
]
