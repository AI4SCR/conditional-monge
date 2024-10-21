from .chemCPA_single_loader import SciPlexCPAModule
from .single_loader import FourIModule, SciPlexModule

DataModuleFactory = {
    "4i": FourIModule,
    "sciplex": SciPlexModule,
    "cpa_sciplex": SciPlexCPAModule,
}
