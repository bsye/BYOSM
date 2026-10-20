import torch

from byosm.jepa import sigreg


def test_sigreg_prefers_isotropic_gaussian():
    torch.manual_seed(0)
    gauss = torch.randn(512, 64)
    collapsed = torch.randn(512, 1).repeat(1, 64) * 0.01
    stretched = torch.randn(512, 64) * torch.linspace(0.1, 3, 64)
    g = sigreg(gauss)
    assert g < sigreg(collapsed) and g < sigreg(stretched)
