import os
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import Bio
import numpy as np
import torch
from Bio.SubsMat import MatrixInfo as matlist
from einops import rearrange
from einops import repeat  # noqa

import helpers.orientation_utils as ori_utils
from common.constants import ATOM_BOND_MAT, ATOM_SEP_MAT, SEQ_SEP_MAT, AA_INDEX_MAP, SS_KEY_MAP
from data.data_io import load_npy
from helpers.orientation_utils import signed_dihedral_4_torch
from helpers.utils import default
from scoring.pyrosetta_terms import get_dssp as rosetta_dssp

blosum80 = matlist.blosum80
default_device = torch.device('cpu')
DEFAULT_SEP_BINS = [-i for i in reversed(range(1, 33))] + [0] + list(range(1, 33))
SMALL_SEP_BINS = [30, 20, 15, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]
SMALL_SEP_BINS = [-i for i in SMALL_SEP_BINS] + [0] + list(reversed(SMALL_SEP_BINS))


def to_tensor(x, dtype=None) -> Optional[torch.Tensor]:
    if x is None:
        return None
    dtype = torch.float32 if dtype is None else dtype
    if torch.is_tensor(x):
        return x.type(dtype)
    else:
        return torch.tensor(x, dtype=dtype)


def get_dssp(N, CA, C, seq, include_ss=False) -> Tuple[Optional[torch.Tensor], ...]:
    ss = None
    if include_ss:
        phi, psi, omega, ss = rosetta_dssp(N, CA, C, seq=seq)
        ss = torch.tensor(SS_KEY_MAP[x] for x in ss).long()
    else:
        N, CA, C = map(lambda x: x.squeeze(), (N, CA, C))
        phi, psi, omega = get_bb_dihedral(N=N, CA=CA, C=C)
    return to_tensor(phi), to_tensor(psi), to_tensor(omega), ss


def get_bb_dihedral(N, CA, C):
    _psi = signed_dihedral_4_torch([N[:-1], CA[:-1], C[:-1], N[1:]])
    phi = torch.zeros_like(N[:, 0])
    psi = torch.zeros_like(N[:, 0])
    omega = torch.zeros_like(N[:, 0])
    phi[1:] = signed_dihedral_4_torch([C[:-1], N[1:], CA[1:], C[1:]])
    psi[:-1] = signed_dihedral_4_torch([N[:-1], CA[:-1], C[:-1], N[1:]])
    omega[:-1] = signed_dihedral_4_torch([CA[:-1], C[:-1], N[1:], CA[1:]])
    return phi, psi, omega


# Utility methods
def detach_array(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach()
    return arr


class EncodingType:
    FOURIER = 'fourier'
    TOKEN = 'token'


class SepType:
    BOND = 'bond'
    SEQ = 'seq'


class TrRosettaOrientationType(Enum):
    PHI, PSI, OMEGA = 'phi', 'psi', 'omega'


TR_ORI_KEYS = [TrRosettaOrientationType.OMEGA, TrRosettaOrientationType.PHI, TrRosettaOrientationType.PSI]


def get_atom_tys_for_ori_key(key):  # noqa
    if key == TrRosettaOrientationType.PHI:
        # N1Ca1Cb1Cb2
        return ['N', 'CA', 'CB', 'CB']
    elif key == TrRosettaOrientationType.OMEGA:
        # Ca1Cb1Cb2Ca2
        return ['CA', 'CB', 'CB', 'CA']
    elif key == TrRosettaOrientationType.PSI:
        # Ca1Cb1Cb2
        return ['CA', 'CB', 'CB']
    else:
        raise Exception(f'invalid key: {key}')


def map_seq_to_int(seq, device=None):
    device = default(device, default_device)
    return torch.tensor([AA_INDEX_MAP[a] for a in seq], device=device).long()


def map_seqs_from_aln(alignment, return_aln=False):
    S1, S2 = alignment
    # convert an aligned seq to a binary vector with 1 indicates aligned and 0 gap
    y = np.array([1 if a != '-' else 0 for a in S2])

    # get the position of each residue in the original sequence, starting from 0.
    ycs = np.cumsum(y) - 1
    np.putmask(ycs, y == 0, -1)

    # map from the 1st seq to the 2nd one. set -1 for an unaligned residue in the 1st sequence
    mapping = [y0 for a, y0 in zip(S1, ycs) if a != '-']
    if return_aln:
        return (S1, S2), np.array(mapping)
    return np.array(mapping)


def map_seqs(target_seq, native_seq, return_aln=False):
    alignment = Bio.pairwise2.align.localds(target_seq, native_seq, blosum80, -5, -0.2)  # noqa
    S1, S2 = alignment[0][0], alignment[0][1]
    return map_seqs_from_aln((S1, S2), return_aln=return_aln)


def align_n_mask_seqs(target_seq, native_seq):
    alignment = Bio.pairwise2.align.localds(target_seq, native_seq, blosum80, -5, -0.2)  # noqa
    S1, S2 = alignment[0][0], alignment[0][1]
    # convert an aligned seq to a binary vector with 1 indicates aligned and 0 gap
    y2 = torch.tensor([1 if a != '-' else 0 for a in S2])
    y1 = torch.tensor([1 if a != '-' else 0 for a in S1])
    # mask for the first sequence (mask1[i] == True <==> S2[i]!='-)
    # mask for second sequence (mask2[i] == True <==> S1[i]!='-')
    return S1, S2, y2 != 0, y1 != 0  # mask1, mask2


def mapping_to_masks(mapping, to_rng):
    to_indices = np.array(list(filter(lambda x: x >= 0, mapping)))
    to_mask = np.zeros((to_rng[1] - to_rng[0]))
    to_mask[(to_indices - to_rng[0]).astype(np.int)] = 1
    return mapping >= 0, to_mask > 0


def fourier_encode(x, num_encodings=4, include_self=True, flatten=True, scale=1):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    scales *= scale
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    x = rearrange(x, 'b m n ... -> b m n (...)') if flatten else x
    return x.to(device)


def fourier_encode_coords(coords, num_encodings, scale=1):
    coord_feats = fourier_encode(
        coords,
        num_encodings=num_encodings,
        include_self=True,
        scale=scale,
    )
    return torch.einsum('...ij-> ...ji', coord_feats).to(coords.device)


def fourier_encode_sep_mat(coords, num_encodings, sep_mat, scale=1, normalize_sep=True, include_self=True):
    n = coords.shape[1]
    sep_mat_copy = sep_mat[:, :n, :n].to(coords.device)  # .type(torch.float64)
    if normalize_sep:
        sep_mat_copy /= torch.max(sep_mat_copy[0, 0, :])
    return fourier_encode(
        sep_mat_copy[:, :, :, None],
        num_encodings=num_encodings,
        include_self=include_self,
        scale=scale,
    )


def fourier_encode_distances(coords, num_encodings, dists=None, scale=1, max_dist=20, normalize_dist=True):
    crds = None if dists is not None else detach_array(coords)
    distances = dists if dists is not None else torch.cdist(crds, crds)
    distances = distances[:, :, :, None]
    distances[distances > max_dist] = max_dist
    distances = distances / max_dist if normalize_dist else distances
    return fourier_encode(
        distances,
        num_encodings=num_encodings,
        include_self=True,
        scale=scale,
    )


def get_orientation_mat(
        N: torch.Tensor,
        CA: torch.Tensor,
        CB: torch.Tensor,
        orientation_type: TrRosettaOrientationType):
    if orientation_type == TrRosettaOrientationType.PSI:
        mat = ori_utils.unsigned_angle_all_torch([CA, CB, CB])
    elif orientation_type == TrRosettaOrientationType.OMEGA:
        mat = ori_utils.signed_dihedral_all_12_torch([CA, CB, CB, CA])
    elif orientation_type == TrRosettaOrientationType.PHI:
        mat = ori_utils.signed_dihedral_all_123_torch([N, CA, CB, CB])
    else:
        raise Exception(f'orientation type {orientation_type} not accepted')
    # expand back to full size
    return mat


def key_encode_mat(mat: torch.Tensor, bin_starts, scale=1):
    assert np.allclose(sorted(bin_starts), bin_starts)
    out_mat = torch.zeros(mat.shape)
    for i, (s, e) in enumerate(zip(bin_starts[:-1], bin_starts[1:])):
        out_mat[torch.logical_and(mat >= s / scale, mat < e / scale)] = i + 1
    out_mat[mat >= bin_starts[-1] / scale] = len(bin_starts)
    out_mat[mat < bin_starts[0] / scale] = 0
    return out_mat.type(torch.long).to(mat.device)


def key_encode_ori_mat(mat, n_bins, min_val=-np.pi, max_val=np.pi):
    mat = mat - min_val
    mat = n_bins * mat / (max_val - min_val)
    mat[mat < 0] = 0
    mat[mat > n_bins - 1] = n_bins - 1
    return mat.long().to(mat.device)


def key_encode_contact_mat(coords, n_bins, dists=None, min_dist=2.4, max_dist=16):
    crds = None if dists is not None else detach_array(coords)
    distances = dists if dists is not None else torch.cdist(crds, crds)
    step = (max_dist - min_dist) / (n_bins - 1)
    # normalize distances so that distance < min dist is zero
    distances = torch.clamp(distances - min_dist, 0, (max_dist - min_dist) - 1e-4)
    # dividing by step should give the bin index of each distance
    return torch.round(distances / step).long().to(dists.device).detach()  # keys for contact mat


def bin_encode_sep_mat(coords, sep_mat, bin_starts=None):
    n = coords.shape[1]
    bin_starts = bin_starts or DEFAULT_SEP_BINS
    return key_encode_mat(sep_mat[:, :n, :n], bin_starts).long()


def key_encode_sep_mat(coords: torch.Tensor, sep_mat: torch.Tensor, bin_starts=None):
    n = coords.shape[1]
    bin_starts = bin_starts or DEFAULT_SEP_BINS
    return key_encode_mat(sep_mat, bin_starts)


def joint_res_atom_key(residues, atoms):
    k = len(atoms)
    return [AA_INDEX_MAP[r] * k + i for r in residues for i in range(k)]


def get_atom_bond_matrix():
    path = Path(os.path.dirname(__file__)) / 'constants' / (ATOM_BOND_MAT + '.npy')
    return load_npy(path)


def get_atom_sep_matrix():
    path = Path(os.path.dirname(__file__)) / 'constants' / (ATOM_SEP_MAT + '.npy')
    return load_npy(path)


def get_seq_sep_matrix():
    path = Path(os.path.dirname(__file__)) / 'constants' / (SEQ_SEP_MAT + '.npy')
    return load_npy(path)


def get_pssm():
    path = Path(os.path.dirname(__file__)) / 'constants' / ('blosum62' + '.npy')
    data = load_npy(path)
    aas, scores = data['aas'], data['scores']
    return aas, scores
