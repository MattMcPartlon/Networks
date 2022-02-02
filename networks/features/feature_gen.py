from typing import Union

from torch import nn

from common.constants import N_AMINO_ACID_KEYS, N_SEC_STRUCT_KEYS, MAX_SEQ_LEN, TEST_MODE
from common.refine_args import RefineArgs
from data.data_types import Sample
from data.data_utils import *
from data.feature_utils import fourier_feature_size
from helpers.orientation_utils import signed_dihedral_4_torch, signed_dihedral_all_12_torch  # noqa
from helpers.torch_utils import safe_cat, _to_device  # noqa
from scoring.pyrosetta_terms import RosettaScore
from torch.nn.functional import one_hot

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
VERBOSE = False
TESTING = TEST_MODE

ATOM_SEP_MAT = torch.tensor(get_atom_sep_matrix()).type(torch.long)
ATOM_SEP_MAT_FLOAT = ATOM_SEP_MAT.clone().type(default_dtype)

SEQ_SEP_MAT = torch.arange(MAX_SEQ_LEN).unsqueeze(0) - torch.arange(MAX_SEQ_LEN).unsqueeze(-1)
SEQ_SEP_MAT_FLOAT = SEQ_SEP_MAT.unsqueeze(0).clone().type(default_dtype)

ATOM_BOND_MATRIX = torch.tensor(get_atom_bond_matrix()).type(torch.long)

BOND_MASK = (ATOM_BOND_MATRIX > 0).bool()
DEFAULT_DEVICE = torch.device("cpu")

ROSETTA_EMBED_SCALE = 2

TEST_MODE = False


class OneHotEmb(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):  # noqa
        super(OneHotEmb, self).__init__()
        self.embedding_dim, self.num_embeddings = num_classes, num_classes

    def forward(self, classes):
        if TEST_MODE:
            assert self.embedding_dim > torch.max(classes) and torch.min(classes) >= 0, \
                f"{self.embedding_dim},{torch.max(classes)}"
        return one_hot(classes, self.embedding_dim)


def _verify_embedding(emb: Union[nn.Embedding, OneHotEmb], emb_input: torch.Tensor, name: str) -> None:
    if TEST_MODE:
        max_expected = emb.num_embeddings
        max_actual = torch.max(emb_input)
        min_actual = torch.min(emb_input)
        assert max_actual < max_expected, f'{type(emb)} max element in embeding input greater than num embeddings for' \
                                          f' {name}. got max: {max_actual}, num embeddings {max_expected}'
        assert min_actual >= 0, f'{type(emb)} got negative element {torch.min(emb_input)} in input embedding for {name}'


def _verify_tensor(*args):
    for a in args:
        if a is not None:
            assert isinstance(a, torch.Tensor), f"expected tensor, got {type(a)}"


class RefineFeatureGenerator(nn.Module):

    def __init__(self, args: RefineArgs, device):
        super().__init__()
        self.args = args
        self.device = device
        sep_bins = DEFAULT_SEP_BINS if not args.use_small_sep_bins else SMALL_SEP_BINS
        self.encoded_seq_sep_mat = key_encode_sep_mat(torch.ones(1, MAX_SEQ_LEN), SEQ_SEP_MAT.unsqueeze(0),
                                                      bin_starts=sep_bins)
        self.encoded_seq_sep_mat = self.encoded_seq_sep_mat.to(device)
        self.atoms, self.n_atoms = args.atom_tys, len(args.atom_tys)
        self.use_rosetta_terms = args.use_rosetta_score_terms
        self.rs = RosettaScore()
        self.rosetta_atom_proj, self.rosetta_pair_proj = None, None
        edge_embeddings, self.edge_feat_dim, self.static_edge_feat_dim = self.init_edge_info()
        atom_embeddings, self.atom_feat_dim, self.static_atom_feat_dim = self.init_atom_info()
        self.variable_edge_feat_dim = self.edge_feat_dim - self.static_edge_feat_dim
        self.variable_atom_feat_dim = self.atom_feat_dim - self.static_atom_feat_dim
        # put everything on the right device
        self.atom_embeddings = atom_embeddings.to(self.device)
        self.edge_embeddings = edge_embeddings.to(self.device)
        self.fourier_encoded_sep_mat = None

        scale = ROSETTA_EMBED_SCALE if args.pre_embed_rosetta_score_terms else 1
        scale = 0 if not self.use_rosetta_terms else scale
        self.pre_rosetta_atom_dim = self.atom_feat_dim - scale * self.rs.num_res_score_terms
        self.pre_rosetta_edge_dim = self.edge_feat_dim - scale * self.rs.num_res_pair_score_terms

    @property
    def dims(self):
        return self.atom_feat_dim, self.edge_feat_dim

    def init_atom_info(self) -> Tuple[nn.ModuleDict, Optional[int], int]:
        args, device = self.args, self.device
        atom_feat_dim = 0
        embeddings = nn.ModuleDict()
        if args.embed_res_ty:
            cls = nn.Embedding if not args.one_hot_features else OneHotEmb
            embeddings['res_embedding'] = cls(N_AMINO_ACID_KEYS,
                                              args.res_ty_embed_dim)

        if args.embed_rel_pos:
            cls = nn.Embedding if not args.one_hot_features else OneHotEmb
            embeddings['rel_pos_embedding'] = cls(
                args.rel_pos_embed_bins, args.rel_pos_embed_dim
            )

        if args.embed_sec_struct and args.include_dssp:
            embeddings['ss_embedding'] = nn.Embedding(N_SEC_STRUCT_KEYS,
                                                      args.sec_struct_embed_dim)

        offset = 0
        if args.embed_dihedral and args.include_dssp:
            dim = args.dihedral_embed_dim
            embeddings['phi_embedding'] = nn.Embedding(args.dihedral_embed_bins + 1, dim)
            embeddings['psi_embedding'] = nn.Embedding(args.dihedral_embed_bins + 1, dim)
            embeddings['omega_embedding'] = nn.Embedding(args.dihedral_embed_bins + 1, dim)
            offset += 3 * dim

        if args.fourier_encode_dihedral and args.include_dssp:
            atom_feat_dim += 3 * (fourier_feature_size(args.dihedral_fourier_feats) - 1)
            offset += 3 * (fourier_feature_size(args.dihedral_fourier_feats) - 1)

        if args.embed_ca_dihedral:
            dim = args.ca_dihedral_embed_dim
            embeddings['ca_dihedral_embedding'] = nn.Embedding(38, dim)
            offset += dim + 2
            atom_feat_dim += 2

        if args.embed_centrality:
            cls = nn.Embedding if not args.one_hot_features else OneHotEmb
            embeddings["centrality_embedding"] = cls(12, args.centrality_embed_dim)
            offset += args.centrality_embed_dim if not args.one_hot_features else 12

        if self.use_rosetta_terms:
            scale = 1
            if args.pre_embed_rosetta_score_terms:
                scale = ROSETTA_EMBED_SCALE
                self.rosetta_atom_proj = nn.Linear(
                    self.rs.num_res_score_terms,
                    scale * self.rs.num_res_score_terms
                )
            atom_feat_dim += scale * self.rs.num_res_score_terms
            offset += scale * self.rs.num_res_score_terms

        atom_feat_dim = atom_feat_dim + sum(e.embedding_dim for e in embeddings.values())
        return embeddings, atom_feat_dim if atom_feat_dim > 0 else None, atom_feat_dim - offset

    def init_edge_info(self) -> Tuple[nn.ModuleDict, Optional[int], int]:
        """Initializes edge information for CoordRefiner

        Sets up embedding nn's for (optional) edge attributes. Calculates edge dimension
        based on the options specified in self.args.

        :return: trailing dimension d of edge data passed to the SE3 Transformer. i.e.
        returns d such that edge data passed to transformer is (b,n,n,d), where b is the
        batch size, and n is the number of atoms in the input.
        """
        # determine the shape of edge features
        args, device = self.args, self.device
        edge_dim, static_dim = 0, 0
        embeddings = nn.ModuleDict()

        # Relative Distance Embedding
        multiple_dist_embed = len(args.rel_dist_atom_tys) > 2
        assert len(args.rel_dist_atom_tys) >= 2, f"{args.rel_dist_atom_tys}"
        assert len(args.rel_dist_atom_tys) % 2 == 0, f"{args.rel_dist_atom_tys}"
        if not multiple_dist_embed:
            cls = nn.Embedding if not args.one_hot_features else OneHotEmb
            # backward compatible - adding multiple atom types for distance embedding
            embeddings['dist_embedding'] = cls(args.num_dist_bins + 1,
                                               args.rel_dist_embed_dim)
        else:
            cls = nn.Embedding if not args.one_hot_features else OneHotEmb
            embeddings['dist_embedding'] = nn.ModuleDict()
            for a1, a2 in zip(args.rel_dist_atom_tys[::2], args.rel_dist_atom_tys[1::2]):
                key = f"rel_dist_embedding_{a1}_{a2}"
                embeddings['dist_embedding'][key] = cls(args.num_dist_bins + 1,
                                                        args.rel_dist_embed_dim)
                edge_dim += args.rel_dist_embed_dim if not args.one_hot_features else args.num_dist_bins + 1

        if args.fourier_encode_rel_dist:
            edge_dim += fourier_feature_size(args.num_fourier_rel_dist_feats)

        # TR-Rosetta orientation embeddings
        ori_embeddings = None
        if args.use_tr_rosetta_ori_features:
            edge_dim += args.ori_embed_dim * len(TR_ORI_KEYS)
            ori_embeddings = nn.ModuleDict()
            for ty in TR_ORI_KEYS:
                ori_embeddings[ty.value] = nn.Embedding(args.ori_embed_bins + 1, args.ori_embed_dim)

        if ori_embeddings is not None:
            embeddings['ori_embeddings'] = ori_embeddings

        if args.fourier_encode_rel_sep:
            edge_dim += fourier_feature_size(args.num_fourier_rel_sep_feats) - 1
            static_dim += fourier_feature_size(args.num_fourier_rel_sep_feats) - 1

        if args.embed_rel_sep and not args.joint_embed_res_pair_n_rel_sep:
            sep_bins = DEFAULT_SEP_BINS if not args.use_small_sep_bins else SMALL_SEP_BINS
            cls = nn.Embedding if not args.one_hot_features else OneHotEmb
            embeddings['sep_embedding'] = cls(len(sep_bins) + 2, args.rel_sep_embed_dim)
            static_dim += args.rel_sep_embed_dim if not args.one_hot_features else len(sep_bins) + 2

        if args.embed_res_pairs and not args.joint_embed_res_pair_n_rel_sep:
            cls = nn.Embedding if not args.one_hot_features else OneHotEmb
            dim = 44 if args.one_hot_features else 22
            embeddings['res_pair_embedding_a'] = cls(dim, args.res_pair_embed_dim)
            embeddings['res_pair_embedding_b'] = cls(dim, args.res_pair_embed_dim)
            static_dim += args.res_pair_embed_dim if not args.one_hot_features else dim
            edge_dim -= args.res_pair_embed_dim if not args.one_hot_features else dim

        if args.joint_embed_res_pair_n_rel_sep:
            dim = max(args.rel_sep_embed_dim, args.res_pair_embed_dim)
            embeddings['sep_embedding'] = nn.Embedding(len(DEFAULT_SEP_BINS) + 2,
                                                       dim)
            embeddings['res_pair_embedding_a'] = nn.Embedding(22, dim)
            embeddings['res_pair_embedding_b'] = nn.Embedding(22, dim)
            edge_dim -= 2 * dim
            static_dim += dim

        edge_dim += sum([e.embedding_dim for e in embeddings.values() if
                         (isinstance(e, nn.Embedding) or isinstance(e, OneHotEmb))])

        if self.use_rosetta_terms:
            scale = 1
            if args.pre_embed_rosetta_score_terms:
                scale = ROSETTA_EMBED_SCALE
                self.rosetta_pair_proj = nn.Linear(
                    self.rs.num_res_pair_score_terms,
                    scale * self.rs.num_res_pair_score_terms
                )
            edge_dim += scale * self.rs.num_res_pair_score_terms

        return embeddings, edge_dim if edge_dim > 0 else None, static_dim

    def get_atom_input(self, sample: Sample, detach_coords=True, split=False
                       ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        phi, psi, omega, ss = sample.dssp
        seq, coords = sample.seq_emb, sample.get_atom_coords(
            initial=True, atom_tys=sample.default_atom_ty(initial=True), flatten=True
        )

        # seq and ss should be tensors, but just in case...
        if ss is not None:
            ss = torch.tensor([SS_KEY_MAP[k] for k in ss], dtype=torch.long) \
                if isinstance(ss, str) else ss
            ss = ss if len(ss.shape) == 2 else ss.unsqueeze(0)
        seq = map_seq_to_int(seq) if isinstance(seq, str) else seq

        # make sure all features have batch dimension!

        seq = seq if len(seq.shape) == 2 else seq.unsqueeze(0)

        phi, psi, omega, ss, seq, coords = _to_device(phi, psi, omega, ss, seq, coords, device=self.device)

        device, args = self.device, self.args
        n, atom_feats, embeddings = coords.shape[1], None, self.atom_embeddings

        if args.embed_res_ty:
            emb = self.atom_embeddings['res_embedding']
            _verify_embedding(emb, seq, 'res_embedding')
            atom_feats = safe_cat(atom_feats, emb(seq.long()), dim=-1)

        if args.embed_rel_pos:
            start, end, length = sample.crop
            rel_pos = torch.arange(start=start, end=end, device=device) / length
            positions = (rel_pos[None, :] * args.rel_pos_embed_bins).long()
            emb = embeddings['rel_pos_embedding']
            _verify_embedding(emb, positions, 'rel_pos_embedding')
            atom_feats = safe_cat(atom_feats, emb(positions), dim=-1)

        if split:
            split_idx = atom_feats.shape[-1]

        if 'ss_embedding' in embeddings and ss is not None:
            emb = embeddings['ss_embedding']
            _verify_embedding(emb, ss, 'ss_embedding')
            atom_feats = safe_cat(atom_feats, emb(ss.long()), dim=-1)
        else:
            assert not args.embed_sec_struct

        if 'cen_dist_embedding' in embeddings:
            # indicating how close each atom is to the proteins core
            emb = embeddings['cen_dist_embedding']
            centroid = torch.mean(coords, dim=1, keepdim=True)
            dists = torch.norm(coords - centroid, dim=-1)
            dists = 10 * dists / (1e-10 + torch.max(dists))
            bins = torch.round(dists)
            _verify_embedding(emb, bins, 'cen_dist_embedding')
            atom_feats = safe_cat(atom_feats, emb(bins.long()), dim=-1)
        else:
            assert not args.embed_cen_dist

        if args.embed_dihedral and args.include_dssp:
            angs = [phi, psi, omega]
            angs = map(lambda a: (((a / np.pi) + 1) / 2) * args.dihedral_embed_bins, angs)
            for ang, ty in zip(angs, "phi psi omega".split(" ")):
                _verify_embedding(embeddings[f'{ty}_embedding'], ang, f'{ty}_embedding')
                atom_feats = safe_cat(
                    atom_feats,
                    embeddings[f'{ty}_embedding'](ang.long()), dim=-1
                )

        if args.fourier_encode_dihedral and args.include_dssp:
            angs = [phi, psi, omega]
            for ang in angs:
                with torch.set_grad_enabled(not detach_coords):
                    enc_ang = fourier_encode(ang.to(device), args.dihedral_fourier_feats, include_self=False).squeeze(
                        -1)
                atom_feats = safe_cat(atom_feats, enc_ang.detach().requires_grad_(True), dim=-1)

        if 'ca_dihedral_embedding' in embeddings:
            with torch.set_grad_enabled(not detach_coords):
                ca_crds = coords.squeeze(-2)
                offset_ca_crds = torch.zeros((4, n, 3), device=device)
                for i in range(4):
                    offset_ca_crds[i, i:n - (3 - i), :] = ca_crds[0, i:n - (3 - i), :]
                angs = signed_dihedral_4_torch(offset_ca_crds).unsqueeze(0)
                fec = fourier_encode(angs, num_encodings=1, include_self=False).squeeze(-1)
                angs = (((angs / np.pi) + 1) / 2) * 36
                angs[:, -1] = 37
                angs[:, :2] = 37

            atom_feats = safe_cat(
                atom_feats,
                embeddings['ca_dihedral_embedding'](angs.to(device).long()), dim=-1
            )
            atom_feats = safe_cat(atom_feats, fec, dim=-1)

        if args.embed_centrality:
            emb = embeddings["centrality_embedding"]
            with torch.set_grad_enabled(not detach_coords):
                # b x n x n
                ca_dists = torch.cdist(coords.squeeze(-2), coords.squeeze(-2))
                centrality = torch.sum((ca_dists <= args.valid_radius).float(), dim=-1)
                centrality = torch.clamp_max((centrality - 1) // 8, 6).long() #TODO: changed from 11 to 6
            _verify_embedding(emb, centrality, "centrality_embedding")
            atom_feats = safe_cat(atom_feats, emb(centrality), dim=-1)

        if not split:
            return atom_feats
        return atom_feats[..., :split_idx], atom_feats[..., split_idx:]  # noqa

    def get_fourier_encoded_sep_mat(self) -> torch.Tensor:
        if self.fourier_encoded_sep_mat is None:
            temp = torch.randn((1, MAX_SEQ_LEN + 1, 1))
            self.fourier_encoded_sep_mat = fourier_encode_sep_mat(
                temp,
                self.args.num_fourier_rel_sep_feats,
                sep_mat=SEQ_SEP_MAT_FLOAT,
                normalize_sep=False,
                include_self=False,
            ).to(self.device)
            self.fourier_encoded_sep_mat.requires_grad_(True)
        return self.fourier_encoded_sep_mat.detach()

    def get_edge_input(self, sample: Sample, detach_coords=True, split=False
                       ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Gets edge input for a data sample

        Given a data sample for refinement (coords, sequence, mask), this method
        generates the SE3-Transformer's edge input for the sample.

        The information gathered in this method is specified in self.args.
        The output from this function is the concatenation of 0 or more of the following:

        - Bond Matrix Embedding
            *Embed bond information (by bond type) between adjacent atoms.
            *This information can (optinally) be added to the Contact Graph before embedding.
            By specifying args.combine_bond_and_contact_embedding, in which case, the embedding
            dimension is taken as max(bond_embed_dim,dist_embed_dim)

        - Distance Matrix Embedding - (Contact graph)
            * Embed pairwise information between all atoms within distance args.max_embed_dist
            * Use args.n_dist_bins + 1 tokens for embedding [key = min(d_ij//args.n_dist_bins,args.n_dist_bins)]

        - Encode distances (fourier encode)
            *

        (5) Generate TrRosetta Orientation information [EDGE]
        (5a) Embed orientation (angle and or dihedral) [EDGE]
        (5b) Encode orientation (fourier encode, angle and or dihedral) [EDGE]
        (7) Embed sequence separation (by seq. sep or #bonds = graph distance)  [EDGE]
        (8) Encode sequence separation (by seq. sep or by #bonds) (fourier encode) [EDGE]
        (9) Embed residue type [ATOM]
        (10) Embed atom type [ATOM]
        (11) encode coords (fourier encoding) [COORD]
        (12) Create adjacency matrix (from rel distance, bond type, etc) [EDGE]
        (13) Embed or encode adjacency matrix [EDGE]
        """

        args, device, n = self.args, self.device, len(sample.seq_str)
        edge_info, dists, bond_mat = None, None, None
        embeddings = self.edge_embeddings

        if args.fourier_encode_rel_sep:
            mat = self.get_fourier_encoded_sep_mat()
            enc = mat[:, :n, :n, :].detach().clone().requires_grad_(True)
            edge_info = safe_cat(edge_info, enc, dim=-1)

        if args.embed_rel_sep and not args.joint_embed_res_pair_n_rel_sep:
            keys = self.encoded_seq_sep_mat[:, :n, :n]
            _verify_embedding(embeddings['sep_embedding'], keys, 'sep_embedding')
            edge_info = safe_cat(edge_info, embeddings['sep_embedding'](keys), dim=-1)

        if args.embed_res_pairs and not args.joint_embed_res_pair_n_rel_sep:
            res_keys = sample.seq_emb
            _verify_embedding(embeddings['res_pair_embedding_a'], res_keys, "res pair embedding a")
            offset = 22 if args.one_hot_features else 0
            res_emb1 = embeddings['res_pair_embedding_a'](res_keys)
            res_emb2 = embeddings['res_pair_embedding_b'](res_keys + offset)
            res_emb = rearrange(res_emb1, 'b n d-> b n () d') + \
                      rearrange(res_emb2, 'b n d-> b () n d')  # noqa
            edge_info = safe_cat(edge_info, res_emb, dim=-1)

        if args.joint_embed_res_pair_n_rel_sep:
            sep_keys = self.encoded_seq_sep_mat[:, :n, :n]
            _verify_embedding(embeddings['sep_embedding'], sep_keys, 'sep_embedding')
            sep_emb = embeddings['sep_embedding'](sep_keys)

            res_keys = sample.seq_emb
            res_emb1 = embeddings['res_pair_embedding_a'](res_keys)
            res_emb2 = embeddings['res_pair_embedding_b'](res_keys)
            res_emb = rearrange(res_emb1, 'b n d-> b n () d') + \
                      rearrange(res_emb2, 'b n d-> b () n d')  # noqa
            edge_info = safe_cat(edge_info, res_emb + sep_emb, dim=-1)

            if split:
                split_idx = edge_info.shape[-1]

        # distance embedding
        for a1, a2 in zip(args.rel_dist_atom_tys[::2], args.rel_dist_atom_tys[1::2]):
            emb = embeddings['dist_embedding']
            key = f"rel_dist_embedding_{a1}_{a2}"
            emb = emb[key] if len(args.rel_dist_atom_tys) > 2 else emb
            a1_coords = sample.get_atom_coords(initial=True, flatten=True, atom_tys=a1)
            a2_coords = sample.get_atom_coords(initial=True, flatten=True, atom_tys=a2) \
                if a1 != a2 else a1_coords
            with torch.set_grad_enabled(not detach_coords):
                dists = torch.cdist(a1_coords, a2_coords)
                dist_keys = key_encode_contact_mat(coords=None,
                                                   dists=dists.detach(),
                                                   n_bins=args.num_dist_bins,
                                                   min_dist=args.min_embed_dist,
                                                   max_dist=args.max_embed_dist)
            _verify_embedding(emb, dist_keys, key)
            edge_info = safe_cat(edge_info, emb(dist_keys.to(device)), dim=-1)

        if args.fourier_encode_rel_dist:
            ca_crds = sample.get_atom_coords(
                initial=True, flatten=True, atom_tys=sample.default_atom_ty(True)
            )
            with torch.set_grad_enabled(not detach_coords):
                enc = fourier_encode_distances(
                    coords=ca_crds,
                    num_encodings=args.num_fourier_rel_dist_feats,
                    dists=dists,
                    max_dist=args.max_embed_dist,
                    normalize_dist=True,
                )
            edge_info = safe_cat(edge_info, enc.requires_grad_(True), dim=-1)

        if args.use_tr_rosetta_ori_features:
            ori_mats = {}
            vCB = "CB" if sample.has_atom(atom_ty="CB", initial=True) else "O"
            N_CA_CB = sample.get_atom_coords(
                initial=True, atom_tys=['N', 'CA', vCB]).squeeze()
            N, CA, CB = N_CA_CB[..., 0, :], N_CA_CB[..., 1, :], N_CA_CB[..., 2, :]
            with torch.set_grad_enabled(not detach_coords):
                for ty in TR_ORI_KEYS:
                    ori_mats[ty.value] = get_orientation_mat(N=N, CA=CA, CB=CB, orientation_type=ty)
            for ty in TR_ORI_KEYS:
                inp = key_encode_ori_mat(ori_mats[ty.value], args.ori_embed_bins, min_val=-np.pi)
                _verify_embedding(embeddings['ori_embeddings'][ty.value], inp, f'ori_embeddings_{ty}')
                edge_info = safe_cat(edge_info, embeddings['ori_embeddings'][ty.value](inp).unsqueeze(0), dim=-1)
        if not split:
            return edge_info
        else:
            return edge_info[..., :split_idx], edge_info[..., split_idx:]  # noqa

    def generate_input(self, refine_sample: Sample, detach_coords=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (1) Align training and coords to native (x)
        (2) Embed bonds [EDGE]
        (3) Embed distances -> Contact graph [EDGE]
        (4) Encode distances (fourier encode) [EDGE]
        (5) Generate TrRosetta Orientation information [EDGE]
        (5a) Embed orientation (angle and or dihedral) [EDGE]
        (5b) Encode orientation (fourier encode, angle and or dihedral) [EDGE]
        (7) Embed sequence separation (by seq. sep or #bonds = graph distance)  [EDGE]
        (8) Encode sequence separation (by seq. sep or by #bonds) (fourier encode) [EDGE]
        (9) Embed residue type [ATOM]
        (10) Embed atom type [ATOM]
        (11) encode coords (fourier encoding) [COORD]
        (12) Create adjacency matrix (from rel distance, bond type, etc) [EDGE]
        (13) Embed or encode adjacency matrix [EDGE]
        (14) add chemical properties to atom embedding type [ATOM]
        (16) Multiple rounds of refinement -> include previous coord diffs
        (17) Secondary structure as atom feature [ATOM]
        (18) dihedral (phi, psi, omega) as atom features [ATOM]
        (19) solvent accessability as atom feature (calc by dssp) [ATOM]
        """
        sample = refine_sample.to_device(self.device)
        edge_feats = self.get_edge_input(sample, detach_coords=detach_coords)
        atom_feats = self.get_atom_input(sample, detach_coords=detach_coords)
        if self.use_rosetta_terms:
            rosetta_atom_feats = sample.rosetta_ens_single
            rosetta_edge_feats = sample.rosetta_ens_pair
            # hopefully normalizing these features before concat
            rosetta_edge_feats = self.rosetta_pair_proj(rosetta_edge_feats.to(self.device))
            rosetta_atom_feats = self.rosetta_atom_proj(rosetta_atom_feats.to(self.device))
            edge_feats = safe_cat(edge_feats, rosetta_edge_feats, dim=-1)
            atom_feats = safe_cat(atom_feats, rosetta_atom_feats, dim=-1)
            # print(atom_feats.shape, edge_feats.shape, rosetta_atom_feats.shape, rosetta_edge_feats.shape, self.dims)
        return atom_feats, edge_feats
