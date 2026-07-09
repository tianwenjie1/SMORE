# coding: utf-8
# rongqing001@e.ntu.edu.sg
r"""
SMORE - Multi-modal Recommender System
Reference:
    ACM WSDM 2025: Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation

Reference Code:
    https://github.com/kennethorq/SMORE
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class SMORE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SMORE, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        self.dropout_rate = config['dropout_rate']
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.image_knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.text_knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda() 

        self.fusion_adj = self.max_pool_fusion()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        self.query_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_f = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_fusion_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.image_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.text_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.fusion_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))

        # ============================================================
        # Innovation 1: Frequency Band Gating (频段门控)
        # ============================================================
        self.freq_band_gating = config['freq_band_gating'] or False
        if self.freq_band_gating:
            freq_dim = self.embedding_dim // 2 + 1
            self.image_band_gate = nn.Sequential(
                nn.Linear(freq_dim, freq_dim),
                nn.Sigmoid()
            )
            self.text_band_gate = nn.Sequential(
                nn.Linear(freq_dim, freq_dim),
                nn.Sigmoid()
            )
            self.fusion_band_gate = nn.Sequential(
                nn.Linear(freq_dim, freq_dim),
                nn.Sigmoid()
            )

        # ============================================================
        # Innovation 2: Modality Reliability Gating (模态可靠性门控)
        # ============================================================
        self.modality_reliability_gating = config['modality_reliability_gating'] or False
        if self.modality_reliability_gating:
            freq_dim = self.embedding_dim // 2 + 1
            self.reliability_estimator = nn.Sequential(
                nn.Linear(freq_dim * 3, freq_dim),
                nn.ReLU(),
                nn.Linear(freq_dim, 3),
            )

        # ============================================================
        # Innovation 3: Modality Dropout Robust Training (模态Dropout鲁棒训练)
        # ============================================================
        self.modality_dropout_rate = config['modality_dropout_rate'] if config['modality_dropout_rate'] is not None else 0.0

        # ============================================================
        # Innovation 4: Graph Edge Reweighting (图边重加权)
        # ============================================================
        self.graph_edge_reweighting = config['graph_edge_reweighting'] or False
        if self.graph_edge_reweighting:
            R_coo = self.interaction_matrix.tocoo()
            self.register_buffer('ui_edge_user', torch.LongTensor(R_coo.row))
            self.register_buffer('ui_edge_item', torch.LongTensor(R_coo.col))
            self.n_edges = len(R_coo.row)
            self.edge_weight_mlp = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, 1),
                nn.Softplus()
            )

        # ============================================================
        # Robustness Evaluation (推理阶段模态扰动, 不影响训练)
        # MQS (Modality Quality Shift) protocol:
        #   normal / drop_image / drop_text / noise_image / noise_text / noise_both
        #   shuffle_image / shuffle_text / mismatch        (use robust_shift_ratio)
        #   tail_noise_image / tail_noise_text / tail_noise_both / tail_missing_image / tail_missing_text
        #   pop_missing_image / pop_missing_text            (popularity-correlated missing)
        # ============================================================
        self.robust_eval_mode = config['robust_eval_mode'] or 'normal'
        self.robust_noise_std = config['robust_noise_std'] if config['robust_noise_std'] is not None else 0.1
        self.robust_shift_ratio = config['robust_shift_ratio'] if config['robust_shift_ratio'] is not None else 0.3
        self.robust_tail_ratio = config['robust_tail_ratio'] if config['robust_tail_ratio'] is not None else 0.3

        # tail item mask: bottom `robust_tail_ratio` items by interaction count
        # (popularity = item degree in the user-item graph)
        item_degree = np.array(self.interaction_matrix.sum(axis=0)).flatten().astype(np.int64)
        n_items = self.n_items
        n_tail = max(1, int(n_items * self.robust_tail_ratio))
        # argsort ascending -> first n_tail are least popular
        tail_idx = np.argsort(item_degree)[:n_tail]
        tail_mask = np.zeros(n_items, dtype=bool)
        tail_mask[tail_idx] = True
        self.register_buffer('tail_mask', torch.from_numpy(tail_mask))
        # popularity-based missing probability (colder -> higher miss prob),
        # normalized so max prob == robust_shift_ratio
        deg = np.maximum(item_degree, 1).astype(np.float64)
        pop_miss_prob = (1.0 / np.log1p(deg))
        if pop_miss_prob.max() > 0:
            pop_miss_prob = pop_miss_prob / pop_miss_prob.max() * self.robust_shift_ratio
        self.register_buffer('pop_miss_prob', torch.from_numpy(pop_miss_prob.astype(np.float32)))
        # item degree (popularity) exposed for Coverage/AvgPopularity metrics
        self.register_buffer('item_degree', torch.from_numpy(item_degree.astype(np.float32)))

        # ============================================================
        # MQR: Modality-Quality Preference Stabilization (training-side)
        # Builds a degraded view each batch and enforces preference stability
        # between clean and degraded views. NOT plain dropout: the degraded view
        # samples a quality *environment* (noise/mismatch/tail-noise) and the
        # stability loss acts on ranking scores, tail-weighted.
        # ============================================================
        self.mqr_enabled = config['mqr_enabled'] or False
        self.mqr_alpha = config['mqr_alpha'] if config['mqr_alpha'] is not None else 0.5
        self.mqr_beta = config['mqr_beta'] if config['mqr_beta'] is not None else 0.2
        self.mqr_tau = config['mqr_tau'] if config['mqr_tau'] is not None else 1.0
        self.mqr_tail_weight = config['mqr_tail_weight'] if config['mqr_tail_weight'] is not None else True
        # naive noise augmentation (ablation baseline, NOT MQR): add noise to
        # features during the clean training forward only.
        self.train_noise_std = config['train_noise_std'] if config['train_noise_std'] is not None else 0.0
        # item-degree-based stability weight: tail items get larger weight
        deg_w = 1.0 / np.log1p(np.maximum(item_degree, 1).astype(np.float64) + 1.0)
        deg_w = deg_w / deg_w.mean()
        self.register_buffer('item_stab_weight', torch.from_numpy(deg_w.astype(np.float32)))


    def pre_epoch_processing(self):
        pass

    def _build_reweighted_adj(self, edge_weights):
        """Build reweighted normalized adjacency matrix for Graph Edge Reweighting.

        Args:
            edge_weights: [n_edges] tensor of learned weights for user-item interactions

        Returns:
            Sparse tensor of reweighted symmetric-normalized adjacency
        """
        n = self.n_users + self.n_items

        # User -> Item edges
        ui_rows = self.ui_edge_user
        ui_cols = self.ui_edge_item + self.n_users
        ui_vals = edge_weights

        # Item -> User edges (symmetric)
        iu_rows = ui_cols
        iu_cols = ui_rows
        iu_vals = edge_weights

        # Combine into full bipartite adjacency
        all_rows = torch.cat([ui_rows, iu_rows])
        all_cols = torch.cat([ui_cols, iu_cols])
        all_vals = torch.cat([ui_vals, iu_vals])

        indices = torch.stack([all_rows, all_cols])
        adj = torch.sparse.FloatTensor(indices, all_vals, torch.Size([n, n]))

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        # Detach degree from autograd to avoid OOM on large graphs during backward.
        # Gradients still flow through edge weights (all_vals); only the degree
        # normalization is treated as fixed (standard approximation in edge-reweighting).
        with torch.no_grad():
            row_sum = torch.sparse.sum(adj, dim=1).to_dense()
            d_inv_sqrt = torch.pow(row_sum + 1e-10, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            norm_factor = d_inv_sqrt[all_rows] * d_inv_sqrt[all_cols]

        all_vals_norm = all_vals * norm_factor
        norm_adj = torch.sparse.FloatTensor(indices, all_vals_norm, torch.Size([n, n]))
        return norm_adj.coalesce()

    def max_pool_fusion(self):
        image_adj = self.image_original_adj.coalesce()
        text_adj = self.text_original_adj.coalesce()

        image_indices = image_adj.indices().to(self.device)
        image_values = image_adj.values().to(self.device)
        text_indices = text_adj.indices().to(self.device)
        text_values = text_adj.values().to(self.device)

        combined_indices = torch.cat((image_indices, text_indices), dim=1)
        combined_indices, unique_idx = torch.unique(combined_indices, dim=1, return_inverse=True)

        combined_values_image = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)
        combined_values_text = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)

        combined_values_image[unique_idx[:image_indices.size(1)]] = image_values
        combined_values_text[unique_idx[image_indices.size(1):]] = text_values
        combined_values, _ = torch.max(torch.stack((combined_values_image, combined_values_text)), dim=0)

        fusion_adj = torch.sparse.FloatTensor(combined_indices, combined_values, image_adj.size()).coalesce()

        return fusion_adj

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def spectrum_convolution(self, image_embeds, text_embeds):
        """
        Modality Denoising & Cross-Modality Fusion
        With optional Frequency Band Gating (Innovation 1) and
        spectral statistics for Modality Reliability Gating (Innovation 2)
        """
        image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')
        text_fft = torch.fft.rfft(text_embeds, dim=1, norm='ortho')

        # ============================================================
        # Innovation 1: Frequency Band Gating
        # Input-dependent gating: emphasize informative frequency bands,
        # suppress noisy ones based on spectrum magnitude
        # ============================================================
        if self.freq_band_gating:
            image_mag = torch.abs(image_fft)   # [n_items, freq_dim]
            text_mag = torch.abs(text_fft)      # [n_items, freq_dim]

            image_gate = self.image_band_gate(image_mag)   # [n_items, freq_dim]
            text_gate = self.text_band_gate(text_mag)       # [n_items, freq_dim]

            # Apply gates before complex weight multiplication
            image_fft = image_fft * image_gate
            text_fft = text_fft * text_gate

        # ============================================================
        # Innovation 2: Compute spectral statistics for reliability gating
        # ============================================================
        spectral_stats = None
        if self.modality_reliability_gating:
            image_mag = torch.abs(image_fft)
            text_mag = torch.abs(text_fft)
            fusion_mag = torch.abs(image_fft * text_fft)
            spectral_stats = torch.cat([image_mag, text_mag, fusion_mag], dim=1)  # [n_items, 3*freq_dim]

        image_complex_weight = torch.view_as_complex(self.image_complex_weight)
        text_complex_weight = torch.view_as_complex(self.text_complex_weight)
        fusion_complex_weight = torch.view_as_complex(self.fusion_complex_weight)

        #   Uni-modal Denoising
        image_conv = torch.fft.irfft(image_fft * image_complex_weight, n=image_embeds.shape[1], dim=1, norm='ortho')
        text_conv = torch.fft.irfft(text_fft * text_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')

        #   Cross-modality fusion
        fusion_fft = image_fft * text_fft * fusion_complex_weight
        if self.freq_band_gating:
            fusion_mag = torch.abs(fusion_fft)
            fusion_gate = self.fusion_band_gate(fusion_mag)
            fusion_fft = fusion_fft * fusion_gate
        fusion_conv = torch.fft.irfft(fusion_fft, n=text_embeds.shape[1], dim=1, norm='ortho')

        return image_conv, text_conv, fusion_conv, spectral_stats
    
    def _apply_mqs(self, image_feats, text_feats, mode, std, ratio):
        """Apply a modality quality shift to item features in-place semantics.
        Returns (image_feats, text_feats) perturbed. Used both at inference
        (eval MQS protocol) and during training (MQR degraded view)."""
        if mode == 'normal':
            return image_feats, text_feats
        has_img = self.v_feat is not None
        has_txt = self.t_feat is not None
        n = image_feats.shape[0] if has_img else text_feats.shape[0]
        device = image_feats.device if has_img else text_feats.device

        # full drop / noise
        if mode == 'drop_image' and has_img:
            image_feats = torch.zeros_like(image_feats)
        if mode == 'drop_text' and has_txt:
            text_feats = torch.zeros_like(text_feats)
        if mode in ('noise_image', 'noise_both') and has_img:
            image_feats = image_feats + torch.randn_like(image_feats) * std
        if mode in ('noise_text', 'noise_both') and has_txt:
            text_feats = text_feats + torch.randn_like(text_feats) * std

        # feature shuffle / mismatch
        if mode in ('shuffle_image', 'shuffle_text', 'mismatch'):
            perm = torch.randperm(n, device=device)
            mask = torch.rand(n, device=device) < ratio
            if mode == 'shuffle_image' and has_img:
                image_feats = torch.where(mask.unsqueeze(1), image_feats[perm], image_feats)
            elif mode == 'shuffle_text' and has_txt:
                text_feats = torch.where(mask.unsqueeze(1), text_feats[perm], text_feats)
            elif mode == 'mismatch' and has_img and has_txt:
                image_feats = torch.where(mask.unsqueeze(1), image_feats[perm], image_feats)

        # tail-only noise / missing
        if mode.startswith('tail_') and self.tail_mask is not None:
            tm = self.tail_mask.to(device)
            if mode in ('tail_noise_image', 'tail_noise_both') and has_img:
                image_feats = torch.where(tm.unsqueeze(1),
                                          image_feats + torch.randn_like(image_feats) * std, image_feats)
            if mode in ('tail_noise_text', 'tail_noise_both') and has_txt:
                text_feats = torch.where(tm.unsqueeze(1),
                                         text_feats + torch.randn_like(text_feats) * std, text_feats)
            if mode == 'tail_missing_image' and has_img:
                image_feats = torch.where(tm.unsqueeze(1), torch.zeros_like(image_feats), image_feats)
            if mode == 'tail_missing_text' and has_txt:
                text_feats = torch.where(tm.unsqueeze(1), torch.zeros_like(text_feats), text_feats)

        # popularity-correlated missing
        if mode in ('pop_missing_image', 'pop_missing_text'):
            prob = self.pop_miss_prob.to(device)
            miss = torch.rand(n, device=device) < prob
            if mode == 'pop_missing_image' and has_img:
                image_feats = torch.where(miss.unsqueeze(1), torch.zeros_like(image_feats), image_feats)
            if mode == 'pop_missing_text' and has_txt:
                text_feats = torch.where(miss.unsqueeze(1), torch.zeros_like(text_feats), text_feats)

        return image_feats, text_feats

    def forward(self, adj, train=False, degrade_env=None):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        # ============================================================
        # MQS perturbation:
        #   - inference (train=False): use self.robust_eval_mode (eval protocol)
        #   - training degraded view (degrade_env set): use the sampled env (MQR)
        # ============================================================
        if degrade_env is not None:
            image_feats, text_feats = self._apply_mqs(
                image_feats, text_feats, degrade_env,
                self.robust_noise_std, self.robust_shift_ratio)
        elif not train and self.robust_eval_mode != 'normal':
            image_feats, text_feats = self._apply_mqs(
                image_feats, text_feats, self.robust_eval_mode,
                self.robust_noise_std, self.robust_shift_ratio)
        elif train and self.train_noise_std > 0:
            # naive noise augmentation (ablation baseline, independent of MQR)
            if self.v_feat is not None:
                image_feats = image_feats + torch.randn_like(image_feats) * self.train_noise_std
            if self.t_feat is not None:
                text_feats = text_feats + torch.randn_like(text_feats) * self.train_noise_std

        #   Spectrum Modality Fusion
        image_conv, text_conv, fusion_conv, spectral_stats = self.spectrum_convolution(image_feats, text_feats)
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_conv))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_conv))
        fusion_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_conv))

        #   User-Item (Behavioral) View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        # ============================================================
        # Innovation 4: Graph Edge Reweighting
        # Learn edge weights from multimodal signals, replacing uniform 0/1
        # ============================================================
        if self.graph_edge_reweighting and train:
            user_repr = self.user_embedding.weight[self.ui_edge_user]
            item_mod_repr = (image_item_embeds[self.ui_edge_item] +
                             text_item_embeds[self.ui_edge_item] +
                             fusion_item_embeds[self.ui_edge_item]) / 3.0
            edge_input = torch.cat([user_repr, item_mod_repr], dim=1)
            edge_weights = self.edge_weight_mlp(edge_input).squeeze(-1)
            gcn_adj = self._build_reweighted_adj(edge_weights)
        else:
            gcn_adj = adj

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(gcn_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        #   Item-Item Modality Specific and Fusion views
        #   Image-view
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        #   Text-view
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        #   Fusion-view
        if self.sparse:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        else:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        # ============================================================
        # Innovation 3: Modality Dropout Robust Training
        # Randomly drop entire modality views during training
        # ============================================================
        if self.training and self.modality_dropout_rate > 0:
            drop_image = torch.rand(1).item() < self.modality_dropout_rate
            drop_text = torch.rand(1).item() < self.modality_dropout_rate
            drop_fusion = torch.rand(1).item() < self.modality_dropout_rate

            # Ensure at least one modality is kept
            if drop_image and drop_text and drop_fusion:
                keep_idx = torch.randint(0, 3, (1,)).item()
                drop_image = (keep_idx != 0)
                drop_text = (keep_idx != 1)
                drop_fusion = (keep_idx != 2)

            if drop_image:
                image_embeds = torch.zeros_like(image_embeds)
            if drop_text:
                text_embeds = torch.zeros_like(text_embeds)
            if drop_fusion:
                fusion_embeds = torch.zeros_like(fusion_embeds)

        #   Modality-aware Preference Module
        fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
        fusion_soft_v = self.softmax(fusion_att_v)
        agg_image_embeds = fusion_soft_v * image_embeds

        fusion_soft_t = self.softmax(fusion_att_t)
        agg_text_embeds = fusion_soft_t * text_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        fusion_prefer = self.gate_fusion_prefer(content_embeds)
        image_prefer, text_prefer, fusion_prefer = self.dropout(image_prefer), self.dropout(text_prefer), self.dropout(fusion_prefer)

        agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
        agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds)
        fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds)

        # ============================================================
        # Innovation 2: Modality Reliability Gating
        # Learn per-item modality reliability weights instead of uniform mean
        # ============================================================
        if self.modality_reliability_gating and spectral_stats is not None:
            reliability_scores = self.reliability_estimator(spectral_stats)   # [n_items, 3]
            reliability_weights = F.softmax(reliability_scores, dim=1)        # [n_items, 3]

            # Users have no spectral features -> uniform weights
            user_weights = torch.ones(self.n_users, 3, device=self.device) / 3.0
            full_weights = torch.cat([user_weights, reliability_weights], dim=0)  # [n_users+n_items, 3]

            stacked = torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds], dim=0)  # [3, n, dim]
            # full_weights: [n, 3] -> [3, n, 1] to match stacked [3, n, dim]
            side_embeds = torch.sum(stacked * full_weights.t().unsqueeze(-1), dim=0)
        else:
            side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0)

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def _sample_mqr_env(self):
        """Sample a modality-quality environment for the degraded view."""
        import random
        return random.choice(['noise_both', 'mismatch', 'tail_noise_both'])

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # ----- clean view -----
        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        total_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss

        # ----- MQR: Modality-Quality Preference Stabilization -----
        # Build a degraded view by sampling a quality environment, then enforce
        # (a) BPR on the degraded view, and (b) preference-stability (ranking
        # consistency) between clean and degraded scores, tail-weighted.
        if self.mqr_enabled:
            env = self._sample_mqr_env()
            ua_d, ia_d, _, _ = self.forward(self.norm_adj, train=True, degrade_env=env)

            u_d = ua_d[users]
            pos_d = ia_d[pos_items]
            neg_d = ia_d[neg_items]

            # (a) BPR on degraded view
            pos_scores_d = torch.sum(torch.mul(u_d, pos_d), dim=1)
            neg_scores_d = torch.sum(torch.mul(u_d, neg_d), dim=1)
            bpr_degraded = -torch.mean(F.logsigmoid(pos_scores_d - neg_scores_d))

            # (b) preference-stability loss: KL between clean & degraded score
            # distributions over the candidate set {pos, neg}, per user.
            pos_scores_c = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), dim=1)
            neg_scores_c = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), dim=1)
            s_c = torch.stack([pos_scores_c, neg_scores_c], dim=0)  # [2, B]
            s_d = torch.stack([pos_scores_d, neg_scores_d], dim=0)  # [2, B]
            logp_c = F.log_softmax(s_c / self.mqr_tau, dim=0)
            p_d = F.softmax(s_d / self.mqr_tau, dim=0)
            per_user_ps = F.kl_div(logp_c, p_d, reduction='none').sum(dim=0)  # [B]

            # tail-sensitive weight: tail items (by pos_item degree) weighted more
            if self.mqr_tail_weight:
                w = self.item_stab_weight.to(per_user_ps.device)[pos_items]
                ps_loss = (per_user_ps * w).mean()
            else:
                ps_loss = per_user_ps.mean()

            total_loss = total_loss + self.mqr_alpha * bpr_degraded + self.mqr_beta * ps_loss

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores