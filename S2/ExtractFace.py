import torch
import numpy as np
import torch.nn as nn



class SurfExtract(nn.Module):

    def __init__(self, confidence_top1=0.8, confidence_top2=0.5, angle=120,
                 quality_weight=0.0, quality_target=0.2):
        super().__init__()
        if not 0 <= confidence_top1 <= 1 or not 0 <= confidence_top2 <= 1:
            raise ValueError("Confidence thresholds must be between 0 and 1.")
        if not 0 < angle < 180:
            raise ValueError("Face angle must be strictly between 0 and 180 degrees.")
        if not 0 <= quality_weight <= 1 or not 0 <= quality_target <= 1:
            raise ValueError("Quality weight and target must be between 0 and 1.")
        self.quality_weight = float(quality_weight)
        self.quality_target = float(quality_target)
        self.conf_1 = float(confidence_top1)
        self.conf_2 = float(confidence_top2)
        self.angle = float(angle)
    
    def _torch_meshgrid_2D(self, input):
        assert(input.dim()==2)
        N, K = input.size()
        expand_3rd = input[...,None].tile(1,1,K)
        expand_2nd = input[:,None,:].tile(1,K,1)
        C = torch.stack([expand_3rd,expand_2nd], dim=3)
        output = torch.reshape(C, [N,-1,2])
        return output   
    
    def _build_candidate_triangles(self, knn_indices):    
        cand_edges = self._torch_meshgrid_2D(knn_indices[:,1:])
        cand_triangles = torch.cat([knn_indices[:,:1].repeat(1,cand_edges.shape[1])[...,None], cand_edges], dim=2)
        return cand_triangles
    
    def forward(self, points, pred_logits, knn_indices):
        N, K = knn_indices.shape
        pred_logits = pred_logits.reshape(N, K-1, K-1)
        pred_logits, indices = torch.sort(pred_logits, dim=-1, descending=True) # sort from confidence high to low
        # Only the first two candidates per edge are consumed below. Keep the
        # original sort so tied logits select the same neighbors as before,
        # but construct just those triangles instead of all (K-1)^2 triples.
        neighbors = knn_indices[:, 1:]
        other_vertices = torch.gather(
            neighbors[:, None, :].expand(-1, K-1, -1), 2, indices[:, :, :2])
        sorted_candidate_triangles = torch.stack([
            knn_indices[:, :1, None].expand(-1, K-1, 2),
            neighbors[:, :, None].expand(-1, -1, 2),
            other_vertices,
        ], dim=-1)
        
        # get flip sign of the top 2 triangles
        triangle_A = sorted_candidate_triangles[:,:,0,:]     # top 1st triangle
        triangle_B = sorted_candidate_triangles[:,:,1,:]     # top 2nd triangle

        center_pt = knn_indices[:,:1]
        edge_pt = triangle_A[:,:,1]
        other_edge_A = triangle_A[:,:,2]
        other_edge_B = triangle_B[:,:,2]
        assert(torch.all(triangle_A[:,:,0]==triangle_B[:,:,0]))
        assert(torch.all(triangle_A[:,:,1]==triangle_B[:,:,1]))

        edge = points[edge_pt] - points[center_pt]
        edge_A = points[other_edge_A] - points[edge_pt]
        edge_B = points[other_edge_B] - points[edge_pt]

        normal_A = torch.cross(edge, edge_A, dim=-1)
        normal_B = torch.cross(edge, edge_B, dim=-1)
        threshold_2 = self.conf_2
        if self.quality_weight:
            # q = 4*sqrt(3)*area / sum(squared edge lengths), in [0, 1].
            # A regular pair may use a lower threshold; a slender pair uses a
            # higher one. The decision uses input geometry, never ground truth.
            edge_sq = edge.square().sum(dim=-1)
            denom_A = edge_sq + edge_A.square().sum(dim=-1) + (edge + edge_A).square().sum(dim=-1)
            denom_B = edge_sq + edge_B.square().sum(dim=-1) + (edge + edge_B).square().sum(dim=-1)
            epsilon = torch.finfo(points.dtype).tiny
            quality_A = 2*np.sqrt(3)*torch.linalg.norm(normal_A, dim=-1)/denom_A.clamp_min(epsilon)
            quality_B = 2*np.sqrt(3)*torch.linalg.norm(normal_B, dim=-1)/denom_B.clamp_min(epsilon)
            pair_quality = torch.minimum(quality_A, quality_B).clamp(0, 1)
            threshold_2 = (self.conf_2 + self.quality_weight*(self.quality_target-pair_quality)).clamp(0, 1)
        normal_A = normal_A/torch.linalg.norm(normal_A, axis=-1, keepdim=True)
        normal_B = normal_B/torch.linalg.norm(normal_B, axis=-1, keepdim=True)
        face_angle_thresh = np.cos(self.angle/180*np.pi)
        flip_sign = torch.sum(normal_A*normal_B, dim=-1)>face_angle_thresh  # true if the two triangles are flipped

        confidence_1 = nn.Sigmoid()(pred_logits[:,:,0])
        confidence_2 = nn.Sigmoid()(pred_logits[:,:,1])
        valid_rows, valid_cols = torch.where((confidence_1>self.conf_1) & (confidence_2>threshold_2) & (~flip_sign))
        pred_triangles = sorted_candidate_triangles[valid_rows, valid_cols,:2,:]
        pred_triangles = pred_triangles.reshape(-1,3)
        pred_triangles = torch.sort(pred_triangles, dim=-1).values
        pred_triangles, counts = torch.unique(pred_triangles, dim=0, return_counts=True)
        return pred_triangles
    
        
