"""Retriangulate overlapping planar patches while checking coverage and seams."""
import itertools
import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def edge_table(faces):
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inverse,kind='stable')
    face_ids=np.tile(np.arange(len(faces)),3)[order]
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    return unique,counts,starts,face_ids


def repair_planar_overlaps(vertices, faces, plane_tolerance=1e-6):
    """Return new indexed faces and diagnostics; do not change vertex positions.

    Replacement patches must preserve projected union area, all used vertices,
    and incident seam edges. The whole result is rejected if indexed boundary
    or non-manifold edge counts increase. Curved/non-planar regions stay intact.
    Shapely is imported only when this optional postprocessing is called.
    """
    import shapely
    if not np.isfinite(plane_tolerance) or not 0<plane_tolerance<=1e-3:
        raise ValueError('Plane tolerance must be in (0, 1e-3].')
    vertices=np.asarray(vertices,dtype=np.float64)
    faces=np.asarray(faces,dtype=np.int64)
    stats=dict(candidate_patches=0,repaired_patches=0,skipped_nonplanar=0,
               skipped_coverage=0,skipped_seams=0,skipped_qhull=0,
               global_reverted=0,removed_overlap_area=0.,input_faces=len(faces),output_faces=len(faces))
    if len(faces)<2:
        return faces.copy(),stats
    diagonal=np.linalg.norm(np.ptp(vertices,axis=0))
    if not diagonal:
        return faces.copy(),stats
    points=(vertices-vertices.mean(axis=0))/diagonal
    cross=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    lengths=np.linalg.norm(cross,axis=1)
    normals=cross/np.maximum(lengths[:,None],np.finfo(float).tiny)
    edges,counts,starts,face_ids=edge_table(faces)
    two=np.where(counts==2)[0]
    a=face_ids[starts[two]]
    b=face_ids[starts[two]+1]
    valid=(np.abs((normals[a]*normals[b]).sum(axis=1))>1-1e-8)&(lengths[a]>1e-15)&(lengths[b]>1e-15)
    aa,bb=a[valid].tolist(),b[valid].tolist()
    for edge in np.where(counts>2)[0]:
        incident=face_ids[starts[edge]:starts[edge]+counts[edge]]
        for a,b in itertools.combinations(incident,2):
            if lengths[a]>1e-15 and lengths[b]>1e-15 and abs(normals[a]@normals[b])>1-1e-8:
                aa.append(int(a));bb.append(int(b))
    graph=coo_matrix((np.ones(len(aa)),(aa,bb)),shape=(len(faces),len(faces))).tocsr()
    _,labels=connected_components(graph,directed=False)
    original_edge_faces={tuple(edge):face_ids[start:start+count] for edge,start,count in zip(edges,starts,counts)}
    output=[]
    for label in np.unique(labels):
        positions=np.where(labels==label)[0]
        group=faces[positions]
        if len(group)<2:
            output.append(group);continue
        ids=np.unique(group)
        normal=normals[positions[0]]
        origin=points[ids].mean(axis=0)
        if np.max(np.abs((points[ids]-origin)@normal))>plane_tolerance:
            stats['skipped_nonplanar']+=1;output.append(group);continue
        u=points[group[0,1]]-points[group[0,0]]
        u=u/np.linalg.norm(u)
        v=np.cross(normal,u)
        xy=np.stack([(points[ids]-origin)@u,(points[ids]-origin)@v],axis=1)
        local=np.searchsorted(ids,group)
        polygons=shapely.polygons(xy[local])
        if not np.all(shapely.is_valid(polygons)):
            stats['skipped_coverage']+=1;output.append(group);continue
        domain=shapely.union_all(polygons)
        area=float(domain.area)
        overlap=float(shapely.area(polygons).sum()-area)
        if area<=1e-15 or overlap<=max(area*1e-8,1e-15):
            output.append(group);continue
        stats['candidate_patches']+=1
        try:
            candidates=Delaunay(xy).simplices
        except QhullError:
            stats['skipped_qhull']+=1;output.append(group);continue
        candidate_polygons=shapely.polygons(xy[candidates])
        tolerance=max(np.sqrt(area)*1e-10,1e-12)
        keep=shapely.covers(domain.buffer(tolerance),candidate_polygons)
        proposed=ids[candidates[keep]]
        new_union=shapely.union_all(candidate_polygons[keep])
        if not len(proposed) or domain.symmetric_difference(new_union).area>max(area*1e-8,1e-15) or not np.array_equal(np.unique(proposed),ids):
            stats['skipped_coverage']+=1;output.append(group);continue
        new_edges,new_counts,_,_=edge_table(proposed)
        new_incidence={tuple(e):int(c) for e,c in zip(new_edges,new_counts)}
        seams_ok=True
        # Reject new edge incidences through an existing other patch, and retain
        # every previously shared seam segment with no increased incidence.
        for edge,count in new_incidence.items():
            old=original_edge_faces.get(edge)
            if old is not None and np.any(labels[old]!=label):
                inside=int((labels[old]==label).sum())
                if count>inside:
                    seams_ok=False;break
        if seams_ok:
            group_edges,_,_,_=edge_table(group)
            for edge in map(tuple,group_edges):
                old=original_edge_faces[edge]
                if np.any(labels[old]!=label) and edge not in new_incidence:
                    seams_ok=False;break
        if not seams_ok:
            stats['skipped_seams']+=1;output.append(group);continue
        output.append(proposed)
        stats['repaired_patches']+=1
        stats['removed_overlap_area']+=overlap*diagonal**2
    if not stats['repaired_patches']:
        return faces.copy(),stats
    result=np.unique(np.sort(np.concatenate(output),axis=1),axis=0)
    _,new_counts,_,_=edge_table(result)
    if (new_counts>2).sum()>(counts>2).sum() or (new_counts==1).sum()>(counts==1).sum():
        stats.update(global_reverted=1,repaired_patches=0,removed_overlap_area=0.)
        return faces.copy(),stats
    stats['output_faces']=len(result)
    return result,stats
