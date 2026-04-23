"""
Threshold Sensitivity Analysis for Bertsekas ALM with KKT Projection Gradient (FILTERED VERSION)
- Tests different threshold values for major/minor path decomposition
- Analyzes impact on compression ratio, accuracy, and computational efficiency
- Compares trade-offs between model complexity and solution quality
- FILTERED: d_multi excludes singleton ODs entirely (dimension reduction)

GRADIENT FORMULA (Corrected):
∇_{x1} L = B1^T ∇_v f_BPR + A1^T(λ_od + ρ_od·od_error) - max{0, λ_major - ρ_major·x1}
∇_θ L = D^T ∇_v f_BPR + M^T(λ_od + ρ_od·od_error) - U_r^T max{0, λ_minor - ρ_minor·x2}

Key change: Uses max{0, λ - ρ·x} instead of (λ + ρ·violation)·mask
"""

import os

# Reduce thread overhead for better wall-clock performance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
import sys
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Threshold for filtering zero/negligible demand OD pairs
NEGLIGIBLE_DEMAND_THRESHOLD = 0.001


def _detect_column(df, candidates, context):
    """Detect column name from candidates"""
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    raise ValueError(
        f"Could not find {context} column. Tried: {candidates}\nAvailable columns: {list(df.columns)}"
    )


def load_gmns_data_with_od(link_file, route_file, demand_file=None, link_perf_file=None):
    """Load GMNS network data with OD demand"""
    links = pd.read_csv(link_file)
    routes = pd.read_csv(route_file)

    if link_perf_file is None:
        link_perf_file = str(Path(link_file).with_name("link_performance_ue.csv"))

    link_perf = None
    if link_perf_file and Path(link_perf_file).exists():
        link_perf = pd.read_csv(link_perf_file)

    # Detect columns
    link_id_col = _detect_column(links, ["link_id", "linkid", "id"], "link_id")
    ref_vol_col = None
    if link_perf is not None:
        perf_link_id_col = _detect_column(link_perf, ["link_id", "linkid", "id"], "link_id")
        ref_vol_col = _detect_column(
            link_perf,
            ["volume", "link_volume", "obs_volume", "ref_volume", "reference_volume"],
            "reference volume",
        )
    else:
        ref_vol_col = _detect_column(
            links,
            ["ref_volume", "reference_volume", "obs_volume", "volume"],
            "reference volume",
        )
    route_id_col = _detect_column(
        routes, ["route_id", "path_id", "agent_id", "id"], "route/path_id"
    )
    link_seq_col = _detect_column(
        routes,
        ["link_ids", "link_id_sequence", "link_sequence", "links"],
        "link sequence",
    )
    path_vol_col = _detect_column(
        routes, ["volume", "flow", "path_volume", "path_flow"], "path volume"
    )

    # Detect OD columns
    try:
        o_zone_col = _detect_column(
            routes, ["o_zone_id", "origin_zone", "origin", "from_zone"], "origin zone"
        )
        d_zone_col = _detect_column(
            routes,
            ["d_zone_id", "destination_zone", "destination", "to_zone"],
            "destination zone",
        )
        print("    Found OD information in route file")
    except ValueError:
        raise ValueError("Route file must contain OD information")

    # Build mappings
    link_ids = sorted(links[link_id_col].unique())
    link_id_to_idx = {lid: idx for idx, lid in enumerate(link_ids)}
    n_links = len(link_ids)

    if link_perf is not None:
        link_vol_map = dict(zip(link_perf[perf_link_id_col], link_perf[ref_vol_col]))
    else:
        link_vol_map = dict(zip(links[link_id_col], links[ref_vol_col]))
    v_ref_raw = np.array([link_vol_map.get(lid, 0.0) for lid in link_ids], dtype=float)

    # Debug: Check if we have meaningful reference volumes
    print(
        f"  Reference link volumes: min={np.min(v_ref_raw):.2f}, max={np.max(v_ref_raw):.2f}, mean={np.mean(v_ref_raw):.2f}"
    )
    print(f"  Zero volume links: {np.sum(v_ref_raw == 0)} / {len(v_ref_raw)}")

    # If reference link volumes are mostly zeros, compute them from path flows
    if np.mean(v_ref_raw) < 0.1 or np.sum(v_ref_raw == 0) > 0.8 * len(v_ref_raw):
        print(
            "  Warning: Reference link volumes mostly zero, will compute from path flows after matrix construction"
        )
        # Keep original for now, will recompute later
        v_ref = v_ref_raw
    else:
        v_ref = v_ref_raw

    routes = routes.sort_values(route_id_col).reset_index(drop=True)
    path_ids = sorted(routes[route_id_col].unique())
    path_id_to_idx = {pid: idx for idx, pid in enumerate(path_ids)}
    n_paths = len(path_ids)

    path_vol_map = dict(zip(routes[route_id_col], routes[path_vol_col]))
    x_ref = np.array([path_vol_map.get(pid, 0.0) for pid in path_ids], dtype=float)

    # Build B matrix
    row_indices, col_indices = [], []
    for _, route in routes.iterrows():
        path_idx = path_id_to_idx[route[route_id_col]]
        link_seq = route[link_seq_col]
        if pd.isna(link_seq):
            continue

        if isinstance(link_seq, str):
            if ";" in link_seq:
                link_ids_list = [
                    int(x.strip()) for x in link_seq.split(";") if x.strip()
                ]
            elif "," in link_seq:
                link_ids_list = [
                    int(x.strip()) for x in link_seq.split(",") if x.strip()
                ]
            else:
                link_ids_list = [int(link_seq.strip())]
        else:
            link_ids_list = [int(link_seq)]

        for link_id in link_ids_list:
            if link_id in link_id_to_idx:
                row_indices.append(path_idx)
                col_indices.append(link_id_to_idx[link_id])

    data = np.ones(len(row_indices), dtype=np.int8)
    B = csr_matrix((data, (row_indices, col_indices)), shape=(n_paths, n_links))

    # Recompute v_ref from path flows if original link volumes are problematic
    if np.mean(v_ref) < 0.1 or np.sum(v_ref == 0) > 0.8 * len(v_ref):
        v_ref_computed = B.T @ x_ref
        if hasattr(v_ref_computed, "toarray"):  # If sparse matrix result
            v_ref_computed = v_ref_computed.toarray().flatten()
        print(
            f"  Recomputed link volumes from path flows: min={np.min(v_ref_computed):.2f}, max={np.max(v_ref_computed):.2f}, mean={np.mean(v_ref_computed):.2f}"
        )
        v_ref = v_ref_computed

    # Capacity
    capacity_col = _detect_column(
        links, ["capacity", "link_capacity", "cap"], "capacity"
    )
    capacity_map = dict(zip(links[link_id_col], links[capacity_col]))
    capacity = np.array(
        [capacity_map.get(lid, 1000.0) for lid in link_ids], dtype=float
    )

    # Free-flow travel time (t_0 = length / free_speed)
    length_col = _detect_column(
        links, ["vdf_length_mi", "length_mi", "length"], "length"
    )
    speed_col = _detect_column(
        links, ["vdf_free_speed_mph", "free_speed_mph", "free_speed"], "free speed"
    )
    length_map = dict(zip(links[link_id_col], links[length_col]))
    speed_map = dict(zip(links[link_id_col], links[speed_col]))

    # Compute t_0 for each link (in minutes: length in miles, speed in mph, multiply by 60)
    t_0 = np.array(
        [
            60.0 * length_map.get(lid, 0.0) / max(speed_map.get(lid, 1.0), 1.0)
            for lid in link_ids
        ],
        dtype=float,
    )

    # OD info
    od_pairs_list = []
    path_od_pairs = []
    for _, route in routes.iterrows():
        o_zone = route[o_zone_col]
        d_zone = route[d_zone_col]
        od_pair = (o_zone, d_zone)
        path_od_pairs.append(od_pair)
        od_pairs_list.append(od_pair)

    od_pairs_unique = sorted(list(set(od_pairs_list)))
    od_pair_to_idx = {od: idx for idx, od in enumerate(od_pairs_unique)}
    n_od = len(od_pairs_unique)
    path_to_od = np.array([od_pair_to_idx[od] for od in path_od_pairs])

    # OD demand
    if demand_file is not None:
        demand_df = pd.read_csv(demand_file)
        o_col = _detect_column(
            demand_df, ["o_zone_id", "origin", "from_zone"], "origin"
        )
        d_col = _detect_column(
            demand_df, ["d_zone_id", "destination", "to_zone"], "destination"
        )
        demand_col = _detect_column(demand_df, ["volume", "demand", "flow"], "demand")

        od_demand = np.zeros(n_od)
        for _, row in demand_df.iterrows():
            od_pair = (row[o_col], row[d_col])
            if od_pair in od_pair_to_idx:
                od_demand[od_pair_to_idx[od_pair]] = row[demand_col]
    else:
        od_demand = np.zeros(n_od)
        for path_idx in range(n_paths):
            od_demand[path_to_od[path_idx]] += x_ref[path_idx]

    paths_by_od = {od_idx: [] for od_idx in range(n_od)}
    for path_idx in range(n_paths):
        paths_by_od[path_to_od[path_idx]].append(path_idx)

    od_info = {
        "od_pairs": od_pairs_unique,
        "od_pair_to_idx": od_pair_to_idx,
        "path_to_od": path_to_od,
        "od_demand": od_demand,
        "paths_by_od": paths_by_od,
        "n_od": n_od,
    }

    return B, x_ref, v_ref, capacity, t_0, od_info, links, routes


def decompose_paths(B, x_ref, od_info, threshold, add_singleton_category=True):
    """Decompose paths into three categories: singleton, major, and minor
    
    Strategy:
    1. First identify singleton OD pairs (ODs with only 1 path)
    2. All paths from singleton ODs go to "singleton" category (regardless of flow)
    3. Remaining paths from multi-path ODs are split by flow threshold:
       - Major: flow >= threshold
       - Minor: flow < threshold (will be compressed via SVD)
    """
    n_od = od_info["n_od"]
    path_to_od = od_info["path_to_od"]
    
    # Step 1: Count paths per OD pair
    od_path_counts = np.zeros(n_od, dtype=int)
    for path_idx in range(len(x_ref)):
        od_idx = path_to_od[path_idx]
        od_path_counts[od_idx] += 1
    
    # Step 2: Identify singleton OD pairs (only 1 path)
    singleton_od_mask = od_path_counts == 1
    
    print(f"  OD analysis: {n_od} total ODs")
    print(f"    Singleton ODs (1 path): {np.sum(singleton_od_mask)}")
    print(f"    Multi-path ODs (2+ paths): {np.sum(~singleton_od_mask)}")
    
    # Step 3: Partition paths into three categories
    singleton_mask = np.zeros(len(x_ref), dtype=bool)
    major_mask = np.zeros(len(x_ref), dtype=bool)
    minor_mask = np.zeros(len(x_ref), dtype=bool)
    
    for path_idx in range(len(x_ref)):
        od_idx = path_to_od[path_idx]
        if singleton_od_mask[od_idx]:
            # Singleton OD: all its paths go to singleton category
            singleton_mask[path_idx] = True
        else:
            # Multi-path OD: partition by flow threshold
            if x_ref[path_idx] > threshold:
                major_mask[path_idx] = True
            else:
                minor_mask[path_idx] = True

    # Ensure each multi-path OD has at least one major path.
    # For any multi-path OD where all paths fell below the threshold,
    # promote the path with the largest reference flow to `major`.
    if add_singleton_category:
        for od_idx in range(n_od):
            if singleton_od_mask[od_idx]:
                continue
            paths = od_info.get("paths_by_od", {}).get(od_idx, [])
            if not paths:
                continue
            # Check if any path for this OD is marked major
            has_major = any(major_mask[p] for p in paths)
            if not has_major:
                # Promote the path with the largest reference flow
                best_path = max(paths, key=lambda p: x_ref[p])
                # Update masks
                if minor_mask[best_path]:
                    minor_mask[best_path] = False
                major_mask[best_path] = True
    
    # Extract path sets
    B_singleton = B[singleton_mask]
    B1 = B[major_mask]
    B2 = B[minor_mask]
    
    x_singleton_ref = x_ref[singleton_mask]
    x1_ref = x_ref[major_mask]
    x2_ref = x_ref[minor_mask]
    
    singleton_indices = np.where(singleton_mask)[0]
    major_indices = np.where(major_mask)[0]
    minor_indices = np.where(minor_mask)[0]
    
    n_singleton = np.sum(singleton_mask)
    s = np.sum(major_mask)
    n_minor = np.sum(minor_mask)

    # Flow distribution across categories
    singleton_flow = np.nansum(x_ref[singleton_mask])
    major_flow = np.nansum(x_ref[major_mask])
    minor_flow = np.nansum(x_ref[minor_mask])
    total_flow = singleton_flow + major_flow + minor_flow
    singleton_flow_pct = 100 * singleton_flow / total_flow if total_flow > 0 else 0.0
    major_flow_pct = 100 * major_flow / total_flow if total_flow > 0 else 0.0
    minor_flow_pct = 100 * minor_flow / total_flow if total_flow > 0 else 0.0

    print("  Path decomposition:")
    print(f"    Singleton paths (from 1-path ODs): {n_singleton}")
    print(f"    Major paths (flow >= {threshold}, multi-path ODs): {s}")
    print(f"    Minor paths (flow < {threshold}, multi-path ODs): {n_minor}")
    print(f"    Total: {n_singleton + s + n_minor}")
    print("  Flow distribution:")
    print(
        f"    Singleton flow: {singleton_flow:.2f} ({singleton_flow_pct:.1f}%)"
    )
    print(f"    Major flow: {major_flow:.2f} ({major_flow_pct:.1f}%)")
    print(f"    Minor flow: {minor_flow:.2f} ({minor_flow_pct:.1f}%)")
    print(f"    Total flow: {total_flow:.2f}")

    # Build A_singleton (singleton paths to OD)
    A_singleton_row, A_singleton_col = [], []
    for col_idx, path_idx in enumerate(singleton_indices):
        A_singleton_row.append(path_to_od[path_idx])
        A_singleton_col.append(col_idx)
    A_singleton = csr_matrix((np.ones(len(A_singleton_row)), (A_singleton_row, A_singleton_col)), shape=(n_od, n_singleton))

    # Build A1 (major paths to OD)
    A1_row, A1_col = [], []
    for col_idx, path_idx in enumerate(major_indices):
        A1_row.append(path_to_od[path_idx])
        A1_col.append(col_idx)
    A1 = csr_matrix((np.ones(len(A1_row)), (A1_row, A1_col)), shape=(n_od, s))

    # Build A2 (minor paths to OD)
    A2_row, A2_col = [], []
    for col_idx, path_idx in enumerate(minor_indices):
        A2_row.append(path_to_od[path_idx])
        A2_col.append(col_idx)
    A2 = csr_matrix((np.ones(len(A2_row)), (A2_row, A2_col)), shape=(n_od, n_minor))

    return {
        "B_singleton": B_singleton,
        "B1": B1,
        "B2": B2,
        "x_singleton_ref": x_singleton_ref,
        "x1_ref": x1_ref,
        "x2_ref": x2_ref,
        "singleton_mask": singleton_mask,
        "major_mask": major_mask,
        "minor_mask": minor_mask,
        "singleton_indices": singleton_indices,
        "major_indices": major_indices,
        "minor_indices": minor_indices,
        "n_singleton": n_singleton,
        "s": s,
        "n_minor": n_minor,
        "total_flow": total_flow,
        "singleton_flow": singleton_flow,
        "major_flow": major_flow,
        "minor_flow": minor_flow,
        "singleton_flow_pct": singleton_flow_pct,
        "major_flow_pct": major_flow_pct,
        "minor_flow_pct": minor_flow_pct,
        "A_singleton": A_singleton,
        "A1": A1,
        "A2": A2,
    }


def compute_svd_compression(B2, x2_ref, rank_pct=0.30, max_rank=50, use_truncated_svd=True):
    """Compute SVD compression using TruncatedSVD for large sparse matrices"""
    n_minor, m = B2.shape
    if n_minor == 0:
        # Return dummy structure for compatibility when no minor paths
        return {
            "U_r": csr_matrix((0, 0), dtype=np.float64),
            "V_r": csr_matrix((0, 0), dtype=np.float64),
            "sigma": np.array([], dtype=np.float64),
            "theta_ref": np.array([], dtype=np.float64),
            "r": 0,
            "compression_ratio": float("inf"),
            "explained_variance_ratio": 1.0,
            "reconstruction_error": 0.0,
            "svd_time": 0.0,
        }

    svd_cpu_start = time.process_time()

    # Determine rank with conservative limits for large matrices
    r = max(1, min(int(rank_pct * n_minor), max_rank, min(n_minor, m) - 1))
    print(f"  Computing SVD with rank {r}...")

    if use_truncated_svd:
        # Convert to float32 for memory efficiency
        B2_float32 = B2.astype(np.float32)

        svd_model = TruncatedSVD(
            n_components=r, random_state=42, algorithm="randomized"
        )

        U_sigma = svd_model.fit_transform(B2_float32)
        S = svd_model.singular_values_
        Vt = svd_model.components_

        U_truncated = (U_sigma / S[np.newaxis, :]).astype(np.float64)
        sigma = S.astype(np.float64)
        Vt_truncated = Vt.astype(np.float64)

        U_r = U_truncated
        V_r = Vt_truncated.T
        # Create dense D matrix for threshold file (different from KKT_stagnation)
        Sigma_r = np.diag(sigma)
        D = V_r @ Sigma_r

        print("    TruncatedSVD completed successfully")
    else:
        # Sparse SVD: svds returns U, sigma, Vt where B2.T = U @ diag(sigma) @ Vt
        # B2.T has shape (n_minor_paths, n_links)
        U_svd, sigma, Vt_svd = svds(B2.T.tocsr(), k=r)

        idx = np.argsort(sigma)[::-1]
        U_svd = U_svd[:, idx]
        sigma = sigma[idx]
        Vt_svd = Vt_svd[idx, :]

        # SVD gives: B2.T = U_svd @ diag(sigma) @ Vt_svd
        # We want: B2.T = U_r @ diag(sigma) @ V_r.T
        # So: U_r = U_svd (shape: n_minor_paths × r)
        #     V_r = Vt_svd.T (shape: n_links × r)
        U_r = U_svd
        V_r = Vt_svd.T
        # Create dense D matrix for threshold file
        Sigma_r = np.diag(sigma)
        D = V_r @ Sigma_r

        print("  Sparse SVD completed successfully")

    theta_ref = U_r.T @ x2_ref

    svd_cpu_time = time.process_time() - svd_cpu_start

    return {
        "U_r": U_r,
        "V_r": V_r,
        "D": D,
        "sigma": sigma,
        "r": r,
        "theta_ref": theta_ref,
        "svd_time": svd_cpu_time,
    }


def bpr_objective(v, capacity, t_0, alpha=0.15, beta=4.0):
    """BPR objective function (congestion component with t_0 scaling)"""
    return np.sum(t_0 * capacity * alpha / (beta + 1) * (v / capacity) ** (beta + 1))


def bpr_gradient(v, capacity, t_0, alpha=0.15, beta=4.0):
    """Gradient of BPR objective (with t_0 scaling)"""
    return t_0 * alpha * (v / capacity) ** beta

################################################################################
# BERTSEKAS ALM OPTIMIZER
################################################################################

class BertsekasALM:
    """Bertsekas ALM optimizer"""

    def __init__(
        self,
        decomp,
        svd_dict,
        capacity,
        t_0,
        od_info,
        v_ref,
        alpha=0.15,
        beta=4.0,
        rho_od_init=1e4,
        rho_nonneg_minor_init=1e5,
        tau=10.0,
        gamma=0.1,
    ):
        self.B1 = decomp["B1"]
        # Ensure B2 exists; default to empty (n_minor x m_links) sparse matrix when missing
        m_links = self.B1.shape[1]
        n_minor = int(decomp.get("n_minor", 0))
        self.B2 = decomp.get("B2", csr_matrix((n_minor, m_links), dtype=np.float64))
        self.A1 = decomp["A1"]
        self.A2 = decomp["A2"]
        self.x2 = np.array([], dtype=np.float64)

        # Handle singleton paths (paths from ODs with only 1 path)
        self.n_singleton = decomp.get("n_singleton", 0)
        if self.n_singleton > 0:
            self.B_singleton = decomp["B_singleton"]
            self.A_singleton = decomp["A_singleton"]
            # Extract singleton OD demands
            path_to_od = od_info["path_to_od"]
            singleton_indices = decomp["singleton_indices"]
            singleton_ods = np.unique([path_to_od[i] for i in singleton_indices])
            d_singleton = od_info["od_demand"][singleton_ods]
            # Pre-compute constant singleton contribution to link flows
            self.v_singleton = self.B_singleton.T @ d_singleton
        else:
            self.B_singleton = None
            self.A_singleton = None
            self.v_singleton = None

        # Handle case where there are no minor paths (n_minor == 0)
        if svd_dict is None or decomp["n_minor"] == 0:
            # No minor paths - create empty structures
            self.U_r = csr_matrix((0, 0), dtype=np.float64)
            self.V_r = csr_matrix((0, 0), dtype=np.float64)
            self.sigma = np.array([], dtype=np.float64)
            self.M = csr_matrix(
                (decomp["A2"].shape[0], 0), dtype=np.float64
            )  # k × 0 matrix
            self.D = csr_matrix((self.B2.shape[0], 0), dtype=np.float64)  # m × 0 matrix
            self.r = 0
            self.theta_ref = np.array([], dtype=np.float64)
        else:
            self.U_r = svd_dict["U_r"]
            self.V_r = svd_dict["V_r"]
            self.sigma = svd_dict["sigma"]
            self.M = self.A2 @ self.U_r
            self.D = self.B2.T @ self.U_r
            self.r = svd_dict["r"]
            self.theta_ref = svd_dict["theta_ref"]

        self.x1_ref = decomp["x1_ref"]
        self.x2_ref = decomp["x2_ref"]
        self.v_ref = v_ref

        self.major_mask = decomp["major_mask"]
        self.minor_mask = decomp["minor_mask"]
        
        # FILTERED VERSION: d_multi only includes multi-path ODs (dimension reduction)
        # Singleton OD conservation is satisfied by construction (x_singleton = d_singleton)
        self.d = od_info["od_demand"].copy()
        self.k_total = len(self.d)  # Total number of ODs (including singleton)
        
        if self.n_singleton > 0:
            path_to_od = od_info["path_to_od"]
            singleton_indices = decomp["singleton_indices"]
            singleton_ods = set([path_to_od[i] for i in singleton_indices])
            # Create boolean mask: True for multi-path ODs, False for singleton ODs
            self.multi_od_mask = np.array([od_idx not in singleton_ods for od_idx in range(self.k_total)])
            # Extract only multi-path OD demands (dimension reduction)
            self.d_multi = self.d[self.multi_od_mask]
            self.k = len(self.d_multi)  # Number of multi-path ODs only
            
            # Filter A1, A2 to only include multi-path OD rows
            self.A1 = self.A1[self.multi_od_mask, :]
            self.A2 = self.A2[self.multi_od_mask, :]
            # Update M = A2 @ U_r with filtered A2
            if self.r > 0:
                self.M = self.A2 @ self.U_r
        else:
            # All ODs are multi-path
            self.multi_od_mask = np.ones(self.k_total, dtype=bool)
            self.d_multi = self.d
            self.k = self.k_total
            # M already computed correctly
            if self.r > 0:
                self.M = self.A2 @ self.U_r

        self.s = decomp["s"]
        self.m = self.B1.shape[1]

        self.capacity = capacity
        self.t_0 = t_0
        self.alpha = alpha
        self.beta = beta

        self.rho_od = rho_od_init
        self.rho_nonneg_minor = rho_nonneg_minor_init
        self.tau = tau
        self.gamma = gamma
        self.MAX_PENALTY = 1e20
        
        # Track previous violations for adaptive penalty updates
        self.prev_od_viol = None
        self.prev_minor_viol = None

        # Initialize Lagrangian multipliers
        # λ_od for OD conservation constraints (dimension: k multi-path ODs)
        self.lambda_od = np.zeros(self.k)
        # λ_m for minor path non-negativity (Full KKT)
        self.lambda_minor = np.zeros(decomp["n_minor"])
        self.bpr_optimal = bpr_objective(v_ref, capacity, t_0, alpha, beta)

        self.history = {
            "outer_iter": [],
            "inner_iter": [],
            "objective": [],
            "bpr_pure": [],
            "bpr_optimal": [],
            "bpr_gap": [],
            "bpr_gap_pct": [],
            "rho_od": [],
            "rho_nonneg_minor": [],
            "od_violation": [],
            "nonneg_minor_violation": [],
            "link_r2": [],
            "link_mae": [],
            "x1_r2": [],
            "x1_mae": [],
            "x2_r2": [],
            "x2_mae": [],
            "travel_time_r2": [],
            "travel_time_mae": [],
            "elapsed_time": [],
            "inner_time": [],
            "outer_time": [],
        }

    def objective_and_gradient_chain_rule(self, z):
        """Compute augmented Lagrangian objective with multipliers and penalties"""
        x1 = z[: self.s]
        theta = z[self.s :] if self.r > 0 else np.array([], dtype=np.float64)

        # Compute link volumes: v = v_singleton + B1^T x1 + D θ
        # v_singleton is a constant (pre-computed in __init__)
        if self.v_singleton is not None:
            v = self.v_singleton + self.B1.T @ x1
        else:
            v = self.B1.T @ x1

        if self.r > 0:
            # minor path contribution computed via chain multiplications
            u = self.U_r @ theta
            v += self.B2.T.dot(u)

        # BPR objective (v includes constant v_singleton contribution)
        f_bpr = bpr_objective(v, self.capacity, self.t_0, self.alpha, self.beta)

        # OD conservation: A1*x1 + A2*(U_r@theta) = d_multi (only multi-path ODs)
        # Singleton ODs excluded - their conservation is satisfied by construction
        if self.r > 0:
            od_flow = self.A1 @ x1 + self.A2 @ u
        else:
            od_flow = self.A1 @ x1
        od_error = od_flow - self.d_multi

        # ALM terms for OD constraints: λ^T * g + ρ/2 * ||g||²
        od_lagrangian = self.lambda_od.T @ od_error
        od_penalty = 0.5 * self.rho_od * np.sum(od_error**2)

        # Non-negativity constraints for minor paths: U_r*θ ≥ 0 (only if minor paths exist)
        # Formula (14): (1/2c) * {(max{0, γ - c·[U_r θ]})² - γ²}
        if self.r > 0:
            max_term_minor = np.maximum(
                0, self.lambda_minor - self.rho_nonneg_minor * u
            )
            minor_penalty_term = (1.0 / (2.0 * self.rho_nonneg_minor)) * (
                np.sum(max_term_minor**2) - np.sum(self.lambda_minor**2)
            )
        else:
            minor_penalty_term = 0.0

        # Total augmented Lagrangian (Formula 14)
        total_obj = (
            f_bpr + od_lagrangian + od_penalty + minor_penalty_term
        )

        # Gradients
        # Note: grad_v computed from v (which includes v_singleton)
        # But ∂v_singleton/∂x1 = 0 and ∂v_singleton/∂θ = 0 (constant doesn't affect gradients)
        grad_v = bpr_gradient(v, self.capacity, self.t_0, self.alpha, self.beta)

        # Gradient w.r.t. x1: ∂f/∂x1 = B1 @ grad_v + A1^T @ (λ + ρ*error)
        grad_x1 = self.B1 @ grad_v
        grad_x1 += self.A1.T @ (self.lambda_od + self.rho_od * od_error)

        # Gradient w.r.t. θ (CORRECTED KKT PROJECTION FORMULA, only if minor paths exist)
        if self.r > 0:
            # tmp = B2 @ grad_v  +  A2.T @ (lambda_od + rho_od * od_error)
            tmp = self.B2 @ grad_v
            tmp += self.A2.T @ (self.lambda_od + self.rho_od * od_error)
            # map back to theta space
            grad_theta = self.U_r.T @ tmp
            # KKT projection gradient: -U_r^T max{0, λ_minor - ρ_minor·x2}
            grad_theta -= self.U_r.T @ np.maximum(
                0, self.lambda_minor - self.rho_nonneg_minor * u
            )
            # Combine gradients
            grad = np.concatenate([grad_x1, grad_theta])
        else:
            grad = grad_x1

        return total_obj, grad

    def objective_and_gradient_direct(self, z):
        """Compute augmented Lagrangian using direct M matrix multiplication
        
        This version explicitly computes M*θ for OD flow calculation.
        Useful when M is sparse and direct multiplication is efficient.
        """
        x1 = z[: self.s]
        theta = z[self.s :] if self.r > 0 else np.array([], dtype=np.float64)

        # Compute link volumes
        if self.v_singleton is not None:
            v = self.v_singleton + self.B1.T @ x1
        else:
            v = self.B1.T @ x1

        if self.r > 0:
            u = self.U_r @ theta
            v += self.D @ theta

        # BPR objective
        f_bpr = bpr_objective(v, self.capacity, self.t_0, self.alpha, self.beta)

        # OD conservation using direct M multiplication: A1*x1 + M*θ = d_multi
        if self.r > 0:
            od_flow = self.A1 @ x1 + self.M @ theta
        else:
            od_flow = self.A1 @ x1
        od_error = od_flow - self.d_multi

        # ALM terms for OD constraints
        od_lagrangian = self.lambda_od.T @ od_error
        od_penalty = 0.5 * self.rho_od * np.sum(od_error**2)

        # Non-negativity constraints for minor paths
        if self.r > 0:
            max_term_minor = np.maximum(
                0, self.lambda_minor - self.rho_nonneg_minor * u
            )
            minor_penalty_term = (1.0 / (2.0 * self.rho_nonneg_minor)) * (
                np.sum(max_term_minor**2) - np.sum(self.lambda_minor**2)
            )
        else:
            minor_penalty_term = 0.0

        # Total augmented Lagrangian
        total_obj = f_bpr + od_lagrangian + od_penalty + minor_penalty_term

        # Gradients
        grad_v = bpr_gradient(v, self.capacity, self.t_0, self.alpha, self.beta)

        # Gradient w.r.t. x1
        grad_x1 = self.B1 @ grad_v
        grad_x1 += self.A1.T @ (self.lambda_od + self.rho_od * od_error)

        # Gradient w.r.t. θ using direct M^T multiplication
        if self.r > 0:
            # BPR gradient contribution: D^T @ grad_v
            grad_theta = self.D.T @ grad_v
            # OD constraint gradient using direct M^T: M^T @ (λ + ρ*error)
            grad_theta += self.M.T @ (self.lambda_od + self.rho_od * od_error)
            # KKT projection for non-negativity
            grad_theta -= self.U_r.T @ np.maximum(
                0, self.lambda_minor - self.rho_nonneg_minor * u
            )
            grad = np.concatenate([grad_x1, grad_theta])
        else:
            grad = grad_x1

        return total_obj, grad

    def objective_and_gradient_direct_new(self, z):
        """Direct objective/gradient with minor allocation reductions.

        Improvements vs objective_and_gradient_direct:
        1. Reuse `dual_od = lambda_od + rho_od * od_error` in both x1/theta gradients.
        2. Reuse `max_term_minor` in both objective penalty and KKT projection gradient.
        """
        x1 = z[: self.s]
        theta = z[self.s :] if self.r > 0 else np.array([], dtype=np.float64)

        # Compute link volumes
        if self.v_singleton is not None:
            v = self.v_singleton + self.B1.T @ x1
        else:
            v = self.B1.T @ x1

        if self.r > 0:
            u = self.U_r @ theta
            v += self.D @ theta

        # BPR objective
        f_bpr = bpr_objective(v, self.capacity, self.t_0, self.alpha, self.beta)

        # OD conservation using direct M multiplication: A1*x1 + M*θ = d_multi
        if self.r > 0:
            od_flow = self.A1 @ x1 + self.M @ theta
        else:
            od_flow = self.A1 @ x1
        od_error = od_flow - self.d_multi

        # ALM terms for OD constraints
        od_lagrangian = self.lambda_od.T @ od_error
        od_penalty = 0.5 * self.rho_od * np.sum(od_error**2)

        # Reuse this vector in both x1/theta gradients
        dual_od = self.lambda_od + self.rho_od * od_error

        # Non-negativity constraints for minor paths
        if self.r > 0:
            # Reuse this vector in both objective penalty and KKT projection gradient
            max_term_minor = np.maximum(
                0, self.lambda_minor - self.rho_nonneg_minor * u
            )
            minor_penalty_term = (1.0 / (2.0 * self.rho_nonneg_minor)) * (
                np.sum(max_term_minor**2) - np.sum(self.lambda_minor**2)
            )
        else:
            max_term_minor = None
            minor_penalty_term = 0.0

        # Total augmented Lagrangian
        total_obj = f_bpr + od_lagrangian + od_penalty + minor_penalty_term

        # Gradients
        grad_v = bpr_gradient(v, self.capacity, self.t_0, self.alpha, self.beta)

        # Gradient w.r.t. x1
        grad_x1 = self.B1 @ grad_v
        grad_x1 += self.A1.T @ dual_od

        # Gradient w.r.t. θ using direct M^T multiplication
        if self.r > 0:
            # BPR gradient contribution: D^T @ grad_v
            grad_theta = self.D.T @ grad_v
            # OD constraint gradient using direct M^T: M^T @ (λ + ρ*error)
            grad_theta += self.M.T @ dual_od
            # KKT projection for non-negativity (reuse max_term_minor)
            grad_theta -= self.U_r.T @ max_term_minor
            grad = np.concatenate([grad_x1, grad_theta])
        else:
            grad = grad_x1

        return total_obj, grad

    def objective_and_gradient_factored(self, z):
        """Compute augmented Lagrangian using factored form V_r @ (sigma * theta) - SPARSE FRIENDLY
        
        MATHEMATICAL FOUNDATION:
        ========================
        The factored form relies on the SVD relationship: B2.T @ U_r = V_r @ diag(sigma)
        
        PROOF:
        ------
        Starting from SVD of B2:
            B2 = U_r @ Sigma @ V_r^T                    [SVD decomposition]
        
        where U_r and V_r have orthonormal columns:
            U_r^T @ U_r = I_r    (r × r identity)
            V_r^T @ V_r = I_r
        
        Taking transpose of B2:
            B2^T = (U_r @ Sigma @ V_r^T)^T
            B2^T = V_r @ Sigma @ U_r^T                  [transpose property]
        
        Right multiply both sides by U_r:
            B2^T @ U_r = V_r @ Sigma @ U_r^T @ U_r
            B2^T @ U_r = V_r @ Sigma @ I_r              [orthonormality]
            B2^T @ U_r = V_r @ Sigma                    [QED]
        
        Therefore: B2^T @ U_r @ theta = V_r @ Sigma @ theta = V_r @ (sigma * theta)
        
        Similarly for gradient: (B2^T @ U_r)^T = Sigma^T @ V_r^T = Sigma @ V_r^T
        So: U_r^T @ B2 @ grad_v = sigma * (V_r^T @ grad_v)
        
        COMPUTATIONAL ADVANTAGE:
        ========================
        The factored form exploits the SVD relationship: B2.T @ U_r = V_r @ diag(sigma)
        
        Chain rule method (objective_and_gradient):
            v += B2.T @ (U_r @ theta)
            grad_theta += U_r.T @ (B2 @ grad_v)
            
        Factored method (this function):
            v += V_r @ (sigma * theta)
            grad_theta += sigma * (V_r.T @ grad_v)
        
        WHY IS THIS FASTER?
        ===================
        1. For SVD compression: V_r is DENSE (n_links × r)
           - No speedup vs chain rule
           - Both methods equivalent in cost
           
        2. For column_subset compression: V_r is SPARSE
           - V_r has only k nonzeros per row (where k << n_links)
           - Example: If k=5, V_r has 99.8% sparsity for n_links=2950
           - Sparse matmul: O(nnz(V_r) * r) vs Dense matmul: O(n_links * n_minor)
           - Speedup: 10-100x depending on sparsity
           
        3. For random_projection compression: V_r is SPARSE
           - V_r typically has ~1% nonzero entries (random sparse projection)
           - Similar speedup to column_subset
        
        COMPLEXITY ANALYSIS:
        ====================
        Assume: n_links=2950, n_minor=24125, r=50
        
        Chain rule:
            Forward:  O(n_minor*r) + O(n_links*n_minor) = O(24125*50) + O(2950*24125)
                    = 1.2M + 71.2M = 72.4M ops
            Backward: O(n_links*n_minor) + O(n_minor*r) = 72.4M ops
            Total: ~145M ops (if B2 is dense)
            
        Factored (sparse V_r with 1% density):
            Forward:  O(r) + O(nnz(V_r)) = 50 + 0.01*2950*50 = 1,525 ops
            Backward: O(nnz(V_r)) = 1,475 ops
            Total: ~3K ops
            Speedup: 48,000x !
            
        Factored (dense V_r for SVD):
            Forward:  O(r) + O(n_links*r) = 50 + 147,500 = 147,550 ops
            Backward: O(n_links*r) = 147,500 ops
            Total: ~295K ops
            Speedup: 500x (avoiding expensive B2 operations)
        
        KEY INSIGHT:
        ============
        Even for dense V_r (SVD), factored form is faster because:
        - Avoids B2 (n_links × n_minor) which is large and potentially dense
        - Uses V_r (n_links × r) where r << n_minor
        - Matrix dimensions: (2950 × 50) vs (2950 × 24125)
        - The factored form "precomputes" the expensive B2.T @ U_r once during setup
        """
        x1 = z[: self.s]
        theta = z[self.s :] if self.r > 0 else np.array([], dtype=np.float64)

        # Compute link volumes: v = v_singleton + B1^T x1 + V_r @ (sigma * theta)
        if self.v_singleton is not None:
            v = self.v_singleton + self.B1.T @ x1
        else:
            v = self.B1.T @ x1

        if self.r > 0:
            # Factored form: V_r @ (sigma * theta) instead of chain rule
            # Preserves sparsity of V_r (critical for column_subset/random_projection)
            theta_scaled = self.sigma * theta  # Element-wise: O(r)
            v += self.V_r @ theta_scaled  # Sparse matrix-vector: O(nnz(V_r))
            # Still need u for non-negativity constraint
            u = self.U_r @ theta

        # BPR objective
        f_bpr = bpr_objective(v, self.capacity, self.t_0, self.alpha, self.beta)

        # OD conservation: A1*x1 + M*θ = d_multi (using precomputed M)
        if self.r > 0:
            od_flow = self.A1 @ x1 + self.M @ theta
        else:
            od_flow = self.A1 @ x1
        od_error = od_flow - self.d_multi

        # ALM terms for OD constraints
        od_lagrangian = self.lambda_od.T @ od_error
        od_penalty = 0.5 * self.rho_od * np.sum(od_error**2)

        # Non-negativity constraints for minor paths
        if self.r > 0:
            max_term_minor = np.maximum(
                0, self.lambda_minor - self.rho_nonneg_minor * u
            )
            minor_penalty_term = (1.0 / (2.0 * self.rho_nonneg_minor)) * (
                np.sum(max_term_minor**2) - np.sum(self.lambda_minor**2)
            )
        else:
            minor_penalty_term = 0.0

        # Total augmented Lagrangian
        total_obj = f_bpr + od_lagrangian + od_penalty + minor_penalty_term

        # Gradients
        grad_v = bpr_gradient(v, self.capacity, self.t_0, self.alpha, self.beta)

        # Gradient w.r.t. x1
        grad_x1 = self.B1 @ grad_v
        grad_x1 += self.A1.T @ (self.lambda_od + self.rho_od * od_error)

        # Gradient w.r.t. θ (using factored form)
        if self.r > 0:
            # BPR gradient: factored form diag(sigma) @ V_r^T @ grad_v
            grad_theta = self.sigma * (self.V_r.T @ grad_v)  # Element-wise scaling after sparse op
            # OD constraint: M^T @ (λ + ρ*error)
            grad_theta += self.M.T @ (self.lambda_od + self.rho_od * od_error)
            # KKT projection for non-negativity
            grad_theta -= self.U_r.T @ np.maximum(
                0, self.lambda_minor - self.rho_nonneg_minor * u
            )
            grad = np.concatenate([grad_x1, grad_theta])
        else:
            grad = grad_x1

        return total_obj, grad

    def objective_and_gradient_mixed(self, z):
        """Compute augmented Lagrangian using MIXED approach - HYBRID BEST OF BOTH
        
        This function combines:
        1. Factored form for LINK VOLUMES: v += V_r @ (sigma * theta)
           - Exploits sparsity in V_r for BPR computation
           - Fast for sparse compression methods
           
        2. Chain rule for OD FLOW: od_flow = A1*x1 + A2*(U_r@theta)
           - Uses chain multiplication through U_r
           - Avoids storing M matrix explicitly
        
        RATIONALE:
        ==========
        - Link volumes dominate computational cost (evaluated every iteration, large gradient)
        - OD flow is cheaper (smaller constraint set, k << m_links)
        - Factored form provides 10-100x speedup for sparse V_r in BPR computation
        - Chain rule for OD avoids storing dense M matrix (memory efficient)
        
        WHEN TO USE:
        ============
        - Sparse compression (column_subset, random_projection) with memory constraints
        - When M = A2 @ U_r would be dense but V_r is sparse
        - Balance between speed (factored BPR) and memory (chain OD)
        """
        x1 = z[: self.s]
        theta = z[self.s :] if self.r > 0 else np.array([], dtype=np.float64)

        # Compute link volumes using FACTORED FORM (sparse-friendly)
        if self.v_singleton is not None:
            v = self.v_singleton + self.B1.T @ x1
        else:
            v = self.B1.T @ x1

        if self.r > 0:
            # Factored form: V_r @ (sigma * theta) - exploits V_r sparsity
            theta_scaled = self.sigma * theta
            v += self.V_r @ theta_scaled
            # Compute u for OD flow and non-negativity constraint
            u = self.U_r @ theta

        # BPR objective
        f_bpr = bpr_objective(v, self.capacity, self.t_0, self.alpha, self.beta)

        # OD conservation using CHAIN RULE (memory efficient)
        if self.r > 0:
            # use chain rule
            od_flow = self.A1 @ x1 + self.A2 @ u  # A2 @ (U_r @ theta)
            # use precomputed M = A2 @ U_r
            # od_flow = self.A1 @ x1 + self.M @ theta
        else:
            od_flow = self.A1 @ x1
        od_error = od_flow - self.d_multi

        # ALM terms for OD constraints
        od_lagrangian = self.lambda_od.T @ od_error
        od_penalty = 0.5 * self.rho_od * np.sum(od_error**2)

        # Non-negativity constraints for minor paths
        if self.r > 0:
            max_term_minor = np.maximum(
                0, self.lambda_minor - self.rho_nonneg_minor * u
            )
            minor_penalty_term = (1.0 / (2.0 * self.rho_nonneg_minor)) * (
                np.sum(max_term_minor**2) - np.sum(self.lambda_minor**2)
            )
        else:
            minor_penalty_term = 0.0

        # Total augmented Lagrangian
        total_obj = f_bpr + od_lagrangian + od_penalty + minor_penalty_term

        # Gradients
        grad_v = bpr_gradient(v, self.capacity, self.t_0, self.alpha, self.beta)

        # Gradient w.r.t. x1
        grad_x1 = self.B1 @ grad_v
        grad_x1 += self.A1.T @ (self.lambda_od + self.rho_od * od_error)

        # Gradient w.r.t. θ using MIXED approach
        if self.r > 0:
            # BPR gradient: FACTORED form (sparse-friendly)
            grad_theta = self.sigma * (self.V_r.T @ grad_v)
            # OD constraint: CHAIN RULE form (memory efficient)
            grad_theta += self.U_r.T @ (self.A2.T @ (self.lambda_od + self.rho_od * od_error))
            # KKT projection for non-negativity
            grad_theta -= self.U_r.T @ np.maximum(
                0, self.lambda_minor - self.rho_nonneg_minor * u
            )
            grad = np.concatenate([grad_x1, grad_theta])
        else:
            grad = grad_x1

        return total_obj, grad

    def compute_violations(self, x1, theta):
        """Compute constraint violations for equality constraints
        
        Returns:
            od_violation: Maximum OD conservation violation
            nonneg_minor_violation: Maximum minor path non-negativity violation
            u: Cached minor path flows (U_r @ theta), or None if no minor paths
            od_flow: Cached OD flows for reuse
        """
        # Handle OD flow computation
        if self.r > 0:
            u = self.U_r @ theta
            od_flow = self.A1 @ x1 + self.A2 @ u
        else:
            u = None
            od_flow = self.A1 @ x1  # Only major paths when no minor paths

        # Equality constraint violation: |od_flow - d_multi|
        od_violation = np.max(np.abs(od_flow - self.d_multi))

        # Handle minor path violations (only if minor paths exist and enabled)
        if self.r > 0:
            self.x2 = u
            nonneg_minor_violation = np.max(-np.minimum(u, 0))
        else:
            nonneg_minor_violation = 0.0  # No minor paths = no violation

        return od_violation, nonneg_minor_violation, u, od_flow

    def get_od_violation_details(self, x1, theta, gamma):
        """Get detailed OD violation information for all OD pairs"""
        # Handle OD flow computation
        if self.r > 0:
            u = self.U_r @ theta
            od_flow = self.A1 @ x1 + self.A2 @ u
        else:
            od_flow = self.A1 @ x1

        # Use d_multi for error calculation (multi-path ODs only)
        od_errors = np.abs(od_flow - self.d_multi)

        # Calculate percentage errors (avoid division by zero)
        od_pct_errors = np.zeros_like(od_errors)
        nonzero_demand_mask = self.d_multi > NEGLIGIBLE_DEMAND_THRESHOLD
        od_pct_errors[nonzero_demand_mask] = (
            od_errors[nonzero_demand_mask] / self.d_multi[nonzero_demand_mask]
        ) * 100

        # Calculate comprehensive statistics for multi-path OD pairs only
        n_od = len(self.d_multi)
        max_abs_viol = np.max(od_errors)
        mean_abs_viol = np.mean(od_errors)
        # Count OD pairs with absolute error > gamma
        num_violated = np.sum(od_errors > gamma)
        # Overall MAE and RMSE
        od_mae = mean_abs_viol

        # R² metric for multi-path OD flows
        ss_res = np.sum((od_flow - self.d_multi) ** 2)
        ss_tot = np.sum((self.d_multi - np.mean(self.d_multi)) ** 2)
        od_r2 = 1 - ss_res / (ss_tot + 1e-10)

        # RMSE
        od_rmse = np.sqrt(np.mean((od_flow - self.d_multi) ** 2))

        # Find worst violations by absolute error (multi-path OD pairs)
        # Top 5 worst by absolute error
        sorted_indices = np.argsort(od_errors)[-5:][::-1]

        details = {
            "max_violation_abs": max_abs_viol,
            "mean_violation_abs": mean_abs_viol,
            "num_violated": num_violated,
            "num_nonzero_od": n_od,
            "od_mae": od_mae,
            "od_r2": od_r2,
            "od_rmse": od_rmse,
            "worst_od_pairs": [],
        }

        for idx in sorted_indices:
            # Only include if absolute error > 0.01
            if od_errors[idx] > 0.01:
                details["worst_od_pairs"].append(
                    {
                        "od_index": int(idx),
                        "demand": float(self.d_multi[idx]),
                        "predicted": float(od_flow[idx]),
                        "error_abs": float(od_errors[idx]),
                        "error_pct": float(
                            od_pct_errors[idx]
                        ),  # Keep for reference but not displayed
                    }
                )

        return details

    def compute_metrics(self, x1, theta, v):
        """Compute comprehensive metrics"""
        # Handle minor path metrics (only if minor paths exist)
        if self.r > 0:
            x2_mae = np.mean(np.abs(self.x2 - self.x2_ref))
            x2_r2 = 1 - np.sum((self.x2 - self.x2_ref) ** 2) / (
                np.sum((self.x2_ref - np.mean(self.x2_ref)) ** 2) + 1e-10
            )
        else:
            # No minor paths - set appropriate values
            self.x2 = np.array([], dtype=np.float64)
            x2_mae = 0.0
            x2_r2 = 1.0  # Perfect fit when no minor paths to predict

        link_mae = np.mean(np.abs(v - self.v_ref))
        link_r2 = 1 - np.sum((v - self.v_ref) ** 2) / (
            np.sum((self.v_ref - np.mean(self.v_ref)) ** 2) + 1e-10
        )

        x1_mae = np.mean(np.abs(x1 - self.x1_ref))
        x1_r2 = 1 - np.sum((x1 - self.x1_ref) ** 2) / (
            np.sum((self.x1_ref - np.mean(self.x1_ref)) ** 2) + 1e-10
        )

        # Weighted combined metrics
        x_full = np.zeros(len(self.major_mask))
        x_full[self.major_mask] = x1
        if self.r > 0 and len(self.x2) > 0:
            x_full[self.minor_mask] = self.x2

        x_ref_full = np.zeros(len(self.major_mask))
        x_ref_full[self.major_mask] = self.x1_ref
        if self.r > 0 and len(self.x2_ref) > 0:
            x_ref_full[self.minor_mask] = self.x2_ref

        bpr_pure = bpr_objective(v, self.capacity, self.t_0, self.alpha, self.beta)
        bpr_gap = bpr_pure - self.bpr_optimal
        bpr_gap_pct = 100 * bpr_gap / self.bpr_optimal

        # Travel time metrics (using BPR function - congestion component with t_0 scaling)
        t_ref = self.t_0 * self.alpha * (self.v_ref / self.capacity) ** self.beta
        t_pred = self.t_0 * self.alpha * (v / self.capacity) ** self.beta

        travel_time_mae = np.mean(np.abs(t_pred - t_ref))
        travel_time_r2 = 1 - np.sum((t_pred - t_ref) ** 2) / (
            np.sum((t_ref - np.mean(t_ref)) ** 2) + 1e-10
        )

        return {
            "link_mae": link_mae,
            "link_r2": link_r2,
            "x1_mae": x1_mae,
            "x1_r2": x1_r2,
            "x2_mae": x2_mae,
            "x2_r2": x2_r2,
            "bpr_pure": bpr_pure,
            "bpr_gap": bpr_gap,
            "bpr_gap_pct": bpr_gap_pct,
            "travel_time_mae": travel_time_mae,
            "travel_time_r2": travel_time_r2,
        }

    def update_multipliers(self, x1, theta, u_cached=None, od_flow_cached=None):
        """Update Lagrangian multipliers using standard ALM update
        
        Args:
            x1: Major path flows
            theta: Latent variables
            u_cached: Pre-computed U_r @ theta (avoids recomputation)
            od_flow_cached: Pre-computed OD flows (avoids recomputation)
        """
        # Use cached values if provided, otherwise compute
        if od_flow_cached is not None:
            od_flow = od_flow_cached
        else:
            if self.r > 0:
                u = u_cached if u_cached is not None else self.U_r @ theta
                od_flow = self.A1 @ x1 + self.A2 @ u
            else:
                od_flow = self.A1 @ x1

        # Standard ALM update for OD conservation: λ^(k+1) = λ^k + ρ*(od_flow - d)
        od_error = od_flow - self.d_multi
        self.lambda_od += self.rho_od * od_error

        # Minor path non-negativity (inequality): KKT projection
        if self.r > 0:
            self.lambda_minor = np.maximum(
                0, self.lambda_minor - self.rho_nonneg_minor * self.x2
            )

    def update_penalties(self, od_viol, minor_viol, eta=0.25):
        """Adaptive penalty update: only increase if violation doesn't decrease sufficiently
        
        Args:
            od_viol: Current OD constraint violation
            minor_viol: Current minor path non-negativity violation
            eta: Reduction factor threshold (default 0.25)
                 Increase penalty only if current_viol > eta * previous_viol
        """
        # OD constraint penalty
        if od_viol >= self.gamma:
            # Check if violation decreased sufficiently
            if self.prev_od_viol is None or od_viol > eta * self.prev_od_viol:
                # Violation didn't decrease enough, increase penalty
                self.rho_od = min(self.rho_od * self.tau, self.MAX_PENALTY)
            # else: keep penalty the same (violation decreased sufficiently)
        
        # Store current violation for next iteration
        self.prev_od_viol = od_viol

        # Minor path non-negativity penalty
        if minor_viol >= self.gamma:
            # Check if violation decreased sufficiently
            if self.prev_minor_viol is None or minor_viol > eta * self.prev_minor_viol:
                # Violation didn't decrease enough, increase penalty
                self.rho_nonneg_minor = min(
                    self.rho_nonneg_minor * self.tau, self.MAX_PENALTY
                )
            # else: keep penalty the same (violation decreased sufficiently)
        
        # Store current violation for next iteration
        self.prev_minor_viol = minor_viol

    def check_convergence(
        self,
        od_viol,
        minor_viol,
        stagnation_tol,
        stagnation_window,
        outer_iter,
    ):
        # CONVERGENCE CHECK 1: Constraint satisfaction
        if od_viol < self.gamma and minor_viol < self.gamma:
            convergence_reason = f"viol < gamma={self.gamma}"
            return True, convergence_reason

        # CONVERGENCE CHECK 2: Stagnation detection
        if outer_iter >= stagnation_window:
            recent_bpr = self.history["bpr_pure"][-stagnation_window:]
            recent_od_viol = self.history["od_violation"][-stagnation_window:]

            # Check if BPR objective has stagnated
            bpr_range = max(recent_bpr) - min(recent_bpr)
            bpr_rel_change = bpr_range / (abs(recent_bpr[0]) + 1e-10)

            # Check if OD violation has stagnated
            od_viol_range = max(recent_od_viol) - min(recent_od_viol)
            od_viol_rel_change = od_viol_range / (recent_od_viol[0] + 1e-10)

            if bpr_rel_change < stagnation_tol and od_viol_rel_change < stagnation_tol:
                convergence_reason = f"Stagnation detected (BPR delta={bpr_rel_change:.2e}, OD delta={od_viol_rel_change:.2e})"
                return True, convergence_reason

        # CONVERGENCE CHECK 3: Penalty maxed out (structural infeasibility)
        if self.rho_od >= self.MAX_PENALTY:
            convergence_reason = (
                f"Penalty maxed (ρ={self.rho_od:.0e}), OD violation is structural"
            )
            return True, convergence_reason

        return False, None

    def update_history(
        self,
        outer_iter,
        result,
        metrics,
        od_viol,
        minor_viol,
        cpu_elapsed,
        inner_cpu_time,
        outer_cpu_time,
    ):
        self.history["outer_iter"].append(outer_iter)
        self.history["inner_iter"].append(result.nit)
        self.history["objective"].append(result.fun)
        self.history["bpr_pure"].append(metrics["bpr_pure"])
        self.history["bpr_optimal"].append(self.bpr_optimal)
        self.history["bpr_gap"].append(metrics["bpr_gap"])
        self.history["bpr_gap_pct"].append(metrics["bpr_gap_pct"])
        self.history["rho_od"].append(self.rho_od)
        self.history["rho_nonneg_minor"].append(self.rho_nonneg_minor)
        self.history["od_violation"].append(od_viol)
        self.history["nonneg_minor_violation"].append(minor_viol)
        self.history["link_r2"].append(metrics["link_r2"])
        self.history["link_mae"].append(metrics["link_mae"])
        self.history["x1_r2"].append(metrics["x1_r2"])
        self.history["x1_mae"].append(metrics["x1_mae"])
        self.history["x2_r2"].append(metrics["x2_r2"])
        self.history["x2_mae"].append(metrics["x2_mae"])
        self.history["travel_time_r2"].append(metrics["travel_time_r2"])
        self.history["travel_time_mae"].append(metrics["travel_time_mae"])
        self.history["elapsed_time"].append(cpu_elapsed)
        self.history["inner_time"].append(inner_cpu_time)
        self.history["outer_time"].append(outer_cpu_time)

    def initialize_solution(
        self, enable_warm_start=False, enable_proportional_cold_start=True
    ):
        if enable_warm_start:
            # warm start
            # print("  Using warm start (0.1) for optimization")
            # x1 = np.maximum(self.x1_ref, 0.1)
            print("  Using warm start (0.0) for optimization")
            x1 = np.copy(self.x1_ref)
        else:
            if enable_proportional_cold_start:
                # cold start: proportional distribution of demand across paths
                print(
                    f"  Using cold start for optimization (proportional distribution, {self.k} OD pairs)"
                )

                # Vectorized approach: use sparse matrix operations
                # A1 is (k × s) sparse matrix where A1[od, path] = 1 if path serves od
                # Count paths per OD: A1.sum(axis=1) gives number of paths for each OD
                paths_per_od = np.asarray(self.A1.sum(axis=1)).flatten()  # (k,)

                # Compute flow per path for each OD: demand / num_paths
                # Avoid division by zero
                flow_per_path_by_od = np.zeros(self.k)
                nonzero_paths = paths_per_od > 0
                flow_per_path_by_od[nonzero_paths] = (
                    self.d_multi[nonzero_paths] / paths_per_od[nonzero_paths]
                )

                # Broadcast to all paths: x1 = A1.T @ flow_per_path_by_od
                # This automatically distributes the flow to the right paths
                x1 = self.A1.T @ flow_per_path_by_od  # (s,)
                if hasattr(x1, "toarray"):
                    x1 = x1.toarray().flatten()

                # Add small noise to break symmetry
                x1 += np.random.uniform(0, 0.01, size=self.s)
            else:
                # print(f"  Using cold start for optimization (0.1, {self.k} OD pairs)")
                # x1 = np.ones(self.s) * 0.1
                print(f"  Using cold start for optimization (0.0, {self.k} OD pairs)")
                x1 = np.zeros(self.s)

        # Handle theta initialization (only if minor paths exist)
        if self.r > 0:
            theta = np.zeros(self.r)
            z = np.concatenate([x1, theta])
            bounds = [(0, None)] * self.s + [(None, None)] * self.r
        else:
            theta = np.array([], dtype=np.float64)
            z = x1  # Only x1 when no minor paths
            bounds = [(0, None)] * self.s  # Only bounds for x1

        return z, x1, theta, bounds

    @staticmethod
    def print_header():
        print(f"\n{'=' * 125}")
        print("BERTSEKAS AUGMENTED LAGRANGIAN METHOD (KKT PROJECTION + STAGNATION)")
        print(f"{'=' * 125}")
        print(
            f"{'Outer':>6} {'Inner':>6} {'Status':>6} {'Objective':>12} {'OD Viol':>10} {'Minor Viol':>11} {'ρ_OD':>10} {'ρ_minor':>10} {'Link R²':>8} {'Inner(s)':>10} {'Outer(s)':>10}"
        )
        print(f"{'-' * 125}")

    def print_iteration_metrics(self, outer_iter, result, metrics, od_viol, minor_viol, inner_time, outer_time, initial_rho_od, initial_rho_nonneg_minor):
        status = "S" if result.success else "F"
        print(
            f"{outer_iter:>6} {result.nit:>6} {status:>6} {result.fun:>12.4e} "
            f"{od_viol:>10.7f} {minor_viol:>10.7f} "
            f"{initial_rho_od:>10.2e} {initial_rho_nonneg_minor:>10.2e} "
            f"{metrics['link_r2']:>8.4f} "
            f"{inner_time:>10.3f} {outer_time:>10.3f}"
        )

    def optimize(
        self,
        max_outer_iter=15,
        max_inner_iter=50,
        stagnation_tol=1e-6,
        stagnation_window=3,
    ):
        """Run Bertsekas ALM with KKT projection and stagnation detection"""
        verbose = True
        cpu_start_time = time.process_time()

        convergence_reason = "Max iterations reached"

        z, x1, theta, bounds = self.initialize_solution(
            enable_warm_start=False, enable_proportional_cold_start=True
        )

        if verbose:
            self.print_header()

        for outer_iter in range(max_outer_iter):
            outer_cpu_start = time.process_time()

            # Store initial penalty values
            initial_rho_od = self.rho_od
            initial_rho_nonneg_minor = self.rho_nonneg_minor

            options = {
                "maxiter": max_inner_iter,
                "disp": False,
                "gtol": 1e-6,
                "ftol": 1e-9,
            }

            inner_cpu_start = time.process_time()
            result = minimize(
                fun=lambda z_: self.objective_and_gradient_mixed(z_),
                x0=z,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options=options,
            )
            inner_cpu_time = time.process_time() - inner_cpu_start

            z = result.x
            x1 = z[: self.s]
            theta = z[self.s :] if self.r > 0 else np.array([], dtype=np.float64)

            # Check convergence and cache intermediate computations
            od_viol, minor_viol, u_cached, od_flow_cached = self.compute_violations(x1, theta)
            has_converged, convergence_reason = self.check_convergence(
                od_viol,
                minor_viol,
                stagnation_tol,
                stagnation_window,
                outer_iter            
            )

            if not has_converged:
                # Reuse cached values to avoid redundant computations
                self.update_multipliers(x1, theta, u_cached=u_cached, od_flow_cached=od_flow_cached)
                self.update_penalties(od_viol, minor_viol, eta=0.25)

            outer_cpu_time = time.process_time() - outer_cpu_start

            cpu_elapsed = time.process_time() - cpu_start_time

            # Compute link volumes using cached u to avoid recomputing U_r @ theta
            if self.v_singleton is not None:
                v = self.v_singleton + self.B1.T @ x1
            else:
                v = self.B1.T @ x1
            
            if self.r > 0:
                # Reuse cached u instead of recomputing U_r @ theta
                v += self.B2.T.dot(u_cached)

            metrics = self.compute_metrics(x1, theta, v)
            self.update_history(
                outer_iter,
                result,
                metrics,
                od_viol,
                minor_viol,
                cpu_elapsed,
                inner_cpu_time,
                outer_cpu_time,
            )

            if verbose:
                self.print_iteration_metrics(
                    outer_iter,
                    result,
                    metrics,
                    od_viol,
                    minor_viol,
                    inner_time=inner_cpu_time,
                    outer_time=outer_cpu_time,
                    initial_rho_od=initial_rho_od,
                    initial_rho_nonneg_minor=initial_rho_nonneg_minor
                )

            if has_converged:
                break

        else:
            # Loop completed without break
            convergence_reason = f"Max iterations ({max_outer_iter}) reached"
            if verbose:
                print(f"\n {convergence_reason}")
                print(
                    f"  Final: OD viol={od_viol:.4f}, Link R²={metrics['link_r2']:.4f}, BPR gap={metrics['bpr_gap_pct']:.2f}%"
                )

        total_time = time.process_time() - cpu_start_time

        # Determine if truly converged vs stopped early
        # Only consider it converged if OD violation < gamma and x2 nonnegativity < gamma
        truly_converged = od_viol < self.gamma and minor_viol < self.gamma

        return {
            "x1": x1,
            "theta": theta,
            "x2": self.x2,
            "v": v,
            "success": result.success,
            "converged": truly_converged,
            "convergence_reason": convergence_reason,
            "outer_iterations": outer_iter + 1,
            "total_inner_iterations": sum(self.history["inner_iter"]),
            "elapsed_time": total_time,
            "final_objective": result.fun,
            "final_metrics": metrics,
            "final_violations": (od_viol, minor_viol),
            "history": self.history,
        }


################################################################################
# THRESHOLD SENSITIVITY ANALYSIS
################################################################################

def analyze_path_flow_distribution(x_ref):
    """ Analyze flow distribution (excluding zeros for percentiles) """
    # Remove NaN values first
    x_valid = x_ref[~np.isnan(x_ref)]
    x_nonzero = x_valid[x_valid > 0]

    if len(x_nonzero) == 0:
        print("\n Warning: All path flows are zero or NaN!")
        return pd.DataFrame()

    # flow_percentiles = np.percentile(x_nonzero, [10, 25, 50, 75, 90, 95, 100])
    flow_percentiles = np.percentile(x_nonzero, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    print("\nPath flow distribution:")
    print(f"  Max: {np.max(x_valid):.2f}")
    print(f"  Min (non-zero): {np.min(x_nonzero):.2f}")
    print(f"  Number of paths: {len(x_ref)}")
    print(f"  Valid paths: {len(x_valid)}")
    print(f"  Non-zero paths: {len(x_nonzero)}")
    if len(x_ref) > len(x_valid):
        print(f"   NaN paths: {len(x_ref) - len(x_valid)}")

    print("\n  Flow percentiles (excluding zeros):")
    print(f"    10th: {flow_percentiles[0]:.2f}")
    print(f"    20th: {flow_percentiles[1]:.2f}")
    print(f"    30th: {flow_percentiles[2]:.2f}")
    print(f"    40th: {flow_percentiles[3]:.2f}")
    print(f"    50th (median): {flow_percentiles[4]:.2f}")
    print(f"    60th: {flow_percentiles[5]:.2f}")
    print(f"    70th: {flow_percentiles[6]:.2f}")
    print(f"    80th: {flow_percentiles[7]:.2f}")
    print(f"    90th: {flow_percentiles[8]:.2f}")
    print(f"    100th: {flow_percentiles[9]:.2f}")
    
    return flow_percentiles


def setup_thresholds(x_ref, od_info=None, num_bins=11):
    """Setup threshold values for major/minor path decomposition
    
    New approach: Adaptively select thresholds that proportionally split paths into K bins
    
    Args:
        x_ref: Reference path flows
        od_info: OD information dict with 'path_to_od' mapping
        num_bins: Number of bins to split paths into (default: 11)
    
    Returns:
        List of threshold values that split the path set into K equal-sized bins
        
    Logic:
        - Each OD must have at least one major path (threshold < max flow for that OD)
        - Use multi-path OD flows only (single-path ODs are not affected by thresholds)
        - Select thresholds at quantiles: 0%, 1/(K-1)*100%, 2/(K-1)*100%, ..., 100%
        - This ensures each bin contains approximately the same number of paths
    """
    flow_percentiles = analyze_path_flow_distribution(x_ref)
    
    if od_info is not None and 'path_to_od' in od_info:
        # Identify multi-path ODs (ODs with more than 1 path)
        path_to_od = od_info['path_to_od']
        od_path_counts = {}
        
        for path_idx, od_idx in enumerate(path_to_od):
            if od_idx not in od_path_counts:
                od_path_counts[od_idx] = []
            od_path_counts[od_idx].append(path_idx)
        
        # Get paths belonging to multi-path ODs
        multi_path_ods = {od_idx: paths for od_idx, paths in od_path_counts.items() if len(paths) > 1}
        multi_path_indices = [path_idx for paths in multi_path_ods.values() for path_idx in paths]
        
        if len(multi_path_indices) > 0:
            # Get flows for multi-path OD paths only
            multi_path_flows = x_ref[multi_path_indices]
            
            # STEP 0: Identify paths that MUST be major (max flow per OD)
            must_be_major_indices = []
            for od_idx, paths in multi_path_ods.items():
                od_flows = x_ref[paths]
                max_flow_path = paths[np.argmax(od_flows)]
                must_be_major_indices.append(max_flow_path)
            
            must_be_major_set = set(must_be_major_indices)
            can_be_minor_indices = np.array([p for p in multi_path_indices if p not in must_be_major_set])
            
            print("\nSTEP 0: Identify mandatory major paths")
            print(f"  Paths that MUST be major (max flow per OD): {len(must_be_major_set)}")
            print(f"  Paths that CAN be minor: {len(can_be_minor_indices)}")
            
            # STEP 1: Find maximum number of minor paths
            # Key insight: Each OD must have ≥1 major path
            # For an OD with N paths, we can have at most N-1 minor paths
            # Therefore: max_minor_paths = sum(N-1) for all multi-path ODs
            #          = total_paths - num_multi_path_ODs
            
            max_minor_paths = len(multi_path_indices) - len(multi_path_ods)
            
            max_multi_flow = np.max(multi_path_flows)
            print("\nMulti-path OD analysis:")
            print(f"  # Multi-path ODs: {len(multi_path_ods)}")
            print(f"  # Paths in multi-path ODs: {len(multi_path_indices)}")
            print(f"  Flow range: [{np.min(multi_path_flows):.2f}, {max_multi_flow:.2f}]")
            print("\nSTEP 1: Maximum number of minor paths")
            print("  Each OD must have ≥1 major path")
            print("  For OD with N paths: max N-1 minor paths")
            print(f"  Total paths in multi-ODs: {len(multi_path_indices)}")
            print(f"  # Multi-path ODs: {len(multi_path_ods)}")
            print(f"  Max possible minor paths = {len(multi_path_indices)} - {len(multi_path_ods)} = {max_minor_paths}")
            
            # STEP 2: Split max_minor_paths into K equal bins using equally-spaced indices
            # Sort flows only from paths that CAN be minor (exclude mandatory major paths)
            can_be_minor_flows = x_ref[can_be_minor_indices]
            sorted_flows = np.sort(can_be_minor_flows)
            
            # Use equally-spaced indices only within max_minor_paths range
            # First threshold is always 0, last is max flow, rest are equally spaced
            indices = np.linspace(0, max_minor_paths-1, num_bins, dtype=int)
            thresholds = sorted_flows[indices].tolist()
            
            # Enforce first threshold = 0 and last threshold = max flow
            thresholds[0] = 0.0
            
            print(f"\nSTEP 2: Split into K={num_bins} bins using equally-spaced indices")
            print(f"  Sorted flows from paths that CAN be minor: {len(sorted_flows)}")
            print(f"  Indices: {indices.tolist()}")
            print(f"  Threshold values: {[f'{t:.6f}' for t in thresholds]}")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_thresholds = []
            for t in thresholds:
                if t not in seen:
                    seen.add(t)
                    unique_thresholds.append(t)
            
            print(f"\nFinal thresholds (after removing duplicates): {len(unique_thresholds)} values")
            print(f"  {[f'{t:.6f}' for t in unique_thresholds]}")

            # unique_thresholds.remove(0.0)  # Remove 0 threshold to avoid all paths being major
            
            return unique_thresholds
        else:
            print("\nWarning: No multi-path ODs found")
    else:
        print("\nWarning: od_info not provided or no 'path_to_od' mapping")
    
    # Fallback: use percentiles from all paths
    print("\nFalling back to percentile-based thresholds from all paths")
    thresholds = [
        0,
        flow_percentiles[0],
        flow_percentiles[1],
        flow_percentiles[2],
        flow_percentiles[3],
        flow_percentiles[4],
        flow_percentiles[5],
        flow_percentiles[6],
        flow_percentiles[7],
        flow_percentiles[8],
        flow_percentiles[9],
    ]
    
    thresholds = sorted(list(set(thresholds)))
    print(f"\nTesting {len(thresholds)} thresholds: {[f'{t:.2f}' for t in thresholds]}")
    return thresholds


def run_threshold_sensitivity_analysis(
    B, x_ref, v_ref, capacity, t_0, od_info, output_dir, rank, thresholds, gamma=0.1
):
    """
    Run threshold sensitivity analysis

    Tests different threshold values to understand:
    1. Impact on compression ratio (major vs minor paths)
    2. Impact on solution quality (R², MAE)
    3. Impact on computational efficiency
    4. Trade-offs between model complexity and accuracy
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 101)
    print(f" THRESHOLD SENSITIVITY ANALYSIS with GAMMA = {gamma}")
    print("=" * 101)

    results = []
    prev_config = None  # Track previous (n_major, n_minor, rank) configuration

    for i, threshold in enumerate(thresholds):
        print(f"\n[{i + 1}/{len(thresholds)}] Testing threshold = {threshold}")
        print("-" * 100)

        # Decompose with this threshold
        decomp = decompose_paths(B, x_ref, od_info, threshold)

        n_major = decomp["s"]
        n_minor = decomp["n_minor"]
        n_total = n_major + n_minor
        major_pct = 100 * n_major / n_total

        # Flow captured by major paths (use nansum to handle NaN values)
        major_flow = np.nansum(decomp["x1_ref"])
        minor_flow = np.nansum(decomp["x2_ref"])
        total_flow = major_flow + minor_flow
        major_flow_pct = 100 * major_flow / total_flow if total_flow > 0 else 0

        print("  Decomposition:")
        print(
            f"    Major: {n_major} paths ({major_pct:.1f}%), flow: {major_flow:.2f} ({major_flow_pct:.1f}%)"
        )
        print(
            f"    Minor: {n_minor} paths ({100 - major_pct:.1f}%), flow: {minor_flow:.2f} ({100 - major_flow_pct:.1f}%)"
        )

        # Handle special case: no minor paths (when threshold is 0 or very low, all paths become major)
        if n_minor == 0:
            print(
                "      No minor paths - running optimization with major paths only (no SVD compression)"
            )
            # Create empty svd_dict for compatibility
            svd_dict = {
                "U_r": csr_matrix((0, 0), dtype=np.float64),
                "D": csr_matrix((0, 0), dtype=np.float64),
                "theta_ref": np.array([], dtype=np.float64),
                "r": 0,
                "compression_ratio": float("inf"),
                "explained_variance_ratio": 1.0,
                "reconstruction_error": 0.0,
                "svd_time": 0.0,
            }
        else:
            # SVD compression for minor paths
            svd_dict = compute_svd_compression(
                decomp["B2"], decomp["x2_ref"], max_rank=rank
            )
            if svd_dict is None:
                print(f"   SVD compression failed - skipping threshold {threshold}")
                continue

        # Check if configuration is same as previous iteration
        current_config = (n_major, n_minor)
        if prev_config is not None and current_config == prev_config:
            print(
                f"     Skipping: Same configuration as previous threshold (n_major={n_major}, n_minor={n_minor})"
            )
            print(
                "     This threshold produces identical decomposition - results would be the same"
            )
            continue

        prev_config = current_config  # Update for next iteration

        # Report compression statistics
        r = svd_dict["r"]
        compression_ratio = n_minor / r if r > 0 else float("inf")
        total_vars = n_major + r

        svd_time = svd_dict["svd_time"]
        if n_minor > 0:
            print(
                f"  SVD: {n_minor} minor paths → {r} latent variables (compression: {compression_ratio:.2f}x, time: {svd_time:.3f}s)"
            )
        print(f"  Total decision variables: {total_vars} (vs {n_total} original paths)")

        # Optimize
        try:
            optimizer = BertsekasALM(
                decomp,
                svd_dict,
                capacity,
                t_0,
                od_info,
                v_ref,
                rho_od_init=1e3,
                rho_nonneg_minor_init=1e3,
                tau=4.0,
                gamma=gamma,
            )

            opt_cpu_start = time.process_time()

            result = optimizer.optimize(
                max_outer_iter=20,
                max_inner_iter=200,
                stagnation_tol=1e-6,
                stagnation_window=3,
            )
            opt_cpu_time = time.process_time() - opt_cpu_start

            final_metrics = result["final_metrics"]
            viol = result["final_violations"]

            # Calculate per-iteration timing metrics
            per_outer_cpu_time = (
                opt_cpu_time / result["outer_iterations"]
                if result["outer_iterations"] > 0
                else 0.0
            )
            per_inner_cpu_time = (
                opt_cpu_time / result["total_inner_iterations"]
                if result["total_inner_iterations"] > 0
                else 0.0
            )

            summary = {
                "threshold": threshold,
                "n_major": n_major,
                "n_minor": n_minor,
                "major_pct": major_pct,
                "major_flow": major_flow,
                "minor_flow": minor_flow,
                "major_flow_pct": major_flow_pct,
                "svd_rank": r,
                "compression_ratio": compression_ratio,
                "total_vars": total_vars,
                "reduction_pct": 100 * (1 - total_vars / n_total),
                "speedup_ub": n_total / total_vars,
                "outer_iterations": result["outer_iterations"],
                "total_inner_iterations": result["total_inner_iterations"],
                "svd_time": svd_time,
                "opt_cpu_time": opt_cpu_time,
                "per_outer_cpu_time": per_outer_cpu_time,
                "per_inner_cpu_time": per_inner_cpu_time,
                "bpr_optimal": optimizer.bpr_optimal,
                "bpr_pure": final_metrics["bpr_pure"],
                "bpr_gap": final_metrics["bpr_gap"],
                "bpr_gap_pct": final_metrics["bpr_gap_pct"],
                "od_violation": viol[0],
                "nonneg_minor_violation": viol[1],
                "link_r2": final_metrics["link_r2"],
                "link_mae": final_metrics["link_mae"],
                "x1_r2": final_metrics["x1_r2"],
                "x1_mae": final_metrics["x1_mae"],
                "x2_r2": final_metrics["x2_r2"],
                "x2_mae": final_metrics["x2_mae"],
                "travel_time_r2": final_metrics["travel_time_r2"],
                "travel_time_mae": final_metrics["travel_time_mae"],
                "converged": (viol[0] < gamma and viol[1] < gamma),
                "convergence_reason": result["convergence_reason"],
            }

            results.append(summary)
            print_summary(summary)

            print_link_volume_analysis(result, v_ref, capacity)
            print_od_violation_analysis(optimizer, result, gamma)

        except Exception as e:
            print(f"    Optimization failed: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "threshold_sensitivity_results.csv", index=False)
    print(f"\n  Saved: {output_dir / 'threshold_sensitivity_results.csv'}")

    print_summary_table(results_df)

    return results_df


def print_link_volume_analysis(result, v_ref, capacity):
    # Analyze link-level deviations and BPR impact
    v = result["v"]
    v_diff = v - v_ref
    v_pct_diff = 100 * v_diff / (v_ref + 1e-10)
    congested_links = v_ref > 0.5 * capacity  # Links at >50% capacity

    print("  Link Volume Analysis:")
    print(f"    Max absolute error: {np.max(np.abs(v_diff)):.2f}")
    print(
        f"    # links with >10% error: {np.sum(np.abs(v_pct_diff) > 10)}/{len(v)}"
    )
    print(f"    # congested links (>50% capacity): {np.sum(congested_links)}")
    if np.sum(congested_links) > 0:
        print(
            f"    Congested links MAE: {np.mean(np.abs(v_diff[congested_links])):.2f}"
        )


def print_od_violation_analysis(optimizer, result, gamma):
    # Print detailed OD violation information
    od_details = optimizer.get_od_violation_details(
        result["x1"], result["theta"], gamma
    )
    print("  OD Constraint Details:")
    print("    Accuracy Metrics:")
    print(f"      R²: {od_details['od_r2']:.6f}")
    print(f"      MAE: {od_details['od_mae']:.4f}")
    print(f"      RMSE: {od_details['od_rmse']:.4f}")
    print("    Violation Metrics:")
    print(f"      Max violation: {od_details['max_violation_abs']:.7f}")
    print(f"      Mean violation: {od_details['mean_violation_abs']:.7f}")
    print(
        f"      # OD pairs violated (>{gamma}): {od_details['num_violated']}/{od_details['num_nonzero_od']}"
    )
    if od_details["worst_od_pairs"]:
        print("    Top violating OD pairs:")
        for i, od_pair in enumerate(od_details["worst_od_pairs"][:3], 1):
            print(
                f"      {i}. OD #{od_pair['od_index']}: demand={od_pair['demand']:.1f}, "
                f"predicted={od_pair['predicted']:.1f}, error={od_pair['error_abs']:.7f}"
            )
    
    # Minor path non-negativity violation details
    if optimizer.r > 0:
        print("  Minor Path Non-negativity Details:")
        x2 = optimizer.x2  # Already computed in get_od_violation_details
        n_minor = len(x2)
        
        # Count violations
        negative_flows = x2 < -gamma
        n_violations = np.sum(negative_flows)
        
        if n_violations > 0:
            min_flow = np.min(x2)
            mean_negative = np.mean(x2[negative_flows]) if n_violations > 0 else 0.0
            max_violation = np.max(-x2[negative_flows])
            
            print(f"    # Minor paths with negative flow: {n_violations}/{n_minor}")
            print(f"    Min flow (most negative): {min_flow:.7f}")
            print(f"    Mean negative flow: {mean_negative:.7f}")
            print(f"    Max violation magnitude: {max_violation:.7f}")
            
            # Show worst violators
            worst_indices = np.argsort(x2)[:min(3, n_violations)]
            print("    Top violating minor paths:")
            for i, idx in enumerate(worst_indices, 1):
                print(f"      {i}. Path #{idx}: flow={x2[idx]:.7f}")
        else:
            print(f"    All {n_minor} minor paths have non-negative flow")
    else:
        print("  Minor Path Non-negativity Details:")
        print("    No minor paths in this decomposition")


def print_summary(summary):
    print("\n  Summary:")
    print(f"    Converged: {summary['converged']}")
    print(
        f"    BPR: {summary['bpr_pure']:.4e} (Optimal: {summary['bpr_optimal']:.4e})"
    )
    print(
        f"    BPR Gap: {summary['bpr_gap_pct']:.3f}% {'(Better!)' if summary['bpr_gap_pct'] < 0 else ''}"
    )
    print(
        f"    Link R²: {summary['link_r2']:.6f}, MAE: {summary['link_mae']:.2f}"
    )
    print(
        f"    Travel Time R²: {summary['travel_time_r2']:.6f}, MAE: {summary['travel_time_mae']:.6f}"
    )
    print(
        f"    Time: SVD={summary['svd_time']:.3f}s (CPU)"
    )
    print(
        f"    Optimization: {summary['opt_cpu_time']:.2f}s (CPU)"
    )
    print(
        f"    Per-iteration: Outer={summary['per_outer_cpu_time']:.3f}s, Inner={summary['per_inner_cpu_time']:.4f}s (CPU)"
    )
    print(f"    Variables reduced by: {summary['reduction_pct']:.1f}%")
    print(f"    Speedup upper bound: {summary['speedup_ub']:.2f}x")


def print_summary_table(results_df):
    if results_df.empty:
        return

    print(f"\n{'=' * 210}")
    print(" SUMMARY TABLE")
    print(f"{'=' * 210}")

    # Calculate speedup relative to first (threshold=0) run
    baseline_time = (
        results_df.loc[0, "opt_cpu_time"] if len(results_df) > 0 else 1.0
    )
    results_df["speedup_cpu_time"] = baseline_time / results_df["opt_cpu_time"]

    # Select columns for summary
    summary_cols = [
        "threshold",
        "n_major",
        "n_minor",
        "svd_rank",
        "compression_ratio",
        "reduction_pct",
        "link_r2",
        "bpr_gap_pct",
        "od_violation",
        "nonneg_minor_violation",  # Add Minor Viol column
        "opt_cpu_time",
        "speedup_cpu_time",
        "speedup_ub",
        "converged",
        "convergence_reason",
    ]

    summary_df = results_df[summary_cols].copy()

    # Format the table
    print(
        f"{'threshold':>10} {'n_major':>12} {'n_minor':>12} {'svd_rank':>9} {'compression_ratio':>18} "
        f"{'reduction_pct':>14} {'link_r2':>9} {'bpr_gap_pct':>12} {'od_violation':>13} {'Minor Viol':>13} "
        f"{'opt_cpu_time':>14} {'speedup_cpu_time':>12} {'speedup_ub':>12} {'converged':>10} {'convergence_reason':<50}"
    )

    for _, row in summary_df.iterrows():
        print(
            f"{row['threshold']:>10.2f} "
            f"{int(row['n_major']):>12,} "
            f"{int(row['n_minor']):>12,} "
            f"{int(row['svd_rank']):>9,} "
            f"{row['compression_ratio']:>18.2f} "
            f"{row['reduction_pct']:>14.2f} "
            f"{row['link_r2']:>9.4f} "
            f"{row['bpr_gap_pct']:>12.3f} "
            f"{row['od_violation']:>13.7f} "
            f"{row['nonneg_minor_violation']:>13.7f} "
            f"{row['opt_cpu_time']:>14.2f} "
            f"{row['speedup_cpu_time']:>12.2f} "
            f"{row['speedup_ub']:>12.2f} "
            f"{str(row['converged']):>10}      "
            f"{row['convergence_reason']:<50}"
        )


################################################################################
# MAIN
################################################################################

if __name__ == "__main__":
    # choose mode from one of the following options: ["tap", "alm"]
    mode = "alm"

    rank = 50
    gamma = 1e-4

    # data_dir = '09_Chicago_Regional'
    # data_dir = "10_Chicago_Sketch/TAPLite/2x"
    data_dir = "chicago_sketch"
    # data_dir = "12_Philadelphia"
    # data_dir = "sioux_falls"

    link_file = f"data/{data_dir}/link.csv"
    link_perf_file = f"data/{data_dir}/link_performance_ue.csv"
    # link_perf_file = None  # No link performance file provided
    # route_file = f"data/{data_dir}/route_assignment.csv"
    route_file = f"data/{data_dir}/columns.csv"
    # demand_file = f"data/{data_dir}/demand.csv"
    demand_file = None  # No demand file provided
    output_dir = f"./test/{data_dir}/rank{rank}"

    if len(sys.argv) > 1:
        link_file = sys.argv[1]
    if len(sys.argv) > 2:
        route_file = sys.argv[2]
    if len(sys.argv) > 3:
        demand_file = sys.argv[3] if sys.argv[3].lower() != "none" else None
    if len(sys.argv) > 4:
        output_dir = sys.argv[4]
    if len(sys.argv) > 5:
        link_perf_file = sys.argv[5]

    print("\n" + "=" * 101)
    print(" COMPRESSED TAP OPTIMIZATION")
    print("=" * 101)

    # Load data
    B, x_ref, v_ref, capacity, t_0, od_info, links, routes = load_gmns_data_with_od(
        link_file, route_file, demand_file, link_perf_file
    )
    thresholds = setup_thresholds(x_ref, od_info)

    # Run threshold sensitivity analysis
    results_df = run_threshold_sensitivity_analysis(
        B, x_ref, v_ref, capacity, t_0, od_info, output_dir, rank, thresholds, gamma
    )

    print("\n" + "=" * 101)
    print(" COMPLETE")
    print("=" * 101 + "\n")
