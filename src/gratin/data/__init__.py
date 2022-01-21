default_graph_info = {
    "edges_per_point": 10,
    "clip_trajs": True,
    "convex_hull": False,
    "scale_types": ["step_std"],
    "normalize_time": True,
    "features_on_edges": False,
    "position_features": True,
    "log_features": True,
    "edge_method": "geom_causal",
    "data_type": "no_features",
}

default_ds_params = {
    "noise_range": (0.0, 0.0),
    "RW_types": ["empty"],
    "force_range": (0.0, 0.0),
    "length_range": (10, 50),
    "time_delta": 1.0,
}
