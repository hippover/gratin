default_graph_info = {
    "edges_per_point": 10,
    "scale_types": ["step_std"],
    "edge_method": "geom_causal",
    "data_type": "no_features",
}

default_ds_params = {
    "dim":2,
    "noise_range": (0.02, 0.05),
    "RW_types": ["fBM", "LW", "sBM", "OU", "CTRW"],
    "length_range": (7, 35),
    "logdiffusion_range":(-2.5,1.1),
    "N": int(1e5),
    "time_delta": 0.03,
}
