from .skeleton_data import *


def TrajData(
    raw_positions, time, graph_info={}, traj_info={}, original_positions=None
) -> SkeletonTrajData:

    assert raw_positions is not None, "raw_positions is None"
    if graph_info["data_type"] == "invariant":
        return InvariantTrajData(
            raw_positions,
            time,
            graph_info=graph_info,
            traj_info=traj_info,
            original_positions=original_positions,
        )

    elif graph_info["data_type"] == "node_features":
        return NodeFeaturesTrajData(
            raw_positions,
            time,
            graph_info=graph_info,
            traj_info=traj_info,
            original_positions=original_positions,
        )

    elif graph_info["data_type"] == "no_features":
        return SkeletonTrajData(
            raw_positions,
            time,
            graph_info=graph_info,
            traj_info=traj_info,
            original_positions=original_positions,
        )

    raise BaseException("Unknown data type : %s" % graph_info["data_type"])


class NodeFeaturesTrajData(SkeletonTrajData):
    def get_node_features(self, positions, graph_info) -> np.array:
        features = []
        N = positions.shape[0]
        reshape = lambda a: np.reshape(a, (N - 1, -1))
        norm = lambda x: x
        if graph_info["log_features"]:
            norm = lambda x: np.log(1e-5 + x)
        # Time
        norm_time = np.arange(N - 1) / (N - 1)
        features.append(reshape(norm_time))
        # get displacements
        dr = get_steps(positions)
        dr_vec = positions[1:] - positions[:-1]

        MD = reshape(np.cumsum(dr) / (1.0 + np.arange(N - 1)))
        MSD = reshape(np.power(np.cumsum(dr**2), 1.0 / 2) / (1.0 + np.arange(N - 1)))
        MQD = reshape(np.power(np.cumsum(dr**4), 1.0 / 4) / (1.0 + np.arange(N - 1)))
        MaxD = reshape(np.maximum.accumulate(dr))

        for scale_name in graph_info["scale_types"]:
            scale = traj_scale(positions, scale_name)
            assert scale > 0.0, "scale not positive %s" % positions

            features.append(np.clip(positions[:-1] / scale, -10, 10))
            features.append(np.clip(dr_vec / scale, -10, 10))

            features.append(norm(MD / scale))
            features.append(norm(MSD / scale))
            features.append(norm(MQD / scale))
            features.append(norm(MaxD / scale))
        features = np.concatenate(features, axis=1)
        assert np.sum(np.isnan(features)) == 0
        assert np.sum(np.isinf(features)) == 0
        return features


class InvariantTrajData(SkeletonTrajData):
    # Invariant by rotation

    def get_edge_features(self, X, positions, edge_index, graph_info):
        L = positions.shape[0]
        E = edge_index.shape[1]
        dt = (edge_index[1] - edge_index[0]) / L

        reshape = lambda a: np.reshape(a, (E, -1))
        DR = np.sqrt(
            np.sum((positions[edge_index[0]] - positions[edge_index[1]]) ** 2, axis=1)
        )
        features = [reshape(dt)]
        for scale_name in graph_info["scale_types"]:
            scale = traj_scale(positions, scale_name)
            features.append(reshape(DR / scale))
        features = np.concatenate(features, axis=1)
        assert np.sum(np.isnan(features)) == 0
        assert np.sum(np.isinf(features)) == 0
        return features

    def get_node_features(self, positions, graph_info) -> np.array:
        features = []
        N = positions.shape[0]
        reshape = lambda a: np.reshape(a, (N - 1, -1))
        norm = lambda x: x
        if graph_info["log_features"]:
            norm = lambda x: np.log(1e-5 + x)
        # Time
        norm_time = np.arange(N - 1) / (N - 1)
        features.append(reshape(norm_time))
        # get displacements
        dr = get_steps(positions)

        MD = reshape(np.cumsum(dr) / (1.0 + np.arange(N - 1)))
        MSD = reshape(np.power(np.cumsum(dr**2), 1.0 / 2) / (1.0 + np.arange(N - 1)))
        MQD = reshape(np.power(np.cumsum(dr**4), 1.0 / 4) / (1.0 + np.arange(N - 1)))
        MaxD = reshape(np.maximum.accumulate(dr))

        for scale_name in graph_info["scale_types"]:
            scale = traj_scale(positions, scale_name)
            assert scale > 0.0, "scale not positive %s" % positions
            features.append(norm(MD / scale))
            features.append(norm(MSD / scale))
            features.append(norm(MQD / scale))
            features.append(norm(MaxD / scale))
        features = np.concatenate(features, axis=1)
        assert np.sum(np.isnan(features)) == 0
        assert np.sum(np.isinf(features)) == 0
        return features
