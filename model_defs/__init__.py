from .td_gcn_contrastive import Model

def get_model(cfg):
    dataset = cfg["dataset"]
    if dataset == "hgr_shrec_2017":
        from graph.shrec17 import get_adjacency
    elif dataset == "ego_gesture":
        from graph.ego_gesture import get_adjacency
    elif dataset in ("ntu60", "ntu120"):
        from graph.ntu_rgb_d import get_adjacency
    elif dataset == "nw_ucla":
        from graph.ucla import get_adjacency
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return Model(
        adjacency=get_adjacency(),
        n_classes=cfg["n_classes"],
        n_joints=cfg["n_joints"],
        num_person=cfg["num_person"],
        seq_len=cfg["seq_len"],
        adaptive=cfg.get("adaptive", True),
        drop_out=cfg.get("drop_out", 0.0),
    )
