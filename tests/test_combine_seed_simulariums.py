import json
from pathlib import Path

from cellpack_analysis.lib.simularium_utils import combine_seed_simulariums

TYPE_MAPPING = {
    "0": {"name": "membrane", "geometry": {"displayType": "OBJ", "url": "mem_mesh_1.obj"}},
    "1": {"name": "nucleus", "geometry": {"displayType": "OBJ", "url": "nuc_mesh_1.obj"}},
    "2": {"name": "membrane_interior_peroxisome", "geometry": {"displayType": "SPHERE"}},
}


def _agent(agent_id: int, type_id: int, x: float) -> list[float]:
    return [1000.0, agent_id, type_id, x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0]


def _write_seed_file(path: Path, xs: list[float]) -> None:
    data = _agent(1, 0, 0.0) + _agent(2, 1, 0.0)
    for i, x in enumerate(xs):
        data += _agent(3 + i, 2, x)
    sim_data = {
        "trajectoryInfo": {"version": 3, "totalSteps": 1, "typeMapping": TYPE_MAPPING},
        "spatialData": {
            "bundleStart": 0,
            "bundleSize": 1,
            "bundleData": [{"frameNumber": 0, "time": 0.0, "data": data}],
        },
        "plotData": {"version": 1, "data": []},
    }
    path.write_text(json.dumps(sim_data))


def test_combine_keeps_every_punctum_and_compartment_in_every_frame(tmp_path: Path) -> None:
    seeds = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]
    seed_paths = []
    for seed, xs in enumerate(seeds):
        seed_paths.append(tmp_path / f"seed_{seed}.simularium")
        _write_seed_file(seed_paths[-1], xs)

    output_path = tmp_path / "all.simularium"
    combine_seed_simulariums(seed_paths, output_path)

    sim_data = json.loads(output_path.read_text())
    frames = sim_data["spatialData"]["bundleData"]
    assert sim_data["trajectoryInfo"]["totalSteps"] == len(frames) == 2
    punctum_ids = []
    for frame_index, (frame, xs) in enumerate(zip(frames, seeds, strict=True)):
        records = [frame["data"][i : i + 11] for i in range(0, len(frame["data"]), 11)]
        assert frame["frameNumber"] == frame_index
        assert [record[2] for record in records] == [0, 1, 2, 2, 2]
        assert [record[3] for record in records if record[2] == 2] == xs
        punctum_ids += [record[1] for record in records if record[2] == 2]
    assert len(set(punctum_ids)) == len(punctum_ids)
