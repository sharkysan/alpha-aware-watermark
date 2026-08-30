import pytest

from watermark_remover.chunks import iter_scene_chunks
from watermark_remover.scenes import Scene


def test_chunks_are_bounded_and_do_not_cross_scene_boundaries():
    scene = Scene(10, 35)
    chunks = iter_scene_chunks(scene, chunk_size=8, temporal_radius=2)

    assert [(c.process_start, c.process_end) for c in chunks] == [
        (10, 18),
        (18, 26),
        (26, 34),
        (34, 35),
    ]
    assert all(c.load_start >= scene.start for c in chunks)
    assert all(c.load_end <= scene.end for c in chunks)
    assert all(c.load_length <= c.process_length + 4 for c in chunks)


@pytest.mark.parametrize("chunk_size,radius", [(0, 2), (4, -1)])
def test_invalid_chunk_arguments_raise(chunk_size, radius):
    with pytest.raises(ValueError):
        iter_scene_chunks(Scene(0, 10), chunk_size, radius)
