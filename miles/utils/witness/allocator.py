from miles.utils.pydantic_utils import FrozenStrictBaseModel


class WitnessInfo(FrozenStrictBaseModel):
    witness_ids: list[int]
    stale_ids: list[int]
    # Allocator counter AFTER this allocation. Persisted with the model (see
    # miles.utils.witness.module) so a resumed run continues allocating where the saved
    # run stopped instead of re-issuing ids whose rows still hold the saved run's state.
    counter: int


class WitnessIdAllocator:
    def __init__(self, *, buffer_size: int) -> None:
        self._buffer_size = buffer_size
        self._counter: int = 0

    def resume(self, counter: int) -> None:
        """Continue from a counter persisted by a previous run (no-op if not ahead)."""
        self._counter = max(self._counter, counter)

    def allocate(self, num_ids: int) -> WitnessInfo:
        assert num_ids <= self._buffer_size, (
            f"num_ids ({num_ids}) exceeds buffer_size ({self._buffer_size}). " f"Increase --witness-buffer-size."
        )
        ids = [(self._counter + i) % self._buffer_size for i in range(num_ids)]
        stale_ids = _compute_stale_ids(
            keep_count=int(self._buffer_size * 0.7),
            counter=self._counter + num_ids,
            buffer_size=self._buffer_size,
        )
        self._counter += num_ids
        return WitnessInfo(witness_ids=ids, stale_ids=stale_ids, counter=self._counter)


def _compute_stale_ids(*, keep_count: int, counter: int, buffer_size: int) -> list[int]:
    if counter == 0:
        return []
    num_stale = buffer_size - min(keep_count, counter, buffer_size)
    if num_stale == 0:
        return []

    head = counter % buffer_size
    return [(head + i) % buffer_size for i in range(num_stale)]
