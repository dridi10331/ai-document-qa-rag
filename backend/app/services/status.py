from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import AsyncIterator

from app.schemas.status import StatusEvent


class StatusHub:
    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue[StatusEvent]]] = defaultdict(list)
        self._last_event: dict[str, StatusEvent] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, document_id: str) -> AsyncIterator[StatusEvent]:
        queue: asyncio.Queue[StatusEvent] = asyncio.Queue()
        async with self._lock:
            self._queues[document_id].append(queue)
            last_event = self._last_event.get(document_id)
            if last_event:
                queue.put_nowait(last_event)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            async with self._lock:
                if queue in self._queues.get(document_id, []):
                    self._queues[document_id].remove(queue)

    async def publish(self, event: StatusEvent) -> None:
        async with self._lock:
            self._last_event[event.document_id] = event
            queues = list(self._queues.get(event.document_id, []))
        for queue in queues:
            queue.put_nowait(event)


status_hub = StatusHub()
