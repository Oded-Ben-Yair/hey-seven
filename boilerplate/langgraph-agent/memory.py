"""Conversation memory and checkpointing for the Casino Host Agent.

For local development, use ``langgraph.checkpoint.memory.MemorySaver`` --
this is the official, fully-compatible checkpointer shipped with LangGraph.

For production with Google Cloud Firestore, use the community package
``langgraph-checkpoint-firestore``::

    pip install langgraph-checkpoint-firestore
    from langgraph_checkpoint_firestore import FirestoreSaver

    db = firestore.Client(project="hey-seven-prod")
    saver = FirestoreSaver(db=db)
    graph = workflow.compile(checkpointer=saver)

The ``FirestoreCheckpointSaver`` class below is a reference implementation
that can be used if the community package is not available. It wraps all
sync Firestore I/O in ``asyncio.to_thread()`` for the async methods (C4/C5
fix), preventing event-loop blocking.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


def get_checkpointer(
    use_firestore: bool = False,
    firestore_db: Any = None,
) -> BaseCheckpointSaver:
    """Factory function to get the appropriate checkpointer.

    Args:
        use_firestore: If True, attempt to use the community Firestore
            checkpointer, falling back to the reference implementation.
        firestore_db: Optional pre-configured Firestore client.

    Returns:
        A BaseCheckpointSaver instance.
    """
    if not use_firestore:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    # Try the community package first (preferred for production)
    try:
        from langgraph_checkpoint_firestore import (  # type: ignore[import-untyped]
            FirestoreSaver,
        )

        if firestore_db:
            return FirestoreSaver(db=firestore_db)
        return FirestoreSaver()
    except ImportError:
        logger.info(
            "langgraph-checkpoint-firestore not installed. "
            "Using reference FirestoreCheckpointSaver."
        )
        return FirestoreCheckpointSaver(db=firestore_db)


class FirestoreCheckpointSaver(BaseCheckpointSaver):
    """Reference LangGraph checkpoint saver backed by Google Cloud Firestore.

    This is a reference implementation. For production, prefer the community
    package ``langgraph-checkpoint-firestore`` which handles edge cases,
    batching, and the 1MB Firestore document limit.

    All async methods use ``asyncio.to_thread()`` to wrap sync Firestore
    calls, preventing event-loop blocking (C4/C5 fix).

    Firestore collection structure::

        checkpoints/{thread_id}/versions/{checkpoint_id}
            - checkpoint: serialized checkpoint data
            - metadata: checkpoint metadata
            - parent_id: parent checkpoint ID (for branching)
            - writes: pending writes for this checkpoint
    """

    def __init__(
        self,
        db: Any = None,
        collection_name: str = "checkpoints",
    ) -> None:
        """Initialize the Firestore checkpoint saver.

        Args:
            db: A google.cloud.firestore.Client instance. If None, will
                attempt to create one using application default credentials.
            collection_name: Firestore collection name for checkpoints.
        """
        super().__init__()
        self.collection_name = collection_name
        self.serde = JsonPlusSerializer()

        if db is not None:
            self.db = db
        else:
            try:
                from google.cloud import firestore  # type: ignore[import-untyped]

                self.db = firestore.Client()
            except Exception:
                logger.warning(
                    "Firestore client not available. Use MemorySaver for "
                    "local development."
                )
                self.db = None

    def _get_thread_ref(self, thread_id: str) -> Any:
        """Get a Firestore document reference for a thread."""
        return self.db.collection(self.collection_name).document(thread_id)

    def _get_checkpoint_ref(self, thread_id: str, checkpoint_id: str) -> Any:
        """Get a Firestore document reference for a specific checkpoint."""
        return (
            self.db.collection(self.collection_name)
            .document(thread_id)
            .collection("versions")
            .document(checkpoint_id)
        )

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve a checkpoint tuple by config.

        Args:
            config: Runnable config containing thread_id and optionally
                checkpoint_id in the configurable dict.

        Returns:
            A CheckpointTuple if found, None otherwise.
        """
        if self.db is None:
            return None

        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_id = configurable.get("checkpoint_id")

        if not thread_id:
            return None

        try:
            if checkpoint_id:
                doc = self._get_checkpoint_ref(thread_id, checkpoint_id).get()
            else:
                query = (
                    self._get_thread_ref(thread_id)
                    .collection("versions")
                    .order_by("ts", direction="DESCENDING")
                    .limit(1)
                )
                docs = list(query.stream())
                if not docs:
                    return None
                doc = docs[0]

            if not doc.exists:
                return None

            data = doc.to_dict()
            checkpoint = self.serde.loads(data["checkpoint"])
            metadata = self.serde.loads(data["metadata"])

            parent_config = None
            if data.get("parent_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": data["parent_id"],
                    }
                }

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint["id"],
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
            )
        except Exception:
            logger.exception(
                "Failed to get checkpoint for thread %s", thread_id
            )
            return None

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint.

        Args:
            config: Runnable config with thread_id.
            checkpoint: The checkpoint data to store.
            metadata: Checkpoint metadata (step, source, writes).
            new_versions: Channel version information.

        Returns:
            Updated config with the new checkpoint_id.
        """
        if self.db is None:
            raise RuntimeError(
                "Firestore client not initialized. Cannot save checkpoint."
            )

        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "default")
        checkpoint_id = checkpoint["id"]
        parent_id = configurable.get("checkpoint_id")

        doc_ref = self._get_checkpoint_ref(thread_id, checkpoint_id)
        doc_ref.set(
            {
                "checkpoint": self.serde.dumps(checkpoint),
                "metadata": self.serde.dumps(metadata),
                "parent_id": parent_id,
                "ts": checkpoint.get("ts", ""),
            }
        )

        # Also update the thread-level doc with latest checkpoint pointer
        self._get_thread_ref(thread_id).set(
            {"latest_checkpoint_id": checkpoint_id, "ts": checkpoint.get("ts", "")},
            merge=True,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending writes for a checkpoint.

        Args:
            config: Runnable config with thread_id and checkpoint_id.
            writes: Sequence of (channel, value) tuples.
            task_id: The task that generated these writes.
            task_path: Optional path for nested graph writes.
        """
        if self.db is None:
            return

        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "default")
        checkpoint_id = configurable.get("checkpoint_id")

        if not checkpoint_id:
            return

        doc_ref = self._get_checkpoint_ref(thread_id, checkpoint_id)
        serialized_writes = [
            {
                "channel": channel,
                "value": self.serde.dumps(value),
                "task_id": task_id,
                "task_path": task_path,
            }
            for channel, value in writes
        ]

        doc_ref.set({"pending_writes": serialized_writes}, merge=True)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread.

        Args:
            config: Runnable config with thread_id.
            filter: Optional metadata filter.
            before: Only return checkpoints before this config.
            limit: Maximum number of checkpoints to return.

        Yields:
            CheckpointTuple for each matching checkpoint, newest first.
        """
        if self.db is None:
            return

        if config is None:
            return

        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")

        if not thread_id:
            return

        try:
            query = (
                self._get_thread_ref(thread_id)
                .collection("versions")
                .order_by("ts", direction="DESCENDING")
            )

            if before:
                before_id = before.get("configurable", {}).get("checkpoint_id")
                if before_id:
                    before_doc = self._get_checkpoint_ref(
                        thread_id, before_id
                    ).get()
                    if before_doc.exists:
                        before_ts = before_doc.to_dict().get("ts", "")
                        query = query.where("ts", "<", before_ts)

            if limit:
                query = query.limit(limit)

            for doc in query.stream():
                data = doc.to_dict()
                checkpoint = self.serde.loads(data["checkpoint"])
                metadata = self.serde.loads(data["metadata"])

                parent_config = None
                if data.get("parent_id"):
                    parent_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": data["parent_id"],
                        }
                    }

                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": checkpoint["id"],
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                )
        except Exception:
            logger.exception(
                "Failed to list checkpoints for thread %s", thread_id
            )

    # ------------------------------------------------------------------
    # Async interface (C4/C5 fix: use asyncio.to_thread to avoid blocking)
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async version of get_tuple. Runs sync I/O in a thread pool."""
        return await asyncio.to_thread(self.get_tuple, config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version of put. Runs sync I/O in a thread pool."""
        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version of put_writes. Runs sync I/O in a thread pool.

        This method was missing in the original implementation (C5 fix).
        Required by LangGraph 1.0 GA for async graph execution.
        """
        await asyncio.to_thread(
            self.put_writes, config, writes, task_id, task_path
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of list.

        Collects sync results in a thread pool, then yields them.
        """
        items = await asyncio.to_thread(
            lambda: list(
                self.list(config, filter=filter, before=before, limit=limit)
            )
        )
        for item in items:
            yield item


# ---------------------------------------------------------------------------
# Player Context Manager
# ---------------------------------------------------------------------------


class PlayerContextManager:
    """Manages player context persistence across conversation sessions.

    Stores and retrieves player profile data in Firestore so that returning
    players have their context immediately available without re-querying.

    Collection structure::

        player_contexts/{player_id}
            - profile: cached player profile
            - last_interaction: ISO timestamp
            - conversation_history_ids: list of thread_ids
            - preferences_overrides: host-set preference overrides
    """

    def __init__(
        self,
        db: Any = None,
        collection_name: str = "player_contexts",
    ) -> None:
        """Initialize the player context manager.

        Args:
            db: Firestore client instance.
            collection_name: Collection name for player contexts.
        """
        self.collection_name = collection_name
        if db is not None:
            self.db = db
        else:
            try:
                from google.cloud import firestore  # type: ignore[import-untyped]

                self.db = firestore.Client()
            except Exception:
                logger.warning(
                    "Firestore client not available for player context."
                )
                self.db = None

    def get_context(self, player_id: str) -> dict[str, Any] | None:
        """Retrieve cached player context.

        Args:
            player_id: The player tracking number.

        Returns:
            Cached player context dict, or None if not found.
        """
        if self.db is None:
            return None

        try:
            doc = (
                self.db.collection(self.collection_name)
                .document(player_id)
                .get()
            )
            if doc.exists:
                return doc.to_dict()
        except Exception:
            logger.exception(
                "Failed to get context for player %s", player_id
            )
        return None

    def save_context(
        self,
        player_id: str,
        profile: dict[str, Any],
        thread_id: str | None = None,
    ) -> None:
        """Save or update player context.

        Args:
            player_id: The player tracking number.
            profile: Player profile data to cache.
            thread_id: Current conversation thread ID to add to history.
        """
        if self.db is None:
            return

        from datetime import datetime, timezone

        from google.cloud import firestore  # type: ignore[import-untyped]

        doc_ref = self.db.collection(self.collection_name).document(player_id)
        update_data: dict[str, Any] = {
            "profile": profile,
            "last_interaction": datetime.now(tz=timezone.utc).isoformat(),
        }

        if thread_id:
            update_data["conversation_history_ids"] = (
                firestore.ArrayUnion([thread_id])
            )

        doc_ref.set(update_data, merge=True)
