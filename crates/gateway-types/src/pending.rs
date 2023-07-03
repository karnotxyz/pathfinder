use crate::reply::PendingBlock;
use pathfinder_common::{BlockHash, BlockTimestamp, StateUpdate};
use std::sync::Arc;
use tokio::sync::RwLock;

struct PendingInner {
    pub block: Arc<PendingBlock>,
    pub state_update: Arc<StateUpdate>,
}

#[derive(Default, Clone)]
pub struct PendingData {
    inner: Arc<RwLock<Option<PendingInner>>>,
}

impl PendingData {
    pub async fn set(&self, block: Arc<PendingBlock>, state_update: Arc<StateUpdate>) {
        *self.inner.write().await = Some(PendingInner {
            block,
            state_update,
        });
    }

    pub async fn clear(&self) {
        *self.inner.write().await = None;
    }

    pub async fn block(&self) -> Option<Arc<PendingBlock>> {
        self.inner
            .read()
            .await
            .as_ref()
            .map(|inner| inner.block.clone())
    }

    pub async fn state_update(&self) -> Option<Arc<StateUpdate>> {
        self.inner
            .read()
            .await
            .as_ref()
            .map(|inner| inner.state_update.clone())
    }

    pub async fn state_update_on_parent_block(
        &self,
    ) -> Option<(BlockHash, BlockTimestamp, Arc<StateUpdate>)> {
        let g = self.inner.read().await;
        let inner = g.as_ref()?;

        Some((
            inner.block.parent_hash,
            inner.block.timestamp,
            inner.state_update.clone(),
        ))
    }
}
