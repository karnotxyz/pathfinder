use std::sync::Arc;

use p2p::libp2p::PeerId;
use p2p::PeerData;
use pathfinder_common::{BlockNumber, ClassHash, SignedBlockHeader};

#[derive(Debug, thiserror::Error)]
pub(super) enum SyncError {
    #[error(transparent)]
    Other(#[from] anyhow::Error),
    #[error("Header signature verification failed")]
    BadHeaderSignature(PeerId),
    #[error("Block hash verification failed")]
    BadBlockHash(PeerId),
    #[error("Discontinuity in header chain")]
    Discontinuity(PeerId),
    #[error("State diff signature verification failed")]
    BadStateDiffSignature(PeerId),
    #[error("State diff commitment mismatch")]
    StateDiffCommitmentMismatch(PeerId),
    #[error("Invalid class definition layout")]
    BadClassLayout(PeerId),
    #[error("Unexpected class definition")]
    UnexpectedClass(PeerId),
    #[error("Event commitment mismatch")]
    EventCommitmentMismatch(PeerId),
    #[error("Transaction commitment mismatch")]
    TransactionCommitmentMismatch(PeerId),
}

impl PartialEq for SyncError {
    fn eq(&self, other: &Self) -> bool {
        todo!();
    }
}

impl SyncError {
    /// Temporary cast to allow refactoring towards SyncError2.
    pub fn into_v2(self) -> PeerData<SyncError2> {
        todo!()
    }

    /// Temporary cast to allow refactoring towards SyncError2.
    pub fn from_v2(v2: PeerData<SyncError2>) -> Self {
        todo!()
    }
}

#[derive(Debug, thiserror::Error, Clone)]
pub(super) enum SyncError2 {
    #[error(transparent)]
    Other(#[from] Arc<anyhow::Error>),
    #[error("Header signature verification failed")]
    BadHeaderSignature,
    #[error("Block hash verification failed")]
    BadBlockHash,
    #[error("Discontinuity in header chain")]
    Discontinuity,
    #[error("State diff signature verification failed")]
    BadStateDiffSignature,
    #[error("State diff commitment mismatch")]
    StateDiffCommitmentMismatch,
    #[error("Invalid class definition layout")]
    BadClassLayout,
    #[error("Unexpected class definition")]
    UnexpectedClass,
    #[error("Event commitment mismatch")]
    EventCommitmentMismatch,
    #[error("Too many events")]
    TooManyEvents,
    #[error("Too few events")]
    TooFewEvents,
}

impl PartialEq for SyncError2 {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Other(x), Self::Other(y)) => x.to_string() == y.to_string(),
            (SyncError2::BadHeaderSignature, SyncError2::BadHeaderSignature) => true,
            (SyncError2::BadBlockHash, SyncError2::BadBlockHash) => true,
            (SyncError2::Discontinuity, SyncError2::Discontinuity) => true,
            (SyncError2::BadStateDiffSignature, SyncError2::BadStateDiffSignature) => true,
            (SyncError2::StateDiffCommitmentMismatch, SyncError2::StateDiffCommitmentMismatch) => {
                true
            }
            (SyncError2::BadClassLayout, SyncError2::BadClassLayout) => true,
            (SyncError2::UnexpectedClass, SyncError2::UnexpectedClass) => true,
            (SyncError2::EventCommitmentMismatch, SyncError2::EventCommitmentMismatch) => true,
            _ => false,
        }
    }
}

impl From<anyhow::Error> for SyncError2 {
    fn from(value: anyhow::Error) -> Self {
        Self::Other(Arc::new(value))
    }
}
