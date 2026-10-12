//! Offline Pathfinder archive-to-pruned SQLite converter.
//!
//! The source database is never opened for writing. The converter creates a
//! new pruning-enabled database, copies the retained blockchain window and a
//! complete state baseline, copies only trie nodes reachable at the boundary,
//! and then replays trie updates to the source head. Replaying the retained
//! window creates the removal journals required for safe future trie pruning.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context};
use clap::{Parser, ValueEnum};
use pathfinder_common::{BlockId, BlockNumber, StateCommitment};
use pathfinder_merkle_tree::starknet_state::update_starknet_state;
use pathfinder_storage::pruning::BlockchainHistoryMode;
use pathfinder_storage::{JournalMode, StorageBuilder, StoredNode, TriePruneMode};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Plan,
    Convert,
}

#[derive(Debug, Parser)]
#[command(about = "Convert an archive Pathfinder SQLite DB into a new pruned DB")]
struct Args {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    target: PathBuf,
    #[arg(long, default_value_t = 300_000)]
    retain_blocks: u64,
    #[arg(long, value_enum, default_value_t = Mode::Plan)]
    mode: Mode,
    #[arg(long, default_value_t = false)]
    stop_after_bootstrap: bool,
    #[arg(long, default_value_t = true)]
    verify_hashes: bool,
    #[arg(long, default_value_t = 100_000)]
    commit_every_nodes: u64,
    #[arg(long, default_value_t = 1_000)]
    report_every_blocks: u64,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    ensure!(args.retain_blocks > 0, "--retain-blocks must be positive");
    ensure!(
        args.commit_every_nodes > 0,
        "--commit-every-nodes must be positive"
    );
    ensure!(
        args.report_every_blocks > 0,
        "--report-every-blocks must be positive"
    );
    if let Some(parent) = args.target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Creating target directory {}", parent.display()))?;
    }
    ensure_distinct_paths(&args.source, &args.target)?;

    let source = open_read_only(&args.source)?;
    let source_version: u32 = source.pragma_query_value(None, "user_version", |r| r.get(0))?;
    let head: u64 = source
        .query_row("SELECT MAX(number) FROM block_headers", [], |r| r.get(0))
        .context("Reading source head")?;
    let boundary = head.saturating_sub(args.retain_blocks);
    let missing_definitions: u64 = source.query_row(
        "SELECT COUNT(*) FROM class_definitions WHERE definition IS NULL",
        [],
        |r| r.get(0),
    )?;

    tracing::info!(
        source = %args.source.display(),
        target = %args.target.display(),
        source_version,
        head,
        boundary,
        retain_blocks = args.retain_blocks,
        missing_definitions,
        "conversion plan"
    );

    for table in ["trie_contracts", "trie_class", "trie_storage"] {
        if matches!(args.mode, Mode::Plan) {
            let (max_idx, rows): (Option<u64>, u64) = source.query_row(
                &format!("SELECT MAX(idx), COUNT(*) FROM {table}"),
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            tracing::info!(table, ?max_idx, rows, "source trie inventory");
        } else {
            let max_idx: Option<u64> = source.query_row(
                &format!("SELECT MAX(idx) FROM {table}"),
                [],
                |r| r.get(0),
            )?;
            tracing::info!(table, ?max_idx, "source trie maximum index");
        }
    }

    if matches!(args.mode, Mode::Plan) {
        return Ok(());
    }

    ensure!(
        missing_definitions == 0,
        "source has {missing_definitions} class definitions with NULL bodies; repair it first"
    );

    let bootstrap_state = target_state(&args.target)?;
    match bootstrap_state.as_deref() {
        None => bootstrap(&args, source_version, head, boundary)?,
        Some("bootstrapped") | Some("replaying") | Some("complete") => {
            tracing::info!(state = ?bootstrap_state, "resuming existing target");
        }
        Some(other) => bail!("target has unknown converter state {other:?}"),
    }

    if args.stop_after_bootstrap {
        tracing::info!("stopping after bootstrap as requested");
        return Ok(());
    }

    replay(&args, head, boundary)?;
    Ok(())
}

fn ensure_distinct_paths(source: &Path, target: &Path) -> anyhow::Result<()> {
    let source = source
        .canonicalize()
        .with_context(|| format!("Canonicalizing source {}", source.display()))?;
    let target = if target.exists() {
        target
            .canonicalize()
            .with_context(|| format!("Canonicalizing target {}", target.display()))?
    } else {
        let parent = target
            .parent()
            .context("Target must have a parent directory")?;
        parent
            .canonicalize()
            .with_context(|| format!("Canonicalizing target parent {}", parent.display()))?
            .join(target.file_name().context("Target must have a file name")?)
    };
    ensure!(source != target, "source and target paths are identical");
    Ok(())
}

fn open_read_only(path: &Path) -> anyhow::Result<Connection> {
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_URI,
    )
    .with_context(|| format!("Opening {} read-only", path.display()))
}

fn target_state(path: &Path) -> anyhow::Result<Option<String>> {
    if !path.exists() || path.metadata()?.len() == 0 {
        return Ok(None);
    }
    let db = Connection::open(path)?;
    let exists: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='archive_conversion')",
        [],
        |r| r.get(0),
    )?;
    if !exists {
        bail!(
            "target {} exists but is not a resumable conversion",
            path.display()
        );
    }
    db.query_row(
        "SELECT value FROM archive_conversion WHERE key='state'",
        [],
        |r| r.get(0),
    )
    .optional()
    .map_err(Into::into)
}

fn bootstrap(args: &Args, source_version: u32, head: u64, boundary: u64) -> anyhow::Result<()> {
    if args.target.exists() {
        ensure!(
            args.target.metadata()?.len() == 0,
            "refusing to overwrite non-empty target {}",
            args.target.display()
        );
        std::fs::remove_file(&args.target).context("Removing empty target placeholder")?;
    }

    tracing::info!("creating pruning-enabled target schema");
    StorageBuilder::file(args.target.clone())
        .journal_mode(JournalMode::WAL)
        .trie_prune_mode(Some(TriePruneMode::Prune {
            num_blocks_kept: args.retain_blocks,
        }))
        .blockchain_history_mode(Some(BlockchainHistoryMode::Prune {
            num_blocks_kept: args.retain_blocks,
        }))
        .migrate()
        .context("Creating target schema")?;

    let mut target = Connection::open_with_flags(
        &args.target,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_URI,
    )?;
    target.busy_timeout(Duration::from_secs(60))?;
    target.execute_batch(
        "PRAGMA foreign_keys=OFF;
         PRAGMA synchronous=NORMAL;
         PRAGMA wal_autocheckpoint=10000;
         CREATE TABLE archive_conversion(key TEXT PRIMARY KEY, value TEXT NOT NULL);
         INSERT INTO archive_conversion VALUES('state', 'bootstrapping');",
    )?;
    for (key, value) in [
        ("source_user_version", source_version.to_string()),
        ("source_head", head.to_string()),
        ("boundary", boundary.to_string()),
        ("retain_blocks", args.retain_blocks.to_string()),
    ] {
        target.execute(
            "INSERT INTO archive_conversion(key, value) VALUES(?, ?)",
            params![key, value],
        )?;
    }

    attach_source(&target, &args.source)?;
    let attached_version: u32 =
        target.pragma_query_value(Some("source"), "user_version", |r| r.get(0))?;
    ensure!(
        attached_version == source_version,
        "source schema changed while bootstrapping"
    );

    for table in [
        "class_definitions",
        "casm_definitions",
        "casm_class_hashes",
        "casm_class_hashes_v2",
        "contract_addresses",
        "storage_addresses",
    ] {
        copy_all(&mut target, table)?;
    }

    copy_filtered(&mut target, "block_headers", "number >= ?", boundary)?;
    copy_filtered(
        &mut target,
        "block_signatures",
        "block_number >= ?",
        boundary,
    )?;
    copy_filtered(&mut target, "transactions", "block_number >= ?", boundary)?;
    copy_filtered(
        &mut target,
        "transaction_hashes",
        "block_number >= ?",
        boundary,
    )?;
    copy_filtered(
        &mut target,
        "redeclared_classes",
        "block_number >= ?",
        boundary,
    )?;
    copy_filtered(&mut target, "event_filters", "to_block >= ?", boundary)?;
    copy_filtered(
        &mut target,
        "class_commitment_leaves",
        "block_number <= ?",
        boundary,
    )?;

    copy_state_window(&mut target, boundary)?;
    copy_l1_state(&mut target, boundary)?;
    copy_singleton_table(&mut target, "refs")?;
    copy_singleton_table(&mut target, "running_event_filter")?;
    copy_boundary_roots(&mut target, boundary)?;
    target.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    target.execute_batch("DETACH DATABASE source;")?;
    drop(target);

    let source = open_read_only(&args.source)?;
    let target = Connection::open(&args.target)?;
    copy_reachable_trie(
        &source,
        &target,
        "trie_contracts",
        root_indices(
            &source,
            "SELECT root_index FROM storage_roots WHERE block_number <= ? ORDER BY block_number DESC LIMIT 1",
            boundary,
        )?,
        args.commit_every_nodes,
    )?;
    copy_reachable_trie(
        &source,
        &target,
        "trie_class",
        root_indices(
            &source,
            "SELECT root_index FROM class_roots WHERE block_number <= ? ORDER BY block_number DESC LIMIT 1",
            boundary,
        )?,
        args.commit_every_nodes,
    )?;
    copy_reachable_trie(
        &source,
        &target,
        "trie_storage",
        root_indices(
            &source,
            "SELECT cr.root_index FROM contract_roots cr JOIN (SELECT contract_address, MAX(block_number) AS block_number FROM contract_roots WHERE block_number <= ? GROUP BY contract_address) latest ON latest.contract_address=cr.contract_address AND latest.block_number=cr.block_number WHERE cr.root_index IS NOT NULL",
            boundary,
        )?,
        args.commit_every_nodes,
    )?;

    target.execute(
        "UPDATE archive_conversion SET value='bootstrapped' WHERE key='state'",
        [],
    )?;
    target.execute(
        "INSERT OR REPLACE INTO archive_conversion VALUES('last_replayed', ?)",
        [boundary.to_string()],
    )?;
    target.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    tracing::info!("bootstrap complete");
    Ok(())
}

fn attach_source(target: &Connection, source: &Path) -> anyhow::Result<()> {
    let uri = format!("file:{}?mode=ro", source.display());
    target
        .execute("ATTACH DATABASE ? AS source", [uri])
        .context("Attaching source database read-only")?;
    Ok(())
}

fn quoted_columns(db: &Connection, schema: &str, table: &str) -> anyhow::Result<String> {
    let mut stmt = db.prepare(&format!("PRAGMA {schema}.table_info('{table}')"))?;
    let columns = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    ensure!(!columns.is_empty(), "table {schema}.{table} has no columns");
    Ok(columns
        .iter()
        .map(|x| format!("\"{}\"", x.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(","))
}

fn copy_all(db: &mut Connection, table: &str) -> anyhow::Result<()> {
    let columns = quoted_columns(db, "main", table)?;
    let started = Instant::now();
    let tx = db.transaction()?;
    let rows = tx.execute(
        &format!("INSERT INTO main.{table} ({columns}) SELECT {columns} FROM source.{table}"),
        [],
    )?;
    tx.commit()?;
    tracing::info!(
        table,
        rows,
        elapsed_s = started.elapsed().as_secs_f64(),
        "copied table"
    );
    Ok(())
}

fn copy_singleton_table(db: &mut Connection, table: &str) -> anyhow::Result<()> {
    let columns = quoted_columns(db, "main", table)?;
    let tx = db.transaction()?;
    tx.execute(&format!("DELETE FROM main.{table}"), [])?;
    tx.execute(
        &format!("INSERT INTO main.{table} ({columns}) SELECT {columns} FROM source.{table}"),
        [],
    )?;
    tx.commit()?;
    Ok(())
}

fn copy_filtered(
    db: &mut Connection,
    table: &str,
    predicate: &str,
    boundary: u64,
) -> anyhow::Result<()> {
    let columns = quoted_columns(db, "main", table)?;
    let started = Instant::now();
    let tx = db.transaction()?;
    let rows = tx.execute(
        &format!(
            "INSERT INTO main.{table} ({columns}) SELECT {columns} FROM source.{table} WHERE {predicate}"
        ),
        [boundary],
    )?;
    tx.commit()?;
    tracing::info!(
        table,
        rows,
        elapsed_s = started.elapsed().as_secs_f64(),
        "copied retained window"
    );
    Ok(())
}

fn copy_state_window(db: &mut Connection, boundary: u64) -> anyhow::Result<()> {
    let specs = [
        ("contract_updates", "contract_address"),
        ("nonce_updates", "contract_address_id"),
        ("storage_updates", "contract_address_id, storage_address_id"),
    ];
    for (table, keys) in specs {
        let columns = quoted_columns(db, "main", table)?;
        let old_columns = columns
            .split(',')
            .map(|column| format!("old.{column}"))
            .collect::<Vec<_>>()
            .join(",");
        let started = Instant::now();
        let join_keys = keys
            .split(", ")
            .map(|key| format!("latest.{key}=old.{key}"))
            .collect::<Vec<_>>()
            .join(" AND ");
        let tx = db.transaction()?;
        let tail_rows = tx.execute(
            &format!(
                "INSERT INTO main.{table} ({columns})
                 SELECT {columns} FROM source.{table} WHERE block_number > ?1"
            ),
            [boundary],
        )?;
        let baseline_rows = tx.execute(
            &format!(
                "INSERT INTO main.{table} ({columns})
                 SELECT {old_columns}
                 FROM source.{table} AS old
                 JOIN (
                     SELECT {keys}, MAX(block_number) AS block_number
                     FROM source.{table}
                     WHERE block_number <= ?1
                     GROUP BY {keys}
                 ) AS latest
                 ON {join_keys} AND latest.block_number=old.block_number"
            ),
            [boundary],
        )?;
        tx.commit()?;
        tracing::info!(
            table,
            tail_rows,
            baseline_rows,
            elapsed_s = started.elapsed().as_secs_f64(),
            "copied state window and baseline"
        );
    }
    Ok(())
}

fn copy_l1_state(db: &mut Connection, boundary: u64) -> anyhow::Result<()> {
    let columns = quoted_columns(db, "main", "l1_state")?;
    let tx = db.transaction()?;
    tx.execute(
        &format!(
            "INSERT INTO main.l1_state ({columns})
             SELECT {columns} FROM source.l1_state WHERE starknet_block_number >= ?1
             UNION ALL
             SELECT {columns} FROM source.l1_state WHERE starknet_block_number = (
                 SELECT MAX(starknet_block_number) FROM source.l1_state WHERE starknet_block_number < ?1
             )"
        ),
        [boundary],
    )?;
    tx.commit()?;
    Ok(())
}

fn copy_boundary_roots(db: &mut Connection, boundary: u64) -> anyhow::Result<()> {
    let tx = db.transaction()?;
    tx.execute(
        "INSERT INTO storage_roots SELECT * FROM source.storage_roots WHERE block_number=(SELECT MAX(block_number) FROM source.storage_roots WHERE block_number <= ?)",
        [boundary],
    )?;
    tx.execute(
        "INSERT INTO class_roots SELECT * FROM source.class_roots WHERE block_number=(SELECT MAX(block_number) FROM source.class_roots WHERE block_number <= ?)",
        [boundary],
    )?;
    tx.execute(
        "INSERT INTO contract_roots SELECT cr.* FROM source.contract_roots cr JOIN (SELECT contract_address, MAX(block_number) AS block_number FROM source.contract_roots WHERE block_number <= ? GROUP BY contract_address) latest ON latest.contract_address=cr.contract_address AND latest.block_number=cr.block_number",
        [boundary],
    )?;
    tx.execute(
        "INSERT INTO contract_state_hashes SELECT cs.* FROM source.contract_state_hashes cs JOIN (SELECT contract_address, MAX(block_number) AS block_number FROM source.contract_state_hashes WHERE block_number <= ? GROUP BY contract_address) latest ON latest.contract_address=cs.contract_address AND latest.block_number=cs.block_number",
        [boundary],
    )?;
    tx.commit()?;
    Ok(())
}

fn root_indices(db: &Connection, query: &str, boundary: u64) -> anyhow::Result<Vec<u64>> {
    let mut stmt = db.prepare(query)?;
    let rows = stmt
        .query_map([boundary], |r| r.get::<_, Option<u64>>(0))?
        .filter_map(|x| x.transpose())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn copy_reachable_trie(
    source: &Connection,
    target: &Connection,
    table: &str,
    roots: Vec<u64>,
    commit_every: u64,
) -> anyhow::Result<()> {
    let max_idx: Option<u64> =
        source.query_row(&format!("SELECT MAX(idx) FROM {table}"), [], |r| r.get(0))?;
    let Some(max_idx) = max_idx else {
        return Ok(());
    };
    ensure!(
        max_idx < usize::MAX as u64,
        "{table} index space is too large for this platform"
    );
    let mut reachable = vec![0u8; (max_idx as usize / 8) + 1];
    for root in &roots {
        ensure!(
            *root <= max_idx,
            "{table} root {root} exceeds max index {max_idx}"
        );
        set_bit(&mut reachable, *root);
    }

    let started = Instant::now();
    let mut scanned = 0u64;
    let mut copied = 0u64;
    let mut source_stmt = source.prepare(&format!(
        "SELECT idx, hash, data FROM {table} ORDER BY idx DESC"
    ))?;
    let mut rows = source_stmt.query([])?;

    target.execute_batch("BEGIN IMMEDIATE")?;
    let mut insert = target.prepare(&format!(
        "INSERT INTO {table}(idx, hash, data) VALUES(?1, ?2, ?3)"
    ))?;
    while let Some(row) = rows.next()? {
        let idx: u64 = row.get(0)?;
        scanned += 1;
        if !get_bit(&reachable, idx) {
            continue;
        }
        let hash: Vec<u8> = row.get(1)?;
        let data: Vec<u8> = row.get(2)?;
        for child in StoredNode::decode(&data)
            .with_context(|| format!("Decoding {table} node {idx}"))?
            .children()
            .into_iter()
            .flatten()
        {
            ensure!(
                child.0 < idx,
                "{table} violates child-before-parent invariant at {idx} -> {}",
                child.0
            );
            set_bit(&mut reachable, child.0);
        }
        insert.execute(params![idx, hash, data])?;
        copied += 1;
        if copied.is_multiple_of(commit_every) {
            drop(insert);
            target.execute_batch("COMMIT; PRAGMA wal_checkpoint(PASSIVE); BEGIN IMMEDIATE")?;
            insert = target.prepare(&format!(
                "INSERT INTO {table}(idx, hash, data) VALUES(?1, ?2, ?3)"
            ))?;
            tracing::info!(
                table,
                scanned,
                copied,
                elapsed_s = started.elapsed().as_secs_f64(),
                "copying reachable trie"
            );
        }
    }
    drop(insert);
    target.execute_batch("COMMIT; PRAGMA wal_checkpoint(TRUNCATE)")?;
    tracing::info!(
        table,
        roots = roots.len(),
        scanned,
        copied,
        elapsed_s = started.elapsed().as_secs_f64(),
        "reachable trie copied"
    );
    Ok(())
}

fn set_bit(bits: &mut [u8], idx: u64) {
    let idx = idx as usize;
    bits[idx / 8] |= 1 << (idx % 8);
}

fn get_bit(bits: &[u8], idx: u64) -> bool {
    let idx = idx as usize;
    bits[idx / 8] & (1 << (idx % 8)) != 0
}

fn replay(args: &Args, head: u64, boundary: u64) -> anyhow::Result<()> {
    let target_meta = Connection::open(&args.target)?;
    let recorded_head: u64 = metadata_u64(&target_meta, "source_head")?;
    let recorded_boundary: u64 = metadata_u64(&target_meta, "boundary")?;
    let recorded_retention: u64 = metadata_u64(&target_meta, "retain_blocks")?;
    ensure!(recorded_head == head, "source head changed since bootstrap");
    ensure!(
        recorded_boundary == boundary,
        "boundary changed since bootstrap"
    );
    ensure!(
        recorded_retention == args.retain_blocks,
        "retention changed since bootstrap"
    );
    let last_replayed = metadata_u64(&target_meta, "last_replayed")?;
    if last_replayed >= head {
        tracing::info!(last_replayed, "conversion already complete");
        return Ok(());
    }
    target_meta.execute(
        "UPDATE archive_conversion SET value='replaying' WHERE key='state'",
        [],
    )?;
    drop(target_meta);

    let source_storage = StorageBuilder::file(args.source.clone())
        .journal_mode(JournalMode::WAL)
        .readonly()?
        .create_read_only_pool(NonZeroU32::new(4).unwrap())?;
    let target_storage = StorageBuilder::file(args.target.clone())
        .journal_mode(JournalMode::WAL)
        .trie_prune_mode(Some(TriePruneMode::Prune {
            num_blocks_kept: args.retain_blocks,
        }))
        .blockchain_history_mode(Some(BlockchainHistoryMode::Prune {
            num_blocks_kept: args.retain_blocks,
        }))
        .migrate()?
        .create_pool(NonZeroU32::new(32).unwrap())?;

    let started = Instant::now();
    for raw in (last_replayed + 1)..=head {
        let block = BlockNumber::new(raw).context("Block number exceeds Pathfinder range")?;
        let state_update = {
            let mut connection = source_storage.connection()?;
            let tx = connection.transaction()?;
            tx.state_update(BlockId::Number(block))?
                .with_context(|| format!("Missing source state update at block {block}"))?
        };
        let header = {
            let mut connection = source_storage.connection()?;
            let tx = connection.transaction()?;
            tx.block_header(BlockId::Number(block))?
                .with_context(|| format!("Missing source header at block {block}"))?
        };

        let mut connection = target_storage.connection()?;
        let tx = connection.transaction()?;
        let (storage_commitment, class_commitment) = update_starknet_state(
            &tx,
            (&state_update).into(),
            args.verify_hashes,
            block,
            target_storage.clone(),
        )
        .with_context(|| format!("Replaying trie update at block {block}"))?;
        let actual = StateCommitment::calculate(
            storage_commitment,
            class_commitment,
            header.starknet_version,
        );
        ensure!(
            actual == header.state_commitment,
            "state commitment mismatch at block {block}: expected {}, got {}",
            header.state_commitment,
            actual
        );
        tx.set_archive_conversion_checkpoint(block)?;
        tx.commit()?;

        if raw.is_multiple_of(args.report_every_blocks) || raw == head {
            let completed = raw - boundary;
            let completed_this_run = raw - last_replayed;
            let rate = completed_this_run as f64 / started.elapsed().as_secs_f64();
            let remaining = head - raw;
            tracing::info!(
                block = raw,
                completed,
                remaining,
                blocks_per_second = rate,
                eta_hours = remaining as f64 / rate / 3600.0,
                "replay progress"
            );
        }
    }

    let target = Connection::open(&args.target)?;
    target.execute(
        "UPDATE archive_conversion SET value='complete' WHERE key='state'",
        [],
    )?;
    target.execute_batch("PRAGMA wal_checkpoint(TRUNCATE); PRAGMA optimize;")?;
    let quick_check: String = target.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    ensure!(
        quick_check == "ok",
        "target quick_check failed: {quick_check}"
    );
    tracing::info!(
        elapsed_s = started.elapsed().as_secs_f64(),
        "conversion complete"
    );
    Ok(())
}

fn metadata_u64(db: &Connection, key: &str) -> anyhow::Result<u64> {
    let value: String = db.query_row(
        "SELECT value FROM archive_conversion WHERE key=?",
        [key],
        |r| r.get(0),
    )?;
    value
        .parse()
        .with_context(|| format!("Invalid converter metadata {key}={value:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reachable_copy_preserves_only_the_root_graph() {
        let source = Connection::open_in_memory().unwrap();
        let target = Connection::open_in_memory().unwrap();
        for db in [&source, &target] {
            db.execute_batch(
                "CREATE TABLE trie_contracts(
                    idx INTEGER PRIMARY KEY,
                    hash BLOB NOT NULL,
                    data BLOB
                );",
            )
            .unwrap();
        }

        // StoredSerde encodes LeafBinary as variant 2, and Binary as variant 0
        // followed by its two small varint child indices.
        source
            .execute(
                "INSERT INTO trie_contracts VALUES(1, x'01', x'02'),
                                                   (2, x'02', x'02'),
                                                   (3, x'03', x'000102'),
                                                   (4, x'04', x'02')",
                [],
            )
            .unwrap();

        copy_reachable_trie(&source, &target, "trie_contracts", vec![3], 2).unwrap();

        let mut stmt = target
            .prepare("SELECT idx FROM trie_contracts ORDER BY idx")
            .unwrap();
        let actual = stmt
            .query_map([], |row| row.get::<_, u64>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual, vec![1, 2, 3]);
    }

    #[test]
    fn bitset_handles_boundaries() {
        let mut bits = vec![0; 3];
        for idx in [0, 7, 8, 17] {
            set_bit(&mut bits, idx);
        }
        for idx in [0, 7, 8, 17] {
            assert!(get_bit(&bits, idx));
        }
        for idx in [1, 6, 9, 16] {
            assert!(!get_bit(&bits, idx));
        }
    }

    #[test]
    fn state_window_keeps_tail_and_one_baseline_per_key() {
        let source_file = tempfile::NamedTempFile::new().unwrap();
        let target_file = tempfile::NamedTempFile::new().unwrap();
        let source = Connection::open(source_file.path()).unwrap();
        let mut target = Connection::open(target_file.path()).unwrap();
        let schema = "
            CREATE TABLE contract_updates(block_number INTEGER, contract_address INTEGER, class_hash INTEGER);
            CREATE TABLE nonce_updates(block_number INTEGER, contract_address_id INTEGER, nonce INTEGER);
            CREATE TABLE storage_updates(block_number INTEGER, contract_address_id INTEGER, storage_address_id INTEGER, storage_value INTEGER);";
        source.execute_batch(schema).unwrap();
        target.execute_batch(schema).unwrap();

        for block in [1, 5, 11] {
            source
                .execute(
                    "INSERT INTO contract_updates VALUES(?, 1, ?)",
                    params![block, block],
                )
                .unwrap();
            source
                .execute(
                    "INSERT INTO nonce_updates VALUES(?, 1, ?)",
                    params![block, block],
                )
                .unwrap();
            source
                .execute(
                    "INSERT INTO storage_updates VALUES(?, 1, 100, ?)",
                    params![block, block],
                )
                .unwrap();
        }
        source
            .execute("INSERT INTO contract_updates VALUES(2, 2, 2)", [])
            .unwrap();
        source
            .execute("INSERT INTO nonce_updates VALUES(2, 2, 2)", [])
            .unwrap();
        source
            .execute("INSERT INTO storage_updates VALUES(2, 2, 200, 2)", [])
            .unwrap();

        attach_source(&target, source_file.path()).unwrap();
        copy_state_window(&mut target, 10).unwrap();

        for table in ["contract_updates", "nonce_updates", "storage_updates"] {
            let mut stmt = target
                .prepare(&format!("SELECT block_number FROM {table} ORDER BY block_number"))
                .unwrap();
            let blocks = stmt
                .query_map([], |row| row.get::<_, u64>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(blocks, vec![2, 5, 11], "table {table}");
        }
    }
}
