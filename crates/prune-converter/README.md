# Pathfinder archive-to-pruned converter

This offline tool converts an immutable archive SQLite database into a separate
pruning-enabled Pathfinder database. It never enables pruning in place.

For a source head `H` and retention `N`, it uses boundary `B = H - N`:

1. Create a new database with both trie and blockchain retention set to `N`.
2. Copy blocks `B..H`, immutable class data, and the latest state value at or
   before `B` for each contract/nonce/storage key.
3. Copy the trie roots at `B` and scan each archive trie once in descending
   node-index order. Because Pathfinder stores children before parents, a
   compact bitset is enough to copy exactly the graph reachable from those
   roots without random traversal reads.
4. Replay `B+1..H` through Pathfinder's native `update_starknet_state`, commit
   each block, and verify every computed state commitment against its header.
   This creates the removal journals needed for safe pruning after conversion.

The source is opened read-only. The target must be empty or contain converter
metadata from a completed bootstrap/replay checkpoint. Use `--mode plan` for a
read-only inventory and `--stop-after-bootstrap` to measure boundary extraction
before beginning the retained-window replay.
