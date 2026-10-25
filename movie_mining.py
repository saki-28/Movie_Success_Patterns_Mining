"""Educational frequent-itemset miners for the movie success project.

All miners use exact transaction counts and the same minimum-count threshold.
The DIC and Pincer implementations are intentionally bounded teaching
variants; see the notebook/report for their scope and interpretation.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import ceil
from time import perf_counter


class VerticalTransactionDB:
    """Transactions plus integer bitmaps for fast exact support counting."""

    def __init__(self, transactions):
        self.transactions = [frozenset(t) for t in transactions]
        self.n = len(self.transactions)
        masks = {}
        for tid, transaction in enumerate(self.transactions):
            bit = 1 << tid
            for item in transaction:
                masks[item] = masks.get(item, 0) | bit
        self.item_masks = masks

    @property
    def items(self):
        return tuple(sorted(self.item_masks))

    def support_count(self, itemset, scope_mask=None):
        iterator = iter(itemset)
        try:
            mask = self.item_masks[next(iterator)]
        except StopIteration:
            return self.n if scope_mask is None else scope_mask.bit_count()
        for item in iterator:
            mask &= self.item_masks.get(item, 0)
            if not mask:
                return 0
        if scope_mask is not None:
            mask &= scope_mask
        return mask.bit_count()


def required_count(n, min_support):
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1].")
    return max(1, ceil(n * min_support - 1e-12))


def _join_level(previous, target_size):
    """Apriori join and subset prune for a target itemset size."""
    previous = set(previous)
    ordered = sorted(tuple(sorted(s)) for s in previous)
    candidates = set()
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            if left[: target_size - 2] != right[: target_size - 2]:
                if left[: target_size - 2] < right[: target_size - 2]:
                    break
                continue
            candidate = frozenset(left + (right[-1],))
            if len(candidate) == target_size and all(
                frozenset(candidate - {item}) in previous for item in candidate
            ):
                candidates.add(candidate)
    return candidates


def _make_result(name, patterns, candidates, details, started):
    return {
        "algorithm": name,
        "patterns": patterns,
        "candidate_count": candidates,
        "details": details,
        "seconds": perf_counter() - started,
    }


def mine_apriori(db, min_support=0.005, max_len=4):
    """Classic level-wise Apriori using exact vertical support intersections."""
    started = perf_counter()
    threshold = required_count(db.n, min_support)
    level = {
        frozenset((item,)): mask.bit_count()
        for item, mask in db.item_masks.items()
        if mask.bit_count() >= threshold
    }
    patterns = dict(level)
    candidate_count = len(db.item_masks)
    k = 2
    while level and k <= max_len:
        candidates = _join_level(level, k)
        candidate_count += len(candidates)
        next_level = {}
        for candidate in candidates:
            count = db.support_count(candidate)
            if count >= threshold:
                next_level[candidate] = count
        patterns.update(next_level)
        level = next_level
        k += 1
    return _make_result(
        "Apriori",
        patterns,
        candidate_count,
        {"min_count": threshold, "max_len": max_len},
        started,
    )


def mine_hash_apriori(db, min_support=0.005, max_len=4, bucket_count=2048):
    """Hash-based Apriori: use first-pass pair buckets to prune C2."""
    started = perf_counter()
    threshold = required_count(db.n, min_support)
    item_to_int = {item: i for i, item in enumerate(sorted(db.item_masks))}
    buckets = [0] * bucket_count
    for transaction in db.transactions:
        encoded = sorted(item_to_int[x] for x in transaction)
        for a, b in combinations(encoded, 2):
            buckets[(a * 31 + b * 17 + 13) % bucket_count] += 1

    level = {
        frozenset((item,)): mask.bit_count()
        for item, mask in db.item_masks.items()
        if mask.bit_count() >= threshold
    }
    frequent_items = sorted(next(iter(s)) for s in level)
    pair_candidates = set()
    for a, b in combinations(frequent_items, 2):
        ia, ib = item_to_int[a], item_to_int[b]
        if ia > ib:
            ia, ib = ib, ia
        bucket = (ia * 31 + ib * 17 + 13) % bucket_count
        if buckets[bucket] >= threshold:
            pair_candidates.add(frozenset((a, b)))
    candidate_count = len(db.item_masks) + len(pair_candidates)
    next_level = {
        pair: count
        for pair in pair_candidates
        if (count := db.support_count(pair)) >= threshold
    }
    patterns = dict(level)
    patterns.update(next_level)
    level = next_level
    k = 3
    while level and k <= max_len:
        candidates = _join_level(level, k)
        candidate_count += len(candidates)
        next_level = {
            candidate: count
            for candidate in candidates
            if (count := db.support_count(candidate)) >= threshold
        }
        patterns.update(next_level)
        level = next_level
        k += 1
    pruned = sum(
        1
        for a, b in combinations(frequent_items, 2)
        if frozenset((a, b)) not in pair_candidates
    )
    return _make_result(
        "Hash Apriori",
        patterns,
        candidate_count,
        {
            "min_count": threshold,
            "bucket_count": bucket_count,
            "pairs_pruned_by_hash": pruned,
            "hash_collisions_possible": True,
            "max_len": max_len,
        },
        started,
    )


def mine_eclat(db, min_support=0.005, max_len=4):
    """Vertical ECLAT: recursively intersect transaction-ID bitmaps."""
    started = perf_counter()
    threshold = required_count(db.n, min_support)
    singles = [
        (item, mask, mask.bit_count())
        for item, mask in db.item_masks.items()
        if mask.bit_count() >= threshold
    ]
    singles.sort(key=lambda row: (row[2], row[0]))
    patterns = {}
    visited = 0

    def visit(prefix, extensions):
        nonlocal visited
        for index, (item, mask, count) in enumerate(extensions):
            current = prefix | {item}
            patterns[frozenset(current)] = count
            visited += 1
            if len(current) >= max_len:
                continue
            children = []
            for other, other_mask, _ in extensions[index + 1 :]:
                intersection = mask & other_mask
                support = intersection.bit_count()
                if support >= threshold:
                    children.append((other, intersection, support))
            if children:
                visit(current, children)

    visit(set(), singles)
    return _make_result(
        "Vertical ECLAT",
        patterns,
        visited,
        {"min_count": threshold, "max_len": max_len, "support_representation": "integer tid bitmaps"},
        started,
    )


class _FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None


def _build_fp_tree(weighted_transactions, min_count):
    frequencies = Counter()
    for transaction, weight in weighted_transactions:
        frequencies.update({item: weight for item in set(transaction)})
    frequencies = {item: count for item, count in frequencies.items() if count >= min_count}
    if not frequencies:
        return None, {}
    root = _FPNode(None, 0, None)
    header = {item: [count, None, None] for item, count in frequencies.items()}
    for transaction, weight in weighted_transactions:
        ordered = sorted(
            (item for item in set(transaction) if item in frequencies),
            key=lambda item: (-frequencies[item], item),
        )
        node = root
        for item in ordered:
            child = node.children.get(item)
            if child is None:
                child = _FPNode(item, 0, node)
                node.children[item] = child
                head, tail = header[item][1], header[item][2]
                if head is None:
                    header[item][1] = child
                else:
                    tail.link = child
                header[item][2] = child
            child.count += weight
            node = child
    return root, header


def mine_fp_growth(db, min_support=0.005, max_len=4):
    """FP-Growth with conditional pattern bases and a maximum pattern length."""
    started = perf_counter()
    threshold = required_count(db.n, min_support)
    weighted = [(transaction, 1) for transaction in db.transactions]
    root, header = _build_fp_tree(weighted, threshold)
    patterns = {}
    nodes_visited = 0

    def mine_tree(current_header, suffix):
        nonlocal nodes_visited
        for item in sorted(current_header, key=lambda x: (current_header[x][0], x)):
            new_suffix = suffix | {item}
            count = current_header[item][0]
            patterns[frozenset(new_suffix)] = count
            nodes_visited += 1
            if len(new_suffix) >= max_len:
                continue
            paths = []
            node = current_header[item][1]
            while node is not None:
                path = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    paths.append((path, node.count))
                node = node.link
            _, conditional_header = _build_fp_tree(paths, threshold)
            if conditional_header:
                mine_tree(conditional_header, new_suffix)

    if root is not None:
        mine_tree(header, set())
    return _make_result(
        "FP-Growth",
        patterns,
        nodes_visited,
        {"min_count": threshold, "max_len": max_len, "candidate_generation": "conditional FP trees"},
        started,
    )


def mine_partitioned_apriori(db, min_support=0.005, max_len=4, partitions=4):
    """Partitioning Apriori: mine local frequent sets, union, then verify globally."""
    started = perf_counter()
    if partitions < 2:
        raise ValueError("Use at least two partitions.")
    local_candidates = set()
    partition_sizes = []
    n = db.n
    for part in range(partitions):
        start = (n * part) // partitions
        stop = (n * (part + 1)) // partitions
        local_transactions = db.transactions[start:stop]
        partition_sizes.append(len(local_transactions))
        if not local_transactions:
            continue
        local_db = VerticalTransactionDB(local_transactions)
        local = mine_apriori(local_db, min_support=min_support, max_len=max_len)
        local_candidates.update(local["patterns"])
    threshold = required_count(n, min_support)
    global_patterns = {
        itemset: count
        for itemset in local_candidates
        if (count := db.support_count(itemset)) >= threshold
    }
    return _make_result(
        "Partitioned Apriori",
        global_patterns,
        len(local_candidates),
        {
            "min_count": threshold,
            "partitions": partitions,
            "partition_sizes": partition_sizes,
            "global_verification": True,
            "max_len": max_len,
        },
        started,
    )


def mine_dic_checkpoint(db, min_support=0.005, max_len=4, checkpoints=4):
    """DIC-style blockwise candidate discovery with exact final verification.

    New levels are introduced at checkpoints using support observed in the
    prefix scanned so far. A final full-database verification makes the result
    exact up to max_len. This is a compact teaching variant of textbook DIC.
    """
    started = perf_counter()
    if checkpoints < 2:
        raise ValueError("Use at least two checkpoints.")
    discovered = {frozenset((item,)) for item in db.item_masks}
    candidate_count = len(discovered)
    discoveries_by_checkpoint = []
    for checkpoint in range(1, checkpoints + 1):
        prefix_n = (db.n * checkpoint) // checkpoints
        prefix_mask = (1 << prefix_n) - 1
        threshold = required_count(prefix_n, min_support)
        additions = 0
        for size in range(1, max_len):
            frequent_prefix = {
                itemset
                for itemset in discovered
                if len(itemset) == size
                and db.support_count(itemset, scope_mask=prefix_mask) >= threshold
            }
            for candidate in _join_level(frequent_prefix, size + 1):
                if candidate not in discovered:
                    discovered.add(candidate)
                    additions += 1
        candidate_count += additions
        discoveries_by_checkpoint.append(additions)
    full_threshold = required_count(db.n, min_support)
    patterns = {
        itemset: count
        for itemset in discovered
        if len(itemset) <= max_len
        and (count := db.support_count(itemset)) >= full_threshold
    }
    return _make_result(
        "DIC checkpoint variant",
        patterns,
        candidate_count,
        {
            "min_count": full_threshold,
            "checkpoints": checkpoints,
            "new_candidates_by_checkpoint": discoveries_by_checkpoint,
            "final_full_support_verification": True,
            "max_len": max_len,
        },
        started,
    )


def mine_aclose(apriori_result):
    """A-Close-style closure filtering over an exact frequent-set support map."""
    started = perf_counter()
    patterns = apriori_result["patterns"]
    by_support = {}
    for itemset, count in patterns.items():
        by_support.setdefault(count, []).append(itemset)
    closed = {}
    for count, group in by_support.items():
        group = sorted(group, key=lambda s: (-len(s), tuple(sorted(s))))
        maximal_same_support = []
        for itemset in group:
            if any(itemset < superset for superset in maximal_same_support):
                continue
            maximal_same_support.append(itemset)
            closed[itemset] = count
    return _make_result(
        "A-Close (closure filtering)",
        closed,
        apriori_result["candidate_count"],
        {
            **apriori_result["details"],
            "input_frequent_sets": len(patterns),
            "closed_sets": len(closed),
            "method": "equal-support closure filtering; educational reference form",
        },
        started,
    )


def mine_pincer(db, min_support=0.005, max_len=4, mfcs_cap=256):
    """Bounded Pincer-style maximal mining with MFS/MFCS border bookkeeping.

    The level-wise candidate frontier guarantees completeness to max_len;
    maximal frequent sets and a capped MFCS border are tracked concurrently.
    This teaching implementation is not the optimized textbook Pincer Search.
    """
    started = perf_counter()
    threshold = required_count(db.n, min_support)
    all_items = set(db.item_masks)
    first = {
        frozenset((item,)): mask.bit_count()
        for item, mask in db.item_masks.items()
        if mask.bit_count() >= threshold
    }
    patterns = dict(first)
    previous = set(first)
    mfcs = {frozenset(all_items)} if all_items else set()
    maximal_frequent = set()
    candidate_count = len(all_items)
    border_updates = 0
    border_truncated = False

    def keep_maximal(collection, itemset):
        if any(itemset < old for old in collection):
            return collection
        return {old for old in collection if not old < itemset} | {itemset}

    for size in range(2, max_len + 1):
        candidates = _join_level(previous, size)
        candidate_count += len(candidates)
        frequent = set()
        infrequent = []
        for candidate in candidates:
            count = db.support_count(candidate)
            if count >= threshold:
                patterns[candidate] = count
                frequent.add(candidate)
                maximal_frequent = keep_maximal(maximal_frequent, candidate)
            else:
                infrequent.append(candidate)
        for infrequent_set in infrequent:
            updated = set(mfcs)
            for maximal_candidate in tuple(mfcs):
                if infrequent_set <= maximal_candidate:
                    updated.discard(maximal_candidate)
                    updated.update(maximal_candidate - {item} for item in infrequent_set)
                    border_updates += 1
            ordered = sorted(updated, key=lambda s: (-len(s), tuple(sorted(s))))
            antichain = []
            for candidate in ordered:
                if candidate and not any(candidate < old for old in antichain):
                    antichain = [old for old in antichain if not old < candidate]
                    antichain.append(candidate)
            if len(antichain) > mfcs_cap:
                antichain = antichain[:mfcs_cap]
                border_truncated = True
            mfcs = set(antichain)
        previous = frequent
        if not previous:
            break
    maximal_patterns = {
        itemset: patterns[itemset]
        for itemset in maximal_frequent
        if itemset in patterns
    }
    # Include maximal singletons when no larger frequent set contains them.
    for itemset, count in first.items():
        if not any(itemset < larger for larger in maximal_patterns):
            maximal_patterns[itemset] = count
    return _make_result(
        "Pincer Search (bounded maximal mode)",
        maximal_patterns,
        candidate_count,
        {
            "min_count": threshold,
            "all_frequent_sets_examined": len(patterns),
            "maximal_frequent_sets": len(maximal_patterns),
            "mfcs_border_updates": border_updates,
            "mfcs_final_size": len(mfcs),
            "mfcs_cap": mfcs_cap,
            "mfcs_border_truncated": border_truncated,
            "max_len": max_len,
        },
        started,
    )


def make_outcome_rules(db, frequent_patterns, outcome_prefix="OUTCOME::", min_confidence=0.0):
    """Build single-outcome association rules and return count-based metrics."""
    outcomes = [item for item in db.item_masks if item.startswith(outcome_prefix)]
    rows = []
    for itemset, joint_count in frequent_patterns.items():
        if len(itemset) < 2:
            continue
        matching = [item for item in itemset if item in outcomes]
        if len(matching) != 1:
            continue
        outcome = matching[0]
        antecedent = frozenset(itemset - {outcome})
        antecedent_count = db.support_count(antecedent)
        if not antecedent_count:
            continue
        confidence = joint_count / antecedent_count
        base_rate = db.support_count((outcome,)) / db.n
        if confidence < min_confidence or base_rate == 0:
            continue
        rows.append(
            {
                "antecedents": tuple(sorted(antecedent)),
                "consequent": outcome,
                "support_count": joint_count,
                "support": joint_count / db.n,
                "confidence": confidence,
                "base_rate": base_rate,
                "lift": confidence / base_rate,
            }
        )
    return rows

