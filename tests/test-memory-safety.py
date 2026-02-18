"""
Memory safety tests for sqlite-vec.

These tests are designed to exercise code paths that were fixed for memory leaks
in PR #258. The tests focus on:

1. vec0_free - freeing partition/auxiliary/metadata column names
2. vec0_init - proper cleanup on error paths
3. vec0_metadata_filter_text - freeing rowids on error
4. vec0Filter_knn - handling array_cleanup and knn_data cleanup on errors
5. vec0_write_metadata_value - freeing zSql after prepare
6. vec0Update_Insert - freeing zSql after prepare
7. vec0Update_Delete_ClearMetadata - freeing zSql and finalizing stmt on error

When run under a memory checker (e.g., valgrind, ASAN), these tests should
not report any memory leaks.
"""

import pytest
import sqlite3
from collections import OrderedDict


# =============================================================================
# Test: vec0_free - Table Creation and Destruction
# =============================================================================


def test_vec0_create_destroy_cycle(db):
    """
    Test that creating and destroying vec0 tables doesn't leak memory.

    This exercises vec0_free which should properly free:
    - partition column names
    - auxiliary column names
    - metadata column names
    """
    for i in range(50):
        db.execute(f"CREATE VIRTUAL TABLE v{i} USING vec0(embedding float[128])")
        db.execute(f"DROP TABLE v{i}")


def test_vec0_create_destroy_with_partition_keys(db):
    """
    Test create/destroy cycle with partition key columns.

    Partition key column names should be freed in vec0_free.
    """
    for i in range(50):
        db.execute(
            f"CREATE VIRTUAL TABLE v{i} USING vec0("
            f"pk1 int partition key, "
            f"pk2 text partition key, "
            f"embedding float[64])"
        )
        db.execute(f"DROP TABLE v{i}")


def test_vec0_create_destroy_with_auxiliary_columns(db):
    """
    Test create/destroy cycle with auxiliary columns.

    Auxiliary column names should be freed in vec0_free.
    """
    for i in range(50):
        db.execute(
            f"CREATE VIRTUAL TABLE v{i} USING vec0("
            f"embedding float[64], "
            f"+aux_int integer, "
            f"+aux_text text, "
            f"+aux_float float, "
            f"+aux_blob blob)"
        )
        db.execute(f"DROP TABLE v{i}")


def test_vec0_create_destroy_with_metadata_columns(db):
    """
    Test create/destroy cycle with metadata columns.

    Metadata column names should be freed in vec0_free.
    """
    for i in range(50):
        db.execute(
            f"CREATE VIRTUAL TABLE v{i} USING vec0("
            f"embedding float[64], "
            f"meta_bool boolean, "
            f"meta_int int, "
            f"meta_float float, "
            f"meta_text text)"
        )
        db.execute(f"DROP TABLE v{i}")


def test_vec0_create_destroy_all_column_types(db):
    """
    Test create/destroy with all column types combined.

    This is the most comprehensive test for vec0_free memory cleanup.
    """
    for i in range(30):
        db.execute(
            f"CREATE VIRTUAL TABLE v{i} USING vec0("
            f"pk int partition key, "
            f"embedding float[32], "
            f"+aux_name text, "
            f"meta_category text, "
            f"meta_score float)"
        )
        db.execute(f"DROP TABLE v{i}")


# =============================================================================
# Test: vec0_init - Error Path Cleanup
# =============================================================================


def test_vec0_init_error_too_many_partition_keys(db):
    """
    Test that vec0_init properly cleans up on error (too many partition keys).

    vec0 allows max 4 partition keys; exceeding this should error cleanly.
    """
    for i in range(20):
        result = exec_safe(
            db,
            """
            CREATE VIRTUAL TABLE v USING vec0(
                p1 int partition key,
                p2 int partition key,
                p3 int partition key,
                p4 int partition key,
                p5 int partition key,
                embedding float[4]
            )
            """
        )
        assert "error" in result or "Error" in result.get("message", "")


def test_vec0_init_error_too_many_metadata_columns(db):
    """
    Test that vec0_init properly cleans up on error (too many metadata columns).

    vec0 allows max 16 metadata columns; exceeding this should error cleanly.
    """
    for i in range(20):
        metadata_cols = ",".join([f"m{x} integer" for x in range(17)])
        result = exec_safe(
            db,
            f"CREATE VIRTUAL TABLE v USING vec0({metadata_cols}, embedding float[4])"
        )
        assert "error" in result or "Error" in result.get("message", "")


def test_vec0_init_error_invalid_syntax(db):
    """
    Test that vec0_init properly cleans up on various syntax errors.
    """
    invalid_schemas = [
        "CREATE VIRTUAL TABLE v USING vec0()",  # No columns
        "CREATE VIRTUAL TABLE v USING vec0(not_a_vector)",  # No vector column
        "CREATE VIRTUAL TABLE v USING vec0(embedding float[0])",  # Zero dimensions
        "CREATE VIRTUAL TABLE v USING vec0(embedding float[-1])",  # Negative dimensions
    ]

    for i in range(10):
        for schema in invalid_schemas:
            result = exec_safe(db, schema)
            # We expect errors; the important thing is no memory leak


# =============================================================================
# Test: vec0Update_Insert - INSERT Operations
# =============================================================================


def test_insert_delete_cycle(db):
    """
    Test INSERT/DELETE operations don't leak memory.

    This exercises vec0Update_Insert and vec0Update_Delete code paths.
    """
    db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

    for i in range(500):
        db.execute(
            "INSERT INTO v(embedding) VALUES (?)",
            [f"[{i}, {i+1}, {i+2}, {i+3}]"]
        )

    db.execute("DELETE FROM v")
    db.execute("DROP TABLE v")


def test_batch_insert_with_metadata(db):
    """
    Test batch inserts with metadata columns.

    This exercises vec0Update_Insert and vec0_write_metadata_value.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[4], "
        "category text, "
        "score float, "
        "active boolean)"
    )

    for i in range(200):
        db.execute(
            "INSERT INTO v(embedding, category, score, active) VALUES (?, ?, ?, ?)",
            [f"[{i}, {i}, {i}, {i}]", f"cat_{i % 10}", float(i) / 100.0, i % 2]
        )

    db.execute("DELETE FROM v")
    db.execute("DROP TABLE v")


def test_batch_insert_with_auxiliary_columns(db):
    """
    Test batch inserts with auxiliary columns.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[4], "
        "+name text, "
        "+data blob)"
    )

    for i in range(200):
        db.execute(
            "INSERT INTO v(embedding, name, data) VALUES (?, ?, ?)",
            [f"[{i}, {i}, {i}, {i}]", f"item_{i}", bytes([i % 256] * 10)]
        )

    db.execute("DELETE FROM v")
    db.execute("DROP TABLE v")


def test_batch_insert_with_partition_keys(db):
    """
    Test batch inserts with partition keys.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "pk int partition key, "
        "embedding float[4])"
    )

    for i in range(200):
        partition = i % 5
        db.execute(
            "INSERT INTO v(pk, embedding) VALUES (?, ?)",
            [partition, f"[{i}, {i}, {i}, {i}]"]
        )

    db.execute("DELETE FROM v")
    db.execute("DROP TABLE v")


def test_insert_with_long_text_metadata(db):
    """
    Test inserts with long text metadata (stored externally).

    Text longer than 12 chars is stored in separate table.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0(embedding float[4], name text, chunk_size=8)"
    )

    for i in range(100):
        # Short text (stored inline)
        db.execute(
            "INSERT INTO v(embedding, name) VALUES (?, ?)",
            [f"[{i}, {i}, {i}, {i}]", f"short{i}"]
        )
        # Long text (stored externally)
        db.execute(
            "INSERT INTO v(embedding, name) VALUES (?, ?)",
            [f"[{i+0.5}, {i+0.5}, {i+0.5}, {i+0.5}]", f"this_is_a_very_long_name_{i}"]
        )

    db.execute("DELETE FROM v")
    db.execute("DROP TABLE v")


# =============================================================================
# Test: vec0Update_Delete_ClearMetadata - DELETE Operations
# =============================================================================


def test_delete_with_metadata_columns(db):
    """
    Test DELETE operations with metadata columns.

    This exercises vec0Update_Delete_ClearMetadata.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[4], "
        "category text, "
        "value int, "
        "chunk_size=8)"
    )

    # Insert data
    for i in range(100):
        db.execute(
            "INSERT INTO v(rowid, embedding, category, value) VALUES (?, ?, ?, ?)",
            [i, f"[{i}, {i}, {i}, {i}]", f"category_name_long_{i}", i * 10]
        )

    # Delete one by one
    for i in range(100):
        db.execute("DELETE FROM v WHERE rowid = ?", [i])

    db.execute("DROP TABLE v")


def test_delete_with_long_text_metadata(db):
    """
    Test DELETE with long text metadata that has external storage.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[4], "
        "description text, "
        "chunk_size=8)"
    )

    # Insert with long text
    for i in range(50):
        long_text = f"this_is_a_long_description_for_item_{i}_with_extra_padding"
        db.execute(
            "INSERT INTO v(rowid, embedding, description) VALUES (?, ?, ?)",
            [i, f"[{i}, {i}, {i}, {i}]", long_text]
        )

    # Delete all
    for i in range(50):
        db.execute("DELETE FROM v WHERE rowid = ?", [i])

    db.execute("DROP TABLE v")


def test_bulk_delete(db):
    """
    Test bulk DELETE operation.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[4], "
        "tag text, "
        "chunk_size=8)"
    )

    # Insert many rows
    for i in range(200):
        db.execute(
            "INSERT INTO v(embedding, tag) VALUES (?, ?)",
            [f"[{i}, {i}, {i}, {i}]", f"tag_{i % 20}"]
        )

    # Bulk delete
    db.execute("DELETE FROM v")
    db.execute("DROP TABLE v")


# =============================================================================
# Test: vec0Filter_knn and vec0_metadata_filter_text - KNN Queries
# =============================================================================


def test_knn_queries_basic(db):
    """
    Test basic KNN queries.

    This exercises vec0Filter_knn code path.
    """
    db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], chunk_size=8)")

    for i in range(100):
        db.execute(
            "INSERT INTO v(embedding) VALUES (?)",
            [f"[{i}, {i}, {i}, {i}]"]
        )

    # Run many KNN queries
    for i in range(50):
        query_vec = f"[{i}, {i}, {i}, {i}]"
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10",
            [query_vec]
        ))

    db.execute("DROP TABLE v")


def test_knn_with_metadata_filters(db):
    """
    Test KNN queries with metadata column filters.

    This exercises vec0_metadata_filter_* functions.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[4], "
        "category text, "
        "score float, "
        "active boolean, "
        "count int, "
        "chunk_size=8)"
    )

    for i in range(100):
        db.execute(
            "INSERT INTO v(embedding, category, score, active, count) VALUES (?, ?, ?, ?, ?)",
            [f"[{i}, {i}, {i}, {i}]", f"cat{i % 5}", float(i) / 10.0, i % 2, i]
        )

    # KNN with various metadata filters
    for i in range(20):
        query_vec = f"[{i * 5}, {i * 5}, {i * 5}, {i * 5}]"

        # Filter by text
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND category = 'cat1'",
            [query_vec]
        ))

        # Filter by boolean
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND active = TRUE",
            [query_vec]
        ))

        # Filter by float
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND score > 5.0",
            [query_vec]
        ))

        # Filter by int
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND count < 50",
            [query_vec]
        ))

    db.execute("DROP TABLE v")


def test_knn_with_text_metadata_filters(db):
    """
    Test KNN queries with text metadata filters (exercises vec0_metadata_filter_text).

    Uses both short and long text values.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[4], "
        "name text, "
        "chunk_size=8)"
    )

    for i in range(100):
        # Alternate between short and long names
        if i % 2 == 0:
            name = f"short{i}"
        else:
            name = f"this_is_a_long_name_for_item_{i}"

        db.execute(
            "INSERT INTO v(embedding, name) VALUES (?, ?)",
            [f"[{i}, {i}, {i}, {i}]", name]
        )

    # KNN with text filters
    for i in range(30):
        query_vec = f"[{i * 3}, {i * 3}, {i * 3}, {i * 3}]"

        # Exact match
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND name = 'short2'",
            [query_vec]
        ))

        # Less than
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND name < 'short50'",
            [query_vec]
        ))

        # Greater than
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND name > 'short50'",
            [query_vec]
        ))

        # Not equal
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND name != 'short0'",
            [query_vec]
        ))

    db.execute("DROP TABLE v")


def test_knn_with_distance_constraints(db):
    """
    Test KNN queries with distance constraints.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0(embedding float[4], tag text, chunk_size=8)"
    )

    for i in range(100):
        db.execute(
            "INSERT INTO v(embedding, tag) VALUES (?, ?)",
            [f"[{i}, {i}, {i}, {i}]", f"tag{i % 10}"]
        )

    for i in range(30):
        query_vec = f"[{i * 3}, {i * 3}, {i * 3}, {i * 3}]"

        # Distance constraints
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND distance < 100",
            [query_vec]
        ))

        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND distance BETWEEN 10 AND 50",
            [query_vec]
        ))

        # Combined with metadata filter
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND tag = 'tag1' AND distance < 200",
            [query_vec]
        ))

    db.execute("DROP TABLE v")


def test_knn_empty_results(db):
    """
    Test KNN queries that return no results (edge case for cleanup).
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0(embedding float[4], category text, chunk_size=8)"
    )

    for i in range(50):
        db.execute(
            "INSERT INTO v(embedding, category) VALUES (?, ?)",
            [f"[{i}, {i}, {i}, {i}]", "existing_category"]
        )

    # Query with filter that matches nothing
    for i in range(20):
        query_vec = f"[{i}, {i}, {i}, {i}]"
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 10 AND category = 'nonexistent'",
            [query_vec]
        ))

    db.execute("DROP TABLE v")


# =============================================================================
# Test: Stress Tests - Rapid Operations
# =============================================================================


def test_rapid_create_insert_drop(db):
    """
    Stress test: rapid create/insert/drop cycles.

    This stresses all allocation/deallocation paths.
    """
    for i in range(30):
        db.execute(
            f"CREATE VIRTUAL TABLE v{i} USING vec0("
            f"pk int partition key, "
            f"embedding float[32], "
            f"+aux text, "
            f"meta_text text, "
            f"meta_int int)"
        )

        for j in range(20):
            db.execute(
                f"INSERT INTO v{i}(pk, embedding, aux, meta_text, meta_int) VALUES (?, ?, ?, ?, ?)",
                [j % 3, f"[{','.join([str(j)] * 32)}]", f"aux_{j}", f"meta_text_{j}", j]
            )

        db.execute(f"DROP TABLE v{i}")


def test_rapid_insert_update_delete(db):
    """
    Stress test: rapid insert/update/delete operations.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[4], "
        "name text, "
        "value int, "
        "chunk_size=8)"
    )

    for cycle in range(10):
        # Insert batch
        for i in range(50):
            rowid = cycle * 50 + i
            db.execute(
                "INSERT INTO v(rowid, embedding, name, value) VALUES (?, ?, ?, ?)",
                [rowid, f"[{i}, {i}, {i}, {i}]", f"name_{rowid}", rowid]
            )

        # Update some
        for i in range(25):
            rowid = cycle * 50 + i
            db.execute(
                "UPDATE v SET name = ?, value = ? WHERE rowid = ?",
                [f"updated_name_{rowid}", rowid * 2, rowid]
            )

        # Delete some
        for i in range(25, 50):
            rowid = cycle * 50 + i
            db.execute("DELETE FROM v WHERE rowid = ?", [rowid])

    db.execute("DROP TABLE v")


def test_mixed_operations_stress(db):
    """
    Stress test: mixed operations (insert, query, update, delete).
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[8], "
        "category text, "
        "score float, "
        "chunk_size=8)"
    )

    for cycle in range(5):
        # Insert phase
        for i in range(30):
            rowid = cycle * 100 + i
            db.execute(
                "INSERT INTO v(rowid, embedding, category, score) VALUES (?, ?, ?, ?)",
                [rowid, f"[{','.join([str(i)] * 8)}]", f"cat{i % 3}", float(i) / 10.0]
            )

        # Query phase
        for i in range(10):
            query_vec = f"[{','.join([str(i * 3)] * 8)}]"
            list(db.execute(
                "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 5 AND category = 'cat1'",
                [query_vec]
            ))

        # Update phase
        for i in range(15):
            rowid = cycle * 100 + i
            db.execute(
                "UPDATE v SET score = ? WHERE rowid = ?",
                [float(i) * 2.0, rowid]
            )

        # Delete phase
        for i in range(15, 30):
            rowid = cycle * 100 + i
            db.execute("DELETE FROM v WHERE rowid = ?", [rowid])

    db.execute("DROP TABLE v")


def test_large_batch_operations(db):
    """
    Test with larger batches to stress memory management.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[64], "
        "meta text)"
    )

    # Large batch insert
    for i in range(1000):
        vec = f"[{','.join([str(float(i) / 1000.0)] * 64)}]"
        db.execute(
            "INSERT INTO v(embedding, meta) VALUES (?, ?)",
            [vec, f"item_{i}"]
        )

    # Many queries
    for i in range(100):
        query_vec = f"[{','.join([str(float(i * 10) / 1000.0)] * 64)}]"
        list(db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = 20",
            [query_vec]
        ))

    # Bulk delete
    db.execute("DELETE FROM v")
    db.execute("DROP TABLE v")


# =============================================================================
# Test: Error Conditions
# =============================================================================


def test_insert_error_invalid_vector_dimension(db):
    """
    Test INSERT with wrong vector dimensions triggers proper cleanup.
    """
    db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

    for i in range(20):
        # Wrong dimension should error
        result = exec_safe(
            db,
            "INSERT INTO v(embedding) VALUES (?)",
            ["[1, 2, 3]"]  # 3 instead of 4
        )
        # Error expected; check no leak

    db.execute("DROP TABLE v")


def test_insert_error_invalid_metadata_type(db):
    """
    Test INSERT with wrong metadata types triggers proper cleanup.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0("
        "embedding float[4], "
        "count int, "
        "active boolean)"
    )

    for i in range(20):
        # Invalid int type
        result = exec_safe(
            db,
            "INSERT INTO v(embedding, count, active) VALUES (?, ?, ?)",
            ["[1, 2, 3, 4]", "not_an_int", True]
        )

        # Invalid boolean type
        result = exec_safe(
            db,
            "INSERT INTO v(embedding, count, active) VALUES (?, ?, ?)",
            ["[1, 2, 3, 4]", 1, "not_a_bool"]
        )

    db.execute("DROP TABLE v")


def test_knn_error_invalid_query_vector(db):
    """
    Test KNN with invalid query vector triggers proper cleanup.
    """
    db.execute(
        "CREATE VIRTUAL TABLE v USING vec0(embedding float[4], tag text)"
    )

    for i in range(50):
        db.execute(
            "INSERT INTO v(embedding, tag) VALUES (?, ?)",
            [f"[{i}, {i}, {i}, {i}]", f"tag{i}"]
        )

    for i in range(20):
        # Wrong dimension query
        result = exec_safe(
            db,
            "SELECT rowid FROM v WHERE embedding MATCH ? AND k = 5",
            ["[1, 2, 3]"]  # 3 instead of 4
        )

    db.execute("DROP TABLE v")


# =============================================================================
# Test: Multiple Tables Simultaneously
# =============================================================================


def test_multiple_tables_concurrent_operations(db):
    """
    Test operations on multiple tables simultaneously.
    """
    # Create multiple tables with different schemas
    db.execute(
        "CREATE VIRTUAL TABLE v1 USING vec0(embedding float[4], tag text)"
    )
    db.execute(
        "CREATE VIRTUAL TABLE v2 USING vec0("
        "pk int partition key, "
        "embedding float[8], "
        "+aux text)"
    )
    db.execute(
        "CREATE VIRTUAL TABLE v3 USING vec0("
        "embedding float[16], "
        "meta_int int, "
        "meta_text text)"
    )

    # Interleaved operations
    for i in range(30):
        db.execute(
            "INSERT INTO v1(embedding, tag) VALUES (?, ?)",
            [f"[{i}, {i}, {i}, {i}]", f"tag{i}"]
        )
        db.execute(
            "INSERT INTO v2(pk, embedding, aux) VALUES (?, ?, ?)",
            [i % 3, f"[{','.join([str(i)] * 8)}]", f"aux{i}"]
        )
        db.execute(
            "INSERT INTO v3(embedding, meta_int, meta_text) VALUES (?, ?, ?)",
            [f"[{','.join([str(i)] * 16)}]", i, f"text{i}"]
        )

    # Queries on all tables
    for i in range(10):
        list(db.execute(
            "SELECT rowid FROM v1 WHERE embedding MATCH '[1,1,1,1]' AND k = 5"
        ))
        list(db.execute(
            "SELECT rowid FROM v2 WHERE embedding MATCH ? AND k = 5",
            [f"[{','.join(['1'] * 8)}]"]
        ))
        list(db.execute(
            "SELECT rowid FROM v3 WHERE embedding MATCH ? AND k = 5",
            [f"[{','.join(['1'] * 16)}]"]
        ))

    # Cleanup
    db.execute("DROP TABLE v1")
    db.execute("DROP TABLE v2")
    db.execute("DROP TABLE v3")


# =============================================================================
# Helper Functions
# =============================================================================


def exec_safe(db, sql, parameters=None):
    """
    Execute SQL and return result dict, catching any errors.
    """
    if parameters is None:
        parameters = []
    try:
        rows = db.execute(sql, parameters).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return {
            "error": e.__class__.__name__,
            "message": str(e),
        }
    a = []
    for row in rows:
        o = OrderedDict()
        for k in row.keys():
            o[k] = row[k]
        a.append(o)
    result = OrderedDict()
    result["sql"] = sql
    result["rows"] = a
    return result
