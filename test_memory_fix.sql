-- Test case for memory leak fixes in sqlite-vec
-- This tests the fixes from PR #258 addressing issues #245, #259, #265

-- Test 1: Basic vec0 table creation and insertion
CREATE VIRTUAL TABLE test_vectors USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[4]
);

-- Insert several vectors (this used to cause segfault on 5th+ insertion in some cases)
INSERT INTO test_vectors(id, embedding) VALUES ('vec1', '[1.0, 2.0, 3.0, 4.0]');
INSERT INTO test_vectors(id, embedding) VALUES ('vec2', '[5.0, 6.0, 7.0, 8.0]');
INSERT INTO test_vectors(id, embedding) VALUES ('vec3', '[9.0, 10.0, 11.0, 12.0]');
INSERT INTO test_vectors(id, embedding) VALUES ('vec4', '[13.0, 14.0, 15.0, 16.0]');
INSERT INTO test_vectors(id, embedding) VALUES ('vec5', '[17.0, 18.0, 19.0, 20.0]');
INSERT INTO test_vectors(id, embedding) VALUES ('vec6', '[21.0, 22.0, 23.0, 24.0]');
INSERT INTO test_vectors(id, embedding) VALUES ('vec7', '[25.0, 26.0, 27.0, 28.0]');
INSERT INTO test_vectors(id, embedding) VALUES ('vec8', '[29.0, 30.0, 31.0, 32.0]');
INSERT INTO test_vectors(id, embedding) VALUES ('vec9', '[33.0, 34.0, 35.0, 36.0]');
INSERT INTO test_vectors(id, embedding) VALUES ('vec10', '[37.0, 38.0, 39.0, 40.0]');

SELECT 'Inserted 10 vectors successfully' AS status;

-- Test 2: KNN query (tests memory handling in vec0Filter_knn)
SELECT id, distance
FROM test_vectors
WHERE embedding MATCH '[1.0, 2.0, 3.0, 4.0]'
ORDER BY distance
LIMIT 3;

-- Test 3: Delete operation (tests memory handling in vec0Update_Delete)
DELETE FROM test_vectors WHERE id = 'vec5';
SELECT 'Deleted vec5 successfully' AS status;

-- Test 4: Verify integrity after operations
SELECT COUNT(*) AS remaining_count FROM test_vectors;

-- Test 5: Multiple insertions and deletions (stress test for memory leaks)
INSERT INTO test_vectors(id, embedding) VALUES ('stress1', '[1.0, 1.0, 1.0, 1.0]');
INSERT INTO test_vectors(id, embedding) VALUES ('stress2', '[2.0, 2.0, 2.0, 2.0]');
DELETE FROM test_vectors WHERE id = 'stress1';
INSERT INTO test_vectors(id, embedding) VALUES ('stress3', '[3.0, 3.0, 3.0, 3.0]');
DELETE FROM test_vectors WHERE id = 'stress2';
INSERT INTO test_vectors(id, embedding) VALUES ('stress4', '[4.0, 4.0, 4.0, 4.0]');
DELETE FROM test_vectors WHERE id = 'stress3';

SELECT 'Stress test passed' AS status;

-- Clean up
DROP TABLE test_vectors;
SELECT 'All tests completed successfully' AS status;
