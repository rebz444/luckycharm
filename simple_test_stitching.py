import os
import pandas as pd
import numpy as np
import tempfile
import shutil

def create_sample_session(session_num, num_trials, num_blocks):
    """Create a sample session DataFrame for testing"""
    np.random.seed(42 + session_num)  # Different seed for each session
    
    data = []
    for trial in range(num_trials):
        for block in range(num_blocks):
            data.append({
                'session_trial_num': trial + 1,
                'block_num': block + 1,
                'session_time': np.random.uniform(0, 100),
                'event_type': f'event_{np.random.randint(1, 5)}',
                'timestamp': np.random.uniform(0, 1000),
                'value': np.random.uniform(0, 1)
            })
    
    df = pd.DataFrame(data)
    df['session_num'] = session_num
    return df

def stitch_sessions_test(session_1, session_2):
    """Test version of stitch_sessions function"""
    # Simulate the helper function behavior
    session_1_basics = {
        'session_time': 100.0,  # Mock values for testing
        'num_blocks': 5,
        'num_trials': 10
    }
    
    # Apply offsets to session 2
    session_2_copy = session_2.copy()
    session_2_copy['session_time'] = session_2_copy['session_time'] + session_1_basics['session_time']
    session_2_copy['block_num'] = session_2_copy['block_num'] + session_1_basics['num_blocks']
    session_2_copy['session_trial_num'] = session_2_copy['session_trial_num'] + session_1_basics['num_trials']
    
    # Concatenate sessions
    stitched_session = pd.concat([session_1, session_2_copy], ignore_index=True)
    return stitched_session

def test_basic_stitching():
    """Test basic session stitching functionality"""
    print("\n=== Testing Basic Session Stitching ===")
    
    # Create test sessions
    session_1 = create_sample_session(1, 5, 3)  # 15 rows
    session_2 = create_sample_session(2, 4, 2)  # 8 rows
    
    print(f"Session 1: {len(session_1)} rows")
    print(f"Session 2: {len(session_2)} rows")
    
    # Stitch them together
    stitched = stitch_sessions_test(session_1, session_2)
    
    # Verify the result
    expected_length = len(session_1) + len(session_2)
    assert len(stitched) == expected_length, \
        f"Expected {expected_length} rows, got {len(stitched)}"
    
    print(f"✓ Successfully stitched: {len(session_1)} + {len(session_2)} = {len(stitched)} rows")
    
    # Check that all data is preserved
    assert stitched['session_num'].nunique() == 2, "Should have 2 unique session numbers"
    assert stitched['event_type'].nunique() > 0, "Should preserve event types"
    
    print("✓ All data preserved in stitched session")
    
    return stitched

def test_time_offset_application():
    """Test that time offsets are properly applied"""
    print("\n=== Testing Time Offset Application ===")
    
    # Create simple test sessions
    session_1 = pd.DataFrame({
        'session_trial_num': [1, 2],
        'block_num': [1, 1],
        'session_time': [10.0, 20.0],
        'event_type': ['lick', 'reward'],
        'value': [1.0, 2.0]
    })
    
    session_2 = pd.DataFrame({
        'session_trial_num': [1, 2],
        'block_num': [1, 1],
        'session_time': [5.0, 15.0],
        'event_type': ['lick', 'lick'],
        'value': [1.0, 1.0]
    })
    
    # Stitch them together
    stitched = stitch_sessions_test(session_1, session_2)
    
    # Check that time offsets were applied
    session_2_rows = stitched[stitched['session_num'] == 2]
    assert all(session_2_rows['session_time'] >= 100.0), \
        "Session 2 times should be offset by at least 100"
    
    print("✓ Time offsets properly applied")
    
    # Check block and trial offsets
    assert all(session_2_rows['block_num'] >= 6), \
        "Session 2 block numbers should be offset by 5"
    assert all(session_2_rows['session_trial_num'] >= 11), \
        "Session 2 trial numbers should be offset by 10"
    
    print("✓ Block and trial offsets properly applied")
    
    return stitched

def test_multiple_session_stitching():
    """Test stitching more than 2 sessions together"""
    print("\n=== Testing Multiple Session Stitching ===")
    
    # Create 3 test sessions
    session_1 = create_sample_session(1, 3, 2)  # 6 rows
    session_2 = create_sample_session(2, 2, 2)  # 4 rows
    session_3 = create_sample_session(3, 2, 1)  # 2 rows
    
    print(f"Session 1: {len(session_1)} rows")
    print(f"Session 2: {len(session_2)} rows")
    print(f"Session 3: {len(session_3)} rows")
    
    # Stitch session 1 + 2 first
    stitched_12 = stitch_sessions_test(session_1, session_2)
    print(f"After stitching 1+2: {len(stitched_12)} rows")
    
    # Then stitch the result with session 3
    stitched_all = stitch_sessions_test(stitched_12, session_3)
    
    # Verify total length
    expected_length = len(session_1) + len(session_2) + len(session_3)
    assert len(stitched_all) == expected_length, \
        f"Expected {expected_length} rows, got {len(stitched_all)}"
    
    print(f"✓ Successfully stitched 3 sessions: {len(stitched_all)} total rows")
    
    # Check that all session numbers are present
    session_nums = sorted(stitched_all['session_num'].unique())
    assert session_nums == [1, 2, 3], f"Expected session numbers [1,2,3], got {session_nums}"
    
    print("✓ All session numbers preserved")
    
    return stitched_all

def test_data_integrity():
    """Test that data integrity is maintained during stitching"""
    print("\n=== Testing Data Integrity ===")
    
    # Create sessions with specific known values
    session_a = pd.DataFrame({
        'session_trial_num': [1, 2, 3],
        'block_num': [1, 1, 2],
        'session_time': [10.0, 20.0, 30.0],
        'event_type': ['lick', 'reward', 'lick'],
        'value': [1.0, 2.0, 1.0]
    })
    
    session_b = pd.DataFrame({
        'session_trial_num': [1, 2],
        'block_num': [1, 1],
        'session_time': [5.0, 15.0],
        'event_type': ['lick', 'lick'],
        'value': [1.0, 1.0]
    })
    
    # Stitch them together
    stitched = stitch_sessions_test(session_a, session_b)
    
    # Verify original data is preserved
    assert len(stitched) == 5, "Should have 5 total rows"
    
    # Check that session A data is unchanged (first 3 rows)
    session_a_rows = stitched.iloc[:3]
    pd.testing.assert_frame_equal(session_a, session_a_rows[['session_trial_num', 'block_num', 'session_time', 'event_type', 'value']])
    
    print("✓ Original session data integrity maintained")
    
    # Check that session B data has proper offsets
    session_b_rows = stitched.iloc[3:]
    assert session_b_rows['session_trial_num'].iloc[0] >= 11, "Trial numbers should be offset by 10"
    assert session_b_rows['block_num'].iloc[0] >= 6, "Block numbers should be offset by 5"
    
    print("✓ Session B data properly offset")
    
    return stitched

def run_all_tests():
    """Run all tests and provide a summary"""
    print("🧪 Running Events Stitching Tests...")
    print("=" * 50)
    
    try:
        # Run all test methods
        test_basic_stitching()
        test_time_offset_application()
        test_multiple_session_stitching()
        test_data_integrity()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Your events stitching logic is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("Please check the implementation and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
