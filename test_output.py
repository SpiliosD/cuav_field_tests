"""Simple test to verify output works."""
import sys

print("TEST: This should appear in terminal", flush=True)
print("TEST: Python version:", sys.version, flush=True)
print("TEST: Script is running!", flush=True, file=sys.stderr)

if __name__ == "__main__":
    print("TEST: __main__ block executed", flush=True)
    print("TEST: All tests passed - output is working", flush=True)

