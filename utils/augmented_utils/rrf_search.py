import subprocess
import json

def do_rrf_search(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return

    try:
        search_results = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Fallback for subprocesses that accidentally print logs before JSON.
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            print("Error parsing search results: empty output")
            if result.stderr:
                print(result.stderr)
            return
        try:
            search_results = json.loads(lines[-1])
        except json.JSONDecodeError:
            print("Error parsing search results:", result.stdout)
            if result.stderr:
                print(result.stderr)
            return
    return search_results