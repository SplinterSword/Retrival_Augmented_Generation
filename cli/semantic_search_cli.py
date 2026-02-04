#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_modal

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    # Verify Modal
    parser.add_argument("command", choices=["verify"], help="Command to execute")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verified = verify_modal()
            
            if verified:
                print("Modal verified successfully")
            else:
                print("Failed to verify modal")
                exit(1)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()