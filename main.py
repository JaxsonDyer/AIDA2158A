"""
Main CLI entrypoint – AIDA 2158A-F Strawberry Harvesting Pipeline
Run a specific module or the full pipeline:

  python main.py --module 1        # Merge dataset + train YOLOv11-seg
  python main.py --module 1b       # Generate ROI crops AND Masks (Automated)
  python main.py --module 3        # Train U-Net
  python main.py --module 4        # Extract peduncle cut lines (skeleton + PCA)
  python main.py --all             # Run 1 → 1b → 3 → 4
"""

import argparse
import sys


def run_module_1():
    from module1_yolo_train import main
    main()


def run_module_1b():
    from module1_roi_crop import main
    main()


def run_module_3():
    from module3_unet_train import train
    train()


def run_module_4():
    from module4_stem_angle import main
    main()


def main():
    parser = argparse.ArgumentParser(
        description="AIDA 2158A-F – Strawberry Harvesting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--module", choices=["1", "1b", "3", "4"],
        help="Module to run (1 | 1b | 3 | 4)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all modules sequentially (100% Automated)",
    )

    args = parser.parse_args()

    if not args.module and not args.all:
        parser.print_help()
        sys.exit(0)

    if args.all:
        print("▶  Module 1: Training YOLOv11-seg ...")
        run_module_1()
        print("\n▶  Module 1b: Generating ROI crops and masks (Automated) ...")
        run_module_1b()
        print("\n▶  Module 3: Training U-Net ...")
        run_module_3()
        print("\n▶  Module 4: Extracting peduncle cut lines ...")
        run_module_4()
        print("\n✅  Full pipeline complete!")
        return

    if args.module == "1":
        run_module_1()
    elif args.module == "1b":
        run_module_1b()
    elif args.module == "3":
        run_module_3()
    elif args.module == "4":
        run_module_4()


if __name__ == "__main__":
    main()
