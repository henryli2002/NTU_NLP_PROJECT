"""
Training-only quick check: verify training pipeline works on selected tasks.
- Runs tiny training (minimal samples, 1 epoch)
- Optionally verifies trained model can be loaded
- Cleans up trained models after check
"""
import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import train_model_on_task

# 默认只测一个轻量模型，避免多模型耗时
DEFAULT_MODEL = "bert-base-uncased"
ALL_TASKS = ["sst2", "stsb", "agnews"]


def cleanup_trained_models():
    """Remove trained model directories"""
    trained_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trained")
    if os.path.isdir(trained_dir):
        try:
            shutil.rmtree(trained_dir)
            print(f"Cleaned up trained models from {trained_dir}")
        except Exception as e:
            print(f"Warning: could not clean up trained models: {e}")


def main():
    parser = argparse.ArgumentParser(description="Training-only quick check")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model")
    parser.add_argument("--tasks", type=str, nargs="+", default=["agnews"],
                        help=f"Tasks to train on: {ALL_TASKS}")
    parser.add_argument("--train_samples", type=int, default=16, help="Training samples per task")
    parser.add_argument("--val_samples", type=int, default=4, help="Validation samples per task")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--verify_load", action="store_true", 
                        help="After training, verify model can be loaded")
    args = parser.parse_args()

    print("\n=== TRAINING QUICK CHECK ===")
    print(f"model={args.model}, tasks={args.tasks}")
    print(f"train_samples={args.train_samples}, val_samples={args.val_samples}")
    print(f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # Reduce warnings noise
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    success_count = 0
    for task in args.tasks:
        try:
            print(f"\n{'='*60}")
            print(f"Training on {task.upper()}")
            print(f"{'='*60}")
            
            result = train_model_on_task(
                model_name=args.model,
                task=task,
                max_train_samples=args.train_samples,
                max_val_samples=args.val_samples,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=None,  # auto-detect
                val_ratio=0.1
            )
            
            print(f"\n✓ Training completed for {task}")
            print(f"  Val metrics: {result.get('val_metrics', {})}")
            print(f"  Test metrics: {result.get('test_metrics', {})}")
            
            # Optionally verify model can be loaded
            if args.verify_load:
                from evaluation.task_evaluator import TaskEvaluator
                model_path = result.get('model_path', '')
                if model_path and os.path.exists(model_path):
                    print(f"\n  Verifying model load from {model_path}...")
                    try:
                        evaluator = TaskEvaluator(
                            model_name=args.model,
                            batch_size=args.batch_size,
                            use_trained_model=True,
                            task=task
                        )
                        print(f"  ✓ Model loaded successfully")
                    except Exception as e:
                        print(f"  ✗ Model load failed: {e}")
            
            success_count += 1
            
        except Exception as e:
            print(f"\n✗ Training failed for {task}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Training check complete: {success_count}/{len(args.tasks)} tasks succeeded")
    print(f"{'='*60}")
    
    # Clean up trained models
    cleanup_trained_models()
    print("\n=== QUICK CHECK DONE ===")


if __name__ == "__main__":
    main()


