import argparse
from test_raw_clip import test_raw_clip
from eval import evaluate_model
def test_and_eval_clip(reference_filepath, test_filepath):
    clip_only_results = test_raw_clip(reference_filepath, test_filepath)
    evaluation_metrics = evaluate_model(clip_only_results)
    print("Evaluation Metrics: ", evaluation_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_filepath", type=str)
    parser.add_argument("--test_filepath", type=str)

    args = parser.parse_args()
    test_and_eval_clip(
        reference_filepath=args.reference_filepath,
        test_filepath=args.test_filepath
    )