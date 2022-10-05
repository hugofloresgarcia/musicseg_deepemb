from pathlib import Path
import json

import matplotlib.pyplot as plt

import musicsections


class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model_deepsim = musicsections.load_deepsim_model("models/deepsim")
        self.model_fewshot = musicsections.load_fewshot_model("models/fewshot")

    def predict(self, audio_path, output_path):
        """Run a single prediction on the model"""
        segmentations, _ = musicsections.segment_file(
            str(audio_path),
            deepsim_model=self.model_deepsim,
            fewshot_model=self.model_fewshot,
            min_duration=8,
            mu=0.5,
            gamma=0.5,
            beats_alg="madmom",
            beats_file=None,
        )

        musicsections.plot_segmentation(
            segmentations, figsize=(10, 3), display_seconds=True
        )

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path / "segmentation.png")

        def jsonify_segments(segments: tuple):
            segs = []
            times, labels = segments
            for time, label in zip(times, labels):
                segs.append({
                    "start": time[0],
                    "end": time[1],
                    "id": label,
                })
            return segs

        def jsonify_segmentation(segmentation: list):
            segs = []
            for seg in segmentation:
                segs.append(jsonify_segments(seg))
            return segs

        with open(output_path / "segmentation.json", "w") as f:
            json.dump(jsonify_segmentation(segmentations), f)

        return output_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=False)
    parser.add_argument("--wav_file_folder", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    predictor = Predictor()
    predictor.setup()
    if args.audio_path:
        predictor.predict(args.audio_path, args.output_path)
    elif args.wav_file_folder:
        for audio_path in Path(args.wav_file_folder).glob("*.wav"):
            predictor.predict(audio_path, Path(args.output_path) / audio_path.stem)
    else:
        raise ValueError("Please provide either --audio_path or --wav_file_folder")
