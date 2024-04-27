import argparse
import random
import regex
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Data sampling')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--sample_size', type=int, default=100, help='sample size for each genre')
    parser.add_argument('--regex', type=str, default='.*', help='regex for filtering files')
    parser.add_argument('--genres', type=str, default=None, help='sample from specific genres, separated by comma')
    return parser.parse_args()

def main(args: argparse.Namespace):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # list folders under data dir
    genres = [dir if os.path.isdir(os.path.join(args.data_dir, dir)) else None for dir in os.listdir(args.data_dir)]
    if args.genres is not None:
        genres = [genre for genre in genres if genre in args.genres.split(",")]
    cnt = 0
    data = {}
    for genre in genres:
        if genre is not None:
            all_files = os.listdir(os.path.join(args.data_dir, genre))
            # regex filter
            all_files = [file for file in all_files if regex.match(args.regex, file)]
            sample_files = random.sample(all_files, min(args.sample_size, len(all_files)))
            # create symlink
            for file in sample_files:
                if os.path.exists(os.path.join(args.output_dir, f"image-{cnt}." + file.split(".")[-1] if file.split(".")[-1] != "" else "jpg")):
                    os.remove(os.path.join(args.output_dir, f"image-{cnt}." + file.split(".")[-1] if file.split(".")[-1] != "" else "jpg"))
                os.symlink(os.path.join(args.data_dir, genre, file), os.path.join(args.output_dir, f"image-{cnt}." + file.split(".")[-1] if file.split(".")[-1] != "" else "jpg"))
                data[genre] = data.get(genre, []) + [(f"image-{cnt}", file)]
                cnt += 1
    # write meta data
    with open(os.path.join(args.output_dir, 'meta.txt'), 'w') as f:
        for genre in data:
            for item in data[genre]:
                f.write(f"{genre} {item[0]} {item[1]}\n")

if __name__ == '__main__':
    args = parse_args()
    main(args)