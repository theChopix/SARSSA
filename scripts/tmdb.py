import os
import time
from pathlib import Path

import pandas as pd
import requests

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
LANGUAGE = "en-US"
TARGET_POSTER_WIDTH = 200
TARGET_POSTER_HEIGHT = 296
BATCH_LOG_EVERY = 200
SLEEP_BETWEEN_REQUESTS = 2 / 30  # ~0.067s → max 30 req/s (2 req per film)

BASE_DIR = Path(__file__).resolve().parent
MOVIES_FILE = BASE_DIR / "ml-32m" / "movies.csv"
LINKS_FILE = BASE_DIR / "ml-32m" / "links.csv"
IMG_FOLDER = BASE_DIR / "img"
PLOTS_FILE = BASE_DIR / "plots.csv"

IMG_FOLDER.mkdir(parents=True, exist_ok=True)


def parse_poster_width(size_value: str) -> int | None:
    if size_value.startswith("w") and size_value[1:].isdigit():
        return int(size_value[1:])
    return None


def choose_poster_size() -> str:
    default_size = "w185"
    url = "https://api.themoviedb.org/3/configuration"
    params = {"api_key": TMDB_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"config request failed: {response.status_code}, fallback {default_size}")
            return default_size

        sizes = response.json().get("images", {}).get("poster_sizes", [])
        fixed_sizes = []
        for value in sizes:
            width = parse_poster_width(value)
            if width is not None:
                fixed_sizes.append((value, width))

        if not fixed_sizes:
            print(f"no fixed poster sizes found, fallback {default_size}")
            return default_size

        selected_size, selected_width = min(
            fixed_sizes,
            key=lambda item: abs(item[1] - TARGET_POSTER_WIDTH),
        )
        print(
            "poster size selected: "
            f"{selected_size} (target {TARGET_POSTER_WIDTH}x{TARGET_POSTER_HEIGHT})"
        )
        return selected_size
    except requests.RequestException as exc:
        print(f"config request error: {exc}, fallback {default_size}")
        return default_size


POSTER_SIZE = choose_poster_size()


def load_already_processed() -> set[int]:
    """Reads movieIds already present in plots.csv – those will be skipped."""
    if not PLOTS_FILE.exists():
        return set()
    try:
        existing = pd.read_csv(PLOTS_FILE, usecols=["movieId"])
        ids = set(existing["movieId"].astype(int).tolist())
        print(f"resume: {len(ids):,} movies already processed, skipping")
        return ids
    except Exception as exc:
        print(f"warning: could not load {PLOTS_FILE}: {exc}")
        return set()


def download_poster(poster_url: str, movie_lens_id: int) -> Path | None:
    file_path = IMG_FOLDER / f"{int(movie_lens_id)}.jpg"
    if file_path.exists():
        return file_path

    try:
        response = requests.get(poster_url, timeout=15)
        if response.status_code == 200:
            file_path.write_bytes(response.content)
            return file_path
        print(f"poster download failed movieId={movie_lens_id} status={response.status_code}")
    except requests.RequestException as exc:
        print(f"poster download error movieId={movie_lens_id}: {exc}")
    return None


def get_tmdb_plot_and_poster_path(tmdb_id: int) -> tuple[str, str | None]:
    if pd.isna(tmdb_id) or int(tmdb_id) == 0:
        return "", None

    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
    params = {"api_key": TMDB_API_KEY, "language": LANGUAGE}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 429:
            print(f"rate limited for tmdbId={tmdb_id}, retrying in 8s")
            time.sleep(8)
            return get_tmdb_plot_and_poster_path(tmdb_id)
        if response.status_code != 200:
            print(f"movie request failed tmdbId={tmdb_id} status={response.status_code}")
            time.sleep(1)
            return "", None

        data = response.json()
        plot = data.get("overview", "") or ""
        poster_path = data.get("poster_path")
        full_poster_url = (
            f"https://image.tmdb.org/t/p/{POSTER_SIZE}{poster_path}" if poster_path else None
        )

        time.sleep(SLEEP_BETWEEN_REQUESTS)  # sleep after request 1 (plot)
        return plot, full_poster_url
    except requests.RequestException as exc:
        print(f"movie request error tmdbId={tmdb_id}: {exc}")
        time.sleep(2)
        return "", None


def append_rows_to_csv(rows: list[dict]) -> None:
    """Appends new rows to plots.csv; writes header only if file does not exist yet."""
    df = pd.DataFrame(rows, columns=["movieId", "plot"])
    write_header = not PLOTS_FILE.exists()
    df.to_csv(PLOTS_FILE, mode="a", index=False, header=write_header)


FLUSH_EVERY = 500  # how often to flush to disk


def main() -> None:
    print("loading MovieLens data")
    movies = pd.read_csv(MOVIES_FILE)
    links = pd.read_csv(LINKS_FILE)
    merged = movies.merge(links, on="movieId", how="left")

    already_done = load_already_processed()
    todo = merged[~merged["movieId"].isin(already_done)]

    total = len(merged)
    remaining = len(todo)
    print(f"total movies: {total:,} | remaining: {remaining:,}")

    if remaining == 0:
        print("all movies already processed, exiting")
        return

    start_time = time.time()
    buffer: list[dict] = []

    for i, (_, row) in enumerate(todo.iterrows()):
        movie_id = int(row["movieId"])
        tmdb_id = row["tmdbId"]

        plot, poster_url = get_tmdb_plot_and_poster_path(tmdb_id)
        if poster_url:
            download_poster(poster_url, movie_id)
            time.sleep(SLEEP_BETWEEN_REQUESTS)  # sleep after request 2 (poster)

        buffer.append({"movieId": movie_id, "plot": plot})

        # periodic flush to disk
        if len(buffer) >= FLUSH_EVERY:
            append_rows_to_csv(buffer)
            buffer.clear()

        if (i + 1) % BATCH_LOG_EVERY == 0 or (i + 1) == remaining:
            elapsed_min = (time.time() - start_time) / 60
            rate = (i + 1) / elapsed_min if elapsed_min > 0 else 0
            print(
                f"progress {i + 1:,}/{remaining:,} elapsed={elapsed_min:.1f}m rate={rate:.0f} movies/min"
            )

    # flush remainder
    if buffer:
        append_rows_to_csv(buffer)

    total_minutes = (time.time() - start_time) / 60
    print(f"done in {total_minutes:.1f}m")
    print(f"images: {IMG_FOLDER}")
    print(f"plots: {PLOTS_FILE}")


if __name__ == "__main__":
    main()
