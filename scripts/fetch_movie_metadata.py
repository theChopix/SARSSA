"""Standalone script to fetch movie metadata from TMDb for MovieLens items.

Reads MovieLens links.csv / movies.csv, fetches rich metadata (plot overview,
director, cast, poster) from TMDb, optionally downloads poster images, and
writes a single JSON file that downstream pipeline plugins consume.

Requires a free TMDb API key — register at https://www.themoviedb.org/ and
set the TMDB_API_KEY environment variable (or pass --api-key).

Usage examples:
    # Fetch all items
    python scripts/fetch_movie_metadata.py \
        --links data/movieLens/ml-32m/links.csv \
        --movies data/movieLens/ml-32m/movies.csv \
        --output data/movieLens/item_descriptions.json

    # Fetch only items that survived filtering + download poster images
    python scripts/fetch_movie_metadata.py \
        --links data/movieLens/ml-32m/links.csv \
        --movies data/movieLens/ml-32m/movies.csv \
        --output data/movieLens/item_descriptions.json \
        --items data/movieLens/filtered_items.npy \
        --download-posters --poster-dir data/movieLens/img
"""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAVE_INTERVAL: int = 100
MAX_RETRIES: int = 5
BACKOFF_BASE: float = 2.0
TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
TMDB_LANGUAGE: str = "en-US"
DEFAULT_POSTER_SIZE: str = "w185"
TARGET_POSTER_WIDTH: int = 200
SLEEP_BETWEEN_REQUESTS: float = 2 / 30  # ~0.067s → stay under 30 req/s


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_links(links_path: str) -> dict[str, dict[str, str]]:
    """Load links.csv and return a mapping from movieId to external IDs.

    Args:
        links_path: Path to the MovieLens links.csv file.

    Returns:
        dict mapping movieId (str) to {"imdbId": str, "tmdbId": str}.

    Raises:
        FileNotFoundError: If links_path does not exist.
    """
    if not os.path.exists(links_path):
        raise FileNotFoundError(f"links.csv not found: {links_path}")

    links: dict[str, dict[str, str]] = {}
    with open(links_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = row["movieId"].strip()
            imdb_id = row.get("imdbId", "").strip()
            tmdb_id = row.get("tmdbId", "").strip()
            links[movie_id] = {"imdbId": imdb_id, "tmdbId": tmdb_id}
    logger.info("Loaded %d links from %s", len(links), links_path)
    return links


def load_movies(movies_path: str) -> dict[str, dict[str, str]]:
    """Load movies.csv and return a mapping from movieId to title + genres.

    Args:
        movies_path: Path to the MovieLens movies.csv file.

    Returns:
        dict mapping movieId (str) to {"title": str, "genres": str}.

    Raises:
        FileNotFoundError: If movies_path does not exist.
    """
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"movies.csv not found: {movies_path}")

    movies: dict[str, dict[str, str]] = {}
    with open(movies_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = row["movieId"].strip()
            movies[movie_id] = {
                "title": row["title"].strip(),
                "genres": row["genres"].strip(),
            }
    logger.info("Loaded %d movies from %s", len(movies), movies_path)
    return movies


def load_item_filter(items_path: str) -> set[str]:
    """Load an items.npy file and return a set of item ID strings.

    Args:
        items_path: Path to an .npy file containing item IDs.

    Returns:
        Set of item ID strings.

    Raises:
        FileNotFoundError: If items_path does not exist.
    """
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"items file not found: {items_path}")

    items = np.load(items_path, allow_pickle=True)
    item_set = {str(i) for i in items}
    logger.info("Loaded %d items from %s for filtering", len(item_set), items_path)
    return item_set


def load_existing_output(output_path: str) -> dict[str, dict]:
    """Load an existing output JSON file for resumable fetching.

    Args:
        output_path: Path to the output JSON file.

    Returns:
        dict of already-fetched entries, or empty dict if file does not exist.
    """
    if os.path.exists(output_path):
        with open(output_path) as f:
            data = json.load(f)
        logger.info("Loaded %d existing entries from %s", len(data), output_path)
        return data
    return {}


def save_output(output_path: str, data: dict[str, dict]) -> None:
    """Atomically save the output JSON file.

    Writes to a temporary file first, then renames to avoid corruption
    if the process is interrupted mid-write.

    Args:
        output_path: Destination path.
        data: The full metadata dict to save.
    """
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)


# ---------------------------------------------------------------------------
# TMDb API helpers
# ---------------------------------------------------------------------------


def choose_poster_size(api_key: str) -> str:
    """Query TMDb configuration API to pick the best poster size.

    Selects the fixed-width size closest to TARGET_POSTER_WIDTH.

    Args:
        api_key: TMDb API key.

    Returns:
        Poster size string (e.g. "w185").
    """
    url = f"{TMDB_BASE_URL}/configuration"
    try:
        resp = requests.get(url, params={"api_key": api_key}, timeout=10)
        if resp.status_code != 200:
            logger.warning(
                "TMDb config request failed (%d), using default %s",
                resp.status_code,
                DEFAULT_POSTER_SIZE,
            )
            return DEFAULT_POSTER_SIZE

        sizes = resp.json().get("images", {}).get("poster_sizes", [])
        fixed_sizes: list[tuple[str, int]] = []
        for s in sizes:
            if s.startswith("w") and s[1:].isdigit():
                fixed_sizes.append((s, int(s[1:])))

        if not fixed_sizes:
            return DEFAULT_POSTER_SIZE

        best, best_w = min(fixed_sizes, key=lambda x: abs(x[1] - TARGET_POSTER_WIDTH))
        logger.info("Selected poster size: %s (width=%d)", best, best_w)
        return best
    except requests.RequestException as e:
        logger.warning("TMDb config error: %s, using default %s", e, DEFAULT_POSTER_SIZE)
        return DEFAULT_POSTER_SIZE


def tmdb_request(url: str, api_key: str) -> dict | None:
    """Make a single TMDb API request with retry + exponential backoff.

    Args:
        url: The full API URL.
        api_key: TMDb API key.

    Returns:
        Parsed JSON dict, or None on failure.
    """
    params = {"api_key": api_key, "language": TMDB_LANGUAGE}

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning("TMDb 429 rate limit, backing off %.1fs", wait)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            logger.warning("TMDb %d for %s", resp.status_code, url)
            time.sleep(1)
            return None
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** (attempt + 1)
            logger.warning("TMDb request error: %s, retrying in %.1fs", e, wait)
            time.sleep(wait)
    logger.error("TMDb request failed after %d retries: %s", MAX_RETRIES, url)
    return None


def fetch_movie_details(tmdb_id: str, api_key: str) -> tuple[str, str | None, int | None]:
    """Fetch plot overview, poster path, and release year from TMDb.

    Args:
        tmdb_id: TMDb movie ID.
        api_key: TMDb API key.

    Returns:
        Tuple of (description, poster_path_or_None, year_or_None).
    """
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    data = tmdb_request(url, api_key)
    if data is None:
        return "", None, None

    description = data.get("overview", "") or ""
    poster_path = data.get("poster_path")
    year = _parse_year(data.get("release_date", ""))
    return description, poster_path, year


def fetch_movie_credits(tmdb_id: str, api_key: str) -> tuple[str, list[str]]:
    """Fetch director and top cast from TMDb credits endpoint.

    Args:
        tmdb_id: TMDb movie ID.
        api_key: TMDb API key.

    Returns:
        Tuple of (director_string, list_of_cast_names).
    """
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}/credits"
    data = tmdb_request(url, api_key)
    if data is None:
        return "", []

    crew = data.get("crew", [])
    directors = [p["name"] for p in crew if p.get("job") == "Director"]
    director = ", ".join(directors)

    cast_list = data.get("cast", [])
    cast_names = [p["name"] for p in cast_list[:5]]

    return director, cast_names


def fetch_movie_keywords(tmdb_id: str, api_key: str) -> list[str]:
    """Fetch keyword tags from TMDb keywords endpoint.

    Args:
        tmdb_id: TMDb movie ID.
        api_key: TMDb API key.

    Returns:
        List of keyword name strings.
    """
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}/keywords"
    data = tmdb_request(url, api_key)
    if data is None:
        return []

    return [kw["name"] for kw in data.get("keywords", [])]


def _parse_year(release_date: str) -> int | None:
    """Extract year from a TMDb release_date string (YYYY-MM-DD).

    Args:
        release_date: Date string in YYYY-MM-DD format, or empty.

    Returns:
        Year as int, or None if parsing fails.
    """
    if release_date and len(release_date) >= 4:
        try:
            return int(release_date[:4])
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Poster download
# ---------------------------------------------------------------------------


def download_poster(poster_url: str, movie_id: str, poster_dir: Path) -> Path | None:
    """Download a poster image to the local poster directory.

    Skips download if the file already exists.

    Args:
        poster_url: Full URL to the poster image.
        movie_id: MovieLens movieId (used as filename).
        poster_dir: Directory to save posters to.

    Returns:
        Path to the saved file, or None on failure.
    """
    file_path = poster_dir / f"{movie_id}.jpg"
    if file_path.exists():
        return file_path

    try:
        resp = requests.get(poster_url, timeout=15)
        if resp.status_code == 200:
            file_path.write_bytes(resp.content)
            return file_path
        logger.debug(
            "Poster download failed movieId=%s status=%d",
            movie_id,
            resp.status_code,
        )
    except requests.RequestException as e:
        logger.debug("Poster download error movieId=%s: %s", movie_id, e)
    return None


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def build_entry(
    movie_id: str,
    description: str,
    director: str,
    cast: list[str],
    keywords: list[str],
    year: int | None,
    image_url: str,
    movies_data: dict[str, dict[str, str]],
) -> dict:
    """Merge fetched metadata with local movies.csv data into a final entry.

    Args:
        movie_id: The MovieLens movieId.
        description: Plot overview text.
        director: Director name(s).
        cast: List of top cast member names.
        keywords: List of TMDb keyword strings.
        year: Release year or None.
        image_url: Full poster image URL.
        movies_data: Mapping from movieId to {title, genres} from movies.csv.

    Returns:
        Complete metadata entry dict.
    """
    local = movies_data.get(movie_id, {})
    title = local.get("title", "")
    genres_raw = local.get("genres", "")
    genres = genres_raw.split("|") if genres_raw and genres_raw != "(no genres listed)" else []

    return {
        "title": title,
        "year": year,
        "description": description,
        "director": director,
        "cast": cast,
        "keywords": keywords,
        "genres": genres,
        "image_url": image_url,
    }


def fetch_all(
    links: dict[str, dict[str, str]],
    movies_data: dict[str, dict[str, str]],
    output_path: str,
    api_key: str,
    item_filter: set[str] | None,
    poster_size: str,
    download_posters: bool,
    poster_dir: Path | None,
) -> None:
    """Fetch metadata for all movies and write to output file.

    For each movie, makes three TMDb API calls (details, credits,
    keywords) and optionally downloads the poster image.

    Args:
        links: Mapping from movieId to {imdbId, tmdbId}.
        movies_data: Mapping from movieId to {title, genres}.
        output_path: Path to the output JSON file.
        api_key: TMDb API key.
        item_filter: Optional set of movieIds to restrict fetching to.
        poster_size: TMDb poster size string (e.g. "w185").
        download_posters: Whether to download poster images.
        poster_dir: Directory to save poster images to.
    """
    # Determine which items to fetch
    movie_ids = sorted(links.keys())
    if item_filter is not None:
        movie_ids = [mid for mid in movie_ids if mid in item_filter]
        logger.info("Filtered to %d items matching --items file", len(movie_ids))

    # Resume from existing output
    result = load_existing_output(output_path)
    already_done = set(result.keys())
    to_fetch = [mid for mid in movie_ids if mid not in already_done]

    logger.info(
        "Total: %d, already fetched: %d, remaining: %d",
        len(movie_ids),
        len(already_done),
        len(to_fetch),
    )

    if not to_fetch:
        logger.info("Nothing to fetch — all items already present in output.")
        return

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if download_posters and poster_dir is not None:
        poster_dir.mkdir(parents=True, exist_ok=True)

    fetched_count = 0
    failed_count = 0
    start_time = time.time()

    for i, movie_id in enumerate(tqdm(to_fetch, desc="Fetching metadata")):
        link = links[movie_id]
        tmdb_id = link.get("tmdbId", "")

        if not tmdb_id:
            failed_count += 1
            continue

        # Fetch details (plot, poster path, year)
        description, poster_path, year = fetch_movie_details(tmdb_id, api_key)

        # Fetch credits (director, cast)
        director, cast_names = fetch_movie_credits(tmdb_id, api_key)

        # Fetch keywords
        keywords = fetch_movie_keywords(tmdb_id, api_key)

        # Build poster URL
        image_url = ""
        if poster_path:
            image_url = f"https://image.tmdb.org/t/p/{poster_size}{poster_path}"

            # Optionally download the poster image
            if download_posters and poster_dir is not None:
                download_poster(image_url, movie_id, poster_dir)
                time.sleep(SLEEP_BETWEEN_REQUESTS)

        result[movie_id] = build_entry(
            movie_id,
            description,
            director,
            cast_names,
            keywords,
            year,
            image_url,
            movies_data,
        )
        fetched_count += 1

        # Periodic save + progress log
        if (i + 1) % SAVE_INTERVAL == 0:
            save_output(output_path, result)
            elapsed_min = (time.time() - start_time) / 60
            rate = (i + 1) / elapsed_min if elapsed_min > 0 else 0
            logger.info(
                "Progress: %d/%d fetched, %d failed (%.0f movies/min)",
                fetched_count,
                len(to_fetch),
                failed_count,
                rate,
            )

    # Final save
    save_output(output_path, result)
    total_min = (time.time() - start_time) / 60
    logger.info(
        "Done in %.1f min. Fetched: %d, failed: %d, total in output: %d",
        total_min,
        fetched_count,
        failed_count,
        len(result),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fetch movie metadata from TMDb for MovieLens items.",
    )
    parser.add_argument(
        "--links",
        required=True,
        help="Path to MovieLens links.csv",
    )
    parser.add_argument(
        "--movies",
        required=True,
        help="Path to MovieLens movies.csv",
    )
    parser.add_argument(
        "--output",
        default="data/movieLens/item_descriptions.json",
        help="Output JSON path (default: data/movieLens/item_descriptions.json)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="TMDb API key. Can also be set via TMDB_API_KEY env var.",
    )
    parser.add_argument(
        "--items",
        default=None,
        help="Optional path to items.npy to restrict fetching to surviving items only",
    )
    parser.add_argument(
        "--download-posters",
        action="store_true",
        default=False,
        help="Download poster images locally",
    )
    parser.add_argument(
        "--poster-dir",
        default="data/movieLens/img",
        help="Directory to save poster images (default: data/movieLens/img)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the metadata fetching script.

    Args:
        argv: Optional argument list for testing (defaults to sys.argv).
    """
    args = parse_args(argv)

    # Resolve API key from arg or env
    api_key = args.api_key or os.environ.get("TMDB_API_KEY")
    if not api_key:
        logger.error(
            "TMDb API key is required. Pass --api-key or set TMDB_API_KEY env var.\n"
            "Get a free key at https://www.themoviedb.org/ → Settings → API"
        )
        raise SystemExit(1)

    # Select optimal poster size from TMDb configuration
    poster_size = choose_poster_size(api_key)

    # Load input data
    links = load_links(args.links)
    movies_data = load_movies(args.movies)

    # Optional item filter
    item_filter: set[str] | None = None
    if args.items:
        item_filter = load_item_filter(args.items)

    # Poster download setup
    poster_dir = Path(args.poster_dir) if args.download_posters else None

    # Fetch
    fetch_all(
        links=links,
        movies_data=movies_data,
        output_path=args.output,
        api_key=api_key,
        item_filter=item_filter,
        poster_size=poster_size,
        download_posters=args.download_posters,
        poster_dir=poster_dir,
    )


if __name__ == "__main__":
    main()
