"""Build one shard of a 640x640 TartanAir V2 dataset on Kaggle: frames, IMU, barometer.

TartanAir V2 has 74 environments. Their front-left frames alone are 833 GB of
640x640 PNG. A Kaggle notebook saves 20 GB of output, so a dataset that covers every
environment has to take a thin, even slice of each one. This reads the published zips
by HTTP range: the central directory once, then one request per kept frame, so a
35 GB zip costs only the frames taken from it.

Selection (``make_plan``) is the same in every shard:
  - every environment gets the same number of frames (water-filling: one too small
    for its share gives all it has, and the rest is shared among the others);
  - inside an environment, trajectories share the quota by length, and the frames
    are evenly spaced with a seeded phase. At about one frame per second the kept
    frames are different views, where 10 Hz neighbours are near copies;
  - no frame closer than ``margin`` frames to either end: a 128-sample IMU window
    centred on it would run off the trajectory, and pairing rejects it.
Environments are packed into ``num_shards`` shards of about equal bytes; each Kaggle
run builds one shard, and its output is one Kaggle dataset.

Per trajectory a shard holds
  image_lcam_front/<index>_lcam_front.<ext>  kept frames, original index in the name
  imu/*.npy, imu/parameter.yaml              the full 100 Hz IMU as published, except
  imu/cam_time.npy                           the kept frames' times only: row i is the
                                             i-th kept frame, as the loader pairs them
  imu/baro.npy, imu/baro.json                synthetic barometer (``synthetic_baro``)
  pose_lcam_front.txt, frames.npy            pose rows and original indices of the kept frames
``imu/`` is written last and renamed into place, so a trajectory cut off by a lost
session is invisible to the loader; the next run redoes it and skips its finished frames.

TartanAir has no barometer. ``baro.npy`` is the clean static pressure that the ISA
troposphere gives for the true height (NED, up = -z of ``pos_global``) above a
per-trajectory site altitude, at the IMU timestamps. Noise, bias drift, lag and a
lower sample rate belong to the corruption stage, as they do for the IMU.

    python3 tools/build_tartanair640.py --shard 0 --num-shards 3 --out /kaggle/working/tartanair640
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import math
import os
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
from PIL import Image

SOURCES = {
    "huggingface": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/",
    "airlab": "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanair_v2/",
}
ENVIRONMENTS = (
    "AbandonedCable", "AbandonedFactory", "AbandonedFactory2", "AbandonedSchool", "AmericanDiner",
    "AmusementPark", "AncientTowns", "Antiquity3D", "Apocalyptic", "ArchVizTinyHouseDay",
    "ArchVizTinyHouseNight", "BrushifyMoon", "CarWelding", "CastleFortress", "CoalMine",
    "ConstructionSite", "CountryHouse", "CyberPunkDowntown", "Cyberpunk", "DesertGasStation",
    "Downtown", "EndofTheWorld", "FactoryWeather", "Fantasy", "ForestEnv", "Gascola",
    "GothicIsland", "GreatMarsh", "HQWesternSaloon", "HongKong", "Hospital", "House",
    "IndustrialHangar", "JapaneseAlley", "JapaneseCity", "MiddleEast", "ModUrbanCity",
    "ModernCityDowntown", "ModularNeighborhood", "ModularNeighborhoodIntExt", "NordicHarbor",
    "Ocean", "Office", "OldBrickHouseDay", "OldBrickHouseNight", "OldIndustrialCity",
    "OldScandinavia", "OldTownFall", "OldTownNight", "OldTownSummer", "OldTownWinter",
    "PolarSciFi", "Prison", "Restaurant", "RetroOffice", "Rome", "Ruins", "SeasideTown",
    "SeasonalForestAutumn", "SeasonalForestSpring", "SeasonalForestSummerNight",
    "SeasonalForestWinter", "SeasonalForestWinterNight", "Sewerage", "ShoreCaves", "Slaughter",
    "SoulCity", "Supermarket", "TerrainBlending", "UrbanConstruction", "VictorianStreet",
    "WaterMillDay", "WaterMillNight", "WesternDesertTown",
)
DIFFICULTIES = ("Data_easy", "Data_hard")
CAMERA = "lcam_front"
FRAME_SIZE = (640, 640)
# Stored bytes per source PNG byte, measured on 17 TartanAir frames: WebP lossless
# 0.76 (decodes as fast as PNG), JPEG q98 4:4:4 0.67 at 45.9 dB. ``calibrate`` measures
# the real ratio before planning; these are only its fallback.
FORMATS = {"png": ("png", 1.0), "webp": ("webp", 0.76), "jpeg98": ("jpg", 0.67)}
# A 128-sample window at 100 Hz spans 1.27 s: frames within 6.35 frames of an end
# can never be centred, so they are never worth downloading.
DEFAULT_MARGIN = 7
# ISA troposphere: p = p0 * (1 - L h / T0) ** (g M / (R L)).
ISA_T0, ISA_LAPSE = 288.15, 0.0065
ISA_EXPONENT = 9.80665 * 0.0289644 / (8.3144598 * ISA_LAPSE)
DONE = "build_done.json"


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


@dataclass(frozen=True)
class Member:
    """Where one file sits inside a zip: enough to fetch it with one range request."""

    name: str
    offset: int
    compressed: int
    size: int
    method: int
    crc: int


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


class RemoteZip:
    """A zip read by byte range, over HTTP(S) or from a local path.

    Hugging Face answers with a redirect to a signed CDN link that lasts about an
    hour. The link is resolved once and reused, so huggingface.co sees a few requests
    per zip rather than one per frame; it is resolved again when it expires.
    """

    REFRESH_SECONDS = 40 * 60

    def __init__(self, location: str, token: str | None = None, retries: int = 6):
        self.location, self.token, self.retries = location, token, retries
        self.local = not location.startswith(("http://", "https://"))
        self._target: str | None = None
        self._resolved = 0.0
        self.size = os.path.getsize(location) if self.local else self._resolve()

    def _resolve(self) -> int:
        headers = {"Range": "bytes=0-0"}
        if self.token and "huggingface.co" in self.location:
            headers["Authorization"] = f"Bearer {self.token}"
        opener = urllib.request.build_opener(_NoRedirect)
        target = self.location
        try:
            with opener.open(urllib.request.Request(target, headers=headers), timeout=60) as response:
                content_range = response.headers["Content-Range"]
        except urllib.error.HTTPError as error:
            if error.code not in (301, 302, 303, 307, 308):
                raise
            # The token stays with huggingface.co; the signed link needs none.
            target = urljoin(target, error.headers["Location"])
            with urllib.request.urlopen(
                urllib.request.Request(target, headers={"Range": "bytes=0-0"}), timeout=60
            ) as response:
                content_range = response.headers["Content-Range"]
        self._target, self._resolved = target, time.monotonic()
        return int(content_range.rsplit("/", 1)[1])

    def read(self, start: int, end: int) -> bytes:
        """Bytes [start, end)."""
        end = min(end, self.size)
        if end <= start:
            return b""
        if self.local:
            with open(self.location, "rb") as handle:
                handle.seek(start)
                return handle.read(end - start)
        error: object = None
        for attempt in range(self.retries):
            try:
                if self._target is None or time.monotonic() - self._resolved > self.REFRESH_SECONDS:
                    self._resolve()
                request = urllib.request.Request(self._target, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:  # a 200 would be the whole zip
                        raise IOError(f"{self.location}: server ignored the byte range ({response.status})")
                    data = response.read()
                if len(data) == end - start:
                    return data
                error = f"short read {len(data)}/{end - start}"
            except urllib.error.HTTPError as exc:
                error = exc
                if exc.code in (401, 403, 410):
                    self._target = None  # the signed link expired
                elif exc.code not in (408, 429, 500, 502, 503, 504):
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
                error = exc
            time.sleep(min(60.0, 2.0 ** attempt))
        raise IOError(f"{self.location} bytes [{start}, {end}): {error}")

    def members(self) -> list[Member]:
        with zipfile.ZipFile(_RangeFile(self)) as archive:
            return [
                Member(info.filename, info.header_offset, info.compress_size, info.file_size,
                       info.compress_type, info.CRC)
                for info in archive.infolist() if not info.is_dir()
            ]

    def member(self, member: Member) -> bytes:
        # Local header (30 bytes) + name + extra field, then the data. The local extra
        # field may differ from the central one, so read a little more and trim.
        blob = self.read(member.offset, member.offset + 30 + 1024 + member.compressed)
        if blob[:4] != b"PK\x03\x04":
            raise IOError(f"{member.name}: no local file header at {member.offset}")
        name_length, extra_length = struct.unpack("<HH", blob[26:30])
        begin = 30 + name_length + extra_length
        body = blob[begin:begin + member.compressed]
        if len(body) < member.compressed:
            body += self.read(member.offset + len(blob), member.offset + begin + member.compressed)
        if member.method == zipfile.ZIP_STORED:
            data = body
        elif member.method == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(body, -15)
        else:
            raise ValueError(f"{member.name}: unsupported compression {member.method}")
        if zlib.crc32(data) != member.crc:
            raise IOError(f"{member.name}: CRC mismatch")
        return data


class _RangeFile(io.RawIOBase):
    """Seekable file over a RemoteZip, so ``zipfile`` can read the central directory."""

    def __init__(self, remote: RemoteZip):
        self.remote, self.position = remote, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.position, 2: self.remote.size}[whence]
        self.position = base + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        end = self.remote.size if size is None or size < 0 else self.position + size
        data = self.remote.read(self.position, end)
        self.position += len(data)
        return data


@dataclass
class TrajectoryEntry:
    environment: str
    difficulty: str
    trajectory: str
    frames: list[tuple[int, Member]] = field(default_factory=list)  # sorted by frame index
    pose: Member | None = None

    @property
    def key(self) -> str:
        return f"{self.environment}/{self.difficulty}/{self.trajectory}"

    @property
    def image_zip(self) -> str:
        return f"{self.environment}/{self.difficulty}/image_{CAMERA}.zip"

    @property
    def imu_zip(self) -> str:
        return f"{self.environment}/{self.difficulty}/imu.zip"

    @property
    def mean_png_bytes(self) -> float:
        return float(np.mean([member.compressed for _, member in self.frames])) if self.frames else 0.0


def _location(source: str, relative: str) -> str:
    return source + relative if source.startswith(("http://", "https://")) else str(Path(source) / relative)


def list_catalog(
    source: str,
    environments: tuple[str, ...] | list[str] = ENVIRONMENTS,
    difficulties: tuple[str, ...] | list[str] = DIFFICULTIES,
    *,
    token: str | None = None,
    threads: int = 16,
) -> tuple[dict[str, TrajectoryEntry], list[str]]:
    """Every trajectory's frames, read from the image zips' central directories."""

    def one(pair: tuple[str, str]) -> tuple[list[TrajectoryEntry], str | None]:
        environment, difficulty = pair
        relative = f"{environment}/{difficulty}/image_{CAMERA}.zip"
        try:
            members = RemoteZip(_location(source, relative), token).members()
        except (FileNotFoundError, urllib.error.HTTPError) as error:
            return [], f"{relative}: {error}"
        entries: dict[str, TrajectoryEntry] = {}
        for member in members:
            parts = member.name.split("/")
            if len(parts) < 4 or parts[0] != environment or parts[1] != difficulty:
                continue
            entry = entries.setdefault(parts[2], TrajectoryEntry(environment, difficulty, parts[2]))
            if parts[3] == f"pose_{CAMERA}.txt" and len(parts) == 4:
                entry.pose = member
            elif parts[3] == f"image_{CAMERA}" and len(parts) == 5 and parts[4].endswith(".png"):
                entry.frames.append((int(parts[4].split("_", 1)[0]), member))
        for entry in entries.values():
            entry.frames.sort(key=lambda item: item[0])
        return [entry for entry in entries.values() if entry.frames], None

    pairs = [(environment, difficulty) for environment in environments for difficulty in difficulties]
    catalog: dict[str, TrajectoryEntry] = {}
    missing: list[str] = []
    with ThreadPoolExecutor(threads) as executor:
        for entries, problem in executor.map(one, pairs):
            if problem:
                missing.append(problem)
            for entry in entries:
                catalog[entry.key] = entry
    return dict(sorted(catalog.items())), missing


def encode_frame(png: bytes, fmt: str, frame_size: tuple[int, int] = FRAME_SIZE) -> bytes:
    with Image.open(io.BytesIO(png)) as image:
        if image.size != (frame_size[1], frame_size[0]):
            raise ValueError(f"frame is {image.size}, expected {frame_size[1]}x{frame_size[0]}")
        if fmt == "png":
            return png
        image = image.convert("RGB")
        buffer = io.BytesIO()
        if fmt == "webp":
            image.save(buffer, "WEBP", lossless=True, method=4)
        elif fmt == "jpeg98":
            image.save(buffer, "JPEG", quality=98, subsampling=0)
        else:
            raise ValueError(f"unknown format {fmt!r}; choose one of {sorted(FORMATS)}")
        return buffer.getvalue()


def calibrate(
    catalog: dict[str, TrajectoryEntry],
    source: str,
    fmt: str,
    *,
    samples: int = 24,
    seed: int = 0,
    token: str | None = None,
    frame_size: tuple[int, int] = FRAME_SIZE,
) -> float:
    """Stored bytes per source PNG byte, measured on a fixed sample of frames.

    Rounded up to 0.01 so that every shard, which measures again, plans the same.
    """
    if fmt == "png":
        return 1.0
    keys = sorted(catalog)
    rng = np.random.default_rng(_seed(seed, "calibrate"))
    chosen = [catalog[keys[index]] for index in rng.choice(len(keys), min(samples, len(keys)), replace=False)]

    def one(entry: TrajectoryEntry) -> tuple[int, int]:
        member = entry.frames[len(entry.frames) // 2][1]
        png = RemoteZip(_location(source, entry.image_zip), token).member(member)
        return len(png), len(encode_frame(png, fmt, frame_size))

    with ThreadPoolExecutor(8) as executor:
        sizes = list(executor.map(one, chosen))
    return math.ceil(100 * sum(stored for _, stored in sizes) / sum(png for png, _ in sizes)) / 100


def make_plan(
    catalog: dict[str, TrajectoryEntry],
    *,
    budget_bytes: float,
    ratio: float,
    margin: int = DEFAULT_MARGIN,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Positions (into each trajectory's sorted frame list) to keep, within ``budget_bytes``."""
    usable = {key: max(0, len(entry.frames) - 2 * margin) for key, entry in catalog.items()}
    by_environment: dict[str, list[str]] = {}
    for key, entry in catalog.items():
        if usable[key]:
            by_environment.setdefault(entry.environment, []).append(key)
    available = {environment: sum(usable[key] for key in keys) for environment, keys in by_environment.items()}
    frame_bytes = {
        environment: ratio * sum(catalog[key].mean_png_bytes * usable[key] for key in keys) / available[environment]
        for environment, keys in by_environment.items()
    }

    def quotas(total: int) -> dict[str, float]:
        # Water-filling: equal shares; an environment that cannot fill its share gives all it has.
        result: dict[str, float] = {}
        remaining = float(total)
        pending = sorted(by_environment, key=lambda environment: (available[environment], environment))
        while pending:
            share = remaining / len(pending)
            if available[pending[0]] <= share:
                environment = pending.pop(0)
                result[environment] = available[environment]
                remaining -= available[environment]
            else:
                result.update({environment: share for environment in pending})
                break
        return result

    def cost(total: int) -> float:
        return sum(count * frame_bytes[environment] for environment, count in quotas(total).items())

    low, high = 0, sum(available.values())
    while low < high:  # largest total frame count whose estimated bytes fit the budget
        middle = (low + high + 1) // 2
        low, high = (middle, high) if cost(middle) <= budget_bytes else (low, middle - 1)

    plan: dict[str, list[int]] = {}
    for environment, quota in quotas(low).items():
        keys = sorted(by_environment[environment])
        exact = [quota * usable[key] / available[environment] for key in keys]
        counts = [int(value) for value in exact]
        leftover = int(round(sum(exact))) - sum(counts)  # largest remainder
        for index in sorted(range(len(keys)), key=lambda item: (counts[item] - exact[item], keys[item]))[:leftover]:
            counts[index] += 1
        for key, count in zip(keys, counts):
            count = min(count, usable[key])
            if count == 0:
                continue
            phase = np.random.default_rng(_seed(seed, "phase", key)).random()
            positions = margin + np.floor((np.arange(count) + phase) * usable[key] / count).astype(int)
            plan[key] = sorted(set(positions.tolist()))
    return dict(sorted(plan.items()))


def environment_bytes(catalog: dict[str, TrajectoryEntry], plan: dict[str, list[int]], ratio: float) -> dict[str, float]:
    totals: dict[str, float] = {}
    for key, positions in plan.items():
        entry = catalog[key]
        stored = ratio * sum(entry.frames[position][1].compressed for position in positions)
        totals[entry.environment] = totals.get(entry.environment, 0.0) + stored
    return totals


def assign_shards(sizes: dict[str, float], num_shards: int) -> dict[str, int]:
    """Largest environment first into the lightest shard; a whole environment per shard."""
    loads = [0.0] * num_shards
    assignment: dict[str, int] = {}
    for environment in sorted(sizes, key=lambda item: (-sizes[item], item)):
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        assignment[environment] = shard
        loads[shard] += sizes[environment]
    return assignment


def synthetic_baro(position_ned: np.ndarray, key: str, seed: int = 0) -> tuple[np.ndarray, dict[str, object]]:
    """Clean static pressure (Pa) along a trajectory, from its true height.

    The site altitude and sea-level pressure are drawn per trajectory, so the
    absolute pressure says nothing about which scene it is: only its changes carry
    the vertical motion (about -12 Pa per metre up).
    """
    rng = np.random.default_rng(_seed(seed, "baro", key))
    site_altitude = float(rng.uniform(0.0, 1500.0))
    sea_level = float(np.clip(rng.normal(101325.0, 800.0), 99000.0, 103500.0))
    height = site_altitude - np.asarray(position_ned, dtype=np.float64)[:, 2]
    pressure = sea_level * (1.0 - ISA_LAPSE * height / ISA_T0) ** ISA_EXPONENT
    return pressure, {
        "model": "ISA troposphere: p = p0 * (1 - L*h/T0) ** (g*M/(R*L))",
        "sea_level_pa": sea_level,
        "site_altitude_m": site_altitude,
        "T0_k": ISA_T0,
        "lapse_k_per_m": ISA_LAPSE,
        "exponent": ISA_EXPONENT,
        "height": "site_altitude_m - pos_global[:, 2] (TartanAir pos_global is NED: z points down)",
        "timestamps": "imu_time.npy (100 Hz)",
        "noise": "none: clean signal; sensor noise, bias drift, lag and rate belong to the corruption stage",
    }


_WORKER_ZIPS: dict[str, RemoteZip] = {}


def _frame_job(job: tuple[str, str | None, Member, str, str, tuple[int, int]]) -> tuple[int, int]:
    location, token, member, destination, fmt, frame_size = job
    if os.path.exists(destination):  # finished by an earlier, interrupted run
        return os.path.getsize(destination), 0
    remote = _WORKER_ZIPS.get(location)
    if remote is None:
        remote = _WORKER_ZIPS[location] = RemoteZip(location, token)
    for attempt in range(3):
        try:
            png = remote.member(member)
            break
        except IOError:
            if attempt == 2:
                raise
    stored = encode_frame(png, fmt, frame_size)
    partial = destination + ".part"
    with open(partial, "wb") as handle:
        handle.write(stored)
    os.replace(partial, destination)
    return len(stored), len(png)


def _load_npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def build_trajectory(
    entry: TrajectoryEntry,
    positions: list[int],
    imu_members: dict[str, Member],
    *,
    image_zip: RemoteZip,
    imu_zip: RemoteZip,
    out_root: Path,
    fmt: str,
    pool,  # noqa: ANN001 - multiprocessing pool
    token: str | None = None,
    seed: int = 0,
    frame_size: tuple[int, int] = FRAME_SIZE,
) -> dict[str, object]:
    target = out_root / entry.environment / entry.difficulty / entry.trajectory
    done = target / DONE
    if done.is_file():
        return json.loads(done.read_text())
    # About a dozen small files per trajectory: one after another they cost a dozen
    # round trips, which is more than the trajectory's frames take.
    wanted = [(name, imu_zip, member) for name, member in imu_members.items()]
    if entry.pose is not None:
        wanted.append((f"pose_{CAMERA}.txt", image_zip, entry.pose))
    with ThreadPoolExecutor(len(wanted)) as executor:
        files = dict(zip([name for name, _, _ in wanted],
                         executor.map(lambda item: item[1].member(item[2]), wanted)))
    arrays = {Path(name).stem: _load_npy(data) for name, data in files.items() if name.endswith(".npy")}
    for required in ("acc", "gyro", "imu_time", "cam_time", "pos_global"):
        if required not in arrays:
            return {"trajectory": entry.key, "skipped": f"imu.zip has no {required}.npy"}
    indices = [index for index, _ in entry.frames]
    if indices != list(range(len(indices))) or len(arrays["cam_time"]) != len(indices):
        return {"trajectory": entry.key, "skipped": (
            f"{len(indices)} frames (index {indices[0]}..{indices[-1]}) vs {len(arrays['cam_time'])} camera times")}
    if len(arrays["pos_global"]) != len(arrays["imu_time"]):
        return {"trajectory": entry.key, "skipped": "pos_global and imu_time differ in length"}

    started = time.perf_counter()
    image_dir = target / f"image_{CAMERA}"
    image_dir.mkdir(parents=True, exist_ok=True)
    extension = FORMATS[fmt][0]
    jobs = [
        (image_zip.location, token, entry.frames[position][1],
         str(image_dir / f"{entry.frames[position][0]:06d}_{CAMERA}.{extension}"), fmt, frame_size)
        for position in positions
    ]
    stored_bytes = downloaded = 0
    for stored, fetched in pool.imap_unordered(_frame_job, jobs, chunksize=1):
        stored_bytes += stored
        downloaded += fetched

    kept = np.asarray([entry.frames[position][0] for position in positions], dtype=np.int64)
    staging, final = target / "imu.part", target / "imu"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    for name, array in arrays.items():
        np.save(staging / f"{name}.npy", array[kept] if name == "cam_time" else array)
    if "parameter.yaml" in files:
        (staging / "parameter.yaml").write_bytes(files["parameter.yaml"])
    pressure, baro = synthetic_baro(arrays["pos_global"], entry.key, seed)
    np.save(staging / "baro.npy", pressure)
    (staging / "baro.json").write_text(json.dumps(baro, indent=2))
    np.save(target / "frames.npy", kept)
    if f"pose_{CAMERA}.txt" in files:
        pose = np.loadtxt(io.BytesIO(files[f"pose_{CAMERA}.txt"]), ndmin=2)
        if len(pose) == len(indices):
            np.savetxt(target / f"pose_{CAMERA}.txt", pose[kept])
    shutil.rmtree(final, ignore_errors=True)
    os.replace(staging, final)
    report = {
        "trajectory": entry.key,
        "frames_available": len(indices),
        "frames_kept": len(positions),
        "stored_bytes": stored_bytes,
        "downloaded_bytes": downloaded,
        "imu_rows": int(len(arrays["imu_time"])),
        "seconds": round(time.perf_counter() - started, 1),
    }
    done.write_text(json.dumps(report))
    return report


def _imu_members(imu_zip: RemoteZip) -> dict[str, dict[str, Member]]:
    """{trajectory: {file name: member}} for the IMU files of one environment/difficulty."""
    result: dict[str, dict[str, Member]] = {}
    for member in imu_zip.members():
        parts = member.name.split("/")
        if len(parts) == 5 and parts[3] == "imu" and (parts[4].endswith(".npy") or parts[4] == "parameter.yaml"):
            result.setdefault(parts[2], {})[parts[4]] = member
    return result


def build_shard(
    catalog: dict[str, TrajectoryEntry],
    plan: dict[str, list[int]],
    assignment: dict[str, int],
    *,
    shard: int,
    source: str,
    out_root: str | Path,
    fmt: str = "webp",
    workers: int = 12,
    cap_bytes: float = 19.3e9,
    ratio: float = 1.0,
    token: str | None = None,
    seed: int = 0,
    frame_size: tuple[int, int] = FRAME_SIZE,
    log=print,  # noqa: ANN001
) -> dict[str, object]:
    """Fetch, encode and write this shard's trajectories; safe to rerun after an interruption."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    keys = [key for key in plan if assignment.get(catalog[key].environment) == shard]
    reports: list[dict[str, object]] = []
    written = 0.0
    started = time.perf_counter()
    remotes: dict[str, RemoteZip] = {}  # one signed link per zip, not one per trajectory
    imu_cache: dict[str, dict[str, dict[str, Member]]] = {}

    def remote(relative: str) -> RemoteZip:
        if relative not in remotes:
            remotes[relative] = RemoteZip(_location(source, relative), token)
        return remotes[relative]

    with get_context("fork").Pool(workers) as pool:
        for number, key in enumerate(keys, 1):
            entry = catalog[key]
            estimate = ratio * sum(entry.frames[position][1].compressed for position in plan[key])
            if not (out_root / entry.environment / entry.difficulty / entry.trajectory / DONE).is_file() \
                    and written + estimate > cap_bytes:
                reports.append({"trajectory": key, "skipped": "shard byte cap reached"})
                log(f"[{number}/{len(keys)}] {key}: skipped, the shard would pass {cap_bytes / 1e9:.1f} GB")
                continue
            if entry.imu_zip not in imu_cache:
                imu_cache[entry.imu_zip] = _imu_members(remote(entry.imu_zip))
            members = imu_cache[entry.imu_zip].get(entry.trajectory)
            if not members:
                report: dict[str, object] = {"trajectory": key, "skipped": "no IMU in imu.zip"}
            else:
                report = build_trajectory(
                    entry, plan[key], members, image_zip=remote(entry.image_zip), imu_zip=remote(entry.imu_zip),
                    out_root=out_root, fmt=fmt, pool=pool, token=token, seed=seed, frame_size=frame_size,
                )
            reports.append(report)
            written += float(report.get("stored_bytes", 0))
            elapsed = time.perf_counter() - started
            done_frames = sum(int(item.get("frames_kept", 0)) for item in reports)
            todo = sum(len(plan[item]) for item in keys[number:])
            rate = done_frames / max(elapsed, 1e-9)
            status = report.get("skipped") or f"{report['frames_kept']} frames, {report['stored_bytes'] / 1e6:.0f} MB"
            log(f"[{number}/{len(keys)}] {key}: {status} | shard {written / 1e9:.2f} GB, "
                f"{rate:.1f} frames/s, ~{todo / max(rate, 1e-9) / 60:.0f} min left")
    return {
        "shard": shard,
        "format": fmt,
        "trajectories": len([item for item in reports if "skipped" not in item]),
        "frames": sum(int(item.get("frames_kept", 0)) for item in reports),
        "stored_bytes": int(written),
        "skipped": [item for item in reports if "skipped" in item],
        "reports": reports,
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--source", default="huggingface", help="huggingface, airlab, a URL prefix or a local directory")
    parser.add_argument("--out", required=True, help="shard root; its children are the environments")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--shard-gb", type=float, default=18.0, help="planned size of one shard")
    parser.add_argument("--cap-gb", type=float, default=19.3, help="hard stop for one shard")
    parser.add_argument("--format", choices=sorted(FORMATS), default="webp")
    parser.add_argument("--environments", nargs="*", default=list(ENVIRONMENTS))
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    source = SOURCES.get(args.source, args.source)
    token = os.environ.get("HF_TOKEN")
    catalog, missing = list_catalog(source, args.environments, token=token)
    for problem in missing:
        print("missing:", problem)
    ratio = calibrate(catalog, source, args.format, seed=args.seed, token=token)
    plan = make_plan(catalog, budget_bytes=args.num_shards * args.shard_gb * 1e9, ratio=ratio,
                     margin=args.margin, seed=args.seed)
    assignment = assign_shards(environment_bytes(catalog, plan, ratio), args.num_shards)
    result = build_shard(catalog, plan, assignment, shard=args.shard, source=source, out_root=args.out,
                         fmt=args.format, workers=args.workers, cap_bytes=args.cap_gb * 1e9, ratio=ratio,
                         token=token, seed=args.seed)
    meta = Path(args.out).parent / "build_meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"shard{args.shard}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
