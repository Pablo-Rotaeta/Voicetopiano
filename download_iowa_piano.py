import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import soundfile as sf
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================

BASE_PAGE = "https://theremin.music.uiowa.edu/MISpiano.html"
BASE_URL = "https://theremin.music.uiowa.edu/"

DOWNLOAD_FOLDER = "iowa_piano_aiff"
WAV_FOLDER = "iowa_piano_wav"

# ============================================
# CREATE FOLDERS
# ============================================

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(WAV_FOLDER, exist_ok=True)

# ============================================
# SCRAPE ALL AIFF LINKS
# ============================================

print("Fetching webpage...")
response = requests.get(BASE_PAGE)
soup = BeautifulSoup(response.text, "html.parser")

links = []

for link in soup.find_all("a"):
    href = link.get("href")
    if href and href.lower().endswith(".aiff"):
        full_url = urljoin(BASE_URL, href)
        links.append(full_url)

print(f"Found {len(links)} AIFF files.")

# ============================================
# DOWNLOAD FILES
# ============================================

print("Downloading files...")

for url in tqdm(links):
    filename = os.path.join(DOWNLOAD_FOLDER, os.path.basename(url))

    if not os.path.exists(filename):
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)

# ============================================
# CONVERT TO WAV
# ============================================

print("Converting to WAV...")

for file in tqdm(os.listdir(DOWNLOAD_FOLDER)):
    if file.lower().endswith(".aiff"):
        input_path = os.path.join(DOWNLOAD_FOLDER, file)
        output_name = os.path.splitext(file)[0] + ".wav"
        output_path = os.path.join(WAV_FOLDER, output_name)

        data, samplerate = sf.read(input_path)
        sf.write(output_path, data, samplerate)

print("Done.")
print(f"WAV files saved to: {WAV_FOLDER}")
