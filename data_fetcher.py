import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

API_KEY = "your_api_key"  # Replace with your actual token
TRAIN_PATH = "data/raw/train(1).csv"
IMAGE_DIR = "data/images/"
ZOOM = 18           
SIZE = "600x600"    
MAX_WORKERS = 10    

def download_one_image(row):
    """Function to download a single image based on a dataframe row."""
    house_id = int(row['id'])
    lat = row['lat']
    lon = row['long']
    
    save_path = os.path.join(IMAGE_DIR, f"{house_id}.jpg")
    
    if os.path.exists(save_path):
        return "skipped"

    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{ZOOM},0/{SIZE}?access_token={API_KEY}"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return "success"
        elif response.status_code == 429:
            return "rate_limited"
        else:
            return f"error_{response.status_code}"
    except Exception as e:
        return f"failed_{str(e)}"

def main():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    df = pd.read_csv(TRAIN_PATH)
    print(f"Total properties to process: {len(df)}")

    results = {"success": 0, "skipped": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {executor.submit(download_one_image, row): row['id'] for _, row in df.iterrows()}
        
        for future in tqdm(as_completed(future_to_id), total=len(df), desc="Fetching Sat-Images"):
            res = future.result()
            if res == "success":
                results["success"] += 1
            elif res == "skipped":
                results["skipped"] += 1
            else:
                results["error"] += 1
                if "rate_limited" in res:
                    time.sleep(2)

    print("\n--- Download Task Complete ---")
    print(f"Successfully downloaded: {results['success']}")
    print(f"Already existed (skipped): {results['skipped']}")
    print(f"Errors/Failed: {results['error']}")

if __name__ == "__main__":
    main()