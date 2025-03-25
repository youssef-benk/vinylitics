import os

def get_mp3(song_name):
    """
    Downloads the first found YouTube result as an MP3 file
    https://pypi.org/project/yt-dlp/
    """
    import yt_dlp
    params = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            # 'preferredquality': '192', # Uncomment to set quality by default it is best
        }],
    }
    with yt_dlp.YoutubeDL(params) as ydl:
        # Use extract_info to download and retrieve file metadata
        info = ydl.extract_info(f"ytsearch1:{song_name}", download=True)
        if 'entries' in info:
            info = info['entries'][0]
        filename = info.get('_filename')
        if not filename:
            title = info.get('title', 'downloaded')
            filename = f"{title}.mp3"
        # Convert filename to an absolute path
        abs_path = os.path.abspath(filename)
        print(f"Downloaded MP3: {abs_path}")
        return abs_path

# Test the function
if __name__ == "__main__":
    import time
    start_time = time.time()

    get_mp3("Blinding lights The Weeknd (Official Audio)") # test to only get mp3
    # get_mp3_features("Blinding lights The Weeknd") # test to get features

    elapsed = time.time() - start_time
    print(f"Time elapsed: {elapsed:.2f} seconds")
