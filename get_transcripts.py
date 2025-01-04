import os
import sys
import time
import json
import subprocess
import numpy as np
from datetime import timedelta
from transformers import pipeline


def check_transcript(out):
    n = len(out['chunks'])
    for i, line in enumerate(out['chunks']):
        begin = line['timestamp'][0]
        end = line['timestamp'][1]
        if i == (n-1):
            if begin is None:
                previous_line = out['chunks'][i - 1]
                begin = previous_line['timestamp'][1]
                line['timestamp'] = (begin, end)
            if end is None:
                command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path_mp4]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                result = json.loads(result.stdout)
                duration = float(result["format"]["duration"])
                end = duration
                line['timestamp'] = (begin, end)
            out['chunks'][i] = line
        if (begin is None) or (end is None):
            return False
    return out

def convert_srt_time(t):
    s = str(timedelta(seconds=t))
    if int(t) != float(t):
        s = s.replace('.', ',')[:-3]
    else:
        s = s + ',000'
    s = s.rjust(12, '0')
    return s

def convert_srt(out):
    srt = []
    for i, line in enumerate(out['chunks']):
        begin = convert_srt_time(line['timestamp'][0])
        end = convert_srt_time(line['timestamp'][1])
        msg = str(i+1) + '\n'
        msg = msg + begin + ' --> ' + end + '\n'
        msg = msg + line['text'].strip() + '\n\n'
        srt.append(msg)
    return srt

def get_candidate_videos(root_path):
    files = []
    for (dir_path, dir_names, file_names) in os.walk(root_path):
        if file_names:
            files_mp4 = [x for x in file_names if x.endswith('.mp4')]
            files_srt = [x for x in file_names if x.endswith('.srt')]
            for file in files_srt:
                mp4 = file[:-4] + '.mp4'
                if mp4 in files_mp4:
                    files_mp4.remove(mp4)
            path_files = [dir_path + '/' + x for x in files_mp4]
            files.extend(path_files)
    return files

def load_completion_time(root_path):
    json_file = root_path + '/time_complete.json'
    with open(json_file, 'r') as f:
        time_complete = json.load(f)
    return time_complete

def save_completion_time(root_path, path_mp4, time_complete, duration):
    json_file = root_path + '/time_complete.json'
    if path_mp4 not in time_complete:
        time_complete[path_mp4] = duration
        with open(json_file, "w") as f:
            json.dump(time_complete, f)
    return time_complete

def save_transcript(out):
    out = check_transcript(out)
    if out:
        path_srt = path_mp4[:-4] + '.srt'
        with open(path_srt, 'w', encoding='utf-8') as f:
            scripts = convert_srt(out)
            f.writelines(scripts)
        return True
    else:
        return False

def estimate_time(i, files, time_complete):
    n = len(files) - i
    times = list(time_complete.values())
    m = sum(times) / len(times)
    total = round(m * n)
    hms = str(timedelta(seconds=total)).rjust(8, '0')
    time = f"remaining estimated time for completion is: {hms}"
    return time

def time_completion_stats(time_complete):
    times = list(time_complete.values())
    mean = np.mean(times).round(2)
    std = np.std(times).round(2)
    median = np.median(times).round(2)
    s = f"average time per video for all files is: median={median} | mean={mean} | sd={std}"
    return s



if __name__ == "__main__":

    run = True

    if run:
        n = int(sys.argv[1])
        root_path = sys.argv[2]

        print("loading transcription model ...")
        transcriber = pipeline(model="openai/whisper-large-v2", device=0)
        print("transcription model loaded")
        
        time_complete = load_completion_time(root_path)
        candidate_files = get_candidate_videos(root_path)
        files = candidate_files[:n]

        start = time.time()
        N = len(candidate_files)
        print(f"transcribing {n} out of {N} files without transcripts\n")

        for i, path_mp4 in enumerate(files):
            print(estimate_time(i, files, time_complete))
            print(f"{i+1}/{n} - getting transcipts for: {path_mp4}")
            s = time.time()
            out = transcriber(path_mp4, chunk_length_s=30, return_timestamps=True)
            e = time.time()
            d = e - s
            print("file completed and execution time:", d, "seconds" + '\n')

            # save completion times
            time_complete = save_completion_time(root_path, path_mp4, time_complete, d)
            # save transcript
            save_flag = save_transcript(out)

        end = time.time()
        dur = round(end - start)
        print(f"Total execution time for {n} files: ", str(timedelta(seconds=dur)).rstrip('0'))
        print(time_completion_stats(time_complete))

