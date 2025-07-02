import requests
import time
import threading

def poll_status(request_id, stop_event):
    last_log_len = 0
    while not stop_event.is_set():
        try:
            s = requests.get("http://127.0.0.1:5000/get_status", params={"request_id": request_id})
            log = s.json()["log"]
            for line in log[last_log_len:]:
                print(line)
            last_log_len = len(log)
        except Exception as e:
            print("Error getting status:", e)
        time.sleep(0.1)

while True:
    prompt = input("\nYou: ")
    if prompt.strip() == "/file":
        filename = input("Enter filename: ")
        with open(filename, "r") as f:
            prompt = f.read()
    if not prompt.strip():
        print("Exiting chat.")
        break

    resp = requests.post("http://127.0.0.1:5000/submit_prompt", json={"prompt": prompt})
    data = resp.json()
    request_id = data["request_id"]

    stop_event = threading.Event()
    status_thread = threading.Thread(target=poll_status, args=(request_id, stop_event))
    status_thread.start()

    # Poll for result
    while True:
        try:
            r = requests.get("http://127.0.0.1:5000/get_result", params={"request_id": request_id})
            result = r.json()
            if result["status"] == "done":
                stop_event.set()
                status_thread.join()
                print("LLM:", result["output"])
                break
        except Exception as e:
            print("Error getting result:", e)
        time.sleep(1)