import requests
import time
import pandas as pd

API_SUBMIT_URL = "http://127.0.0.1:5000/submit_prompt"
API_STATUS_URL = "http://127.0.0.1:5000/get_status"
API_RESULT_URL = "http://127.0.0.1:5000/get_result"

def get_llm_response(prompt):
    llm_prompt = f"Answer only 'Yes' or 'No' to the following question:\n{prompt}"
    resp = requests.post(API_SUBMIT_URL, json={"prompt": llm_prompt})

    print("Response status code:", resp.status_code)
    #print("Response text:", resp.text)  # add this line to see raw response

    try:
        data = resp.json()
    except Exception as e:
        print("Failed to parse JSON:", e)
        return "N"  # or handle error appropriately

    request_id = data["request_id"]

    while True:
        try:
            r = requests.get(API_RESULT_URL, params={"request_id": request_id})
            result = r.json()
            if result["status"] == "done":
                output = result["output"].strip().lower()
                if output.startswith("yes"):
                    return "Y"
                elif output.startswith("no"):
                    return "N"
                else:
                    return "N"
            time.sleep(1)
        except Exception as e:
            print("Error getting result:", e)
            time.sleep(1)


def main():
    df = pd.read_csv("classified_texts.csv")

    model_outputs = []
    for idx, row in df.iterrows():
        prompt = row["inner_text"]
        print(f"Processing row {idx}: {prompt[:50]}...")
        llm_answer = get_llm_response(prompt)
        print(f"Model answer: {llm_answer}")
        model_outputs.append(llm_answer)

    df["model_electric_car"] = model_outputs

    # Create comparison report
    df["correct"] = df["electric_car"].str.upper() == df["model_electric_car"]
    df[["inner_text", "electric_car", "model_electric_car", "correct"]].to_csv("comparison_report.csv", index=False)

    print("Done! Outputs saved to comparison_report.csv")

if __name__ == "__main__":
    main()
