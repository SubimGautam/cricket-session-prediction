import requests

def get_all_test_matches(api_key):
    url = "https://api.cricapi.com/v1/matches"
    offset = 0
    all_test_matches = []

    while True:
        params = {
            "apikey": api_key,
            "offset": offset
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print("❌ Request error:", e)
            break

        if data.get("status") != "success":
            print("❌ API error:", data)
            break

        matches = data.get("data", [])
        if not matches:
            break

        for match in matches:
            if match.get("matchType") == "test":
                all_test_matches.append({
                    "id": match.get("id"),
                    "name": match.get("name"),
                    "date": match.get("date")
                })

        offset += 25

    return all_test_matches