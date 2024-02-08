from getpass import getpass
# Python 3
from urllib.parse import urlparse, parse_qs


import requests

user = input("Username: ")
pw = getpass()
def get_token():
    return requests.post("https://auth.windenergy.dtu.dk/auth/realms/DTU-Wind-Energy/protocol/openid-connect/token",
        data={
            "username": user,
            "password": pw,
            "client_id": "science_gwa_staging",
            "grant_type": "password",
            "client_secret": "38d1bfd0-c068-4d17-8f42-5651c8a2dba3"
            },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    ).json()["access_token"]

with open("download_list.txt") as in_file:
    # loop over each line in the file to get the URL strings
    token = get_token()
    for url_str in in_file:
       

        # Parse the URL string to get arguments for requests
        url_parse = urlparse(url_str)
        url = url_parse._replace(query=None).geturl()
        query = parse_qs(url_parse.query)

        # Get the file name
        fname = f"{query['name'][0]}_py.nc"
        print(f"Downloading file... {fname}")

        # Get content from web
        r = requests.get(url, params=query, headers={"Authorization": f"Bearer {token}"})

        # Write to file
        with open(fname, "wb") as f:
            f.write(r.content)
