#!/bin/bash
# Store username and password
read -p "Username: " user
read -sp "Password: " pass

while read url || [[ -n $url ]];
do
    out_f="`echo $url | cut -d '=' -f 2 | cut -d '&' -f 1`.nc"
    echo "Downloading file... ${out_f}"

    # Get token requires jq
    export TOKEN=$(curl -s -X POST "https://auth.windenergy.dtu.dk/auth/realms/DTU-Wind-Energy/protocol/openid-connect/token" -H "Content-Type: application/x-www-form-urlencoded" -d "username=$user" -d "password=$pass" -d "client_id=science_gwa_staging" -d "grant_type=password" -d "client_secret=38d1bfd0-c068-4d17-8f42-5651c8a2dba3" | jq -r ".access_token")

    # Download NetCDF
    curl -s -X GET ${url} -H "Authorization: Bearer $TOKEN" --output ${out_f}
done < download_list.txt

