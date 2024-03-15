from simple_youtube_api.Channel import Channel
from simple_youtube_api.LocalVideo import LocalVideo

# loggin into the channel
channel = Channel()
channel.login("/SYSTEMS/PROG/SHELL/VIDEO/YOUTUBE-UPLOAD/client_secret.json", "/SYSTEMS/PROG/SHELL/VIDEO/YOUTUBE-UPLOAD/credentials.storage")

# setting up the video that is going to be uploaded
video = LocalVideo(file_path = "/DATA/OUTPUT/VIDEO/202403/15/Mp4_For_OUTPUT_gk2a_ami_le1b_rgb-true_fd010ge_%Y%m%d%H%M.srv.png_202403150000_202403140000-202403150000_KST.mp4")

# setting snippet
video.set_title("""
[Daily] 3월 14, 2024 GK2A Satellite RGB True | Full Disk | 20240314
""")

video.set_description("""
[Characteristics]
   - Video editing of GK2A Satellite RGB True Image

[Materials]
   - Satellite Name: GK2A (GEO-KOMPSAT 2A, Cheollian 2A)
   - Sensor Name: AMI (Advanced Meteorological Imager)
   - Level: Level 1B 
   - Type: RGB True Image
   - Area: Full Disk
   - Resolution: 0.5, 1.0, 2.0 km
   - Projection: LCC (Lambert Conformal Conic) / GEOS (GEOstationary Satellite)
   - Period: 3월 14, 2024
   - Source: National Meteorological Satellite Center (http://datasvc.nmsc.kma.go.kr/datasvc/html/main/main.do)
   - Related blog: https://shlee1990.tistory.com

[Donation Link] Toonation
   - https://toon.at/donate/637793859903715183

#Shorts
""")

# video.set_tags(["this", "tag"])
# video.set_category("gaming")
# video.set_default_language("en-US")

# setting status
video.set_embeddable(True)
# video.set_license("creativeCommon")
# video.set_privacy_status("private")
video.set_public_stats_viewable(True)

# setting thumbnail
# video.set_thumbnail_path('test_thumb.png')

# uploading video and printing the results
video = channel.upload_video(video)
print(video.id)
print(video)

# liking video
video.like()
