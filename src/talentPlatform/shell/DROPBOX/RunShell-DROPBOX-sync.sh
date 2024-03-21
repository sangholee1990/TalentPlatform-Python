#!/bin/bash

ulimit -s unlimited

rclone sync "dropbox:/솔라미" "shlee:/솔라미" &
rclone sync "dropbox:/비디와이드" "shlee:/비디와이드" &
