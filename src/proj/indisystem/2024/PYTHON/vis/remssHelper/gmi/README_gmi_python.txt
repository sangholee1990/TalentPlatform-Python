This directory contains the python code for reading GMI (F35) data.
Four files are needed: bytemaps.py, gmi_daily_v8.py, gmi_averaged_v8.py and example_usage.py

In order to test the programs, you need to 
1) download the appropriate verify.txt file located in the GMI/support_v08.2/verify/ directory
2) download the 4 files required to test:
For gmi:
Time              Directory                        File
daily passes      gmi/bmaps_v0N.N/y2014/m05/       F35_20140519vN.N.gz
3-day mean        gmi/bmaps_v0N.N/y2014/m05/       F35_20140519vN.N_d3d.gz
weekly mean       gmi/bmaps_v0N.N/weeks/           F35_20140524vN.N.gz
monthly mean      gmi/bmaps_v0N.N/y2014/m05/       F35_201405vN.N.gz

Each of these files contains GMI data for:
the day May 19th, 2014 (daily)
the days May 17th, 18th and 19th, 2014 (3-day mean)
May 18th (Sunday) to May 24th (Saturday), 2015 (weekly mean) 
or the month of May 2014 (monthly mean) 

3) place these files from step 1 and 2 in the same directory as the programs   
   
First run the daily and averaged routines to be sure they execute correctly.  You will get a 'verification failed' message if there is a problem.  If they work correctly, the message 'all tests completed successfully' will be displayed.

After confirming the routines work, use the example_usage.py routine as your base program and adapt to your needs.  This code shows you how to call the needed subroutines and diplays an example image.  Once you change the program, make sure to run it on the test files and check that the results match those listed in the verify_gmi_v8.2.txt file.

If you have any questions regarding these programs 
or the RSS binary data files, contact RSS support:
http://www.remss.com/support