This directory contains the python code for reading ASCAT data.
Four files are needed: bytemaps.py, ascat_daily.py, ascat_averaged.py and example_usage_v02.1.py

In order to test the programs, you need to 
1) download the appropriate verify.txt file located in the scatterometer_bmap_support/ascat_verify directory
2) download the 4 files required to test:
For ascat:
   daily file	ascat_20071022_v02.1.gz
   3-day file	ascat_20071022_v02.1_3day.gz
   weekly file	ascat_20071027_v02.1.gz
   monthly file ascat_200710_v02.1.gz
3) place these files from step 1 and 2 in the same directory as the programs   
   
First run the daily and averaged routines to be sure they execute correctly.  You will get a 'verification failed' message if there is a problem.  If they work correctly, the message 'all tests completed successfully' will be displayed.

After confirming the routines work, use the example_usage.py routine as your base program and adapt to your needs.  This code shows you how to call the needed subroutines and displays an example image.  Once you change the program, make sure to run it on the test files and check that the results match those listed in the ascat_verify_v02.1.txt file.

If you have any questions regarding these programs 
or the RSS binary data files, contact RSS support:
http://www.remss.com/support