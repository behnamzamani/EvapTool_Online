EvapTool ver. 2.0
Copyright (C) 2026 Behnam Zamani

This program is licensed under the GNU Affero General Public License v3.0.
See the LICENSE file for details.

To use the program, run it in python and enter the lat/lon coordinates of your desired location (WGS84) in the prompt (example: 52.3759,9.7320).
The program downloads the recent daily meteorological data from closest DWD weather stations to the desired location. In case no data available for any data type, the program searches other nearby stations for that data type and merges to the originally-downloaded data (closest station).
Data are downloaded automatically from DWD FTP server.
Daily evaporation is computed by Penman and Priestley-Taylor approaches.
water balance calculated as difference between precipitation and evaporation.
