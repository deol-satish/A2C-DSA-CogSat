%% P02_GStations
% Ground Stations in Australia
GsLocations = {
    'Sydney',       -33.8688, 151.2093;
    'Melbourne',    -37.8136, 144.9631;
    'Brisbane',     -27.4698, 153.0251;
    'Perth',        -31.9505, 115.8605;
    'Adelaide',     -34.9285, 138.6007;
    'Hobart',       -42.8821, 147.3272;
    'Darwin',       -12.4634, 130.8456;
    'Canberra',     -35.2809, 149.1300;
    'Cairns',       -16.9203, 145.7710;
    'Gold_Coast',   -28.0167, 153.4000;
    'Newcastle',    -32.9283, 151.7817;
    'Geelong',      -38.1499, 144.3617;
    'Sunshine_Coast', -26.6500, 153.0667;
    'Mandurah',     -32.5366, 115.7447;
    'Victor_Harbor', -35.5500, 138.6167;
    'Launceston',   -41.4333, 147.1667;
    'Katherine',    -14.4667, 132.2667;
    'Wollongong',   -34.4244, 150.8939;
    'Townsville',   -19.2500, 146.8167;
    'Toowoomba',    -27.5667, 151.9500
    'Rockhampton',  -23.3750, 150.5117;
    'Ballarat',     -37.5622, 143.8503;
    'Mackay',       -21.1410, 149.1860;
    'Albury',       -36.0737, 146.9135;
    'Shepparton',   -36.3805, 145.3992;
    'Bunbury',      -33.3271, 115.6417;
    'Geraldton',    -28.7682, 114.6144;
    'Alice_Springs',-23.6980, 133.8807;
    'Port_Macquarie',-31.4300, 152.9080;
    'Mildura',      -34.1855, 142.1625};

% NumLeoUser = size(GsLocations,1)/2;     % 10 uesers each with seperate channel (10 Channels)
% NumGeoUser = size(GsLocations,1)/2;     % 10 uesers each with seperate channel (10 Channels)
NumGS = NumLeoUser + NumGeoUser;        % Total ground stations
GSLEOFilter = logical([ones(NumLeoUser,1); zeros(NumGeoUser,1) ]);
GSGEOFilter = logical([zeros(NumLeoUser,1); ones(NumGeoUser,1) ]);

%% Convert to ECEF Cartesian coordinates (X, Y, Z in meters)
[xECEFgs, yECEFgs, zECEFgs] = geodetic2ecef(E, cell2mat(GsLocations(:,2)), cell2mat(GsLocations(:,3)), 0);
% Combine into one matrix for convenience
GSECEF = [xECEFgs, yECEFgs, zECEFgs];  % 20 × 3 matrix