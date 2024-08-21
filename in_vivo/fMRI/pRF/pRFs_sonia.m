group = load('group_V1_sortedVox.mat');
data = abs(group.group(4).conds(1).voxXYSD);
ecc = [];
for v = 1:numel(data(:,1))
    ecc = [ecc, sqrt(data(v,1) ^ 2 + data(v,2) ^ 2)];
end
size = data(:,3)';

% pixels to degrees
ppd = 412 / 9;
ecc_deg = ecc / ppd;
size_deg = size / ppd;

% plot eccen v SD
scatter(ecc_deg, size_deg, 'blue')
X = [ones(length(x), 1) ecc_deg'];
slope_sd = X\size_deg';
slope_points = X*slope_sd;
hold on
plot(ecc_deg, slope_points, 'red')
xlabel('eccentricity (degrees)')
ylabel('pRF size (SD)')
title(sprintf('intercept = %f, slope = %f', slope_sd))
close all

% SD to FWHM
size_deg_fwhm = size_deg / 2.355;
scatter(ecc_deg, size_deg_fwhm, 'blue')
slope_fwhm = X\size_deg_fwhm';
slope_points = X*slope_fwhm;
hold on
plot(ecc_deg, slope_points, 'red')
xlabel('eccentricity (degrees)')
ylabel('pRF size (FWHM)')
title(sprintf('intercept = %f, slope = %f', slope_fwhm))
close all

% allow for voxel size differences
my_slope = slope_fwhm * ((2*2) / (3*3));
sizes = [1, 1; 1, 2; 1, 3; 1, 4] * my_slope;

