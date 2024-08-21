%function voxelwiseMaster_4_eccenBins(expt,subjNums,ROI,cutoff,plotInfo)


clear all;% close all;
expt = 'LGNedgeAll'; subjNums = 1:6; ROIs = {'LGN' 'V1' 'V2' 'V3'}; cutoff = 0;

load(['exptParams/' expt]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% what to plot?                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eccenParams.fitName = 'sG_multiBars';
eccenParams.subjs = exptSubjs(subjNums);%
if cutoff > 1 eccenParams.limitBy = 'perc'; else eccenParams.limitBy = 'R2'; end
eccenParams.minEccen= .5;
eccenParams.minSize = 0.3;
eccenParams.edgeEccen = stimEdge;
eccenParams.ppd = 45.73;
eccenParams.whichSize = 'FWHM'; % 'FWHM' or 'SD'
fontSize = 24; titleSize = 36;
sameColor = [25 25 112]/255;
condColors = {[76 0 153]/255 [255 128 0]/255 [204 0 0]/255 [0 128 0]/255};
hems = {'lh' 'rh' };%

figure;set(gcf,'color',[1 1 1],'Units', 'Normalized', 'OuterPosition', [.1 .1 .2 .4]); set(gca,'FontSize',16);

for r = 1:length(ROIs)
    eccenParams.ROI = ROIs{r};
    voxCount = 0;
    voxInfo = struct;
    for s = 1:length(eccenParams.subjs)
        subj = eccenParams.subjs{s}
        
        % now we load in the data from both hemispheres, and threshold across
        % them, not within them (controls for somewhat large fit quality
        % differences across hems in some subjects
        for n = 1:length(hems)
            
            clear PRFfits; clear GLM;
            
            fitDir = [fMRIdir '/' expt '/' subj '/matlabAnalysis/modelFit/' eccenParams.fitName '/'];
            fitFile = [hems{n} eccenParams.ROI '/fits_' eccenParams.fitName '_' hems{n} eccenParams.ROI '.mat'];
            load([fitDir fitFile]);
            
            dataDir = [fMRIdir '/' expt '/' subj '/matlabAnalysis/GLMresults/'];
            dataFile = ['voxelWise_GLM_' hems{n} eccenParams.ROI '.mat'];
            load([dataDir dataFile]);
            
            if n ==1
                bilatFits = PRFfits;
                bilatGLM = GLM;
            elseif n ==2
                bilatFits = [bilatFits PRFfits];
                bilatGLM = [bilatGLM GLM];
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % which voxels are we plotting? trim edge values, R2 cutoff                    %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if strcmp(eccenParams.limitBy,'perc')==1
            [plotVox,eccenParams.R2cutoff] = analysisMaster_pickPRFs(bilatFits,cutoff,'perc');
            eccenParams.percCutoff=cutoff;
        else [plotVox,~] = analysisMaster_pickPRFs(bilatFits,cutoff,'R2');
            eccenParams.percCutoff=[]; eccenParams.R2cutoff = cutoff;
        end
        
        clear PRFfits; clear GLM;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % proceed with good fits!                                                      %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        for v = 1:length(plotVox)
            if bilatFits(plotVox(v)).rad/eccenParams.ppd > eccenParams.minEccen && bilatFits(plotVox(v)).SD/eccenParams.ppd > eccenParams.minSize
                voxCount = voxCount+1; % across subjects
                voxInfo.eccens(voxCount) = bilatFits(plotVox(v)).rad/eccenParams.ppd;
                voxInfo.Xs(voxCount) = bilatFits(plotVox(v)).XY(1)./eccenParams.ppd;
                voxInfo.Ys(voxCount) = bilatFits(plotVox(v)).XY(2)./eccenParams.ppd;
                voxInfo.SDs(voxCount) = bilatFits(plotVox(v)).SD/eccenParams.ppd;
                voxInfo.FWHMs(voxCount) = bilatFits(plotVox(v)).SD/eccenParams.ppd * 1.1774;
                voxInfo.R2(voxCount) = bilatFits(plotVox(v)).r2;
                voxInfo.betas(voxCount,:) = bilatGLM(plotVox(v)).betas;
                eval(['voxInfo.size(voxCount) = voxInfo.' eccenParams.whichSize 's(voxCount);']);
                
                % distance from edge
                if voxInfo.eccens(voxCount) < eccenParams.edgeEccen
                    voxInfo.distances(voxCount) = eccenParams.edgeEccen - (voxInfo.eccens(voxCount) + voxInfo.size(voxCount));
                else  voxInfo.distances(voxCount) =  (voxInfo.eccens(voxCount) - voxInfo.size(voxCount)) - eccenParams.edgeEccen; end
                
                voxInfo.voxInd(voxCount) = bilatFits(plotVox(v)).volumeInd;
            end
        end
    end
    
    % size by eccen
    
    scatter([voxInfo.eccens],[voxInfo.size],30,condColors{r});l = lsline; set(l,'LineWidth',3); hold on;
    xlabel('Radius (dva)'); ylabel(['Center ' eccenParams.whichSize ' (dva)']); title('Center SD by Eccentricity');%ylim([0 2.5]);
    axis square;
    
    xlabel('Max Binned Eccentricity (dva)'); ylabel(['Width of PRF (' eccenParams.whichSize ' in dva)']);
    title('pRF Sizes Accross Eccen');
    
    eval([eccenParams.ROI '= voxInfo;']);
    eval([eccenParams.ROI '.params = eccenParams;']);
end

text = [];
for r = 1:length(ROIs)
    text = [text ROIs{r} ' '];
end
if length(eccenParams.subjs)>1
    if cutoff > 1
        suptitle([text 'across N = ' num2str(length(eccenParams.subjs)) ' (' num2str(eccenParams.percCutoff) 'th percentile)']);
    else
        suptitle([text 'across N = ' num2str(length(eccenParams.subjs)) ' (R2 cutoff = ' num2str(eccenParams.R2cutoff) ')']);
    end
else
    if cutoff > 1
        suptitle([text ', Subj = ' eccenParams.subjs{1} ' (' num2str(eccenParams.percCutoff) 'th percentile)']);
    else
        suptitle([text ', Subj = ' eccenParams.subjs{1} ' (R2 cutoff = ' num2str(eccenParams.R2cutoff) ')']);
    end
end
