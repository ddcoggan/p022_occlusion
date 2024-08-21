%% Summary:
% Texture- & Shape-biases (Gerihos et al., 2018)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;
%addpath /home/tonglab/Documents/Project/MATLAB/utils
% cmap = cbrewer('qual','Set1',9); 
cmap = zeros(2,3);
cmap(1,:) = [255 190 10]/255; % unoccluded
cmap(2,:) = [230 50 75]/255; % occluded

categories = {'airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', ...
    'chair', 'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck'};


%% CNN 
cnn_path = {'DNN/data/alexnet/imagenet1000/fromPretrained/naturalTextured/texture_shape_bias.csv'};
cnn_model_decision = zeros(numel(cnn_path), 16, 80); % 0, wrong; 1, texture; 2, shape; 3, both correct
num_check = ones(numel(cnn_path), 16);

for c = 1:numel(cnn_path)
    decision_matrix = readtable(cnn_path{c});
    
    for row = 1:size(decision_matrix,1)
        answer = decision_matrix{row,5}{1};
        disp(answer)
        tmp = strsplit(decision_matrix{row,8}{1}, {'_', '-', '.', '/'});
        shape_category = regexprep(tmp{end-2}, '\d+(?:_(?=\d))?', '');
        texture_category = regexprep(tmp{end-1}, '\d+(?:_(?=\d))?', '');        
        shape_category_id = find(contains(categories, shape_category));
        
        if ~strcmp(answer, texture_category) && ~strcmp(answer, shape_category)
            cnn_model_decision(c, shape_category_id, num_check(c, shape_category_id)) = 0;
        elseif strcmp(answer, texture_category) && ~strcmp(answer, shape_category)
            cnn_model_decision(c, shape_category_id, num_check(c, shape_category_id)) = 1;
        elseif ~strcmp(answer, texture_category) && strcmp(answer, shape_category)
            cnn_model_decision(c, shape_category_id, num_check(c, shape_category_id)) = 2;
        elseif strcmp(answer, texture_category) && strcmp(answer, shape_category)
            cnn_model_decision(c, shape_category_id, num_check(c, shape_category_id)) = 3;
        end        
        num_check(c, shape_category_id) = num_check(c, shape_category_id) + 1;
    end
end

%% mr-CNN
mrcnn_path = {'DNN/data/alexnet/imagenet1000/fromPretrained/naturalTextured/texture_shape_bias.csv'};

mrcnn_model_decision = zeros(numel(mrcnn_path), 16, 80); % 0, wrong; 1, texture; 2, shape; 3, both correct
num_check = ones(numel(mrcnn_path), 16);

for c = 1:numel(mrcnn_path)
    decision_matrix = readtable(mrcnn_path{c});
    
    for row = 1:size(decision_matrix,1)
        answer = decision_matrix{row,5}{1};
        tmp = strsplit(decision_matrix{row,8}{1}, {'_', '-', '.', '/'});
        shape_category = regexprep(tmp{end-2}, '\d+(?:_(?=\d))?', '');
        texture_category = regexprep(tmp{end-1}, '\d+(?:_(?=\d))?', '');        
        shape_category_id = find(contains(categories, shape_category));
        
        if ~strcmp(answer, texture_category) && ~strcmp(answer, shape_category)
            mrcnn_model_decision(c, shape_category_id, num_check(c, shape_category_id)) = 0;
        elseif strcmp(answer, texture_category) && ~strcmp(answer, shape_category)
            mrcnn_model_decision(c, shape_category_id, num_check(c, shape_category_id)) = 1;
        elseif ~strcmp(answer, texture_category) && strcmp(answer, shape_category)
            mrcnn_model_decision(c, shape_category_id, num_check(c, shape_category_id)) = 2;
        elseif strcmp(answer, texture_category) && strcmp(answer, shape_category)
            mrcnn_model_decision(c, shape_category_id, num_check(c, shape_category_id)) = 3;
        end        
        num_check(c, shape_category_id) = num_check(c, shape_category_id) + 1;
    end
end

%% Calculate bias-to-texture
cnn_bias_to_texture = zeros(numel(cnn_path), 16);
for c = 1:numel(cnn_path)
    for i = 1:16
        cnn_bias_to_texture(c, i) = numel(find(cnn_model_decision(c, i, :)==1)) / (numel(find(cnn_model_decision(c, i, :)==1)) + numel(find(cnn_model_decision(c, i, :)==2)));
    end    
end

mrcnn_bias_to_texture = zeros(numel(mrcnn_path), 16);
for c = 1:numel(mrcnn_path)
    for i = 1:16
        mrcnn_bias_to_texture(c, i) = numel(find(mrcnn_model_decision(c, i, :)==1)) / (numel(find(mrcnn_model_decision(c, i, :)==1)) + numel(find(mrcnn_model_decision(c, i, :)==2)));
    end    
end

%% Visualize

f=figure('units','normalized','outerposition',[0 0 0.55 0.55]);

cnn_bias_to_texture_average = mean(cnn_bias_to_texture, 1);
[sort_val, sort_idx] = sort(cnn_bias_to_texture_average);

errorbar(mean(cnn_bias_to_texture(:,sort_idx),1), 1:16, std(cnn_bias_to_texture(:,sort_idx),[],1), '.', 'horizontal', 'linewidth',1, 'color',cmap(1,:), 'marker','none', 'capsize',0); hold on;
errorbar(mean(mrcnn_bias_to_texture(:,sort_idx),1), 1:16, std(mrcnn_bias_to_texture(:,sort_idx),[],1), '.', 'horizontal', 'linewidth',1, 'color',cmap(2,:), 'marker','none', 'capsize',0); hold on;

p2 = plot(1-mean(cnn_bias_to_texture(:,sort_idx),1), 1:16, '^', 'markersize', 9, 'linewidth',1, 'color',cmap(1,:), 'markerfacecolor',cmap(1,:)); hold on;
p3 = plot(1-mean(mrcnn_bias_to_texture(:,sort_idx),1), 1:16, 'o', 'markersize', 9, 'linewidth',1, 'color',cmap(2,:), 'markerfacecolor',cmap(2,:)); hold on;

legend([p2 p3], {'unoccluded-trained CNN', 'naturalTypes-trained CNN'}, 'box','off', 'fontsize',15, 'orientation','horizontal');

% set(gca, 'linewidth',1.5, 'fontsize',14, 'ygrid','on', 'GridLineStyle','-', 'GridAlpha',0.1); axis square;

axis tight; xlabel('Shape bias'); 
xlim([-.1, 1.1]); ylim([0,17]); yticks(1:16); yticklabels(categories(sort_idx));

ax = gca(); 
ax.TickDir = 'out'; ax.LineWidth = 1; ax.FontSize = 15; ax.YGrid = 'on'; ax.GridAlpha = 0.1;
addlistener ( ax, 'MarkedClean', @(obj,event) change_ticks(ax) );
    
drawnow();

saveas(f,'DNN/analysis/results/shapeTextureBias/alexnet/imagenet1000/Geirhos/shapeTextureBias.png')
close f

